// Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
// SPDX-License-Identifier: AGPL-3.0-only

/*
 * bench_rknn_perf.cpp — RKNN INT8 模型评测：纯 NPU + 端到端 + ROI/SIMD 后处理
 *
 * 用法:
 *   ./bench_rknn_perf --model X.rknn [--warmup 10] [--runs 200]
 *                     [--core 0|1|2|all] [--postproc
 * predist|predfl|seg_predist|seg_predfl|none]
 *                     [--conf-thr 0.25] [--sram off|private|shared] [--json
 * OUT.json]
 *   ./bench_rknn_perf --model X.rknn --fps-workers 3 --fps-seconds 10
 *                     [--fps-core-map 0,1,2] [--postproc ...] [--sram ...]
 * [--json OUT.json]
 *
 * 输出:
 *   - 纯 NPU 时间       RKNN_QUERY_PERF_RUN
 *   - 端到端时间        输入同步 + rknn_run + 按需输出同步 + 后处理
 *   - 后处理时间        若 --postproc 非 none，则使用 NEON
 * 加速候选筛选、dequant 与 mask
 *   - 离线 FPS          多 RKNN context 并发循环，不接摄像头/ROS/真实预处理
 *
 * 后处理路线 (NEON best-class 阈值扫描 + candidate-first decode):
 *   predist (reg_max=1):
 *     scores: [1,nc,A] i8  → INT8 best-class 阈值扫描
 *     boxes:  [1,4,A]  i8  → 仅对候选 anchor dequant ltrb
 *   predfl (reg_max=16):
 *     scores: [1,nc,A] i8  → INT8 best-class 阈值扫描
 *     boxes:  [1,64,A] i8  → 仅对候选 anchor softmax(16) → · arange(16) → ltrb
 *   seg_predist:
 *     score_sum 先筛 anchor，再扫描 survivor 的分类并按需处理 mask
 *   seg_predfl:
 *     score_sum 先筛 anchor，再扫描 survivor 的分类/DFL 并按需处理 mask
 */

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <getopt.h>
#include <pthread.h>
#include <time.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "postprocess/five_output_runtime.hpp"
#include "postprocess/roi_mask_decoder.hpp"
#include "postprocess/seg_class_selector.hpp"
#include "rknn_api.h"

#if !defined(__aarch64__) || !defined(__ARM_NEON)
#error "bench_rknn_perf requires AArch64 NEON"
#endif

#include <arm_neon.h>

#define HAVE_NEON 1

static constexpr int kMaxNmsCandidates = 30000;

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ---------- 后处理实现 ---------- */

/* 每线程 buffer 池，避免每次 malloc；FPS 模式下各 worker 独立使用。 */
typedef struct {
  float x1, y1, x2, y2;
  float score;
  int cls;
  int anchor;
} candidate_t;

static thread_local float *g_mb = nullptr, *g_pb = nullptr;
static thread_local float *g_roi_storage = nullptr, *g_resize_storage = nullptr;
static thread_local float *g_dfl_logits = nullptr;
static thread_local uint8_t *g_binary_storage = nullptr;
static thread_local uint8_t *g_suppressed = nullptr;
static thread_local candidate_t *g_candidates = nullptr;
static thread_local paddleyolo_rknn::postprocess::ClassSeed *g_class_seeds =
    nullptr;
static thread_local int *g_score_sum_survivors = nullptr;
static thread_local int *g_mask_candidate_indices = nullptr;
static thread_local int g_lm = 0, g_lp = 0;
static thread_local int g_lroi = 0, g_lresize = 0, g_lbinary = 0;
static thread_local int g_ldfl_logits = 0;
static thread_local int g_lsuppressed = 0;
static thread_local int g_lcandidates = 0;
static thread_local int g_lclass_seeds = 0;
static thread_local int g_lscore_sum_survivors = 0;
static thread_local int g_lmask_candidate_indices = 0;
static thread_local int g_last_kept = 0;
static thread_local int g_last_candidates = 0;
static thread_local int g_last_sum_scanned = 0;
static thread_local int g_last_score_sum_applied = 0;
static thread_local int g_last_class_anchors = 0;
static thread_local long g_last_class_values = 0;
static thread_local long g_last_nms_pairs = 0;
static thread_local long g_last_mask_pixels = 0;
static thread_local long g_last_mask_active = 0;
static thread_local long g_last_proto_roi_area = 0;
static thread_local uint64_t g_last_mask_hash = 0;
static thread_local const char *g_last_mask_mode = "none";
static thread_local const char *g_last_fetch_outcome = "none";
static thread_local bool g_last_staged_failed = false;
static thread_local size_t g_last_sync_bytes = 0;
static thread_local unsigned int g_last_ready_mask = 0;
static thread_local double g_last_sync_ms = 0.0;
static thread_local double g_last_mask_verify_ms = 0.0;
static thread_local int g_input_h = 640, g_input_w = 640;
static thread_local volatile float g_sink = 0.f;
static thread_local cv::Mat g_roi_logits;
static thread_local cv::Mat g_mask_resized;
static thread_local cv::Mat g_mask_binary;
static bool g_score_sum_enabled = true;
static bool g_mask_verify_enabled = false;
static bool g_mask_all_classes = false;
static std::vector<int> g_mask_class_ids{0};
static const char *g_mask_class_ids_spec = "0";
static int g_mask_output_width = 640;
static int g_mask_output_height = 480;

/**
 * @brief 释放当前线程持有的后处理 scratch buffer。
 */
static void free_postproc_buffers(void) {
  free(g_mb);
  free(g_pb);
  free(g_roi_storage);
  free(g_resize_storage);
  free(g_binary_storage);
  free(g_dfl_logits);
  free(g_suppressed);
  free(g_candidates);
  free(g_class_seeds);
  free(g_score_sum_survivors);
  free(g_mask_candidate_indices);
  g_mb = nullptr;
  g_pb = nullptr;
  g_roi_storage = nullptr;
  g_resize_storage = nullptr;
  g_binary_storage = nullptr;
  g_dfl_logits = nullptr;
  g_suppressed = nullptr;
  g_candidates = nullptr;
  g_class_seeds = nullptr;
  g_score_sum_survivors = nullptr;
  g_mask_candidate_indices = nullptr;
  g_lm = 0;
  g_lp = 0;
  g_lroi = 0;
  g_lresize = 0;
  g_lbinary = 0;
  g_ldfl_logits = 0;
  g_lsuppressed = 0;
  g_lcandidates = 0;
  g_lclass_seeds = 0;
  g_lscore_sum_survivors = 0;
  g_lmask_candidate_indices = 0;
  g_last_kept = 0;
  g_last_candidates = 0;
  g_last_sum_scanned = 0;
  g_last_score_sum_applied = 0;
  g_last_class_anchors = 0;
  g_last_class_values = 0;
  g_last_mask_pixels = 0;
  g_last_nms_pairs = 0;
  g_last_mask_active = 0;
  g_last_proto_roi_area = 0;
  g_last_mask_hash = 0;
  g_last_mask_mode = "none";
  g_last_fetch_outcome = "none";
  g_last_staged_failed = false;
  g_last_sync_bytes = 0;
  g_last_ready_mask = 0;
  g_last_sync_ms = 0.0;
  g_last_mask_verify_ms = 0.0;
  g_roi_logits.release();
  g_mask_resized.release();
  g_mask_binary.release();
  g_sink = 0.f;
}

static size_t round_up_64(size_t bytes) { return (bytes + 63U) & ~size_t{63U}; }

static float *ensure(float **p, int *cap, int need) {
  if (need > *cap) {
    free(*p);
    size_t bytes = round_up_64(sizeof(float) * static_cast<size_t>(need));
    *p = (float *)aligned_alloc(64, bytes);
    *cap = *p != nullptr ? need : 0;
  }
  return *p;
}

static int8_t *ensure_i8(int8_t **p, int *cap, int need) {
  if (need > *cap) {
    free(*p);
    size_t bytes = round_up_64(sizeof(int8_t) * static_cast<size_t>(need));
    *p = (int8_t *)aligned_alloc(64, bytes);
    *cap = *p != nullptr ? need : 0;
  }
  return *p;
}

static uint8_t *ensure_u8(uint8_t **p, int *cap, int need) {
  return (uint8_t *)ensure_i8((int8_t **)p, cap, need);
}

static int *ensure_int(int **p, int *cap, int need) {
  if (need > *cap) {
    free(*p);
    const size_t bytes = round_up_64(sizeof(int) * static_cast<size_t>(need));
    *p = static_cast<int *>(aligned_alloc(64, bytes));
    *cap = *p != nullptr ? need : 0;
  }
  return *p;
}

static candidate_t *ensure_candidates(int need) {
  if (need > g_lcandidates) {
    free(g_candidates);
    const size_t bytes =
        round_up_64(static_cast<size_t>(need) * sizeof(candidate_t));
    g_candidates = (candidate_t *)aligned_alloc(64, bytes);
    g_lcandidates = g_candidates != nullptr ? need : 0;
  }
  return g_candidates;
}

/**
 * @brief 确保 best-class 分类候选 scratch 容量充足。
 * @param need 需要的候选数。
 * @return 当前线程的候选缓冲区。
 */
static paddleyolo_rknn::postprocess::ClassSeed *ensure_class_seeds(int need) {
  if (need > g_lclass_seeds) {
    free(g_class_seeds);
    const size_t bytes =
        round_up_64(static_cast<size_t>(need) * sizeof(*g_class_seeds));
    g_class_seeds = static_cast<paddleyolo_rknn::postprocess::ClassSeed *>(
        aligned_alloc(64, bytes));
    g_lclass_seeds = g_class_seeds != nullptr ? need : 0;
  }
  return g_class_seeds;
}

static inline float dequant_i8_value(int8_t value, int32_t zp, float scale) {
  return ((float)((int32_t)value - zp)) * scale;
}

static bool candidate_score_greater(const candidate_t &a,
                                    const candidate_t &b) {
  if (a.score != b.score)
    return a.score > b.score;
  if (a.anchor != b.anchor)
    return a.anchor < b.anchor;
  return a.cls < b.cls;
}

static void anchor_geometry(int anchor, float *cx, float *cy, float *stride) {
  static const int strides[] = {8, 16, 32};
  int offset = anchor;
  for (size_t level = 0; level < sizeof(strides) / sizeof(strides[0]);
       level++) {
    int s = strides[level];
    int gh = g_input_h / s;
    int gw = g_input_w / s;
    int count = gh * gw;
    if (offset < count) {
      *cx = (float)(offset % gw) + 0.5f;
      *cy = (float)(offset / gw) + 0.5f;
      *stride = (float)s;
      return;
    }
    offset -= count;
  }
  *cx = *cy = 0.5f;
  *stride = 32.f;
}

static inline float32x4_t neon_exp_fast4(float32x4_t x) {
  x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-88.0f)), vdupq_n_f32(88.0f));
  const float32x4_t y = vmulq_n_f32(x, 1.4426950408889634f);
  const int32x4_t exponent = vcvtmq_s32_f32(y);
  const float32x4_t fraction = vsubq_f32(y, vcvtq_f32_s32(exponent));
  float32x4_t polynomial =
      vmlaq_n_f32(vdupq_n_f32(0.0096181291f), fraction, 0.0013333558f);
  polynomial = vmlaq_f32(vdupq_n_f32(0.0555041087f), polynomial, fraction);
  polynomial = vmlaq_f32(vdupq_n_f32(0.2402265070f), polynomial, fraction);
  polynomial = vmlaq_f32(vdupq_n_f32(0.6931471805f), polynomial, fraction);
  polynomial = vmlaq_f32(vdupq_n_f32(1.0f), polynomial, fraction);
  const int32x4_t exponent_bits =
      vshlq_n_s32(vaddq_s32(exponent, vdupq_n_s32(127)), 23);
  return vmulq_f32(polynomial, vreinterpretq_f32_s32(exponent_bits));
}

static bool decode_dfl_expectation4(const int8_t *boxes,
                                    const rknn_tensor_attr *attr, int anchor,
                                    int anchors, int reg_max, int transposed,
                                    float output[4]) {
  float *logits = ensure(&g_dfl_logits, &g_ldfl_logits, 4 * reg_max);
  if (logits == nullptr)
    return false;
  const int channels = 4 * reg_max;
  const float zero_point = static_cast<float>(attr->zp);
  if (transposed) {
    const int8_t *source = boxes + static_cast<size_t>(anchor) * channels;
    for (int channel = 0; channel < channels; ++channel) {
      logits[channel] =
          (static_cast<float>(source[channel]) - zero_point) * attr->scale;
    }
  } else {
    for (int channel = 0; channel < channels; ++channel) {
      logits[channel] =
          (static_cast<float>(
               boxes[static_cast<size_t>(channel) * anchors + anchor]) -
           zero_point) *
          attr->scale;
    }
  }
  for (int side = 0; side < 4; ++side) {
    const float *side_logits = logits + side * reg_max;
    int bin = 0;
    float32x4_t maximum_v = vdupq_n_f32(-INFINITY);
    for (; bin + 4 <= reg_max; bin += 4) {
      maximum_v = vmaxq_f32(maximum_v, vld1q_f32(side_logits + bin));
    }
    float maximum = vmaxvq_f32(maximum_v);
    for (; bin < reg_max; ++bin)
      maximum = std::max(maximum, side_logits[bin]);

    float32x4_t weighted_v = vdupq_n_f32(0.0f);
    float32x4_t sum_v = vdupq_n_f32(0.0f);
    const float32x4_t lane = {0.0f, 1.0f, 2.0f, 3.0f};
    bin = 0;
    for (; bin + 4 <= reg_max; bin += 4) {
      const float32x4_t weight = neon_exp_fast4(
          vsubq_f32(vld1q_f32(side_logits + bin), vdupq_n_f32(maximum)));
      const float32x4_t index =
          vaddq_f32(lane, vdupq_n_f32(static_cast<float>(bin)));
      weighted_v = vmlaq_f32(weighted_v, index, weight);
      sum_v = vaddq_f32(sum_v, weight);
    }
    float weighted = vaddvq_f32(weighted_v);
    float sum = vaddvq_f32(sum_v);
    for (; bin < reg_max; ++bin) {
      const float weight = expf(side_logits[bin] - maximum);
      weighted += weight * static_cast<float>(bin);
      sum += weight;
    }
    output[side] = sum > 0.0f ? weighted / sum : 0.0f;
  }
  return true;
}

static float candidate_iou(const candidate_t *a, const candidate_t *b) {
  float x1 = fmaxf(a->x1, b->x1);
  float y1 = fmaxf(a->y1, b->y1);
  float x2 = fminf(a->x2, b->x2);
  float y2 = fminf(a->y2, b->y2);
  float iw = fmaxf(0.f, x2 - x1);
  float ih = fmaxf(0.f, y2 - y1);
  float inter = iw * ih;
  float area_a = fmaxf(0.f, a->x2 - a->x1) * fmaxf(0.f, a->y2 - a->y1);
  float area_b = fmaxf(0.f, b->x2 - b->x1) * fmaxf(0.f, b->y2 - b->y1);
  float denom = area_a + area_b - inter;
  return denom > 0.f ? inter / denom : 0.f;
}

static int nms_and_cap(candidate_t *candidates, int count, float iou_thr,
                       int max_det) {
  const int nms_count = std::min(count, kMaxNmsCandidates);
  if (count > nms_count) {
    std::nth_element(candidates, candidates + nms_count, candidates + count,
                     candidate_score_greater);
  }
  std::sort(candidates, candidates + nms_count, candidate_score_greater);
  uint8_t *suppressed = ensure_u8(&g_suppressed, &g_lsuppressed, nms_count);
  if (nms_count > 0 && suppressed == nullptr)
    return -1;
  memset(suppressed, 0, (size_t)nms_count);
  int kept = 0;
  g_last_nms_pairs = 0;
  for (int i = 0; i < nms_count; i++) {
    if (suppressed[i])
      continue;
    candidate_t selected = candidates[i];
    candidates[kept++] = selected;
    if (kept >= max_det)
      break;
    for (int j = i + 1; j < nms_count; j++) {
      if (!suppressed[j] && selected.cls == candidates[j].cls) {
        g_last_nms_pairs++;
        if (candidate_iou(&selected, &candidates[j]) <= iou_thr)
          continue;
        suppressed[j] = 1;
      }
    }
  }
  return kept;
}

static int decode_candidates_from_seeds(
    const rknn_output *outs, const rknn_tensor_attr *attrs, bool use_dfl,
    const paddleyolo_rknn::postprocess::ClassSeed *class_seeds, int count,
    float iou_thr, int max_det) {
  const int8_t *boxes = (const int8_t *)outs[0].buf;
  int anchors = attrs[1].dims[2];
  int box_ch = use_dfl ? attrs[0].dims[2] : 4;
  int transposed = use_dfl ? 1 : 0;
  if (use_dfl && attrs[0].dims[2] == (uint32_t)anchors) {
    box_ch = attrs[0].dims[1];
    transposed = 0;
  }
  int reg_max = box_ch / 4;
  candidate_t *candidates = ensure_candidates(count);
  if (count > 0 &&
      (boxes == nullptr || candidates == nullptr || class_seeds == nullptr)) {
    g_last_candidates = 0;
    g_last_kept = 0;
    return -1;
  }

  int cached_anchor = -1;
  float cached_ltrb[4] = {};
  for (int index = 0; index < count; ++index) {
    const auto &seed = class_seeds[index];
    if (seed.anchor != cached_anchor) {
      if (use_dfl) {
        if (!decode_dfl_expectation4(boxes, &attrs[0], seed.anchor, anchors,
                                     reg_max, transposed, cached_ltrb)) {
          g_last_candidates = 0;
          g_last_kept = 0;
          return -1;
        }
      } else {
        for (int side = 0; side < 4; ++side) {
          cached_ltrb[side] =
              dequant_i8_value(boxes[(size_t)side * anchors + seed.anchor],
                               attrs[0].zp, attrs[0].scale);
        }
      }
      cached_anchor = seed.anchor;
    }
    float cx, cy, stride;
    anchor_geometry(seed.anchor, &cx, &cy, &stride);
    candidate_t *candidate = &candidates[index];
    candidate->x1 = (cx - cached_ltrb[0]) * stride;
    candidate->y1 = (cy - cached_ltrb[1]) * stride;
    candidate->x2 = (cx + cached_ltrb[2]) * stride;
    candidate->y2 = (cy + cached_ltrb[3]) * stride;
    candidate->score = seed.confidence;
    candidate->cls = seed.class_id;
    candidate->anchor = seed.anchor;
  }
  g_last_candidates = count;
  g_last_kept =
      count > 0 ? nms_and_cap(candidates, count, iou_thr, max_det) : 0;
  return g_last_kept;
}

static int collect_candidates(const rknn_output *outs,
                              const rknn_tensor_attr *attrs, bool use_dfl,
                              bool use_score_sum, float conf_thr, float iou_thr,
                              int max_det) {
  g_last_candidates = 0;
  g_last_kept = 0;
  g_last_nms_pairs = 0;
  g_last_sum_scanned = 0;
  g_last_score_sum_applied = 0;
  g_last_class_anchors = 0;
  g_last_class_values = 0;
  const int8_t *scores = (const int8_t *)outs[1].buf;
  const int anchors = attrs[1].dims[2];
  const int classes = attrs[1].dims[1];
  const int max_candidates = anchors;
  auto *class_seeds = ensure_class_seeds(max_candidates);

  paddleyolo_rknn::postprocess::ScoreSumInt8View score_sum_view;
  const paddleyolo_rknn::postprocess::ScoreSumInt8View *score_sum_ptr = nullptr;
  if (use_score_sum && g_score_sum_enabled) {
    score_sum_view.data = static_cast<const int8_t *>(outs[4].buf);
    score_sum_view.quant = {attrs[4].scale, attrs[4].zp};
    score_sum_ptr = &score_sum_view;
  }
  paddleyolo_rknn::postprocess::ClassSelectionStats selection_stats;
  const int count = paddleyolo_rknn::postprocess::SelectBestClassSeedsInt8(
      scores, classes, anchors, {attrs[1].scale, attrs[1].zp}, conf_thr,
      score_sum_ptr, class_seeds, static_cast<size_t>(max_candidates),
      &selection_stats);
  if (count < 0) {
    fprintf(stderr, "class selector failed: %d\n", count);
    g_last_candidates = 0;
    g_last_kept = 0;
    return 0;
  }
  g_last_sum_scanned = selection_stats.score_sum_scanned;
  g_last_score_sum_applied = selection_stats.used_score_sum ? 1 : 0;
  g_last_class_anchors = selection_stats.class_anchors_scanned;
  g_last_class_values = static_cast<long>(selection_stats.class_values_scanned);

  return decode_candidates_from_seeds(outs, attrs, use_dfl, class_seeds, count,
                                      iou_thr, max_det);
}

static cv::Rect make_proto_roi(const candidate_t &candidate, int proto_h,
                               int proto_w) {
  const float x_factor =
      static_cast<float>(proto_w) / static_cast<float>(g_input_w);
  const float y_factor =
      static_cast<float>(proto_h) / static_cast<float>(g_input_h);
  const float left =
      std::clamp(candidate.x1, 0.0f, static_cast<float>(g_input_w));
  const float top =
      std::clamp(candidate.y1, 0.0f, static_cast<float>(g_input_h));
  const float right =
      std::clamp(candidate.x2, 0.0f, static_cast<float>(g_input_w));
  const float bottom =
      std::clamp(candidate.y2, 0.0f, static_cast<float>(g_input_h));
  const int x1 = std::clamp(static_cast<int>(left * x_factor), 0, proto_w);
  const int y1 = std::clamp(static_cast<int>(top * y_factor), 0, proto_h);
  const int x2 = std::clamp(static_cast<int>(right * x_factor), 0, proto_w);
  const int y2 = std::clamp(static_cast<int>(bottom * y_factor), 0, proto_h);
  return cv::Rect(x1, y1, std::max(0, x2 - x1), std::max(0, y2 - y1));
}

static cv::Size make_mask_output_size(const candidate_t &candidate) {
  const float left =
      std::clamp(candidate.x1, 0.0f, static_cast<float>(g_input_w - 1));
  const float top =
      std::clamp(candidate.y1, 0.0f, static_cast<float>(g_input_h - 1));
  const float right =
      std::clamp(candidate.x2, 0.0f, static_cast<float>(g_input_w - 1));
  const float bottom =
      std::clamp(candidate.y2, 0.0f, static_cast<float>(g_input_h - 1));
  const float width_scale =
      static_cast<float>(g_mask_output_width) / static_cast<float>(g_input_w);
  const float height_scale =
      static_cast<float>(g_mask_output_height) / static_cast<float>(g_input_h);
  return cv::Size(
      static_cast<int>(std::max(0.0f, right - left) * width_scale),
      static_cast<int>(std::max(0.0f, bottom - top) * height_scale));
}

static bool mask_class_enabled(const int class_id) {
  return g_mask_all_classes ||
         std::find(g_mask_class_ids.begin(), g_mask_class_ids.end(),
                   class_id) != g_mask_class_ids.end();
}

static int collect_mask_candidate_indices(int *indices, const int capacity) {
  if (indices == nullptr || capacity < g_last_kept)
    return -1;
  int count = 0;
  for (int i = 0; i < g_last_kept; ++i) {
    if (mask_class_enabled(g_candidates[i].cls))
      indices[count++] = i;
  }
  return count;
}

static void update_mask_stats(const cv::Mat &binary, uint64_t *hash,
                              long *active) {
  const size_t bytes = binary.total() * binary.elemSize();
  const uint8_t *data = binary.ptr<uint8_t>();
  for (size_t i = 0; i < bytes; ++i) {
    const uint8_t value = data[i];
    *active += value != 0U ? 1 : 0;
    *hash ^= value;
    *hash *= UINT64_C(1099511628211);
  }
}

static bool run_segmentation_tail(
    const rknn_output *outs, const rknn_tensor_attr *attrs,
    const int *mask_candidate_indices, const int mask_candidate_count,
    const paddleyolo_rknn::postprocess::Nc1hwc2Int8View *native_proto =
        nullptr) {
  g_last_mask_pixels = 0;
  g_last_mask_active = 0;
  g_last_proto_roi_area = 0;
  g_last_mask_hash = UINT64_C(1469598103934665603);
  g_last_mask_mode = "none";
  g_last_mask_verify_ms = 0.0;
  if (mask_candidate_count <= 0)
    return true;

  const int anchors = attrs[2].dims[2];
  const int nm = attrs[2].dims[1];
  const int proto_h = attrs[3].dims[2];
  const int proto_w = attrs[3].dims[3];
  const int proto_pixels = proto_h * proto_w;
  if (native_proto == nullptr || native_proto->data == nullptr)
    return false;

  float *coeff = ensure(&g_mb, &g_lm, mask_candidate_count * nm);
  const int8_t *coeff_q = static_cast<const int8_t *>(outs[2].buf);
  if (coeff == nullptr || coeff_q == nullptr) {
    g_last_mask_mode = "allocation_failed";
    return false;
  }
  for (int i = 0; i < mask_candidate_count; ++i) {
    const candidate_t &candidate = g_candidates[mask_candidate_indices[i]];
    const int anchor = candidate.anchor;
    for (int channel = 0; channel < nm; ++channel) {
      coeff[static_cast<size_t>(i) * nm + channel] = dequant_i8_value(
          coeff_q[static_cast<size_t>(channel) * anchors + anchor], attrs[2].zp,
          attrs[2].scale);
    }
    g_last_proto_roi_area += make_proto_roi(candidate, proto_h, proto_w).area();
  }

  const auto decode_path =
      paddleyolo_rknn::postprocess::SelectRoiMaskDecodePath(
          g_last_proto_roi_area);
  const bool use_int8_roi =
      decode_path == paddleyolo_rknn::postprocess::RoiMaskDecodePath::kInt8;
  g_last_mask_mode = use_int8_roi ? "roi_tiled_i8" : "roi_tiled_f32";
  float *proto_f32 = nullptr;
  if (!use_int8_roi) {
    proto_f32 = ensure(&g_pb, &g_lp, nm * proto_pixels);
    if (!paddleyolo_rknn::postprocess::DequantizeNc1hwc2Int8ToNchwFloat32(
            *native_proto, attrs[3].scale, attrs[3].zp, proto_f32,
            static_cast<size_t>(nm) * proto_pixels)) {
      g_last_mask_mode = "proto_dequant_failed";
      return false;
    }
  }

  long active_pixels = 0;
  uint64_t mask_hash = UINT64_C(1469598103934665603);
  for (int i = 0; i < mask_candidate_count; ++i) {
    const candidate_t &candidate = g_candidates[mask_candidate_indices[i]];
    const cv::Rect roi = make_proto_roi(candidate, proto_h, proto_w);
    const cv::Size output_size = make_mask_output_size(candidate);
    if (roi.empty() || output_size.width <= 0 || output_size.height <= 0)
      continue;

    float *roi_storage = ensure(&g_roi_storage, &g_lroi, roi.area());
    float *resize_storage = ensure(&g_resize_storage, &g_lresize,
                                   output_size.width * output_size.height);
    uint8_t *binary_storage = ensure_u8(&g_binary_storage, &g_lbinary,
                                        output_size.width * output_size.height);
    if (roi_storage == nullptr || resize_storage == nullptr ||
        binary_storage == nullptr) {
      g_last_mask_mode = "allocation_failed";
      return false;
    }
    g_roi_logits = cv::Mat(roi.height, roi.width, CV_32F, roi_storage);
    g_mask_resized =
        cv::Mat(output_size.height, output_size.width, CV_32F, resize_storage);
    g_mask_binary =
        cv::Mat(output_size.height, output_size.width, CV_8U, binary_storage);

    const float *coeff_row = coeff + static_cast<size_t>(i) * nm;
    if (use_int8_roi) {
      paddleyolo_rknn::postprocess::ComputeRoiMaskInt8Nc1hwc2(
          *native_proto, coeff_row, attrs[3].scale, attrs[3].zp, roi,
          g_roi_logits,
          paddleyolo_rknn::postprocess::RoiMaskActivation::kSigmoid);
    } else {
      paddleyolo_rknn::postprocess::ComputeRoiMaskFloat32(
          proto_f32, nm, proto_h, proto_w, coeff_row, roi, g_roi_logits,
          paddleyolo_rknn::postprocess::RoiMaskActivation::kSigmoid);
    }
    cv::resize(g_roi_logits, g_mask_resized, output_size, 0.0, 0.0,
               cv::INTER_LINEAR);
    paddleyolo_rknn::postprocess::AssignBinaryMaskFromProbabilityMat(
        g_mask_resized, 127, g_mask_binary);
    g_last_mask_pixels += static_cast<long>(g_mask_binary.total());
    if (g_mask_verify_enabled) {
      const double verify_start = now_ms();
      update_mask_stats(g_mask_binary, &mask_hash, &active_pixels);
      g_last_mask_verify_ms += now_ms() - verify_start;
    }
  }
  g_last_mask_active = active_pixels;
  g_last_mask_hash = mask_hash;
  g_sink += static_cast<float>(active_pixels) * 1e-12f;
  return true;
}

static double postproc_detect(const rknn_output *outs,
                              const rknn_tensor_attr *attrs, bool use_dfl,
                              float conf_thr, float iou_thr, int max_det) {
  double start = now_ms();
  collect_candidates(outs, attrs, use_dfl, false, conf_thr, iou_thr, max_det);
  g_sink += (float)g_last_kept * 1e-9f;
  return now_ms() - start;
}

static double postproc_predist(const rknn_output *outs,
                               const rknn_tensor_attr *attrs, uint32_t n_out,
                               float conf_thr, float iou_thr, int max_det) {
  if (n_out != 2)
    return 0.0;
  return postproc_detect(outs, attrs, false, conf_thr, iou_thr, max_det);
}

static double postproc_predfl(const rknn_output *outs,
                              const rknn_tensor_attr *attrs, uint32_t n_out,
                              float conf_thr, float iou_thr, int max_det) {
  if (n_out != 2)
    return 0.0;
  return postproc_detect(outs, attrs, true, conf_thr, iou_thr, max_det);
}

typedef enum {
  PP_NONE,
  PP_PREDIST,
  PP_PREDFL,
  PP_SEG_PREDIST,
  PP_SEG_PREDFL
} pp_mode_t;

/**
 * @brief 校验后处理模式要求的输出数量。
 * @param mode 后处理模式。
 * @param n_output RKNN 模型输出数量。
 * @return 契约满足时返回 0，否则返回 -1。
 */
static int validate_output_count(pp_mode_t mode, uint32_t n_output) {
  if ((mode == PP_SEG_PREDIST || mode == PP_SEG_PREDFL) && n_output != 5) {
    fprintf(stderr, "seg postproc requires exactly 5 outputs, got %u\n",
            n_output);
    return -1;
  }
  return 0;
}

static int validate_quantized_output(const rknn_tensor_attr *attr,
                                     uint32_t index) {
  if (attr->type != RKNN_TENSOR_INT8 ||
      attr->qnt_type != RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC ||
      attr->scale <= 0.f) {
    fprintf(stderr,
            "output[%u] must be affine INT8 with positive scale, got type=%d "
            "qnt=%d scale=%g\n",
            index, attr->type, attr->qnt_type, attr->scale);
    return -1;
  }
  if (attr->fmt != RKNN_TENSOR_NCHW && attr->fmt != RKNN_TENSOR_UNDEFINED) {
    fprintf(stderr,
            "output[%u] must use logical NCHW/UNDEFINED layout, got fmt=%d\n",
            index, attr->fmt);
    return -1;
  }
  return 0;
}

static int validate_output_contract(pp_mode_t mode, uint32_t n_output,
                                    const rknn_tensor_attr *attrs) {
  if (mode == PP_NONE)
    return 0;
  uint32_t expected =
      (mode == PP_SEG_PREDIST || mode == PP_SEG_PREDFL) ? 5u : 2u;
  if (n_output != expected) {
    fprintf(stderr, "postproc route requires exactly %u outputs, got %u\n",
            expected, n_output);
    return -1;
  }
  for (uint32_t i = 0; i < n_output; i++) {
    if (validate_quantized_output(&attrs[i], i) < 0)
      return -1;
  }

  if (attrs[1].n_dims != 3 || attrs[1].dims[0] != 1 || attrs[1].dims[1] < 1 ||
      attrs[1].dims[2] < 1) {
    fprintf(stderr, "output[1] must be cls [1,nc,A]\n");
    return -1;
  }
  int anchors = attrs[1].dims[2];
  int expected_anchors = 0;
  static const int strides[] = {8, 16, 32};
  for (size_t i = 0; i < sizeof(strides) / sizeof(strides[0]); i++) {
    expected_anchors += (g_input_h / strides[i]) * (g_input_w / strides[i]);
  }
  if (anchors != expected_anchors) {
    fprintf(stderr,
            "anchor count mismatch: output=%d expected=%d for input=%dx%d\n",
            anchors, expected_anchors, g_input_w, g_input_h);
    return -1;
  }

  bool dfl = mode == PP_PREDFL || mode == PP_SEG_PREDFL;
  if (attrs[0].n_dims != 3 || attrs[0].dims[0] != 1) {
    fprintf(stderr, "output[0] must be rank-3 with batch=1\n");
    return -1;
  }
  if (dfl) {
    bool transposed = attrs[0].dims[1] == (uint32_t)anchors &&
                      attrs[0].dims[2] > 4 && attrs[0].dims[2] % 4 == 0;
    bool channel_first = mode == PP_PREDFL &&
                         attrs[0].dims[2] == (uint32_t)anchors &&
                         attrs[0].dims[1] > 4 && attrs[0].dims[1] % 4 == 0;
    if (!transposed && !channel_first) {
      fprintf(stderr, "DFL output[0] must be [1,A,4*reg_max]%s\n",
              mode == PP_PREDFL ? " or [1,4*reg_max,A]" : "");
      return -1;
    }
  } else if (attrs[0].dims[1] != 4 || attrs[0].dims[2] != (uint32_t)anchors) {
    fprintf(stderr, "distance output[0] must be [1,4,A]\n");
    return -1;
  }

  if (expected == 5u) {
    if (attrs[2].n_dims != 3 || attrs[2].dims[0] != 1 || attrs[2].dims[1] < 1 ||
        attrs[2].dims[2] != (uint32_t)anchors) {
      fprintf(stderr, "output[2] must be mask coeff [1,nm,A]\n");
      return -1;
    }
    int nm = attrs[2].dims[1];
    if (attrs[3].n_dims != 4 || attrs[3].dims[0] != 1 ||
        attrs[3].dims[1] != (uint32_t)nm || attrs[3].dims[2] < 1 ||
        attrs[3].dims[3] < 1) {
      fprintf(stderr, "output[3] must be proto [1,nm,H,W] with matching nm\n");
      return -1;
    }
    if (attrs[4].n_dims != 3 || attrs[4].dims[0] != 1 ||
        attrs[4].dims[1] != 1 || attrs[4].dims[2] != (uint32_t)anchors) {
      fprintf(stderr, "output[4] must be score_sum [1,1,A]\n");
      return -1;
    }
  }
  return 0;
}

static pp_mode_t parse_pp(const char *s) {
  if (!s || !strcmp(s, "none"))
    return PP_NONE;
  if (!strcmp(s, "predist"))
    return PP_PREDIST;
  if (!strcmp(s, "predfl"))
    return PP_PREDFL;
  if (!strcmp(s, "seg_predist"))
    return PP_SEG_PREDIST;
  if (!strcmp(s, "seg_predfl"))
    return PP_SEG_PREDFL;
  return PP_NONE;
}

static bool parse_core(const char *s, int *mask) {
  if (s == nullptr || mask == nullptr)
    return false;
  if (!strcmp(s, "all")) {
    *mask = RKNN_NPU_CORE_0_1_2;
    return true;
  }
  if (!strcmp(s, "0")) {
    *mask = RKNN_NPU_CORE_0;
    return true;
  }
  if (!strcmp(s, "1")) {
    *mask = RKNN_NPU_CORE_1;
    return true;
  }
  if (!strcmp(s, "2")) {
    *mask = RKNN_NPU_CORE_2;
    return true;
  }
  return false;
}

static const char *normalize_sram_mode(const char *s) {
  if (!s || !strcmp(s, "off") || !strcmp(s, "private") ||
      !strcmp(s, "shared")) {
    return s ? s : "off";
  }
  return "off";
}

static uint32_t parse_sram_flags(const char *s) {
  const char *mode = normalize_sram_mode(s);
  if (!strcmp(mode, "off"))
    return 0;
#ifdef RKNN_FLAG_ENABLE_SRAM
  uint32_t flags = RKNN_FLAG_ENABLE_SRAM;
#ifdef RKNN_FLAG_SHARE_SRAM
  if (!strcmp(mode, "shared"))
    flags |= RKNN_FLAG_SHARE_SRAM;
#endif
  return flags;
#else
  (void)mode;
  return 0;
#endif
}

static bool is_shared_sram_mode(const char *s) {
  return s && !strcmp(normalize_sram_mode(s), "shared");
}

static bool flags_request_shared_sram(uint32_t flags) {
#ifdef RKNN_FLAG_SHARE_SRAM
  return (flags & RKNN_FLAG_SHARE_SRAM) != 0;
#else
  (void)flags;
  return false;
#endif
}

#define MAX_IO 16
#define MAX_FPS_WORKERS 32

/**
 * @brief 单个离线 FPS worker 的运行参数和统计结果。
 */
typedef struct {
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  rknn_context ctx;
  int ready;
  int failed;
} fps_shared_state_t;

typedef struct {
  const void *model_buf;
  size_t model_size;
  int worker_id;
  int core_mask;
  uint32_t init_flags;
  bool shared_sram_mode;
  fps_shared_state_t *share_state;
  pp_mode_t pp;
  float conf_thr;
  float iou_thr;
  int max_det;
  int warmup;
  pthread_barrier_t *ready_barrier;
  pthread_barrier_t *start_barrier;
  pthread_barrier_t *done_barrier;
  pthread_barrier_t *release_barrier;
  double *end_ms;
  long frames;
  double sum_e2e_ms;
  double sum_pp_ms;
  double sum_sync_ms;
  double sum_mask_verify_ms;
  uint64_t sum_sync_bytes;
  double min_e2e_ms;
  double max_e2e_ms;
  int last_kept;
  unsigned int last_ready_mask;
  const char *last_fetch_outcome;
  int ret;
} fps_worker_arg_t;

/**
 * @brief 按指定 route 执行一次 C/NEON 后处理并返回耗时。
 * @param pp 后处理 route。
 * @param outs RKNN 原始输出。
 * @param out_attrs RKNN 输出 tensor 属性。
 * @param n_output 输出 tensor 数量。
 * @param conf_thr 置信度阈值。
 * @return 后处理耗时，单位 ms。
 */
static double run_postproc(pp_mode_t pp, const rknn_output *outs,
                           const rknn_tensor_attr *out_attrs, uint32_t n_output,
                           float conf_thr, float iou_thr, int max_det) {
  switch (pp) {
  case PP_PREDIST:
    return postproc_predist(outs, out_attrs, n_output, conf_thr, iou_thr,
                            max_det);
  case PP_PREDFL:
    return postproc_predfl(outs, out_attrs, n_output, conf_thr, iou_thr,
                           max_det);
  default:
    return 0.0;
  }
}

static void update_full_io_stats(
    const paddleyolo_rknn::postprocess::FiveOutputRuntime &runtime) {
  const auto &stats = runtime.Stats();
  g_last_sync_bytes = stats.native_bytes;
  g_last_ready_mask = stats.ready_mask;
  g_last_sync_ms = stats.sync_ms;
}

static void mark_staged_failure(const char *outcome) {
  g_last_fetch_outcome = outcome;
  g_last_staged_failed = true;
}

static double run_staged_seg_postproc(
    paddleyolo_rknn::postprocess::FiveOutputRuntime *runtime,
    const rknn_tensor_attr *attrs, bool use_dfl, float conf_thr, float iou_thr,
    int max_det) {
  using paddleyolo_rknn::postprocess::ClassifyAnchorsBestClassInt8;
  using paddleyolo_rknn::postprocess::ClassSelectionStats;
  using paddleyolo_rknn::postprocess::CollectScoreSumSurvivorsInt8;
  using paddleyolo_rknn::postprocess::FiveOutputTensor;
  using paddleyolo_rknn::postprocess::kScoreSumPrescreenUnavailable;
  using paddleyolo_rknn::postprocess::ScoreSumInt8View;

  const double start = now_ms();
  g_last_candidates = 0;
  g_last_kept = 0;
  g_last_nms_pairs = 0;
  g_last_sum_scanned = 0;
  g_last_score_sum_applied = 0;
  g_last_class_anchors = 0;
  g_last_class_values = 0;
  g_last_mask_pixels = 0;
  g_last_mask_active = 0;
  g_last_proto_roi_area = 0;
  g_last_mask_hash = UINT64_C(1469598103934665603);
  g_last_mask_mode = "none";
  g_last_mask_verify_ms = 0.0;
  g_last_fetch_outcome = "pending";
  g_last_staged_failed = false;

  rknn_output *outputs = runtime->Outputs();
  const int anchors = attrs[1].dims[2];
  const int classes = attrs[1].dims[1];
  const int max_candidates = anchors;
  auto *class_seeds = ensure_class_seeds(max_candidates);
  int *survivors =
      ensure_int(&g_score_sum_survivors, &g_lscore_sum_survivors, anchors);
  if (class_seeds == nullptr || survivors == nullptr) {
    mark_staged_failure("allocation_failed");
    update_full_io_stats(*runtime);
    return now_ms() - start;
  }

  int survivor_count = kScoreSumPrescreenUnavailable;
  if (g_score_sum_enabled) {
    if (runtime->Prepare(FiveOutputTensor::kScoreSum) != 0) {
      mark_staged_failure("score_sum_sync_failed");
      update_full_io_stats(*runtime);
      return now_ms() - start;
    }
    ScoreSumInt8View score_sum{static_cast<const int8_t *>(outputs[4].buf),
                               {attrs[4].scale, attrs[4].zp}};
    survivor_count = CollectScoreSumSurvivorsInt8(
        score_sum, anchors, conf_thr, survivors, static_cast<size_t>(anchors));
    if (survivor_count < 0 && survivor_count != kScoreSumPrescreenUnavailable) {
      mark_staged_failure("score_sum_selection_failed");
      update_full_io_stats(*runtime);
      return std::max(0.0, now_ms() - start - g_last_sync_ms);
    }
    if (survivor_count >= 0) {
      g_last_score_sum_applied = 1;
      g_last_sum_scanned = anchors;
      if (survivor_count == 0) {
        g_last_fetch_outcome = "no_score_sum_survivors";
        update_full_io_stats(*runtime);
        return std::max(0.0, now_ms() - start - g_last_sync_ms);
      }
    }
  }

  if (runtime->Prepare(FiveOutputTensor::kClass) != 0) {
    mark_staged_failure("class_sync_failed");
    update_full_io_stats(*runtime);
    return now_ms() - start;
  }
  ClassSelectionStats selection_stats{};
  int seed_count = 0;
  if (survivor_count >= 0) {
    seed_count = ClassifyAnchorsBestClassInt8(
        static_cast<const int8_t *>(outputs[1].buf), classes, anchors,
        {attrs[1].scale, attrs[1].zp}, conf_thr, survivors,
        static_cast<size_t>(survivor_count), class_seeds,
        static_cast<size_t>(max_candidates), &selection_stats);
    selection_stats.used_score_sum = true;
    selection_stats.score_sum_scanned = anchors;
  } else {
    seed_count = paddleyolo_rknn::postprocess::SelectBestClassSeedsInt8(
        static_cast<const int8_t *>(outputs[1].buf), classes, anchors,
        {attrs[1].scale, attrs[1].zp}, conf_thr, nullptr, class_seeds,
        static_cast<size_t>(max_candidates), &selection_stats);
  }
  if (seed_count < 0) {
    mark_staged_failure("classification_failed");
    update_full_io_stats(*runtime);
    return std::max(0.0, now_ms() - start - g_last_sync_ms);
  }
  g_last_class_anchors = selection_stats.class_anchors_scanned;
  g_last_class_values = static_cast<long>(selection_stats.class_values_scanned);
  if (seed_count == 0) {
    g_last_fetch_outcome = "no_class_seeds";
    update_full_io_stats(*runtime);
    return std::max(0.0, now_ms() - start - g_last_sync_ms);
  }

  if (runtime->Prepare(FiveOutputTensor::kBox) != 0) {
    mark_staged_failure("box_sync_failed");
    update_full_io_stats(*runtime);
    return now_ms() - start;
  }
  if (decode_candidates_from_seeds(outputs, attrs, use_dfl, class_seeds,
                                   seed_count, iou_thr, max_det) < 0) {
    mark_staged_failure("candidate_decode_failed");
    update_full_io_stats(*runtime);
    return std::max(0.0, now_ms() - start - g_last_sync_ms);
  }
  if (g_last_kept <= 0) {
    g_last_fetch_outcome = "no_boxes";
    update_full_io_stats(*runtime);
    return std::max(0.0, now_ms() - start - g_last_sync_ms);
  }

  int *mask_candidate_indices = ensure_int(
      &g_mask_candidate_indices, &g_lmask_candidate_indices, g_last_kept);
  const int mask_candidate_count =
      collect_mask_candidate_indices(mask_candidate_indices, g_last_kept);
  if (mask_candidate_count < 0) {
    mark_staged_failure("mask_class_selection_failed");
    update_full_io_stats(*runtime);
    return std::max(0.0, now_ms() - start - g_last_sync_ms);
  }
  if (mask_candidate_count == 0) {
    g_last_fetch_outcome = "no_mask_classes";
    update_full_io_stats(*runtime);
    return std::max(0.0, now_ms() - start - g_last_sync_ms);
  }

  if (runtime->Prepare(FiveOutputTensor::kMaskCoeff) != 0 ||
      runtime->Prepare(FiveOutputTensor::kProto) != 0) {
    mark_staged_failure("mask_sync_failed");
    update_full_io_stats(*runtime);
    return now_ms() - start;
  }
  if (!run_segmentation_tail(outputs, attrs, mask_candidate_indices,
                             mask_candidate_count, &runtime->ProtoView())) {
    mark_staged_failure("mask_decode_failed");
    update_full_io_stats(*runtime);
    return std::max(0.0, now_ms() - start - g_last_sync_ms);
  }
  g_last_fetch_outcome = "masks_required";
  update_full_io_stats(*runtime);
  return std::max(0.0,
                  now_ms() - start - g_last_sync_ms - g_last_mask_verify_ms);
}

static bool staged_postproc_failed(void) { return g_last_staged_failed; }

static bool parse_mask_class_ids(const char *spec) {
  if (spec == nullptr || *spec == '\0')
    return false;
  if (strcmp(spec, "all") == 0) {
    g_mask_all_classes = true;
    g_mask_class_ids.clear();
    return true;
  }
  std::vector<int> parsed;
  const char *cursor = spec;
  while (*cursor != '\0') {
    char *end = nullptr;
    const long value = strtol(cursor, &end, 10);
    if (end == cursor || value < 0 || value > INT_MAX ||
        (*end != '\0' && *end != ',')) {
      return false;
    }
    const int class_id = static_cast<int>(value);
    if (std::find(parsed.begin(), parsed.end(), class_id) == parsed.end())
      parsed.push_back(class_id);
    if (*end == '\0')
      break;
    cursor = end + 1;
    if (*cursor == '\0')
      return false;
  }
  if (parsed.empty())
    return false;
  g_mask_all_classes = false;
  g_mask_class_ids = std::move(parsed);
  return true;
}

static bool parse_mask_output_size(const char *spec) {
  if (spec == nullptr || *spec == '\0')
    return false;
  char *width_end = nullptr;
  const long width = strtol(spec, &width_end, 10);
  if (width_end == spec || (*width_end != 'x' && *width_end != 'X'))
    return false;
  char *height_end = nullptr;
  const long height = strtol(width_end + 1, &height_end, 10);
  if (height_end == width_end + 1 || *height_end != '\0' || width <= 0 ||
      height <= 0 || width > INT_MAX || height > INT_MAX) {
    return false;
  }
  g_mask_output_width = static_cast<int>(width);
  g_mask_output_height = static_cast<int>(height);
  return true;
}

/**
 * @brief 解析 FPS worker 的 core 轮转列表。
 * @param spec 逗号分隔的 core 列表，例如 `0,1,2`。
 * @param masks 输出 core mask 数组。
 * @param max_masks `masks` 容量。
 * @return 成功解析出的 core 数量；0 表示使用 `--core` 的默认绑定。
 */
static int parse_core_map(const char *spec, int *masks, int max_masks) {
  if (!spec || !*spec || max_masks <= 0)
    return 0;

  char tmp[128];
  const size_t spec_length = strlen(spec);
  if (spec_length >= sizeof(tmp) || spec[0] == ',' ||
      spec[spec_length - 1] == ',' || strstr(spec, ",,") != nullptr)
    return -1;
  strncpy(tmp, spec, sizeof(tmp) - 1);
  tmp[sizeof(tmp) - 1] = '\0';

  int n = 0;
  char *tok = strtok(tmp, ",");
  while (tok && n < max_masks) {
    if (!parse_core(tok, &masks[n]))
      return -1;
    ++n;
    tok = strtok(NULL, ",");
  }
  if (tok != nullptr)
    return -1;
  return n;
}

/**
 * @brief 初始化一个独立 RKNN context 及其 dummy 输入。
 * @param ctx 输出 RKNN context。
 * @param model_buf RKNN 模型文件内容。
 * @param model_size RKNN 模型字节数。
 * @param core_mask NPU core 绑定 mask。
 * @param init_flags 传给 `rknn_init` 的扩展 flags。
 * @param share_ctx shared SRAM 的源 RKNN context；不共享时为 0。
 * @param pp 后处理模式，用于校验模型输出契约。
 * @param io 输出输入/输出数量。
 * @param in_attrs 输出输入 tensor 属性。
 * @param out_attrs 输出输出 tensor 属性。
 * @param dummy 输出 dummy 输入 buffer，调用者负责释放。
 * @param in_size 输出 dummy 输入字节数。
 * @param inp 输出 RKNN 输入描述。
 * @return 0 表示成功；负数表示 RKNN 或内存错误。
 */
static int
setup_context(rknn_context *ctx, const void *model_buf, size_t model_size,
              int core_mask, uint32_t init_flags, rknn_context share_ctx,
              pp_mode_t pp, rknn_input_output_num *io,
              rknn_tensor_attr *in_attrs, rknn_tensor_attr *out_attrs,
              void **dummy, size_t *in_size, rknn_input *inp,
              paddleyolo_rknn::postprocess::FiveOutputRuntime *full_io) {
  void *model_copy = malloc(model_size);
  if (!model_copy) {
    fprintf(stderr, "malloc model copy failed\n");
    return -1;
  }
  memcpy(model_copy, model_buf, model_size);
  rknn_init_extend init_extend;
  memset(&init_extend, 0, sizeof(init_extend));
  init_extend.ctx = share_ctx;
  int ret = rknn_init(ctx, model_copy, model_size, init_flags,
                      init_flags ? &init_extend : NULL);
  free(model_copy);
  if (ret < 0) {
    fprintf(stderr, "rknn_init: %d\n", ret);
    return ret;
  }
  ret = rknn_set_core_mask(*ctx, static_cast<rknn_core_mask>(core_mask));
  if (ret != 0) {
    fprintf(stderr, "rknn_set_core_mask: %d\n", ret);
    return -1;
  }

  memset(io, 0, sizeof(*io));
  ret = rknn_query(*ctx, RKNN_QUERY_IN_OUT_NUM, io, sizeof(*io));
  if (ret < 0) {
    fprintf(stderr, "RKNN_QUERY_IN_OUT_NUM: %d\n", ret);
    return ret;
  }
  if (io->n_input < 1 || io->n_input > MAX_IO || io->n_output > MAX_IO) {
    fprintf(stderr, "unsupported io count: input=%u output=%u\n", io->n_input,
            io->n_output);
    return -1;
  }
  if (validate_output_count(pp, io->n_output) < 0) {
    return -1;
  }

  for (uint32_t i = 0; i < io->n_input; i++) {
    memset(&in_attrs[i], 0, sizeof(in_attrs[i]));
    in_attrs[i].index = i;
    ret = rknn_query(*ctx, RKNN_QUERY_INPUT_ATTR, &in_attrs[i],
                     sizeof(in_attrs[i]));
    if (ret < 0) {
      fprintf(stderr, "RKNN_QUERY_INPUT_ATTR[%u]: %d\n", i, ret);
      return ret;
    }
  }
  for (uint32_t i = 0; i < io->n_output; i++) {
    memset(&out_attrs[i], 0, sizeof(out_attrs[i]));
    out_attrs[i].index = i;
    ret = rknn_query(*ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attrs[i],
                     sizeof(out_attrs[i]));
    if (ret < 0) {
      fprintf(stderr, "RKNN_QUERY_OUTPUT_ATTR[%u]: %d\n", i, ret);
      return ret;
    }
  }

  if (in_attrs[0].n_dims != 4 || in_attrs[0].dims[0] != 1) {
    fprintf(stderr, "input[0] must be a static rank-4 tensor with batch=1\n");
    return -1;
  }
  if (in_attrs[0].fmt == RKNN_TENSOR_NHWC) {
    g_input_h = in_attrs[0].dims[1];
    g_input_w = in_attrs[0].dims[2];
  } else if (in_attrs[0].fmt == RKNN_TENSOR_NCHW) {
    g_input_h = in_attrs[0].dims[2];
    g_input_w = in_attrs[0].dims[3];
  } else {
    fprintf(stderr, "input[0] must use NCHW or NHWC layout\n");
    return -1;
  }
  if (validate_output_contract(pp, io->n_output, out_attrs) < 0)
    return -1;

  *in_size = 1;
  for (uint32_t d = 0; d < in_attrs[0].n_dims; d++) {
    *in_size *= in_attrs[0].dims[d];
  }
  *dummy = calloc(*in_size, 1);
  if (!*dummy) {
    fprintf(stderr, "calloc input failed\n");
    return -1;
  }

  memset(inp, 0, sizeof(*inp));
  inp[0].index = 0;
  inp[0].buf = *dummy;
  inp[0].size = *in_size;
  inp[0].pass_through = 0;
  inp[0].type = RKNN_TENSOR_UINT8;
  inp[0].fmt = RKNN_TENSOR_NHWC;
  if (pp == PP_SEG_PREDIST || pp == PP_SEG_PREDFL) {
    if (full_io == nullptr ||
        !full_io->Initialize(*ctx, in_attrs[0], out_attrs, io->n_output) ||
        !full_io->SetInput(*dummy, *in_size)) {
      fprintf(stderr, "five-output full-IO initialization failed\n");
      return -1;
    }
  }
  return 0;
}

/**
 * @brief 执行一次离线帧：输入设置、推理、取输出和可选后处理。
 * @param ctx RKNN context。
 * @param inp RKNN 输入描述。
 * @param io 输入/输出数量。
 * @param out_attrs 输出 tensor 属性。
 * @param pp 后处理 route。
 * @param conf_thr 置信度阈值。
 * @param e2e_ms 输出单帧端到端耗时，单位 ms。
 * @param pp_ms 输出后处理耗时，单位 ms。
 * @return 0 表示成功；负数表示 RKNN 错误。
 */
static int
run_one_iteration(rknn_context ctx, const rknn_input *inp,
                  const rknn_input_output_num *io,
                  const rknn_tensor_attr *out_attrs, pp_mode_t pp,
                  float conf_thr, float iou_thr, int max_det,
                  paddleyolo_rknn::postprocess::FiveOutputRuntime *full_io,
                  double *e2e_ms, double *pp_ms) {
  if (full_io != nullptr && full_io->IsInitialized()) {
    const double start = now_ms();
    int ret = full_io->Run();
    if (ret != 0)
      return ret;
    const double post_ms = run_staged_seg_postproc(
        full_io, out_attrs, pp == PP_SEG_PREDFL, conf_thr, iou_thr, max_det);
    if (staged_postproc_failed())
      return -1;
    if (pp_ms)
      *pp_ms = post_ms;
    if (e2e_ms)
      *e2e_ms = std::max(0.0, now_ms() - start - g_last_mask_verify_ms);
    return 0;
  }
  rknn_output outs[MAX_IO];
  double t0 = now_ms();
  int ret = rknn_inputs_set(ctx, 1, (rknn_input *)inp);
  if (ret < 0)
    return ret;
  ret = rknn_run(ctx, NULL);
  if (ret < 0)
    return ret;
  memset(outs, 0, sizeof(outs));
  for (uint32_t o = 0; o < io->n_output; o++) {
    outs[o].index = o;
    outs[o].want_float = 0;
  }
  ret = rknn_outputs_get(ctx, io->n_output, outs, NULL);
  if (ret < 0)
    return ret;
  double post_ms = run_postproc(pp, outs, out_attrs, io->n_output, conf_thr,
                                iou_thr, max_det);
  ret = rknn_outputs_release(ctx, io->n_output, outs);
  double t2 = now_ms();
  if (ret < 0)
    return ret;

  if (pp_ms)
    *pp_ms = post_ms;
  if (e2e_ms)
    *e2e_ms = t2 - t0;
  return 0;
}

/**
 * @brief 离线 FPS worker 主函数。
 * @param opaque `fps_worker_arg_t*`。
 * @return 始终返回 NULL，错误码写入 worker 参数。
 */
static void *fps_worker_main(void *opaque) {
  fps_worker_arg_t *arg = (fps_worker_arg_t *)opaque;
  rknn_context ctx = 0;
  rknn_context share_ctx = 0;
  uint32_t init_flags = arg->init_flags;
  rknn_input_output_num io;
  rknn_tensor_attr in_attrs[MAX_IO];
  rknn_tensor_attr out_attrs[MAX_IO];
  void *dummy = NULL;
  size_t in_size = 0;
  rknn_input inp[1];
  paddleyolo_rknn::postprocess::FiveOutputRuntime full_io;

  if (arg->shared_sram_mode) {
    if (arg->worker_id == 0) {
      init_flags = parse_sram_flags("private");
    } else {
      fps_shared_state_t *state = arg->share_state;
      pthread_mutex_lock(&state->mutex);
      while (!state->ready && !state->failed) {
        pthread_cond_wait(&state->cond, &state->mutex);
      }
      if (!state->ready || state->ctx == 0) {
        arg->ret = -1;
        pthread_mutex_unlock(&state->mutex);
        pthread_barrier_wait(arg->ready_barrier);
        pthread_barrier_wait(arg->start_barrier);
        goto done;
      }
      share_ctx = state->ctx;
      pthread_mutex_unlock(&state->mutex);
    }
  }

  arg->ret =
      setup_context(&ctx, arg->model_buf, arg->model_size, arg->core_mask,
                    init_flags, share_ctx, arg->pp, &io, in_attrs, out_attrs,
                    &dummy, &in_size, inp, &full_io);
  if (arg->shared_sram_mode && arg->worker_id == 0) {
    fps_shared_state_t *state = arg->share_state;
    pthread_mutex_lock(&state->mutex);
    state->ctx = ctx;
    state->ready = arg->ret >= 0;
    state->failed = arg->ret < 0;
    pthread_cond_broadcast(&state->cond);
    pthread_mutex_unlock(&state->mutex);
  }
  if (arg->ret < 0) {
    pthread_barrier_wait(arg->ready_barrier);
    pthread_barrier_wait(arg->start_barrier);
    goto done;
  }

  for (int w = 0; w < arg->warmup; w++) {
    double e2e_ms = 0.0, pp_ms = 0.0;
    arg->ret = run_one_iteration(ctx, inp, &io, out_attrs, arg->pp,
                                 arg->conf_thr, arg->iou_thr, arg->max_det,
                                 &full_io, &e2e_ms, &pp_ms);
    if (arg->ret < 0) {
      pthread_barrier_wait(arg->ready_barrier);
      pthread_barrier_wait(arg->start_barrier);
      goto done;
    }
  }

  pthread_barrier_wait(arg->ready_barrier);
  pthread_barrier_wait(arg->start_barrier);

  arg->min_e2e_ms = DBL_MAX;
  while (now_ms() < *arg->end_ms) {
    double e2e_ms = 0.0, pp_ms = 0.0;
    arg->ret = run_one_iteration(ctx, inp, &io, out_attrs, arg->pp,
                                 arg->conf_thr, arg->iou_thr, arg->max_det,
                                 &full_io, &e2e_ms, &pp_ms);
    if (arg->ret < 0)
      goto done;
    arg->frames++;
    arg->sum_e2e_ms += e2e_ms;
    arg->sum_pp_ms += pp_ms;
    arg->sum_sync_ms += g_last_sync_ms;
    arg->sum_mask_verify_ms += g_last_mask_verify_ms;
    arg->sum_sync_bytes += g_last_sync_bytes;
    if (e2e_ms < arg->min_e2e_ms)
      arg->min_e2e_ms = e2e_ms;
    if (e2e_ms > arg->max_e2e_ms)
      arg->max_e2e_ms = e2e_ms;
    arg->last_kept = g_last_kept;
    arg->last_ready_mask = g_last_ready_mask;
    arg->last_fetch_outcome = g_last_fetch_outcome;
  }

done:
  if (arg->done_barrier) {
    pthread_barrier_wait(arg->done_barrier);
  }
  if (dummy)
    free(dummy);
  if (arg->release_barrier && arg->worker_id == 0) {
    // shared SRAM owner 必须最后释放，避免消费者仍引用源 context。
    pthread_barrier_wait(arg->release_barrier);
    full_io.Release();
    if (ctx)
      rknn_destroy(ctx);
  } else {
    full_io.Release();
    if (ctx)
      rknn_destroy(ctx);
    if (arg->release_barrier)
      pthread_barrier_wait(arg->release_barrier);
  }
  free_postproc_buffers();
  return NULL;
}

/**
 * @brief 运行多 context 离线 FPS 模式并输出文本/JSON 结果。
 * @param model_path 模型路径，仅用于报告。
 * @param model_buf RKNN 模型文件内容。
 * @param model_size RKNN 模型字节数。
 * @param json_out JSON 输出路径；NULL 表示不输出。
 * @param pp_str 后处理 route 字符串。
 * @param pp 后处理 route 枚举。
 * @param conf_thr 置信度阈值。
 * @param warmup 每个 worker 的预热轮次。
 * @param core_str 默认 core 绑定。
 * @param fps_core_map worker core 轮转列表。
 * @param sram_str SRAM 模式字符串，仅用于报告。
 * @param init_flags 传给 `rknn_init` 的扩展 flags。
 * @param fps_workers worker/context 数量。
 * @param fps_seconds 计时秒数。
 * @return 0 表示成功；非 0 表示参数或 RKNN 错误。
 */
static int run_fps_mode(const char *model_path, const void *model_buf,
                        size_t model_size, const char *json_out,
                        const char *pp_str, pp_mode_t pp, float conf_thr,
                        float iou_thr, int max_det, int warmup,
                        const char *core_str, const char *fps_core_map,
                        const char *sram_str, uint32_t init_flags,
                        int fps_workers, double fps_seconds) {
  if (fps_workers <= 0 || fps_workers > MAX_FPS_WORKERS) {
    fprintf(stderr, "--fps-workers must be in [1,%d]\n", MAX_FPS_WORKERS);
    return 2;
  }
  if (fps_seconds <= 0.0) {
    fprintf(stderr, "--fps-seconds must be > 0\n");
    return 2;
  }

  int mapped_cores[MAX_FPS_WORKERS];
  int map_count = parse_core_map(fps_core_map, mapped_cores, MAX_FPS_WORKERS);
  if (map_count < 0) {
    fprintf(stderr, "invalid --fps-core-map value: %s\n",
            fps_core_map ? fps_core_map : "");
    return 2;
  }
  int default_core = RKNN_NPU_CORE_AUTO;
  if (!parse_core(core_str, &default_core)) {
    fprintf(stderr, "invalid --core value: %s\n", core_str ? core_str : "");
    return 2;
  }
  pthread_t threads[MAX_FPS_WORKERS];
  fps_worker_arg_t args[MAX_FPS_WORKERS];
  pthread_barrier_t ready_barrier;
  pthread_barrier_t start_barrier;
  pthread_barrier_t done_barrier;
  pthread_barrier_t release_barrier;
  fps_shared_state_t share_state;
  double end_ms = 0.0;
  bool shared_sram_mode =
      is_shared_sram_mode(sram_str) && flags_request_shared_sram(init_flags);

  pthread_barrier_init(&ready_barrier, NULL, (unsigned)fps_workers + 1);
  pthread_barrier_init(&start_barrier, NULL, (unsigned)fps_workers + 1);
  if (shared_sram_mode) {
    pthread_barrier_init(&done_barrier, NULL, (unsigned)fps_workers);
    pthread_barrier_init(&release_barrier, NULL, (unsigned)fps_workers);
    memset(&share_state, 0, sizeof(share_state));
    pthread_mutex_init(&share_state.mutex, NULL);
    pthread_cond_init(&share_state.cond, NULL);
  }

  for (int i = 0; i < fps_workers; i++) {
    memset(&args[i], 0, sizeof(args[i]));
    args[i].model_buf = model_buf;
    args[i].model_size = model_size;
    args[i].worker_id = i;
    args[i].core_mask =
        map_count > 0 ? mapped_cores[i % map_count] : default_core;
    args[i].init_flags = init_flags;
    args[i].shared_sram_mode = shared_sram_mode;
    args[i].share_state = shared_sram_mode ? &share_state : NULL;
    args[i].pp = pp;
    args[i].conf_thr = conf_thr;
    args[i].iou_thr = iou_thr;
    args[i].max_det = max_det;
    args[i].warmup = warmup;
    args[i].ready_barrier = &ready_barrier;
    args[i].start_barrier = &start_barrier;
    args[i].done_barrier = shared_sram_mode ? &done_barrier : NULL;
    args[i].release_barrier = shared_sram_mode ? &release_barrier : NULL;
    args[i].end_ms = &end_ms;
    args[i].min_e2e_ms = DBL_MAX;
    int ret = pthread_create(&threads[i], NULL, fps_worker_main, &args[i]);
    if (ret != 0) {
      fprintf(stderr, "pthread_create[%d]: %d\n", i, ret);
      exit(1);
    }
  }

  pthread_barrier_wait(&ready_barrier);
  double start_ms = now_ms();
  end_ms = start_ms + fps_seconds * 1000.0;
  pthread_barrier_wait(&start_barrier);

  for (int i = 0; i < fps_workers; i++) {
    pthread_join(threads[i], NULL);
  }
  double finish_ms = now_ms();

  pthread_barrier_destroy(&ready_barrier);
  pthread_barrier_destroy(&start_barrier);
  if (shared_sram_mode) {
    pthread_barrier_destroy(&done_barrier);
    pthread_barrier_destroy(&release_barrier);
    pthread_cond_destroy(&share_state.cond);
    pthread_mutex_destroy(&share_state.mutex);
  }

  long total_frames = 0;
  double total_e2e_ms = 0.0, total_pp_ms = 0.0, total_sync_ms = 0.0;
  double total_mask_verify_ms = 0.0;
  uint64_t total_sync_bytes = 0;
  double min_e2e_ms = DBL_MAX, max_e2e_ms = 0.0;
  int ret_code = 0;
  for (int i = 0; i < fps_workers; i++) {
    if (args[i].ret < 0)
      ret_code = 1;
    total_frames += args[i].frames;
    total_e2e_ms += args[i].sum_e2e_ms;
    total_pp_ms += args[i].sum_pp_ms;
    total_sync_ms += args[i].sum_sync_ms;
    total_mask_verify_ms += args[i].sum_mask_verify_ms;
    total_sync_bytes += args[i].sum_sync_bytes;
    if (args[i].frames > 0 && args[i].min_e2e_ms < min_e2e_ms) {
      min_e2e_ms = args[i].min_e2e_ms;
    }
    if (args[i].max_e2e_ms > max_e2e_ms)
      max_e2e_ms = args[i].max_e2e_ms;
  }
  if (ret_code)
    return ret_code;

  double measured_s = (finish_ms - start_ms) / 1000.0;
  double fps = measured_s > 0.0 ? (double)total_frames / measured_s : 0.0;
  double avg_e2e_ms =
      total_frames > 0 ? total_e2e_ms / (double)total_frames : 0.0;
  double avg_pp_ms =
      total_frames > 0 ? total_pp_ms / (double)total_frames : 0.0;
  double avg_sync_ms =
      total_frames > 0 ? total_sync_ms / (double)total_frames : 0.0;
  double avg_mask_verify_ms =
      total_frames > 0 ? total_mask_verify_ms / (double)total_frames : 0.0;
  double avg_sync_bytes =
      total_frames > 0 ? (double)total_sync_bytes / (double)total_frames : 0.0;
  if (min_e2e_ms == DBL_MAX)
    min_e2e_ms = 0.0;

  printf("\n=== bench_rknn_perf offline FPS ===\n");
  printf("model:       %s\n", model_path);
  printf("workers:     %d   seconds: %.3f   warmup: %d\n", fps_workers,
         fps_seconds, warmup);
  printf("core:        %s   fps_core_map: %s   neon: %d\n", core_str,
         fps_core_map && *fps_core_map ? fps_core_map : "(none)", HAVE_NEON);
  printf("sram:        %s   init_flags: 0x%08x\n", sram_str, init_flags);
  printf("postproc:    %s\n", pp_str);
  printf("score_sum:   %s\n", g_score_sum_enabled ? "on" : "off");
  printf("mask_verify: %s\n", g_mask_verify_enabled ? "on" : "off");
  printf("mask_classes: %s\n", g_mask_class_ids_spec);
  printf("mask_size:    %dx%d\n", g_mask_output_width, g_mask_output_height);
  printf("frames:      %ld\n", total_frames);
  printf("offline_fps: %.2f\n", fps);
  printf("e2e_ms:      avg=%.3f  min=%.3f  max=%.3f\n", avg_e2e_ms, min_e2e_ms,
         max_e2e_ms);
  printf("postproc_ms: %.3f\n", avg_pp_ms);
  if (pp == PP_SEG_PREDIST || pp == PP_SEG_PREDFL) {
    printf("output_sync: avg_ms=%.3f  avg_native_bytes=%.0f\n", avg_sync_ms,
           avg_sync_bytes);
    printf("mask_verify: avg_ms=%.3f (excluded from postproc/e2e)\n",
           avg_mask_verify_ms);
  }
  for (int i = 0; i < fps_workers; i++) {
    double worker_avg =
        args[i].frames > 0 ? args[i].sum_e2e_ms / (double)args[i].frames : 0.0;
    printf("worker[%d]:   frames=%ld  avg_e2e_ms=%.3f  last_kept=%d  "
           "ready=0x%02x  outcome=%s\n",
           i, args[i].frames, worker_avg, args[i].last_kept,
           args[i].last_ready_mask,
           args[i].last_fetch_outcome ? args[i].last_fetch_outcome : "none");
  }

  if (json_out) {
    FILE *jp = fopen(json_out, "w");
    if (jp) {
      fprintf(jp,
              "{\"mode\":\"fps\",\"model\":\"%s\",\"core\":\"%s\","
              "\"fps_core_map\":\"%s\",\"sram\":\"%s\",\"init_flags\":%u,"
              "\"postproc\":\"%s\","
              "\"workers\":%d,\"seconds\":%.4f,\"measured_seconds\":%.4f,"
              "\"warmup\":%d,\"neon\":%d,\"frames\":%ld,"
              "\"offline_fps\":%.4f,\"e2e_avg_ms\":%.4f,"
              "\"e2e_min_ms\":%.4f,\"e2e_max_ms\":%.4f,"
              "\"postproc_ms\":%.4f,\"output_sync_ms\":%.4f,"
              "\"native_sync_bytes\":%.0f,\"score_sum\":\"%s\","
              "\"mask_verify\":\"%s\",\"mask_verify_ms\":%.4f,"
              "\"mask_class_ids\":\"%s\",\"mask_output_size\":\"%dx%d\","
              "\"ready_mask\":%u,\"fetch_outcome\":\"%s\","
              "\"worker_frames\":[",
              model_path, core_str, fps_core_map ? fps_core_map : "", sram_str,
              init_flags, pp_str, fps_workers, fps_seconds, measured_s, warmup,
              HAVE_NEON, total_frames, fps, avg_e2e_ms, min_e2e_ms, max_e2e_ms,
              avg_pp_ms, avg_sync_ms, avg_sync_bytes,
              g_score_sum_enabled ? "on" : "off",
              g_mask_verify_enabled ? "on" : "off", avg_mask_verify_ms,
              g_mask_class_ids_spec, g_mask_output_width, g_mask_output_height,
              fps_workers > 0 ? args[0].last_ready_mask : 0U,
              fps_workers > 0 && args[0].last_fetch_outcome
                  ? args[0].last_fetch_outcome
                  : "none");
      for (int i = 0; i < fps_workers; i++) {
        fprintf(jp, "%s%ld", i ? "," : "", args[i].frames);
      }
      fprintf(jp, "],\"worker_ready_masks\":[");
      for (int i = 0; i < fps_workers; i++) {
        fprintf(jp, "%s%u", i ? "," : "", args[i].last_ready_mask);
      }
      fprintf(jp, "],\"worker_fetch_outcomes\":[");
      for (int i = 0; i < fps_workers; i++) {
        fprintf(jp, "%s\"%s\"", i ? "," : "",
                args[i].last_fetch_outcome ? args[i].last_fetch_outcome
                                           : "none");
      }
      fprintf(jp, "]}\n");
      fclose(jp);
    }
  }
  return 0;
}

static void usage(const char *prog) {
  fprintf(
      stderr,
      "用法: %s --model M.rknn [选项]\n"
      "  --warmup N        预热轮次 (默认 10)\n"
      "  --runs   N        计时轮次 (默认 200)\n"
      "  --core   0|1|2|all   NPU 核心选择 (默认 all)\n"
      "  --postproc predist|predfl|seg_predist|seg_predfl|none  CPU 后处理 "
      "(默认 none)\n"
      "  --conf-thr F      后处理置信度阈值 (默认 0.25)\n"
      "  --iou-thr F       NMS IoU 阈值 (默认 0.45)\n"
      "  --max-det N       NMS 后最多保留实例数 (默认 300)\n"
      "  --score-sum on|off  分割模型 score_sum 预筛 A/B (默认 on)\n"
      "  --mask-verify on|off  mask 像素计数/hash 校验 (默认 off)\n"
      "  --mask-class-ids all|L  生成 mask 的类别 ID，如 0,1 (默认 0)\n"
      "  --mask-output-size WxH  mask 输出坐标尺寸 (默认 640x480)\n"
      "  --input F.rgb     单帧 HWC RGB uint8 原始输入；仅延迟模式使用\n"
      "  --sram off|private|shared  RKNN SRAM 初始化策略 (默认 off)\n"
      "  --fps-workers N   离线 FPS worker/context 数；设置后进入 FPS 模式\n"
      "  --fps-seconds F   离线 FPS 计时秒数 (默认 10)\n"
      "  --fps-core-map L  FPS worker core 轮转列表，如 0,1,2 或 all,all\n"
      "  --json   F.json   写出机器可读结果\n",
      prog);
}

int main(int argc, char **argv) {
  const char *model_path = NULL;
  const char *input_path = NULL;
  const char *json_out = NULL;
  const char *pp_str = "none";
  const char *core_str = "all";
  const char *fps_core_map = NULL;
  const char *sram_str = "off";
  const char *score_sum_str = "on";
  const char *mask_verify_str = "off";
  const char *mask_class_ids_str = "0";
  const char *mask_output_size_str = "640x480";
  int warmup = 10, runs = 200;
  int fps_workers = 0;
  double fps_seconds = 10.0;
  float conf_thr = 0.25f;
  float iou_thr = 0.45f;
  int max_det = 300;

  static struct option opts[] = {
      {"model", required_argument, 0, 'm'},
      {"warmup", required_argument, 0, 'w'},
      {"runs", required_argument, 0, 'r'},
      {"core", required_argument, 0, 'c'},
      {"postproc", required_argument, 0, 'p'},
      {"conf-thr", required_argument, 0, 't'},
      {"iou-thr", required_argument, 0, 1003},
      {"max-det", required_argument, 0, 1004},
      {"score-sum", required_argument, 0, 1005},
      {"input", required_argument, 0, 1006},
      {"mask-verify", required_argument, 0, 1007},
      {"mask-class-ids", required_argument, 0, 1008},
      {"mask-output-size", required_argument, 0, 1009},
      {"json", required_argument, 0, 'j'},
      {"sram", required_argument, 0, 's'},
      {"fps-workers", required_argument, 0, 1000},
      {"fps-seconds", required_argument, 0, 1001},
      {"fps-core-map", required_argument, 0, 1002},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}};
  int opt;
  while ((opt = getopt_long(argc, argv, "m:w:r:c:p:t:j:s:h", opts, NULL)) !=
         -1) {
    switch (opt) {
    case 'm':
      model_path = optarg;
      break;
    case 'w':
      warmup = atoi(optarg);
      break;
    case 'r':
      runs = atoi(optarg);
      break;
    case 'c':
      core_str = optarg;
      break;
    case 'p':
      pp_str = optarg;
      break;
    case 't':
      conf_thr = (float)atof(optarg);
      break;
    case 'j':
      json_out = optarg;
      break;
    case 's':
      sram_str = normalize_sram_mode(optarg);
      break;
    case 1000:
      fps_workers = atoi(optarg);
      break;
    case 1001:
      fps_seconds = atof(optarg);
      break;
    case 1002:
      fps_core_map = optarg;
      break;
    case 1003:
      iou_thr = (float)atof(optarg);
      break;
    case 1004:
      max_det = atoi(optarg);
      break;
    case 1005:
      score_sum_str = optarg;
      break;
    case 1006:
      input_path = optarg;
      break;
    case 1007:
      mask_verify_str = optarg;
      break;
    case 1008:
      mask_class_ids_str = optarg;
      break;
    case 1009:
      mask_output_size_str = optarg;
      break;
    case 'h':
      usage(argv[0]);
      return 0;
    default:
      usage(argv[0]);
      return 2;
    }
  }
  if (!model_path) {
    usage(argv[0]);
    return 2;
  }
  if (warmup < 0 || runs <= 0 || !std::isfinite(conf_thr) || conf_thr < 0.f ||
      conf_thr > 1.f || !std::isfinite(iou_thr) || iou_thr < 0.f ||
      iou_thr > 1.f || max_det <= 0 || fps_workers < 0 ||
      !std::isfinite(fps_seconds) || fps_seconds <= 0.0) {
    fprintf(stderr, "invalid numeric CLI value\n");
    return 2;
  }
  if (strcmp(pp_str, "none") && strcmp(pp_str, "predist") &&
      strcmp(pp_str, "predfl") && strcmp(pp_str, "seg_predist") &&
      strcmp(pp_str, "seg_predfl")) {
    fprintf(stderr, "invalid --postproc value: %s\n", pp_str);
    return 2;
  }
  if (strcmp(core_str, "0") && strcmp(core_str, "1") && strcmp(core_str, "2") &&
      strcmp(core_str, "all")) {
    fprintf(stderr, "invalid --core value: %s\n", core_str);
    return 2;
  }
  if (strcmp(score_sum_str, "on") && strcmp(score_sum_str, "off")) {
    fprintf(stderr, "invalid --score-sum value: %s\n", score_sum_str);
    return 2;
  }
  if (strcmp(mask_verify_str, "on") && strcmp(mask_verify_str, "off")) {
    fprintf(stderr, "invalid --mask-verify value: %s\n", mask_verify_str);
    return 2;
  }
  if (!parse_mask_class_ids(mask_class_ids_str)) {
    fprintf(stderr, "invalid --mask-class-ids value: %s\n", mask_class_ids_str);
    return 2;
  }
  if (!parse_mask_output_size(mask_output_size_str)) {
    fprintf(stderr, "invalid --mask-output-size value: %s\n",
            mask_output_size_str);
    return 2;
  }
  if (fps_workers > 0 && input_path) {
    fprintf(stderr,
            "--input is only supported in single-context latency mode\n");
    return 2;
  }
  g_score_sum_enabled = !strcmp(score_sum_str, "on");
  g_mask_verify_enabled = !strcmp(mask_verify_str, "on");
  g_mask_class_ids_spec = mask_class_ids_str;
  pp_mode_t pp = parse_pp(pp_str);
  uint32_t init_flags = parse_sram_flags(sram_str);

  FILE *fp = fopen(model_path, "rb");
  if (!fp) {
    perror("fopen");
    return 1;
  }
  fseek(fp, 0, SEEK_END);
  size_t sz = ftell(fp);
  rewind(fp);
  void *buf = malloc(sz);
  if (fread(buf, 1, sz, fp) != sz) {
    fprintf(stderr, "read fail\n");
    free(buf);
    fclose(fp);
    return 1;
  }
  fclose(fp);

  if (fps_workers == 1 && is_shared_sram_mode(sram_str) &&
      flags_request_shared_sram(init_flags)) {
    fprintf(stderr, "single-worker --sram shared has no consumer context; "
                    "using private SRAM\n");
    sram_str = "private";
    init_flags = parse_sram_flags(sram_str);
  }

  if (fps_workers > 0) {
    int ret = run_fps_mode(model_path, buf, sz, json_out, pp_str, pp, conf_thr,
                           iou_thr, max_det, warmup, core_str, fps_core_map,
                           sram_str, init_flags, fps_workers, fps_seconds);
    free(buf);
    return ret;
  }

  rknn_context ctx;
  rknn_init_extend init_extend;
  memset(&init_extend, 0, sizeof(init_extend));
  if (is_shared_sram_mode(sram_str) && flags_request_shared_sram(init_flags)) {
    fprintf(stderr, "single-context --sram shared has no source context; using "
                    "private SRAM\n");
    sram_str = "private";
    init_flags = parse_sram_flags("private");
  }
  int ret =
      rknn_init(&ctx, buf, sz, init_flags, init_flags ? &init_extend : NULL);
  free(buf);
  if (ret < 0) {
    fprintf(stderr, "rknn_init: %d\n", ret);
    return 1;
  }
  int core_mask = RKNN_NPU_CORE_AUTO;
  if (!parse_core(core_str, &core_mask)) {
    fprintf(stderr, "invalid --core value: %s\n", core_str);
    rknn_destroy(ctx);
    return 2;
  }
  ret = rknn_set_core_mask(ctx, static_cast<rknn_core_mask>(core_mask));
  if (ret != 0) {
    fprintf(stderr, "rknn_set_core_mask: %d\n", ret);
    rknn_destroy(ctx);
    return 1;
  }

  rknn_sdk_version sdk_ver;
  memset(&sdk_ver, 0, sizeof(sdk_ver));
  int sdk_ret =
      rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver));

  rknn_input_output_num io;
  memset(&io, 0, sizeof(io));
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
  if (ret < 0) {
    fprintf(stderr, "RKNN_QUERY_IN_OUT_NUM: %d\n", ret);
    rknn_destroy(ctx);
    return 1;
  }
  if (io.n_input < 1 || io.n_input > MAX_IO || io.n_output > MAX_IO) {
    fprintf(stderr, "unsupported io count: input=%u output=%u\n", io.n_input,
            io.n_output);
    rknn_destroy(ctx);
    return 1;
  }
  if (validate_output_count(pp, io.n_output) < 0) {
    rknn_destroy(ctx);
    return 1;
  }

  rknn_tensor_attr in_attrs[MAX_IO];
  for (uint32_t i = 0; i < io.n_input && i < MAX_IO; i++) {
    memset(&in_attrs[i], 0, sizeof(in_attrs[i]));
    in_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attrs[i],
                     sizeof(in_attrs[i]));
    if (ret < 0) {
      fprintf(stderr, "RKNN_QUERY_INPUT_ATTR[%u]: %d\n", i, ret);
      rknn_destroy(ctx);
      return 1;
    }
  }
  rknn_tensor_attr out_attrs[MAX_IO];
  for (uint32_t i = 0; i < io.n_output && i < MAX_IO; i++) {
    memset(&out_attrs[i], 0, sizeof(out_attrs[i]));
    out_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attrs[i],
                     sizeof(out_attrs[i]));
    if (ret < 0) {
      fprintf(stderr, "RKNN_QUERY_OUTPUT_ATTR[%u]: %d\n", i, ret);
      rknn_destroy(ctx);
      return 1;
    }
  }

  if (in_attrs[0].n_dims != 4 || in_attrs[0].dims[0] != 1) {
    fprintf(stderr, "input[0] must be a static rank-4 tensor with batch=1\n");
    rknn_destroy(ctx);
    return 1;
  }
  if (in_attrs[0].fmt == RKNN_TENSOR_NHWC) {
    g_input_h = in_attrs[0].dims[1];
    g_input_w = in_attrs[0].dims[2];
  } else if (in_attrs[0].fmt == RKNN_TENSOR_NCHW) {
    g_input_h = in_attrs[0].dims[2];
    g_input_w = in_attrs[0].dims[3];
  } else {
    fprintf(stderr, "input[0] must use NCHW or NHWC layout\n");
    rknn_destroy(ctx);
    return 1;
  }
  if (validate_output_contract(pp, io.n_output, out_attrs) < 0) {
    rknn_destroy(ctx);
    return 1;
  }

  size_t in_size = 1;
  for (uint32_t d = 0; d < in_attrs[0].n_dims; d++)
    in_size *= in_attrs[0].dims[d];
  void *dummy = calloc(in_size, 1);
  if (!dummy) {
    fprintf(stderr, "input allocation failed: %zu bytes\n", in_size);
    rknn_destroy(ctx);
    return 1;
  }
  if (input_path) {
    FILE *input_file = fopen(input_path, "rb");
    if (!input_file) {
      perror("fopen input");
      free(dummy);
      rknn_destroy(ctx);
      return 1;
    }
    size_t input_read = fread(dummy, 1, in_size, input_file);
    int trailing = fgetc(input_file);
    fclose(input_file);
    if (input_read != in_size || trailing != EOF) {
      fprintf(stderr, "input size mismatch: expected exactly %zu bytes\n",
              in_size);
      free(dummy);
      rknn_destroy(ctx);
      return 1;
    }
  }
  rknn_input inp[1];
  memset(inp, 0, sizeof(inp));
  inp[0].index = 0;
  inp[0].buf = dummy;
  inp[0].size = in_size;
  inp[0].pass_through = 0;
  inp[0].type = RKNN_TENSOR_UINT8;
  inp[0].fmt = RKNN_TENSOR_NHWC;

  paddleyolo_rknn::postprocess::FiveOutputRuntime full_io;
  if (pp == PP_SEG_PREDIST || pp == PP_SEG_PREDFL) {
    if (!full_io.Initialize(ctx, in_attrs[0], out_attrs, io.n_output) ||
        !full_io.SetInput(dummy, in_size)) {
      fprintf(stderr, "five-output full-IO initialization failed\n");
      full_io.Release();
      free(dummy);
      rknn_destroy(ctx);
      return 1;
    }
  }

  rknn_output outs[MAX_IO];

  for (int w = 0; w < warmup; w++) {
    if (full_io.IsInitialized()) {
      ret = full_io.Run();
      if (ret != 0) {
        fprintf(stderr, "warmup full-IO run: %d\n", ret);
        full_io.Release();
        free(dummy);
        rknn_destroy(ctx);
        return 1;
      }
      run_staged_seg_postproc(&full_io, out_attrs, pp == PP_SEG_PREDFL,
                              conf_thr, iou_thr, max_det);
      if (staged_postproc_failed()) {
        fprintf(stderr, "warmup staged postprocess failed: %s\n",
                g_last_fetch_outcome);
        full_io.Release();
        free(dummy);
        rknn_destroy(ctx);
        return 1;
      }
      continue;
    }
    ret = rknn_inputs_set(ctx, 1, inp);
    if (ret < 0) {
      fprintf(stderr, "warmup rknn_inputs_set: %d\n", ret);
      free(dummy);
      rknn_destroy(ctx);
      return 1;
    }
    ret = rknn_run(ctx, NULL);
    if (ret < 0) {
      fprintf(stderr, "warmup rknn_run: %d\n", ret);
      free(dummy);
      rknn_destroy(ctx);
      return 1;
    }
    memset(outs, 0, sizeof(outs));
    for (uint32_t o = 0; o < io.n_output; o++) {
      outs[o].index = o;
      outs[o].want_float = 0;
    }
    ret = rknn_outputs_get(ctx, io.n_output, outs, NULL);
    if (ret < 0) {
      fprintf(stderr, "warmup rknn_outputs_get: %d\n", ret);
      free(dummy);
      rknn_destroy(ctx);
      return 1;
    }
    run_postproc(pp, outs, out_attrs, io.n_output, conf_thr, iou_thr, max_det);
    ret = rknn_outputs_release(ctx, io.n_output, outs);
    if (ret < 0) {
      fprintf(stderr, "warmup rknn_outputs_release: %d\n", ret);
      free(dummy);
      rknn_destroy(ctx);
      return 1;
    }
  }

  /* 收集每帧时间用于百分位统计 */
  double *npu_times = (double *)malloc(runs * sizeof(double));
  double *io_wall_times = (double *)malloc(runs * sizeof(double));
  double *e2e_times = (double *)malloc(runs * sizeof(double));
  if (npu_times == nullptr || io_wall_times == nullptr ||
      e2e_times == nullptr) {
    fprintf(stderr, "timing allocation failed: runs=%d\n", runs);
    free(npu_times);
    free(io_wall_times);
    free(e2e_times);
    full_io.Release();
    free_postproc_buffers();
    free(dummy);
    rknn_destroy(ctx);
    return 1;
  }
  double sum_npu_ms = 0, sum_io_wall_ms = 0, sum_e2e_ms = 0, sum_pp_ms = 0;
  double sum_output_sync_ms = 0.0, sum_native_sync_bytes = 0.0;
  double sum_mask_verify_ms = 0.0;
  double max_e2e = 0, min_e2e = 1e9;
  for (int i = 0; i < runs; i++) {
    double t0 = now_ms();
    if (full_io.IsInitialized()) {
      ret = full_io.Run();
      if (ret != 0) {
        fprintf(stderr, "full-IO run: %d\n", ret);
        free(npu_times);
        free(io_wall_times);
        free(e2e_times);
        full_io.Release();
        free_postproc_buffers();
        free(dummy);
        rknn_destroy(ctx);
        return 1;
      }
      const double run_end = now_ms();
      double pp_ms = run_staged_seg_postproc(
          &full_io, out_attrs, pp == PP_SEG_PREDFL, conf_thr, iou_thr, max_det);
      if (staged_postproc_failed()) {
        fprintf(stderr, "staged postprocess failed: %s\n",
                g_last_fetch_outcome);
        free(npu_times);
        free(io_wall_times);
        free(e2e_times);
        full_io.Release();
        free_postproc_buffers();
        free(dummy);
        rknn_destroy(ctx);
        return 1;
      }
      const double e2e = std::max(0.0, now_ms() - t0 - g_last_mask_verify_ms);

      rknn_perf_run perf_run;
      memset(&perf_run, 0, sizeof(perf_run));
      ret = rknn_query(ctx, RKNN_QUERY_PERF_RUN, &perf_run, sizeof(perf_run));
      if (ret < 0 || perf_run.run_duration <= 0) {
        fprintf(stderr, "RKNN_QUERY_PERF_RUN failed: ret=%d duration=%lldus\n",
                ret, (long long)perf_run.run_duration);
        free(npu_times);
        free(io_wall_times);
        free(e2e_times);
        full_io.Release();
        free_postproc_buffers();
        free(dummy);
        rknn_destroy(ctx);
        return 1;
      }
      const double npu_ms = (double)perf_run.run_duration / 1000.0;
      const double io_wall_ms = (run_end - t0) + g_last_sync_ms;
      npu_times[i] = npu_ms;
      io_wall_times[i] = io_wall_ms;
      e2e_times[i] = e2e;
      sum_npu_ms += npu_ms;
      sum_io_wall_ms += io_wall_ms;
      sum_pp_ms += pp_ms;
      sum_output_sync_ms += g_last_sync_ms;
      sum_native_sync_bytes += static_cast<double>(g_last_sync_bytes);
      sum_mask_verify_ms += g_last_mask_verify_ms;
      sum_e2e_ms += e2e;
      if (e2e > max_e2e)
        max_e2e = e2e;
      if (e2e < min_e2e)
        min_e2e = e2e;
      continue;
    }
    ret = rknn_inputs_set(ctx, 1, inp);
    if (ret < 0) {
      fprintf(stderr, "rknn_inputs_set: %d\n", ret);
      free(npu_times);
      free(io_wall_times);
      free(e2e_times);
      free_postproc_buffers();
      free(dummy);
      rknn_destroy(ctx);
      return 1;
    }
    ret = rknn_run(ctx, NULL);
    if (ret < 0) {
      fprintf(stderr, "rknn_run: %d\n", ret);
      free(npu_times);
      free(io_wall_times);
      free(e2e_times);
      free_postproc_buffers();
      free(dummy);
      rknn_destroy(ctx);
      return 1;
    }
    memset(outs, 0, sizeof(outs));
    for (uint32_t o = 0; o < io.n_output; o++) {
      outs[o].index = o;
      outs[o].want_float = 0;
    }
    ret = rknn_outputs_get(ctx, io.n_output, outs, NULL);
    if (ret < 0) {
      fprintf(stderr, "rknn_outputs_get: %d\n", ret);
      free(npu_times);
      free(io_wall_times);
      free(e2e_times);
      free_postproc_buffers();
      free(dummy);
      rknn_destroy(ctx);
      return 1;
    }
    double t1 = now_ms();

    rknn_perf_run perf_run;
    memset(&perf_run, 0, sizeof(perf_run));
    ret = rknn_query(ctx, RKNN_QUERY_PERF_RUN, &perf_run, sizeof(perf_run));
    if (ret < 0 || perf_run.run_duration <= 0) {
      fprintf(stderr, "RKNN_QUERY_PERF_RUN failed: ret=%d duration=%lldus\n",
              ret, (long long)perf_run.run_duration);
      const int release_ret = rknn_outputs_release(ctx, io.n_output, outs);
      if (release_ret < 0) {
        fprintf(stderr, "rknn_outputs_release after query failure: %d\n",
                release_ret);
      }
      free(npu_times);
      free(io_wall_times);
      free(e2e_times);
      free_postproc_buffers();
      free(dummy);
      rknn_destroy(ctx);
      return 1;
    }
    double npu_ms = (double)perf_run.run_duration / 1000.0;
    double io_wall_ms = t1 - t0;
    npu_times[i] = npu_ms;
    io_wall_times[i] = io_wall_ms;
    sum_npu_ms += npu_ms;
    sum_io_wall_ms += io_wall_ms;

    double pp_ms = run_postproc(pp, outs, out_attrs, io.n_output, conf_thr,
                                iou_thr, max_det);
    sum_pp_ms += pp_ms;

    double release_start = now_ms();
    ret = rknn_outputs_release(ctx, io.n_output, outs);
    double release_ms = now_ms() - release_start;
    if (ret < 0) {
      fprintf(stderr, "rknn_outputs_release: %d\n", ret);
      free(npu_times);
      free(io_wall_times);
      free(e2e_times);
      free_postproc_buffers();
      free(dummy);
      rknn_destroy(ctx);
      return 1;
    }
    double e2e = io_wall_ms + pp_ms + release_ms;
    e2e_times[i] = e2e;
    sum_e2e_ms += e2e;
    if (e2e > max_e2e)
      max_e2e = e2e;
    if (e2e < min_e2e)
      min_e2e = e2e;
  }

  /* 排序算百分位 */
  for (int i = 0; i < runs - 1; i++)
    for (int j = i + 1; j < runs; j++)
      if (npu_times[j] < npu_times[i]) {
        double t = npu_times[i];
        npu_times[i] = npu_times[j];
        npu_times[j] = t;
      }
  for (int i = 0; i < runs - 1; i++)
    for (int j = i + 1; j < runs; j++)
      if (io_wall_times[j] < io_wall_times[i]) {
        double t = io_wall_times[i];
        io_wall_times[i] = io_wall_times[j];
        io_wall_times[j] = t;
      }
  for (int i = 0; i < runs - 1; i++)
    for (int j = i + 1; j < runs; j++)
      if (e2e_times[j] < e2e_times[i]) {
        double t = e2e_times[i];
        e2e_times[i] = e2e_times[j];
        e2e_times[j] = t;
      }

  double npu_avg_ms = sum_npu_ms / runs;
  double npu_p50 = npu_times[runs / 2];
  double npu_p90 = npu_times[(int)(runs * 0.9)];
  double npu_best = npu_times[0];
  double io_wall_avg_ms = sum_io_wall_ms / runs;
  double io_wall_p50 = io_wall_times[runs / 2];
  double io_wall_p90 = io_wall_times[(int)(runs * 0.9)];
  double io_wall_best = io_wall_times[0];
  double e2e_avg_ms = sum_e2e_ms / runs;
  double e2e_p50 = e2e_times[runs / 2];
  double e2e_p90 = e2e_times[(int)(runs * 0.9)];
  double e2e_best = e2e_times[0];
  double pp_avg_ms = sum_pp_ms / runs;
  double output_sync_avg_ms = sum_output_sync_ms / runs;
  double native_sync_bytes_avg = sum_native_sync_bytes / runs;
  double mask_verify_avg_ms = sum_mask_verify_ms / runs;

  printf("\n=== bench_rknn_perf ===\n");
  printf("model:       %s\n", model_path);
  printf("core:        %s   neon: %d   runs: %d  warmup: %d\n", core_str,
         HAVE_NEON, runs, warmup);
  printf("sram:        %s   init_flags: 0x%08x\n", sram_str, init_flags);
  printf("postproc:    %s\n", pp_str);
  printf("score_sum:   %s\n", score_sum_str);
  printf("mask_verify: %s\n", mask_verify_str);
  printf("mask_classes: %s\n", g_mask_class_ids_spec);
  printf("mask_size:    %dx%d\n", g_mask_output_width, g_mask_output_height);
  if (sdk_ret == 0) {
    printf("rknn_sdk:    api=%s   drv=%s\n", sdk_ver.api_version,
           sdk_ver.drv_version);
  }
  printf("npu_pure_ms: best=%.3f  P50=%.3f  P90=%.3f  avg=%.3f  "
         "(RKNN_QUERY_PERF_RUN)\n",
         npu_best, npu_p50, npu_p90, npu_avg_ms);
  printf("io_wall_ms:  best=%.3f  P50=%.3f  P90=%.3f  avg=%.3f\n", io_wall_best,
         io_wall_p50, io_wall_p90, io_wall_avg_ms);
  printf("postproc_ms: %.3f\n", pp_avg_ms);
  if (pp != PP_NONE) {
    printf("postproc_kept: %d\n", g_last_kept);
    printf("postproc_candidates: %d  nms_pairs=%ld  score_sum_applied=%d  "
           "score_sum_scanned=%d  class_anchors=%d  class_values=%ld\n",
           g_last_candidates, g_last_nms_pairs, g_last_score_sum_applied,
           g_last_sum_scanned, g_last_class_anchors, g_last_class_values);
    if (pp == PP_SEG_PREDIST || pp == PP_SEG_PREDFL) {
      printf("mask_tail: mode=%s proto_roi_area=%ld resized_pixels=%ld "
             "active_pixels=%ld hash=%016llx verify_avg_ms=%.3f\n",
             g_last_mask_mode, g_last_proto_roi_area, g_last_mask_pixels,
             g_last_mask_active, (unsigned long long)g_last_mask_hash,
             mask_verify_avg_ms);
      printf("output_sync: avg_ms=%.3f avg_native_bytes=%.0f ready=0x%02x "
             "outcome=%s\n",
             output_sync_avg_ms, native_sync_bytes_avg, g_last_ready_mask,
             g_last_fetch_outcome);
    }
  }
  printf("e2e_ms:      best=%.3f  P50=%.3f  P90=%.3f  avg=%.3f  (%.1f FPS "
         "best, %.1f FPS P50)\n",
         e2e_best, e2e_p50, e2e_p90, e2e_avg_ms, 1000.0 / e2e_best,
         1000.0 / e2e_p50);

  free(npu_times);
  free(io_wall_times);
  free(e2e_times);

  if (json_out) {
    FILE *jp = fopen(json_out, "w");
    if (jp) {
      fprintf(jp,
              "{\"model\":\"%s\",\"core\":\"%s\",\"sram\":\"%s\","
              "\"init_flags\":%u,\"postproc\":\"%s\",\"score_sum\":\"%s\","
              "\"runs\":%d,\"warmup\":%d,\"neon\":%d,"
              "\"rknn_api_version\":\"%s\",\"rknn_drv_version\":\"%s\","
              "\"npu_pure_best_ms\":%.4f,\"npu_pure_p50_ms\":%.4f,\"npu_pure_"
              "p90_ms\":%.4f,\"npu_pure_avg_ms\":%.4f,"
              "\"io_wall_best_ms\":%.4f,\"io_wall_p50_ms\":%.4f,\"io_wall_p90_"
              "ms\":%.4f,\"io_wall_avg_ms\":%.4f,"
              "\"postproc_ms\":%.4f,\"postproc_candidates\":%d,\"postproc_"
              "kept\":%d,\"nms_pairs\":%ld,"
              "\"score_sum_applied\":%d,\"score_sum_scanned\":%d,"
              "\"class_anchors_scanned\":%d,\"class_values_scanned\":%ld,"
              "\"output_sync_ms\":%.4f,\"native_sync_bytes\":%.0f,"
              "\"ready_mask\":%u,\"fetch_outcome\":\"%s\","
              "\"mask_verify\":\"%s\",\"mask_verify_ms\":%.4f,"
              "\"mask_class_ids\":\"%s\",\"mask_output_size\":\"%dx%d\","
              "\"mask_mode\":\"%s\",\"mask_proto_roi_area\":%ld,"
              "\"mask_resized_pixels\":%ld,\"mask_active_pixels\":%ld,\"mask_"
              "hash\":\"%016llx\","
              "\"e2e_best_ms\":%.4f,\"e2e_p50_ms\":%.4f,\"e2e_p90_ms\":%.4f,"
              "\"e2e_avg_ms\":%.4f}\n",
              model_path, core_str, sram_str, init_flags, pp_str, score_sum_str,
              runs, warmup, HAVE_NEON,
              sdk_ret == 0 ? sdk_ver.api_version : "unknown",
              sdk_ret == 0 ? sdk_ver.drv_version : "unknown", npu_best, npu_p50,
              npu_p90, npu_avg_ms, io_wall_best, io_wall_p50, io_wall_p90,
              io_wall_avg_ms, pp_avg_ms, g_last_candidates, g_last_kept,
              g_last_nms_pairs, g_last_score_sum_applied, g_last_sum_scanned,
              g_last_class_anchors, g_last_class_values, output_sync_avg_ms,
              native_sync_bytes_avg, g_last_ready_mask, g_last_fetch_outcome,
              mask_verify_str, mask_verify_avg_ms, g_mask_class_ids_spec,
              g_mask_output_width, g_mask_output_height, g_last_mask_mode,
              g_last_proto_roi_area, g_last_mask_pixels, g_last_mask_active,
              (unsigned long long)g_last_mask_hash, e2e_best, e2e_p50, e2e_p90,
              e2e_avg_ms);
      fclose(jp);
    }
  }

  full_io.Release();
  free_postproc_buffers();
  free(dummy);
  rknn_destroy(ctx);
  return 0;
}
