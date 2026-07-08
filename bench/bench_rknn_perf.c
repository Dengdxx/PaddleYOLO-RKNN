// Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
// SPDX-License-Identifier: AGPL-3.0-only

/*
 * bench_rknn_perf.c — RKNN INT8 模型评测：纯 NPU + 端到端 + 可选 NEON 后处理
 *
 * 用法:
 *   ./bench_rknn_perf --model X.rknn [--warmup 10] [--runs 200]
 *                     [--core 0|1|2|all] [--postproc predist|predfl|seg_predist|seg_predfl|none]
 *                     [--conf-thr 0.25] [--sram off|private|shared] [--json OUT.json]
 *   ./bench_rknn_perf --model X.rknn --fps-workers 3 --fps-seconds 10
 *                     [--fps-core-map 0,1,2] [--postproc ...] [--sram ...] [--json OUT.json]
 *
 * 输出:
 *   - 纯 NPU 时间       壁钟测量 (rknn_inputs_set + rknn_run + rknn_outputs_get)
 *   - 端到端时间        rknn_inputs_set + rknn_run + rknn_outputs_get + 后处理
 *   - 后处理时间        若 --postproc 非 none，则使用 NEON 加速 dequant/score-max/DFL
 *   - 离线 FPS          多 RKNN context 并发循环，不接摄像头/ROS/真实预处理
 *
 * 后处理路线 (NEON score-max + candidate-first decode; DFL softmax 标量但候选裁剪):
 *   predist (reg_max=1):
 *     scores: [1,nc,A] i8  → INT8 per-anchor max → quantized logit(conf) 比较
 *     boxes:  [1,4,A]  i8  → 仅对候选 anchor dequant ltrb
 *   predfl (reg_max=16):
 *     scores: [1,nc,A] i8  → INT8 per-anchor max → quantized logit(conf) 比较
 *     boxes:  [1,64,A] i8  → 仅对候选 anchor softmax(16) → · arange(16) → ltrb
 *   seg_predist:
 *     在 predist 之上额外只 gather/dequant 候选 mask_coeff，proto 有候选时再 dequant
 *   seg_predfl:
 *     在 predfl 之上额外只 gather/dequant 候选 mask_coeff，proto 有候选时再 dequant
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <float.h>
#include <time.h>
#include <math.h>
#include <getopt.h>
#include <pthread.h>
#include "rknn_api.h"

#if defined(__aarch64__) || defined(__ARM_NEON)
#  include <arm_neon.h>
#  define HAVE_NEON 1
#else
#  define HAVE_NEON 0
#endif

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

#if HAVE_NEON
static inline void neon_load_dequant16_i8_f32(const int8_t *src,
                                              int16x8_t vzp,
                                              float32x4_t vscale,
                                              float32x4_t *f0,
                                              float32x4_t *f1,
                                              float32x4_t *f2,
                                              float32x4_t *f3) {
    int8x16_t v = vld1q_s8(src);
    int16x8_t lo = vsubq_s16(vmovl_s8(vget_low_s8(v)), vzp);
    int16x8_t hi = vsubq_s16(vmovl_s8(vget_high_s8(v)), vzp);
    *f0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),  vscale);
    *f1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), vscale);
    *f2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),  vscale);
    *f3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), vscale);
}

/* Dequantize int8 → fp32: dst[i] = (src[i] - zp) * scale, NEON 加速 */
static void neon_dequant_i8_f32(const int8_t *src, float *dst, size_t n,
                                int32_t zp, float scale) {
    const int16x8_t vzp = vdupq_n_s16((int16_t)zp);
    const float32x4_t vscale = vdupq_n_f32(scale);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        float32x4_t f0, f1, f2, f3;
        neon_load_dequant16_i8_f32(src + i, vzp, vscale, &f0, &f1, &f2, &f3);
        vst1q_f32(dst + i + 0,  f0);
        vst1q_f32(dst + i + 4,  f1);
        vst1q_f32(dst + i + 8,  f2);
        vst1q_f32(dst + i + 12, f3);
    }
    for (; i < n; i++) dst[i] = (src[i] - zp) * scale;
}
#else
static void neon_dequant_i8_f32(const int8_t *src, float *dst, size_t n,
                                int32_t zp, float scale) {
    for (size_t i = 0; i < n; i++) dst[i] = (src[i] - zp) * scale;
}
#endif

static inline float logit_threshold(float conf_thr) {
    if (conf_thr <= 0.f) return -INFINITY;
    if (conf_thr >= 1.f) return INFINITY;
    return logf(conf_thr / (1.f - conf_thr));
}

/* ---------- 后处理实现 ---------- */

/* 每线程 buffer 池，避免每次 malloc；FPS 模式下各 worker 独立使用。 */
static _Thread_local float *g_bf = NULL, *g_mb = NULL, *g_pb = NULL;
static _Thread_local int *g_keep = NULL;
static _Thread_local int8_t *g_qmax = NULL;
static _Thread_local int g_lb = 0, g_lm = 0, g_lp = 0, g_lk = 0, g_lq = 0;
static _Thread_local int g_last_kept = 0;
static _Thread_local volatile float g_sink = 0.f;

/**
 * @brief 释放当前线程持有的后处理 scratch buffer。
 */
static void free_postproc_buffers(void) {
    free(g_bf); free(g_mb); free(g_pb); free(g_keep); free(g_qmax);
    g_bf = NULL; g_mb = NULL; g_pb = NULL; g_keep = NULL; g_qmax = NULL;
    g_lb = 0; g_lm = 0; g_lp = 0; g_lk = 0; g_lq = 0;
    g_last_kept = 0; g_sink = 0.f;
}

static float *ensure(float **p, int *cap, int need) {
    if (need > *cap) {
        free(*p);
        size_t bytes = sizeof(float) * (size_t)((need + 15) & ~15);
        *p = (float *)aligned_alloc(64, bytes);
        *cap = need;
    }
    return *p;
}

static int *ensure_i32(int **p, int *cap, int need) {
    if (need > *cap) {
        free(*p);
        size_t bytes = sizeof(int) * (size_t)((need + 15) & ~15);
        *p = (int *)aligned_alloc(64, bytes);
        *cap = need;
    }
    return *p;
}

static int8_t *ensure_i8(int8_t **p, int *cap, int need) {
    if (need > *cap) {
        free(*p);
        size_t bytes = sizeof(int8_t) * (size_t)((need + 63) & ~63);
        *p = (int8_t *)aligned_alloc(64, bytes);
        *cap = need;
    }
    return *p;
}

static int collect_scores_over_threshold_i8(const int8_t *scores,
                                            int nc,
                                            int anchors,
                                            int32_t zp,
                                            float scale,
                                            float conf_thr,
                                            int **keep_out) {
    int *keep = ensure_i32(&g_keep, &g_lk, anchors);

    if (conf_thr <= 0.f) {
        for (int a = 0; a < anchors; a++) keep[a] = a;
        if (keep_out) *keep_out = keep;
        g_last_kept = anchors;
        return anchors;
    }
    if (conf_thr >= 1.f) {
        if (keep_out) *keep_out = keep;
        g_last_kept = 0;
        return 0;
    }

    if (scale <= 0.f) {
        if (keep_out) *keep_out = keep;
        g_last_kept = 0;
        return 0;
    }

    const float thr = logit_threshold(conf_thr);
    int q_thr_i = (int)ceilf(thr / scale + (float)zp);
    if (q_thr_i <= INT8_MIN) {
        for (int a = 0; a < anchors; a++) keep[a] = a;
        if (keep_out) *keep_out = keep;
        g_last_kept = anchors;
        return anchors;
    }
    if (q_thr_i > INT8_MAX) {
        if (keep_out) *keep_out = keep;
        g_last_kept = 0;
        return 0;
    }

    int8_t *max_q = ensure_i8(&g_qmax, &g_lq, anchors);
    memset(max_q, INT8_MIN, (size_t)anchors);

#if HAVE_NEON
    for (int c = 0; c < nc; c++) {
        const int8_t *row = scores + (size_t)c * anchors;
        int a = 0;
        for (; a + 16 <= anchors; a += 16) {
            int8x16_t cur = vld1q_s8(max_q + a);
            int8x16_t val = vld1q_s8(row + a);
            vst1q_s8(max_q + a, vmaxq_s8(cur, val));
        }
        for (; a < anchors; a++) {
            if (row[a] > max_q[a]) max_q[a] = row[a];
        }
    }
#else
    for (int c = 0; c < nc; c++) {
        const int8_t *row = scores + (size_t)c * anchors;
        for (int a = 0; a < anchors; a++) {
            if (row[a] > max_q[a]) max_q[a] = row[a];
        }
    }
#endif

    const int8_t q_thr = (int8_t)q_thr_i;
    int kept = 0;
    for (int a = 0; a < anchors; a++) {
        if (max_q[a] >= q_thr) keep[kept++] = a;
    }
    if (keep_out) *keep_out = keep;
    g_last_kept = kept;
    return kept;
}

static inline float dequant_i8_scalar(int8_t value, int32_t zp, float scale) {
    return ((float)((int32_t)value - zp)) * scale;
}

static inline bool prefer_full_anchor_path(int kept, int anchors) {
    return kept > anchors / 4;
}

static float dfl_checksum_all_i8(const rknn_output *out,
                                 const rknn_tensor_attr *attr,
                                 int box_ch,
                                 int anchors,
                                 int reg_max,
                                 int transposed) {
    int box_n = box_ch * anchors;
    float *bf = ensure(&g_bf, &g_lb, box_n);
    neon_dequant_i8_f32((const int8_t *)out->buf, bf, box_n, attr->zp, attr->scale);

    float dfl_checksum = 0.f;
    for (int a = 0; a < anchors; a++) {
        for (int side = 0; side < 4; side++) {
            float maxv = -INFINITY;
            for (int k = 0; k < reg_max; k++) {
                float v;
                if (transposed) {
                    v = bf[a * box_ch + side * reg_max + k];
                } else {
                    v = bf[(side * reg_max + k) * anchors + a];
                }
                if (v > maxv) maxv = v;
            }
            float sum = 0.f, acc = 0.f;
            for (int k = 0; k < reg_max; k++) {
                float e;
                if (transposed) {
                    e = expf(bf[a * box_ch + side * reg_max + k] - maxv);
                } else {
                    e = expf(bf[(side * reg_max + k) * anchors + a] - maxv);
                }
                sum += e; acc += e * (float)k;
            }
            dfl_checksum += acc / sum;
        }
    }
    return dfl_checksum;
}

static float dfl_checksum_kept_i8(const rknn_output *out,
                                  const rknn_tensor_attr *attr,
                                  int anchors,
                                  int reg_max,
                                  const int *keep,
                                  int kept,
                                  int transposed) {
    float dfl_checksum = 0.f;
    const int8_t *boxes = (const int8_t *)out->buf;
    int box_ch = 4 * reg_max;
    for (int i = 0; i < kept; i++) {
        int a = keep[i];
        for (int side = 0; side < 4; side++) {
            float maxv = -INFINITY;
            for (int k = 0; k < reg_max; k++) {
                float v;
                if (transposed) {
                    v = dequant_i8_scalar(boxes[(size_t)a * box_ch + side * reg_max + k],
                                          attr->zp, attr->scale);
                } else {
                    int ch = side * reg_max + k;
                    v = dequant_i8_scalar(boxes[(size_t)ch * anchors + a],
                                          attr->zp, attr->scale);
                }
                if (v > maxv) maxv = v;
            }
            float sum = 0.f, acc = 0.f;
            for (int k = 0; k < reg_max; k++) {
                float v;
                if (transposed) {
                    v = dequant_i8_scalar(boxes[(size_t)a * box_ch + side * reg_max + k],
                                          attr->zp, attr->scale);
                } else {
                    int ch = side * reg_max + k;
                    v = dequant_i8_scalar(boxes[(size_t)ch * anchors + a],
                                          attr->zp, attr->scale);
                }
                float e = expf(v - maxv);
                sum += e; acc += e * (float)k;
            }
            dfl_checksum += acc / sum;
        }
    }
    return dfl_checksum;
}

static double dequant_mask_coeff_kept_i8(const rknn_output *out,
                                         const rknn_tensor_attr *attr,
                                         int anchors,
                                         const int *keep,
                                         int kept) {
    if (!out || !out->buf || !attr || !keep || kept <= 0) return 0.0;
    int nm = attr->dims[1];
    int need = nm * kept;
    float *mb = ensure(&g_mb, &g_lm, need);
    const int8_t *src = (const int8_t *)out->buf;
    double t0 = now_ms();
    for (int i = 0; i < kept; i++) {
        int a = keep[i];
        float *dst_row = mb + (size_t)i * nm;
        for (int mc = 0; mc < nm; mc++) {
            dst_row[mc] = dequant_i8_scalar(src[(size_t)mc * anchors + a], attr->zp, attr->scale);
        }
    }
    double t1 = now_ms();
    g_sink += mb[0] * 1e-12f;
    return t1 - t0;
}

static double dequant_mask_coeff_adaptive_i8(const rknn_output *out,
                                            const rknn_tensor_attr *attr,
                                            int anchors,
                                            const int *keep,
                                            int kept) {
    if (!out || !out->buf || !attr || kept <= 0) return 0.0;
    int nm = attr->dims[1];
    if (prefer_full_anchor_path(kept, anchors)) {
        int mc_n = nm * anchors;
        float *mb = ensure(&g_mb, &g_lm, mc_n);
        double t0 = now_ms();
        neon_dequant_i8_f32((const int8_t *)out->buf, mb, mc_n, attr->zp, attr->scale);
        double t1 = now_ms();
        g_sink += mb[0] * 1e-12f;
        return t1 - t0;
    }
    return dequant_mask_coeff_kept_i8(out, attr, anchors, keep, kept);
}

/* predist: reg_max=1 */
static double postproc_predist(const rknn_output *outs,
                               const rknn_tensor_attr *attrs,
                               uint32_t n_out, float conf_thr) {
    if (n_out < 2) return 0.0;
    int A  = attrs[0].dims[2];
    int NC = attrs[1].dims[1];

    double t0 = now_ms();
    int *keep = NULL;
    int kept = collect_scores_over_threshold_i8((const int8_t *)outs[1].buf,
                                                NC, A, attrs[1].zp,
                                                attrs[1].scale, conf_thr,
                                                &keep);
    const int8_t *boxes = (const int8_t *)outs[0].buf;
    float checksum = 0.f;
    for (int i = 0; i < kept; i++) {
        int a = keep[i];
        for (int side = 0; side < 4; side++) {
            checksum += dequant_i8_scalar(boxes[(size_t)side * A + a], attrs[0].zp, attrs[0].scale);
        }
    }
    double t1 = now_ms();
    g_sink += checksum * 1e-12f + (float)kept * 1e-9f;
    return t1 - t0;
}

/* predfl: reg_max=16，自动检测 legacy [1,4*reg_max,N] 或转置 [1,N,4*reg_max] */
static double postproc_predfl(const rknn_output *outs,
                              const rknn_tensor_attr *attrs,
                              uint32_t n_out, float conf_thr) {
    if (n_out < 2) return 0.0;
    int box_ch, A;
    int NC = attrs[1].dims[1];
    int transposed;
    /* 转置: dims[2] > 4 且整除 4 且 dims[1] > dims[2]（N >> 4*reg_max） */
    if (attrs[0].dims[2] > 4 && attrs[0].dims[2] % 4 == 0 && attrs[0].dims[1] > attrs[0].dims[2]) {
        box_ch = attrs[0].dims[2];
        A      = attrs[0].dims[1];
        transposed = 1;
    } else {
        box_ch = attrs[0].dims[1];
        A      = attrs[0].dims[2];
        transposed = 0;
    }
    int reg_max = box_ch / 4;

    double t0 = now_ms();
    int *keep = NULL;
    int kept = collect_scores_over_threshold_i8((const int8_t *)outs[1].buf,
                                                NC, A, attrs[1].zp,
                                                attrs[1].scale, conf_thr,
                                                &keep);

    float dfl_checksum = prefer_full_anchor_path(kept, A)
                             ? dfl_checksum_all_i8(&outs[0], &attrs[0], box_ch, A, reg_max, transposed)
                             : dfl_checksum_kept_i8(&outs[0], &attrs[0], A, reg_max, keep, kept, transposed);
    double t1 = now_ms();
    g_sink += dfl_checksum * 1e-12f + (float)kept * 1e-9f;
    return t1 - t0;
}

/* seg_predist: predist + dequant mask_coeff/proto */
static double postproc_seg_predist(const rknn_output *outs,
                                   const rknn_tensor_attr *attrs,
                                   uint32_t n_out, float conf_thr) {
    if (n_out < 4) return 0.0;
    double t = postproc_predist(outs, attrs, 2, conf_thr);
    int nm = attrs[2].dims[1];
    int A  = attrs[2].dims[2];
    int Hp = attrs[3].dims[2];
    int Wp = attrs[3].dims[3];
    int proto_n = nm * Hp * Wp;
    double extra_ms = 0.0;
    if (g_last_kept > 0) {
        extra_ms += dequant_mask_coeff_adaptive_i8(&outs[2], &attrs[2], A, g_keep, g_last_kept);
        float *pb = ensure(&g_pb, &g_lp, proto_n);
        double t0 = now_ms();
        neon_dequant_i8_f32((const int8_t *)outs[3].buf, pb, proto_n,
                            attrs[3].zp, attrs[3].scale);
        double t1 = now_ms();
        extra_ms += t1 - t0;
    }
    return t + extra_ms;
}

/* seg_predfl: 5 输出，转置 DFL + score_sum
 * [0]=DFL [1,N,4*reg_max], [1]=cls [1,nc,N], [2]=coeff [1,nm,N],
 * [3]=proto [1,nm,H,W], [4]=score_sum [1,1,N]
 * 仅用于 5 输出新格式 bench；4 输出 legacy 模型请用 --postproc predfl
 */
static double postproc_seg_predfl(const rknn_output *outs,
                                  const rknn_tensor_attr *attrs,
                                  uint32_t n_out, float conf_thr) {
    if (n_out < 5) return 0.0;
    int coeff_idx = 2;
    int proto_idx = 3;

    double t = postproc_predfl(outs, attrs, 2, conf_thr);
    int nm = attrs[coeff_idx].dims[1];
    int A  = attrs[coeff_idx].dims[2];
    int Hp = attrs[proto_idx].dims[2];
    int Wp = attrs[proto_idx].dims[3];
    int proto_n = nm * Hp * Wp;
    double extra_ms = 0.0;
    if (g_last_kept > 0) {
        extra_ms += dequant_mask_coeff_adaptive_i8(&outs[coeff_idx], &attrs[coeff_idx],
                                                   A, g_keep, g_last_kept);
        float *pb = ensure(&g_pb, &g_lp, proto_n);
        double t0 = now_ms();
        neon_dequant_i8_f32((const int8_t *)outs[proto_idx].buf, pb, proto_n,
                            attrs[proto_idx].zp, attrs[proto_idx].scale);
        double t1 = now_ms();
        extra_ms += t1 - t0;
    }
    return t + extra_ms;
}

typedef enum { PP_NONE, PP_PREDIST, PP_PREDFL, PP_SEG_PREDIST, PP_SEG_PREDFL } pp_mode_t;

static pp_mode_t parse_pp(const char *s) {
    if (!s || !strcmp(s, "none")) return PP_NONE;
    if (!strcmp(s, "predist"))     return PP_PREDIST;
    if (!strcmp(s, "predfl"))      return PP_PREDFL;
    if (!strcmp(s, "seg_predist")) return PP_SEG_PREDIST;
    if (!strcmp(s, "seg_predfl"))  return PP_SEG_PREDFL;
    return PP_NONE;
}

static int parse_core(const char *s) {
    if (!s)                  return RKNN_NPU_CORE_AUTO;
    if (!strcmp(s, "all"))   return RKNN_NPU_CORE_0_1_2;
    if (!strcmp(s, "0"))     return RKNN_NPU_CORE_0;
    if (!strcmp(s, "1"))     return RKNN_NPU_CORE_1;
    if (!strcmp(s, "2"))     return RKNN_NPU_CORE_2;
    return RKNN_NPU_CORE_AUTO;
}

static const char *normalize_sram_mode(const char *s) {
    if (!s || !strcmp(s, "off") || !strcmp(s, "private") || !strcmp(s, "shared")) {
        return s ? s : "off";
    }
    return "off";
}

static uint32_t parse_sram_flags(const char *s) {
    const char *mode = normalize_sram_mode(s);
    if (!strcmp(mode, "off")) return 0;
#ifdef RKNN_FLAG_ENABLE_SRAM
    uint32_t flags = RKNN_FLAG_ENABLE_SRAM;
#ifdef RKNN_FLAG_SHARE_SRAM
    if (!strcmp(mode, "shared")) flags |= RKNN_FLAG_SHARE_SRAM;
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
    int warmup;
    pthread_barrier_t *ready_barrier;
    pthread_barrier_t *start_barrier;
    pthread_barrier_t *done_barrier;
    double *end_ms;
    long frames;
    double sum_e2e_ms;
    double sum_pp_ms;
    double min_e2e_ms;
    double max_e2e_ms;
    int last_kept;
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
static double run_postproc(pp_mode_t pp,
                           const rknn_output *outs,
                           const rknn_tensor_attr *out_attrs,
                           uint32_t n_output,
                           float conf_thr) {
    switch (pp) {
        case PP_PREDIST:
            return postproc_predist(outs, out_attrs, n_output, conf_thr);
        case PP_PREDFL:
            return postproc_predfl(outs, out_attrs, n_output, conf_thr);
        case PP_SEG_PREDIST:
            return postproc_seg_predist(outs, out_attrs, n_output, conf_thr);
        case PP_SEG_PREDFL:
            return postproc_seg_predfl(outs, out_attrs, n_output, conf_thr);
        default:
            return 0.0;
    }
}

/**
 * @brief 解析 FPS worker 的 core 轮转列表。
 * @param spec 逗号分隔的 core 列表，例如 `0,1,2`。
 * @param masks 输出 core mask 数组。
 * @param max_masks `masks` 容量。
 * @return 成功解析出的 core 数量；0 表示使用 `--core` 的默认绑定。
 */
static int parse_core_map(const char *spec, int *masks, int max_masks) {
    if (!spec || !*spec || max_masks <= 0) return 0;

    char tmp[128];
    strncpy(tmp, spec, sizeof(tmp) - 1);
    tmp[sizeof(tmp) - 1] = '\0';

    int n = 0;
    char *tok = strtok(tmp, ",");
    while (tok && n < max_masks) {
        masks[n++] = parse_core(tok);
        tok = strtok(NULL, ",");
    }
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
 * @param io 输出输入/输出数量。
 * @param in_attrs 输出输入 tensor 属性。
 * @param out_attrs 输出输出 tensor 属性。
 * @param dummy 输出 dummy 输入 buffer，调用者负责释放。
 * @param in_size 输出 dummy 输入字节数。
 * @param inp 输出 RKNN 输入描述。
 * @return 0 表示成功；负数表示 RKNN 或内存错误。
 */
static int setup_context(rknn_context *ctx,
                         const void *model_buf,
                         size_t model_size,
                         int core_mask,
                         uint32_t init_flags,
                         rknn_context share_ctx,
                         rknn_input_output_num *io,
                         rknn_tensor_attr *in_attrs,
                         rknn_tensor_attr *out_attrs,
                         void **dummy,
                         size_t *in_size,
                         rknn_input *inp) {
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
    rknn_set_core_mask(*ctx, core_mask);

    memset(io, 0, sizeof(*io));
    ret = rknn_query(*ctx, RKNN_QUERY_IN_OUT_NUM, io, sizeof(*io));
    if (ret < 0) {
        fprintf(stderr, "RKNN_QUERY_IN_OUT_NUM: %d\n", ret);
        return ret;
    }
    if (io->n_input < 1 || io->n_input > MAX_IO || io->n_output > MAX_IO) {
        fprintf(stderr, "unsupported io count: input=%u output=%u\n",
                io->n_input, io->n_output);
        return -1;
    }

    for (uint32_t i = 0; i < io->n_input; i++) {
        memset(&in_attrs[i], 0, sizeof(in_attrs[i]));
        in_attrs[i].index = i;
        ret = rknn_query(*ctx, RKNN_QUERY_INPUT_ATTR, &in_attrs[i], sizeof(in_attrs[i]));
        if (ret < 0) {
            fprintf(stderr, "RKNN_QUERY_INPUT_ATTR[%u]: %d\n", i, ret);
            return ret;
        }
    }
    for (uint32_t i = 0; i < io->n_output; i++) {
        memset(&out_attrs[i], 0, sizeof(out_attrs[i]));
        out_attrs[i].index = i;
        ret = rknn_query(*ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attrs[i], sizeof(out_attrs[i]));
        if (ret < 0) {
            fprintf(stderr, "RKNN_QUERY_OUTPUT_ATTR[%u]: %d\n", i, ret);
            return ret;
        }
    }

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
static int run_one_iteration(rknn_context ctx,
                             const rknn_input *inp,
                             const rknn_input_output_num *io,
                             const rknn_tensor_attr *out_attrs,
                             pp_mode_t pp,
                             float conf_thr,
                             double *e2e_ms,
                             double *pp_ms) {
    rknn_output outs[MAX_IO];
    double t0 = now_ms();
    int ret = rknn_inputs_set(ctx, 1, (rknn_input *)inp);
    if (ret < 0) return ret;
    ret = rknn_run(ctx, NULL);
    if (ret < 0) return ret;
    memset(outs, 0, sizeof(outs));
    for (uint32_t o = 0; o < io->n_output; o++) {
        outs[o].index = o;
        outs[o].want_float = 0;
    }
    ret = rknn_outputs_get(ctx, io->n_output, outs, NULL);
    if (ret < 0) return ret;
    double t1 = now_ms();

    double post_ms = run_postproc(pp, outs, out_attrs, io->n_output, conf_thr);
    rknn_outputs_release(ctx, io->n_output, outs);

    if (pp_ms) *pp_ms = post_ms;
    if (e2e_ms) *e2e_ms = (t1 - t0) + post_ms;
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

    arg->ret = setup_context(&ctx, arg->model_buf, arg->model_size, arg->core_mask,
                             init_flags, share_ctx,
                             &io, in_attrs, out_attrs, &dummy, &in_size, inp);
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
                                     arg->conf_thr, &e2e_ms, &pp_ms);
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
                                     arg->conf_thr, &e2e_ms, &pp_ms);
        if (arg->ret < 0) goto done;
        arg->frames++;
        arg->sum_e2e_ms += e2e_ms;
        arg->sum_pp_ms += pp_ms;
        if (e2e_ms < arg->min_e2e_ms) arg->min_e2e_ms = e2e_ms;
        if (e2e_ms > arg->max_e2e_ms) arg->max_e2e_ms = e2e_ms;
        arg->last_kept = g_last_kept;
    }

done:
    if (arg->done_barrier) {
        pthread_barrier_wait(arg->done_barrier);
    }
    if (dummy) free(dummy);
    if (ctx) rknn_destroy(ctx);
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
static int run_fps_mode(const char *model_path,
                        const void *model_buf,
                        size_t model_size,
                        const char *json_out,
                        const char *pp_str,
                        pp_mode_t pp,
                        float conf_thr,
                        int warmup,
                        const char *core_str,
                        const char *fps_core_map,
                        const char *sram_str,
                        uint32_t init_flags,
                        int fps_workers,
                        double fps_seconds) {
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
    int default_core = parse_core(core_str);
    pthread_t threads[MAX_FPS_WORKERS];
    fps_worker_arg_t args[MAX_FPS_WORKERS];
    pthread_barrier_t ready_barrier;
    pthread_barrier_t start_barrier;
    pthread_barrier_t done_barrier;
    fps_shared_state_t share_state;
    double end_ms = 0.0;
    bool shared_sram_mode = is_shared_sram_mode(sram_str) && flags_request_shared_sram(init_flags);

    pthread_barrier_init(&ready_barrier, NULL, (unsigned)fps_workers + 1);
    pthread_barrier_init(&start_barrier, NULL, (unsigned)fps_workers + 1);
    if (shared_sram_mode) {
        pthread_barrier_init(&done_barrier, NULL, (unsigned)fps_workers);
        memset(&share_state, 0, sizeof(share_state));
        pthread_mutex_init(&share_state.mutex, NULL);
        pthread_cond_init(&share_state.cond, NULL);
    }

    for (int i = 0; i < fps_workers; i++) {
        memset(&args[i], 0, sizeof(args[i]));
        args[i].model_buf = model_buf;
        args[i].model_size = model_size;
        args[i].worker_id = i;
        args[i].core_mask = map_count > 0 ? mapped_cores[i % map_count] : default_core;
        args[i].init_flags = init_flags;
        args[i].shared_sram_mode = shared_sram_mode;
        args[i].share_state = shared_sram_mode ? &share_state : NULL;
        args[i].pp = pp;
        args[i].conf_thr = conf_thr;
        args[i].warmup = warmup;
        args[i].ready_barrier = &ready_barrier;
        args[i].start_barrier = &start_barrier;
        args[i].done_barrier = shared_sram_mode ? &done_barrier : NULL;
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
        pthread_cond_destroy(&share_state.cond);
        pthread_mutex_destroy(&share_state.mutex);
    }

    long total_frames = 0;
    double total_e2e_ms = 0.0, total_pp_ms = 0.0;
    double min_e2e_ms = DBL_MAX, max_e2e_ms = 0.0;
    int ret_code = 0;
    for (int i = 0; i < fps_workers; i++) {
        if (args[i].ret < 0) ret_code = 1;
        total_frames += args[i].frames;
        total_e2e_ms += args[i].sum_e2e_ms;
        total_pp_ms += args[i].sum_pp_ms;
        if (args[i].frames > 0 && args[i].min_e2e_ms < min_e2e_ms) {
            min_e2e_ms = args[i].min_e2e_ms;
        }
        if (args[i].max_e2e_ms > max_e2e_ms) max_e2e_ms = args[i].max_e2e_ms;
    }
    if (ret_code) return ret_code;

    double measured_s = (finish_ms - start_ms) / 1000.0;
    double fps = measured_s > 0.0 ? (double)total_frames / measured_s : 0.0;
    double avg_e2e_ms = total_frames > 0 ? total_e2e_ms / (double)total_frames : 0.0;
    double avg_pp_ms = total_frames > 0 ? total_pp_ms / (double)total_frames : 0.0;
    if (min_e2e_ms == DBL_MAX) min_e2e_ms = 0.0;

    printf("\n=== bench_rknn_perf offline FPS ===\n");
    printf("model:       %s\n", model_path);
    printf("workers:     %d   seconds: %.3f   warmup: %d\n",
           fps_workers, fps_seconds, warmup);
    printf("core:        %s   fps_core_map: %s   neon: %d\n",
           core_str, fps_core_map && *fps_core_map ? fps_core_map : "(none)", HAVE_NEON);
    printf("sram:        %s   init_flags: 0x%08x\n", sram_str, init_flags);
    printf("postproc:    %s\n", pp_str);
    printf("frames:      %ld\n", total_frames);
    printf("offline_fps: %.2f\n", fps);
    printf("e2e_ms:      avg=%.3f  min=%.3f  max=%.3f\n",
           avg_e2e_ms, min_e2e_ms, max_e2e_ms);
    printf("postproc_ms: %.3f\n", avg_pp_ms);
    for (int i = 0; i < fps_workers; i++) {
        double worker_avg = args[i].frames > 0
                                ? args[i].sum_e2e_ms / (double)args[i].frames
                                : 0.0;
        printf("worker[%d]:   frames=%ld  avg_e2e_ms=%.3f  last_kept=%d\n",
               i, args[i].frames, worker_avg, args[i].last_kept);
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
                "\"postproc_ms\":%.4f,\"worker_frames\":[",
                model_path, core_str, fps_core_map ? fps_core_map : "", sram_str, init_flags,
                pp_str, fps_workers, fps_seconds, measured_s, warmup, HAVE_NEON,
                total_frames, fps, avg_e2e_ms, min_e2e_ms, max_e2e_ms, avg_pp_ms);
            for (int i = 0; i < fps_workers; i++) {
                fprintf(jp, "%s%ld", i ? "," : "", args[i].frames);
            }
            fprintf(jp, "]}\n");
            fclose(jp);
        }
    }
    return 0;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "用法: %s --model M.rknn [选项]\n"
        "  --warmup N        预热轮次 (默认 10)\n"
        "  --runs   N        计时轮次 (默认 200)\n"
        "  --core   0|1|2|all   NPU 核心选择 (默认 all)\n"
        "  --postproc predist|predfl|seg_predist|seg_predfl|none  CPU 后处理 (默认 none)\n"
        "  --conf-thr F      后处理置信度阈值 (默认 0.25)\n"
        "  --sram off|private|shared  RKNN SRAM 初始化策略 (默认 off)\n"
        "  --fps-workers N   离线 FPS worker/context 数；设置后进入 FPS 模式\n"
        "  --fps-seconds F   离线 FPS 计时秒数 (默认 10)\n"
        "  --fps-core-map L  FPS worker core 轮转列表，如 0,1,2 或 all,all\n"
        "  --json   F.json   写出机器可读结果\n",
        prog);
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *json_out   = NULL;
    const char *pp_str     = "none";
    const char *core_str   = "all";
    const char *fps_core_map = NULL;
    const char *sram_str   = "off";
    int warmup = 10, runs = 200;
    int fps_workers = 0;
    double fps_seconds = 10.0;
    float conf_thr = 0.25f;

    static struct option opts[] = {
        {"model",     required_argument, 0, 'm'},
        {"warmup",    required_argument, 0, 'w'},
        {"runs",      required_argument, 0, 'r'},
        {"core",      required_argument, 0, 'c'},
        {"postproc",  required_argument, 0, 'p'},
        {"conf-thr",  required_argument, 0, 't'},
        {"json",      required_argument, 0, 'j'},
        {"sram",      required_argument, 0, 's'},
        {"fps-workers", required_argument, 0, 1000},
        {"fps-seconds", required_argument, 0, 1001},
        {"fps-core-map", required_argument, 0, 1002},
        {"help",      no_argument,       0, 'h'},
        {0,0,0,0}
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "m:w:r:c:p:t:j:s:h", opts, NULL)) != -1) {
        switch (opt) {
            case 'm': model_path = optarg; break;
            case 'w': warmup     = atoi(optarg); break;
            case 'r': runs       = atoi(optarg); break;
            case 'c': core_str   = optarg; break;
            case 'p': pp_str     = optarg; break;
            case 't': conf_thr   = (float)atof(optarg); break;
            case 'j': json_out   = optarg; break;
            case 's': sram_str   = normalize_sram_mode(optarg); break;
            case 1000: fps_workers = atoi(optarg); break;
            case 1001: fps_seconds = atof(optarg); break;
            case 1002: fps_core_map = optarg; break;
            case 'h': usage(argv[0]); return 0;
            default:  usage(argv[0]); return 2;
        }
    }
    if (!model_path) { usage(argv[0]); return 2; }
    pp_mode_t pp = parse_pp(pp_str);
    uint32_t init_flags = parse_sram_flags(sram_str);

    FILE *fp = fopen(model_path, "rb");
    if (!fp) { perror("fopen"); return 1; }
    fseek(fp, 0, SEEK_END); size_t sz = ftell(fp); rewind(fp);
    void *buf = malloc(sz);
    if (fread(buf, 1, sz, fp) != sz) { fprintf(stderr, "read fail\n"); free(buf); fclose(fp); return 1; }
    fclose(fp);

    if (fps_workers == 1 && is_shared_sram_mode(sram_str) && flags_request_shared_sram(init_flags)) {
        fprintf(stderr, "single-worker --sram shared has no consumer context; using private SRAM\n");
        sram_str = "private";
        init_flags = parse_sram_flags(sram_str);
    }

    if (fps_workers > 0) {
        int ret = run_fps_mode(model_path, buf, sz, json_out, pp_str, pp, conf_thr,
                               warmup, core_str, fps_core_map, sram_str, init_flags, fps_workers,
                               fps_seconds);
        free(buf);
        return ret;
    }

    rknn_context ctx;
    rknn_init_extend init_extend;
    memset(&init_extend, 0, sizeof(init_extend));
    if (is_shared_sram_mode(sram_str) && flags_request_shared_sram(init_flags)) {
        fprintf(stderr, "single-context --sram shared has no source context; using private SRAM\n");
        sram_str = "private";
        init_flags = parse_sram_flags("private");
    }
    int ret = rknn_init(&ctx, buf, sz, init_flags, init_flags ? &init_extend : NULL);
    free(buf);
    if (ret < 0) { fprintf(stderr, "rknn_init: %d\n", ret); return 1; }
    rknn_set_core_mask(ctx, parse_core(core_str));

    rknn_sdk_version sdk_ver;
    memset(&sdk_ver, 0, sizeof(sdk_ver));
    int sdk_ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver));

    rknn_input_output_num io;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
    if (io.n_input < 1 || io.n_input > MAX_IO || io.n_output > MAX_IO) {
        fprintf(stderr, "unsupported io count: input=%u output=%u\n",
                io.n_input, io.n_output);
        rknn_destroy(ctx);
        return 1;
    }

    rknn_tensor_attr in_attrs[MAX_IO];
    for (uint32_t i = 0; i < io.n_input && i < MAX_IO; i++) {
        memset(&in_attrs[i], 0, sizeof(in_attrs[i])); in_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attrs[i], sizeof(in_attrs[i]));
    }
    rknn_tensor_attr out_attrs[MAX_IO];
    for (uint32_t i = 0; i < io.n_output && i < MAX_IO; i++) {
        memset(&out_attrs[i], 0, sizeof(out_attrs[i])); out_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attrs[i], sizeof(out_attrs[i]));
    }

    size_t in_size = 1;
    for (uint32_t d = 0; d < in_attrs[0].n_dims; d++) in_size *= in_attrs[0].dims[d];
    void *dummy = calloc(in_size, 1);
    rknn_input inp[1]; memset(inp, 0, sizeof(inp));
    inp[0].index = 0; inp[0].buf = dummy; inp[0].size = in_size;
    inp[0].pass_through = 0; inp[0].type = RKNN_TENSOR_UINT8; inp[0].fmt = RKNN_TENSOR_NHWC;

    rknn_output outs[MAX_IO];

    for (int w = 0; w < warmup; w++) {
        rknn_inputs_set(ctx, 1, inp);
        rknn_run(ctx, NULL);
        memset(outs, 0, sizeof(outs));
        for (uint32_t o = 0; o < io.n_output; o++) { outs[o].index = o; outs[o].want_float = 0; }
        rknn_outputs_get(ctx, io.n_output, outs, NULL);
        rknn_outputs_release(ctx, io.n_output, outs);
    }

    /* 收集每帧时间用于百分位统计 */
    double *npu_times = (double *)malloc(runs * sizeof(double));
    double *e2e_times = (double *)malloc(runs * sizeof(double));
    double sum_npu_ms = 0, sum_e2e_ms = 0, sum_pp_ms = 0;
    double max_e2e = 0, min_e2e = 1e9;
    for (int i = 0; i < runs; i++) {
        double t0 = now_ms();
        rknn_inputs_set(ctx, 1, inp);
        rknn_run(ctx, NULL);
        memset(outs, 0, sizeof(outs));
        for (uint32_t o = 0; o < io.n_output; o++) { outs[o].index = o; outs[o].want_float = 0; }
        rknn_outputs_get(ctx, io.n_output, outs, NULL);
        double t1 = now_ms();

        double npu_ms = t1 - t0;
        npu_times[i] = npu_ms;
        sum_npu_ms += npu_ms;

        double pp_ms = run_postproc(pp, outs, out_attrs, io.n_output, conf_thr);
        sum_pp_ms += pp_ms;

        double e2e = npu_ms + pp_ms;
        e2e_times[i] = e2e;
        sum_e2e_ms += e2e;
        if (e2e > max_e2e) max_e2e = e2e;
        if (e2e < min_e2e) min_e2e = e2e;

        rknn_outputs_release(ctx, io.n_output, outs);
    }

    /* 排序算百分位 */
    for (int i = 0; i < runs - 1; i++)
        for (int j = i + 1; j < runs; j++)
            if (npu_times[j] < npu_times[i]) {
                double t = npu_times[i]; npu_times[i] = npu_times[j]; npu_times[j] = t;
            }
    for (int i = 0; i < runs - 1; i++)
        for (int j = i + 1; j < runs; j++)
            if (e2e_times[j] < e2e_times[i]) {
                double t = e2e_times[i]; e2e_times[i] = e2e_times[j]; e2e_times[j] = t;
            }

    double npu_avg_ms = sum_npu_ms / runs;
    double npu_p50 = npu_times[runs / 2];
    double npu_p90 = npu_times[(int)(runs * 0.9)];
    double npu_best = npu_times[0];
    double e2e_avg_ms = sum_e2e_ms / runs;
    double e2e_p50 = e2e_times[runs / 2];
    double e2e_p90 = e2e_times[(int)(runs * 0.9)];
    double e2e_best = e2e_times[0];
    double pp_avg_ms  = sum_pp_ms / runs;

    printf("\n=== bench_rknn_perf ===\n");
    printf("model:       %s\n", model_path);
    printf("core:        %s   neon: %d   runs: %d  warmup: %d\n",
           core_str, HAVE_NEON, runs, warmup);
    printf("sram:        %s   init_flags: 0x%08x\n", sram_str, init_flags);
    printf("postproc:    %s\n", pp_str);
    if (sdk_ret == 0) {
        printf("rknn_sdk:    api=%s   drv=%s\n", sdk_ver.api_version, sdk_ver.drv_version);
    }
    printf("npu_wall_ms: best=%.3f  P50=%.3f  P90=%.3f  avg=%.3f  (%.1f FPS best, %.1f FPS P50)\n",
           npu_best, npu_p50, npu_p90, npu_avg_ms, 1000.0 / npu_best, 1000.0 / npu_p50);
    printf("postproc_ms: %.3f\n", pp_avg_ms);
    if (pp != PP_NONE) {
        printf("postproc_kept: %d\n", g_last_kept);
    }
    printf("e2e_ms:      best=%.3f  P50=%.3f  P90=%.3f  avg=%.3f  (%.1f FPS best, %.1f FPS P50)\n",
           e2e_best, e2e_p50, e2e_p90, e2e_avg_ms, 1000.0 / e2e_best, 1000.0 / e2e_p50);

    free(npu_times);
    free(e2e_times);

    if (json_out) {
        FILE *jp = fopen(json_out, "w");
        if (jp) {
            fprintf(jp,
                "{\"model\":\"%s\",\"core\":\"%s\",\"sram\":\"%s\","
                "\"init_flags\":%u,\"postproc\":\"%s\","
                "\"runs\":%d,\"warmup\":%d,\"neon\":%d,"
                "\"rknn_api_version\":\"%s\",\"rknn_drv_version\":\"%s\","
                "\"npu_wall_best_ms\":%.4f,\"npu_wall_p50_ms\":%.4f,\"npu_wall_p90_ms\":%.4f,\"npu_wall_avg_ms\":%.4f,"
                "\"postproc_ms\":%.4f,\"postproc_kept\":%d,"
                "\"e2e_best_ms\":%.4f,\"e2e_p50_ms\":%.4f,\"e2e_p90_ms\":%.4f,\"e2e_avg_ms\":%.4f}\n",
                model_path, core_str, sram_str, init_flags, pp_str, runs, warmup, HAVE_NEON,
                sdk_ret == 0 ? sdk_ver.api_version : "unknown",
                sdk_ret == 0 ? sdk_ver.drv_version : "unknown",
                npu_best, npu_p50, npu_p90, npu_avg_ms,
                pp_avg_ms, g_last_kept,
                e2e_best, e2e_p50, e2e_p90, e2e_avg_ms);
            fclose(jp);
        }
    }

    free_postproc_buffers();
    free(dummy);
    rknn_destroy(ctx);
    return 0;
}
