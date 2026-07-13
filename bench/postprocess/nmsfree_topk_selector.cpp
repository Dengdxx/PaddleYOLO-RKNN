// Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
// SPDX-License-Identifier: AGPL-3.0-only

#include "postprocess/nmsfree_topk_selector.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define PADDLEYOLO_RKNN_NMSFREE_HAS_NEON 1
#else
#define PADDLEYOLO_RKNN_NMSFREE_HAS_NEON 0
#endif

namespace paddleyolo_rknn::postprocess {
namespace {

/**
 * @brief 按分数降序、原始顺序升序比较 stable TopK 元素。
 * @param lhs 左侧元素。
 * @param rhs 右侧元素。
 * @return lhs 应排在 rhs 之前时返回 true。
 */
bool StableScoreGreater(const NmsFreeRankedValue &lhs, const NmsFreeRankedValue &rhs) noexcept {
  if (lhs.value == rhs.value)
    return lhs.original_index < rhs.original_index;
  return lhs.value > rhs.value;
}

/**
 * @brief 计算稳定 sigmoid。
 * @param value logit。
 * @return sigmoid 概率。
 */
float StableSigmoid(const float value) noexcept {
  if (value >= 0.0F) {
    const float z = std::exp(-value);
    return 1.0F / (1.0F + z);
  }
  const float z = std::exp(value);
  return z / (1.0F + z);
}

/**
 * @brief 把概率阈值转换为 logit 阈值。
 * @param probability `[0, 1]` 概率。
 * @return 等价 logit 阈值。
 */
float LogitThreshold(const float probability) noexcept {
  if (probability <= 0.0F)
    return -std::numeric_limits<float>::infinity();
  if (probability >= 1.0F)
    return std::numeric_limits<float>::infinity();
  return std::log(probability / (1.0F - probability));
}

/**
 * @brief 跨类别扫描每个 anchor 的最大 INT8 logit。
 * @details AArch64 每次并行处理 16 个 anchor；其他平台使用标量路径。
 * @param logits 布局为 `[classes, anchors]` 的 logits。
 * @param classes 类别数。
 * @param anchors anchor 数。
 * @param[out] maximums 长度为 anchors 的最大值数组。
 */
void CollectAnchorMaxLogits(const std::int8_t *logits, const int classes, const int anchors,
                            std::int8_t *maximums) noexcept {
  std::fill_n(maximums, anchors, std::numeric_limits<std::int8_t>::min());
  for (int class_id = 0; class_id < classes; ++class_id) {
    const auto *class_logits = logits + static_cast<std::size_t>(class_id) * anchors;
    int anchor = 0;
#if PADDLEYOLO_RKNN_NMSFREE_HAS_NEON
    for (; anchor + 16 <= anchors; anchor += 16) {
      const int8x16_t previous = vld1q_s8(maximums + anchor);
      const int8x16_t current = vld1q_s8(class_logits + anchor);
      vst1q_s8(maximums + anchor, vmaxq_s8(previous, current));
    }
#endif
    for (; anchor < anchors; ++anchor)
      maximums[anchor] = std::max(maximums[anchor], class_logits[anchor]);
  }
}

}  // namespace

int SelectNmsFreeTopKSeedsInt8(const std::int8_t *logits, const int classes, const int anchors,
                               const AffineInt8Quant quant, const float confidence_threshold,
                               const int max_det, const NmsFreeTopKScratch scratch,
                               ClassSeed *output, const std::size_t output_capacity,
                               NmsFreeTopKStats *stats) noexcept {
  NmsFreeTopKStats local_stats{};
  if (stats != nullptr)
    *stats = local_stats;
  if (logits == nullptr || classes <= 0 || anchors <= 0 || max_det <= 0 || output == nullptr ||
      scratch.anchor_max_logits == nullptr || scratch.anchor_ranking == nullptr ||
      scratch.gathered_ranking == nullptr || !(quant.scale > 0.0F) || !std::isfinite(quant.scale) ||
      !std::isfinite(confidence_threshold) || confidence_threshold < 0.0F ||
      confidence_threshold > 1.0F) {
    return kNmsFreeTopKInvalidArgument;
  }

  const int topk = std::min(max_det, anchors);
  if (static_cast<std::size_t>(classes) >
      std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(topk)) {
    return kNmsFreeTopKInvalidArgument;
  }
  const std::size_t gathered_count = static_cast<std::size_t>(topk) * classes;
  if (scratch.anchor_max_capacity < static_cast<std::size_t>(anchors) ||
      scratch.anchor_ranking_capacity < static_cast<std::size_t>(anchors) ||
      scratch.gathered_ranking_capacity < gathered_count ||
      output_capacity < static_cast<std::size_t>(topk)) {
    return kNmsFreeTopKInsufficientCapacity;
  }

  CollectAnchorMaxLogits(logits, classes, anchors, scratch.anchor_max_logits);
  local_stats.anchor_values_scanned = anchors * classes;
  for (int anchor = 0; anchor < anchors; ++anchor) {
    scratch.anchor_ranking[anchor] = NmsFreeRankedValue{scratch.anchor_max_logits[anchor], anchor};
  }
  std::partial_sort(scratch.anchor_ranking, scratch.anchor_ranking + topk,
                    scratch.anchor_ranking + anchors, StableScoreGreater);
  local_stats.selected_anchors = topk;

  std::size_t gathered_index = 0;
  for (int gather_position = 0; gather_position < topk; ++gather_position) {
    const int anchor = scratch.anchor_ranking[gather_position].original_index;
    for (int class_id = 0; class_id < classes; ++class_id) {
      scratch.gathered_ranking[gathered_index] =
          NmsFreeRankedValue{logits[static_cast<std::size_t>(class_id) * anchors + anchor],
                             gather_position * classes + class_id};
      ++gathered_index;
    }
  }
  local_stats.gathered_values = static_cast<std::int64_t>(gathered_count);
  std::partial_sort(scratch.gathered_ranking, scratch.gathered_ranking + topk,
                    scratch.gathered_ranking + gathered_count, StableScoreGreater);

  const float logit_threshold = LogitThreshold(confidence_threshold);
  int output_count = 0;
  for (int index = 0; index < topk; ++index) {
    const auto &ranked = scratch.gathered_ranking[index];
    const float logit =
        (static_cast<float>(ranked.value) - static_cast<float>(quant.zero_point)) * quant.scale;
    if (!(logit > logit_threshold))
      break;
    const int gather_position = ranked.original_index / classes;
    const int class_id = ranked.original_index % classes;
    output[output_count++] = ClassSeed{scratch.anchor_ranking[gather_position].original_index,
                                       class_id, StableSigmoid(logit)};
  }

  if (stats != nullptr)
    *stats = local_stats;
  return output_count;
}

}  // namespace paddleyolo_rknn::postprocess
