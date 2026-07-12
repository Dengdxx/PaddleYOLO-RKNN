// Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
// SPDX-License-Identifier: AGPL-3.0-only

#include "postprocess/seg_class_selector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#if !defined(__aarch64__) || !defined(__ARM_NEON)
#error "seg_class_selector requires AArch64 NEON"
#endif

#include <arm_neon.h>

namespace paddleyolo_rknn::postprocess {
namespace {

/**
 * @brief 将概率阈值转换为稳定 logit 阈值。
 * @param confidence_threshold 概率阈值。
 * @return logit 阈值。
 */
float LogitThreshold(const float confidence_threshold) noexcept {
  if (confidence_threshold <= 0.0F)
    return -std::numeric_limits<float>::infinity();
  if (confidence_threshold >= 1.0F)
    return std::numeric_limits<float>::infinity();
  return std::log(confidence_threshold / (1.0F - confidence_threshold));
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
 * @brief 查找第一个严格通过浮点阈值的 INT8 值。
 * @param threshold logit 阈值。
 * @param quant 分类量化参数。
 * @return 最小通过值；128 表示不可能通过。
 */
int QuantizedStrictGreaterMin(const float threshold,
                              const AffineInt8Quant quant) noexcept {
  if (threshold == -std::numeric_limits<float>::infinity()) {
    return std::numeric_limits<std::int8_t>::min();
  }
  if (threshold == std::numeric_limits<float>::infinity() ||
      !(quant.scale > 0.0F) || !std::isfinite(quant.scale)) {
    return static_cast<int>(std::numeric_limits<std::int8_t>::max()) + 1;
  }
  for (int q = std::numeric_limits<std::int8_t>::min();
       q <= std::numeric_limits<std::int8_t>::max(); ++q) {
    const float value =
        (static_cast<float>(q) - static_cast<float>(quant.zero_point)) *
        quant.scale;
    if (value > threshold)
      return q;
  }
  return static_cast<int>(std::numeric_limits<std::int8_t>::max()) + 1;
}

/**
 * @brief 计算带一个量化档安全余量的 score_sum 下界。
 * @param threshold 概率阈值。
 * @param quant score_sum 量化参数。
 * @return 最小存活值；-128 表示禁用预筛。
 */
int QuantizedPrescreenMin(const float threshold,
                          const AffineInt8Quant quant) noexcept {
  if (!(threshold > 0.0F) || !(quant.scale > 0.0F) ||
      !std::isfinite(quant.scale)) {
    return std::numeric_limits<std::int8_t>::min();
  }
  const double boundary =
      static_cast<double>(threshold) / static_cast<double>(quant.scale) +
      static_cast<double>(quant.zero_point);
  if (boundary <=
          static_cast<double>(std::numeric_limits<std::int8_t>::min()) ||
      boundary >
          static_cast<double>(std::numeric_limits<std::int8_t>::max()) + 1.0) {
    return std::numeric_limits<std::int8_t>::min();
  }
  const int base = static_cast<int>(std::ceil(boundary));
  return std::clamp(base - 1,
                    static_cast<int>(std::numeric_limits<std::int8_t>::min()),
                    static_cast<int>(std::numeric_limits<std::int8_t>::max()));
}

/**
 * @brief 向调用方缓冲区追加一个精确候选。
 * @param anchor anchor 索引。
 * @param class_id 类别索引。
 * @param quantized_logit INT8 logit。
 * @param min_q INT8 预筛下界。
 * @param logit_threshold 精确 logit 阈值。
 * @param quant 分类量化参数。
 * @param[out] output 候选缓冲区。
 * @param output_capacity 缓冲区容量。
 * @param[out] count 已写入候选数。
 * @return 容量充足时返回 true。
 */
bool AppendSeed(const int anchor, const int class_id,
                const std::int8_t quantized_logit, const int min_q,
                const float logit_threshold, const AffineInt8Quant quant,
                ClassSeed *output, const std::size_t output_capacity,
                int *count) noexcept {
  if (static_cast<int>(quantized_logit) < min_q)
    return true;
  const float logit = (static_cast<float>(quantized_logit) -
                       static_cast<float>(quant.zero_point)) *
                      quant.scale;
  if (!(logit > logit_threshold))
    return true;
  if (static_cast<std::size_t>(*count) >= output_capacity)
    return false;
  output[*count] = ClassSeed{anchor, class_id, StableSigmoid(logit)};
  ++(*count);
  return true;
}

/**
 * @brief 在指定区间收集第五输出存活 anchor。
 * @param data score_sum 数据。
 * @param begin 起始 anchor。
 * @param end 结束 anchor。
 * @param min_q 最小存活量化值。
 * @param[out] output 输出缓冲区。
 * @param output_capacity 输出容量。
 * @return 已写入数量；容量不足返回 `kClassSelectionInsufficientCapacity`。
 */
int CollectSurvivorsRange(const std::int8_t *data, const int begin,
                          const int end, const int min_q, int *output,
                          const std::size_t output_capacity) noexcept {
  int count = 0;
  const int8x16_t threshold = vdupq_n_s8(static_cast<std::int8_t>(min_q));
  int anchor = begin;
  for (; anchor + 16 <= end; anchor += 16) {
    const uint8x16_t enabled = vcgeq_s8(vld1q_s8(data + anchor), threshold);
    std::uint8_t lanes[16];
    vst1q_u8(lanes, enabled);
    for (int lane = 0; lane < 16; ++lane) {
      if (lanes[lane] == 0U)
        continue;
      if (static_cast<std::size_t>(count) >= output_capacity)
        return kClassSelectionInsufficientCapacity;
      output[count++] = anchor + lane;
    }
  }
  for (; anchor < end; ++anchor) {
    if (static_cast<int>(data[anchor]) < min_q)
      continue;
    if (static_cast<std::size_t>(count) >= output_capacity)
      return kClassSelectionInsufficientCapacity;
    output[count++] = anchor;
  }
  return count;
}

/**
 * @brief 使用已计算阈值精确分类指定 anchor。
 * @param logits 分类 logits。
 * @param classes 类别数。
 * @param anchors anchor 总数。
 * @param quant 分类量化参数。
 * @param min_q 最小通过量化值。
 * @param logit_threshold 精确 logit 阈值。
 * @param anchor_indices 待分类 anchor。
 * @param anchor_count anchor 数量。
 * @param[out] output 候选输出。
 * @param output_capacity 输出容量。
 * @param[out] count 已写入数量。
 * @return 成功返回 true，容量不足返回 false。
 */
bool ClassifyAnchorsBestClass(const std::int8_t *logits, const int classes,
                              const int anchors, const AffineInt8Quant quant,
                              const int min_q, const float logit_threshold,
                              const int *anchor_indices,
                              const std::size_t anchor_count, ClassSeed *output,
                              const std::size_t output_capacity,
                              int *count) noexcept {
  for (std::size_t index = 0; index < anchor_count; ++index) {
    const int anchor = anchor_indices[index];
    int best_class = 0;
    std::int8_t best_value = logits[anchor];
    for (int class_id = 1; class_id < classes; ++class_id) {
      const std::int8_t value =
          logits[static_cast<std::size_t>(class_id) * anchors + anchor];
      if (value > best_value) {
        best_value = value;
        best_class = class_id;
      }
    }
    if (!AppendSeed(anchor, best_class, best_value, min_q, logit_threshold,
                    quant, output, output_capacity, count))
      return false;
  }
  return true;
}

} // namespace

int CollectScoreSumSurvivorsInt8(const ScoreSumInt8View &score_sum,
                                 const int anchors,
                                 const float confidence_threshold, int *output,
                                 const std::size_t output_capacity) noexcept {
  if (score_sum.data == nullptr || output == nullptr || anchors <= 0 ||
      !std::isfinite(confidence_threshold) || !(score_sum.quant.scale > 0.0F) ||
      !std::isfinite(score_sum.quant.scale)) {
    return kClassSelectionInvalidArgument;
  }
  const int min_q =
      QuantizedPrescreenMin(confidence_threshold, score_sum.quant);
  if (min_q <= static_cast<int>(std::numeric_limits<std::int8_t>::min()))
    return kScoreSumPrescreenUnavailable;
  return CollectSurvivorsRange(score_sum.data, 0, anchors, min_q, output,
                               output_capacity);
}

int ClassifyAnchorsBestClassInt8(
    const std::int8_t *logits, const int classes, const int anchors,
    const AffineInt8Quant logit_quant, const float confidence_threshold,
    const int *anchor_indices, const std::size_t anchor_count,
    ClassSeed *output, const std::size_t output_capacity,
    ClassSelectionStats *stats) noexcept {
  ClassSelectionStats local_stats{};
  if (stats != nullptr)
    *stats = local_stats;
  if (logits == nullptr || output == nullptr || classes <= 0 || anchors <= 0 ||
      (anchor_count != 0U && anchor_indices == nullptr) ||
      !(logit_quant.scale > 0.0F) || !std::isfinite(logit_quant.scale) ||
      !std::isfinite(confidence_threshold)) {
    return kClassSelectionInvalidArgument;
  }
  for (std::size_t index = 0; index < anchor_count; ++index) {
    if (anchor_indices[index] < 0 || anchor_indices[index] >= anchors)
      return kClassSelectionInvalidArgument;
  }

  const float logit_threshold = LogitThreshold(confidence_threshold);
  const int min_q = QuantizedStrictGreaterMin(logit_threshold, logit_quant);
  if (min_q > static_cast<int>(std::numeric_limits<std::int8_t>::max()))
    return 0;

  int count = 0;
  local_stats.class_anchors_scanned = static_cast<int>(anchor_count);
  local_stats.class_values_scanned =
      static_cast<std::int64_t>(anchor_count) * classes;
  if (!ClassifyAnchorsBestClass(logits, classes, anchors, logit_quant, min_q,
                                logit_threshold, anchor_indices, anchor_count,
                                output, output_capacity, &count)) {
    if (stats != nullptr)
      *stats = local_stats;
    return kClassSelectionInsufficientCapacity;
  }
  if (stats != nullptr)
    *stats = local_stats;
  return count;
}

int SelectBestClassSeedsInt8(
    const std::int8_t *logits, const int classes, const int anchors,
    const AffineInt8Quant logit_quant, const float confidence_threshold,
    const ScoreSumInt8View *score_sum, ClassSeed *output,
    const std::size_t output_capacity, ClassSelectionStats *stats) noexcept {
  ClassSelectionStats local_stats{};
  if (stats != nullptr)
    *stats = local_stats;
  if (logits == nullptr || output == nullptr || classes <= 0 || anchors <= 0 ||
      !(logit_quant.scale > 0.0F) || !std::isfinite(logit_quant.scale) ||
      !std::isfinite(confidence_threshold)) {
    return kClassSelectionInvalidArgument;
  }

  const float logit_threshold = LogitThreshold(confidence_threshold);
  const int min_q = QuantizedStrictGreaterMin(logit_threshold, logit_quant);
  if (min_q > static_cast<int>(std::numeric_limits<std::int8_t>::max()))
    return 0;

  int count = 0;
  int sum_min_q = std::numeric_limits<std::int8_t>::min();
  if (score_sum != nullptr && score_sum->data != nullptr) {
    sum_min_q = QuantizedPrescreenMin(confidence_threshold, score_sum->quant);
  }

  if (sum_min_q > static_cast<int>(std::numeric_limits<std::int8_t>::min())) {
    local_stats.used_score_sum = true;
    local_stats.score_sum_scanned = anchors;
    constexpr int kSurvivorChunkSize = 128;
    std::array<int, kSurvivorChunkSize> survivors{};
    for (int begin = 0; begin < anchors; begin += kSurvivorChunkSize) {
      const int end = std::min(begin + kSurvivorChunkSize, anchors);
      const int survivor_count =
          CollectSurvivorsRange(score_sum->data, begin, end, sum_min_q,
                                survivors.data(), survivors.size());
      if (survivor_count < 0)
        return survivor_count;
      local_stats.class_anchors_scanned += survivor_count;
      local_stats.class_values_scanned +=
          static_cast<std::int64_t>(survivor_count) * classes;
      if (!ClassifyAnchorsBestClass(logits, classes, anchors, logit_quant,
                                    min_q, logit_threshold, survivors.data(),
                                    survivor_count, output, output_capacity,
                                    &count)) {
        if (stats != nullptr)
          *stats = local_stats;
        return kClassSelectionInsufficientCapacity;
      }
    }
    if (stats != nullptr)
      *stats = local_stats;
    return count;
  }

  local_stats.class_anchors_scanned = anchors;
  local_stats.class_values_scanned =
      static_cast<std::int64_t>(anchors) * classes;
  constexpr int kAnchorChunkSize = 128;
  std::array<int, kAnchorChunkSize> anchors_chunk{};
  for (int begin = 0; begin < anchors; begin += kAnchorChunkSize) {
    const int end = std::min(begin + kAnchorChunkSize, anchors);
    for (int anchor = begin; anchor < end; ++anchor)
      anchors_chunk[static_cast<std::size_t>(anchor - begin)] = anchor;
    if (!ClassifyAnchorsBestClass(logits, classes, anchors, logit_quant, min_q,
                                  logit_threshold, anchors_chunk.data(),
                                  static_cast<std::size_t>(end - begin), output,
                                  output_capacity, &count)) {
      if (stats != nullptr)
        *stats = local_stats;
      return kClassSelectionInsufficientCapacity;
    }
  }
  if (stats != nullptr)
    *stats = local_stats;
  return count;
}

} // namespace paddleyolo_rknn::postprocess
