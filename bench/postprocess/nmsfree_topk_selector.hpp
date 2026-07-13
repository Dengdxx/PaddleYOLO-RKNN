// Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "postprocess/seg_class_selector.hpp"

#include <cstddef>
#include <cstdint>

namespace paddleyolo_rknn::postprocess {

/** @brief YOLO26 NMS-free 两阶段 TopK 的读取量统计。 */
struct NmsFreeTopKStats {
  int anchor_values_scanned{0};     ///< 第一阶段读取的分类 logit 数。
  int selected_anchors{0};          ///< 第一阶段入选的 anchor 数。
  std::int64_t gathered_values{0};  ///< 第二阶段读取的 logit 数。
};

/** @brief stable TopK 内部使用的量化分数与原始顺序。 */
struct NmsFreeRankedValue {
  std::int8_t value{0};   ///< 量化 logit。
  int original_index{0};  ///< 原始顺序，同分时用于稳定排序。
};

/** @brief YOLO26 NMS-free TopK 选择器需要的调用方 scratch 缓冲区。 */
struct NmsFreeTopKScratch {
  std::int8_t *anchor_max_logits{nullptr};        ///< 长度至少为 anchors。
  std::size_t anchor_max_capacity{0};             ///< anchor_max_logits 容量。
  NmsFreeRankedValue *anchor_ranking{nullptr};    ///< 长度至少为 anchors。
  std::size_t anchor_ranking_capacity{0};         ///< anchor_ranking 容量。
  NmsFreeRankedValue *gathered_ranking{nullptr};  ///< 长度至少为 TopK×classes。
  std::size_t gathered_ranking_capacity{0};       ///< gathered_ranking 容量。
};

inline constexpr int kNmsFreeTopKInvalidArgument = -1;       ///< 参数无效。
inline constexpr int kNmsFreeTopKInsufficientCapacity = -2;  ///< 缓冲区容量不足。

/**
 * @brief 从 YOLO26 O2O 头的 INT8 logits 中执行两阶段 stable exact TopK。
 * @details 语义与 YOLO26 `_postprocess_export()` 一致：先按 anchor 取
 * 跨类别最大分数 TopK，再对入选 anchor 的全部类别展平 TopK。
 * 排序在同分时保持原始顺序，因此允许同一 anchor 以多个类别入选；
 * 本函数不执行 NMS。sigmoid 为单调函数，所以直接在量化 logit
 * 域完成 TopK，仅对最终候选反量化并计算置信度。
 *
 * @param logits 分类 logits，布局为 `[classes, anchors]`。
 * @param classes 类别数，必须大于零。
 * @param anchors anchor 数，必须大于零。
 * @param quant logits 的 affine INT8 量化参数。
 * @param confidence_threshold sigmoid 置信度阈值，范围 `[0, 1]`。
 * @param max_det 两阶段 TopK 上限，必须大于零。
 * @param scratch 调用方提供的复用缓冲区。
 * @param[out] output 最终候选，按分数降序且同分稳定。
 * @param output_capacity output 容量，至少为 `min(max_det, anchors)`。
 * @param[out] stats 可选的读取量统计。
 * @return 候选数；负值表示参数或容量错误。
 */
int SelectNmsFreeTopKSeedsInt8(const std::int8_t *logits, int classes, int anchors,
                               AffineInt8Quant quant, float confidence_threshold, int max_det,
                               NmsFreeTopKScratch scratch, ClassSeed *output,
                               std::size_t output_capacity,
                               NmsFreeTopKStats *stats = nullptr) noexcept;

}  // namespace paddleyolo_rknn::postprocess
