// Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstddef>
#include <cstdint>

namespace paddleyolo_rknn::postprocess {

/** @brief affine INT8 量化参数。 */
struct AffineInt8Quant {
  float scale{1.0F};          ///< 量化比例。
  std::int32_t zero_point{0}; ///< 量化零点。
};

/** @brief 第五输出 score_sum 的只读视图。 */
struct ScoreSumInt8View {
  const std::int8_t *data{nullptr}; ///< `[anchors]` INT8 数据。
  AffineInt8Quant quant{};          ///< score_sum 量化参数。
};

/** @brief 每 anchor 最优类别分类阶段产生的候选。 */
struct ClassSeed {
  int anchor{-1};         ///< anchor 索引。
  int class_id{-1};       ///< 类别索引。
  float confidence{0.0F}; ///< sigmoid 置信度。
};

/** @brief 分类读取量与第五输出命中统计。 */
struct ClassSelectionStats {
  int score_sum_scanned{0};             ///< 扫描的 score_sum 数量。
  int class_anchors_scanned{0};         ///< 扫描分类的 anchor 数量。
  std::int64_t class_values_scanned{0}; ///< 读取的分类 logit 数量。
  bool used_score_sum{false};           ///< 是否实际启用安全预筛。
};

inline constexpr int kClassSelectionInvalidArgument = -1; ///< 参数无效。
inline constexpr int kClassSelectionInsufficientCapacity =
    -2; ///< 输出容量不足。
inline constexpr int kScoreSumPrescreenUnavailable =
    -3; ///< 第五输出无法安全预筛。

/**
 * @brief 仅扫描第五输出并收集需要精确分类的 anchor。
 * @param score_sum 第五输出及其量化参数。
 * @param anchors anchor 数，必须大于零。
 * @param confidence_threshold sigmoid 置信度阈值。
 * @param[out] output 调用方提供的 anchor 索引缓冲区，结果按升序排列。
 * @param output_capacity output 容量。
 * @return 存活 anchor 数；负值为 `kClassSelectionInvalidArgument`、
 * `kClassSelectionInsufficientCapacity` 或 `kScoreSumPrescreenUnavailable`。
 */
int CollectScoreSumSurvivorsInt8(const ScoreSumInt8View &score_sum, int anchors,
                                 float confidence_threshold, int *output,
                                 std::size_t output_capacity) noexcept;

/**
 * @brief 对指定 anchor 执行精确 best-class 分类。
 * @details 每个 anchor 仅保留最大 logit 对应的第一个类别，与 Python
 * 五输出解码及常规 YOLO `multi_label=false` 语义一致。最终判断始终使用
 * 反量化 logit 和严格浮点阈值。
 *
 * @param logits INT8 分类 logits，布局为 `[classes,anchors]`。
 * @param classes 类别数，必须大于零。
 * @param anchors anchor 数，必须大于零。
 * @param logit_quant 分类 logits 量化参数。
 * @param confidence_threshold sigmoid 置信度阈值。
 * @param anchor_indices 待分类的 anchor 索引。
 * @param anchor_count anchor_indices 元素数。
 * @param[out] output 调用方提供的候选缓冲区。
 * @param output_capacity output 容量。
 * @param[out] stats 可选读取量统计。
 * @return 候选数；负值为 `kClassSelectionInvalidArgument` 或
 * `kClassSelectionInsufficientCapacity`。
 */
int ClassifyAnchorsBestClassInt8(const std::int8_t *logits, int classes,
                                 int anchors, AffineInt8Quant logit_quant,
                                 float confidence_threshold,
                                 const int *anchor_indices,
                                 std::size_t anchor_count, ClassSeed *output,
                                 std::size_t output_capacity,
                                 ClassSelectionStats *stats = nullptr) noexcept;

/**
 * @brief 从 INT8 `[classes,anchors]` logits 中选择 best-class 候选。
 * @details
 * score_sum 可安全预筛时，先以 NEON 顺序扫描第五输出，仅读取
 * survivor 对应的全部类别 logits。缺少第五输出或量化上端无法
 * 安全判定时，回退为完整 anchor 扫描。每个 anchor 仅输出最大类别，
 * 最终始终使用精确反量化值与 sigmoid 严格判断阈值。
 *
 * @param logits INT8 分类 logits，布局为 `[classes,anchors]`。
 * @param classes 类别数，必须大于零。
 * @param anchors anchor 数，必须大于零。
 * @param logit_quant 分类 logits 量化参数。
 * @param confidence_threshold sigmoid 置信度阈值。
 * @param score_sum 可选第五输出；为空时完整扫描。
 * @param[out] output 调用方提供的候选缓冲区。
 * @param output_capacity output 容量，应不小于 `anchors`。
 * @param[out] stats 可选读取量统计。
 * @return 候选数；负值为 `kClassSelectionInvalidArgument` 或
 * `kClassSelectionInsufficientCapacity`。
 */
int SelectBestClassSeedsInt8(const std::int8_t *logits, int classes,
                             int anchors, AffineInt8Quant logit_quant,
                             float confidence_threshold,
                             const ScoreSumInt8View *score_sum,
                             ClassSeed *output, std::size_t output_capacity,
                             ClassSelectionStats *stats = nullptr) noexcept;

} // namespace paddleyolo_rknn::postprocess
