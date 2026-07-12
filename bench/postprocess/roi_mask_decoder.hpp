// Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file roi_mask_decoder.hpp
 * @brief YOLOv8 五输出分割模型的 ROI mask 解码内核。
 */
#pragma once

#include "postprocess/five_output_runtime.hpp"

#include <cstddef>
#include <cstdint>

#include <opencv2/core.hpp>

namespace paddleyolo_rknn::postprocess {

/** @brief ROI mask 自动解码可选的两条优化路径。 */
enum class RoiMaskDecodePath {
  kInt8,   ///< 小 ROI 直接消费原生 INT8 proto。
  kFloat32 ///< 大 ROI 先融合反量化为 NCHW FP32 proto。
};

/** @brief ROI mask 内核输出激活方式。 */
enum class RoiMaskActivation {
  kNone,   ///< 输出原始 logits。
  kSigmoid ///< 在最终写回时输出 sigmoid 概率。
};

/** @brief 自动模式选择 INT8 ROI 路径的最大 proto ROI 总面积。 */
inline constexpr std::uint64_t kAutoRoiInt8MaxArea = 8000U;

/**
 * @brief 按本帧 proto ROI 总面积选择 mask 解码路径。
 * @param total_roi_area 所有待解码 ROI 的面积之和。
 * @return 面积不大于阈值时返回 INT8 路径，否则返回 FP32 路径。
 */
RoiMaskDecodePath
SelectRoiMaskDecodePath(std::uint64_t total_roi_area) noexcept;

/**
 * @brief ROI 内从 FP32 NCHW proto 逐通道累加 mask。
 * @param proto FP32 proto 首地址，布局为 `[channels, proto_h, proto_w]`。
 * @param channels mask proto 通道数。
 * @param proto_h proto 高度。
 * @param proto_w proto 宽度。
 * @param coeff 单个检测框的 mask 系数，长度为 channels。
 * @param roi proto 坐标系下的有效 ROI。
 * @param[out] logits 输出 ROI logits 或概率，类型为
 * `CV_32F`；尺寸匹配时复用内存。
 * @param activation 最终写回时使用的激活方式。
 */
void ComputeRoiMaskFloat32(
    const float *proto, int channels, int proto_h, int proto_w,
    const float *coeff, const cv::Rect &roi, cv::Mat &logits,
    RoiMaskActivation activation = RoiMaskActivation::kNone);

/**
 * @brief 直接从 RKNN 原生 `NC1HWC2` INT8 proto 计算 ROI mask。
 * @details 支持 C2=8/16、宽度 stride 和末通道 padding；每个元素严格按
 * `(q - zero_point) -> float -> scale -> coeff` 的顺序计算。
 * @param proto 原生 `NC1HWC2` proto 只读视图。
 * @param coeff 单个检测框的 mask 系数，长度为 `proto.channels`。
 * @param scale proto 量化 scale。
 * @param zero_point proto 量化 zero-point。
 * @param roi proto 坐标系下的有效 ROI。
 * @param[out] logits 输出 ROI logits 或概率，类型为
 * `CV_32F`；尺寸匹配时复用内存。
 * @param activation 最终写回时使用的激活方式。
 */
void ComputeRoiMaskInt8Nc1hwc2(
    const Nc1hwc2Int8View &proto, const float *coeff, float scale,
    std::int32_t zero_point, const cv::Rect &roi, cv::Mat &logits,
    RoiMaskActivation activation = RoiMaskActivation::kNone);

/**
 * @brief 将原生 `NC1HWC2` INT8 proto 融合转换为逻辑 NCHW FP32。
 * @param proto 原生 `NC1HWC2` proto 只读视图。
 * @param scale proto 量化 scale。
 * @param zero_point proto 量化 zero-point。
 * @param[out] output 调用方预分配的 NCHW FP32 缓冲区。
 * @param output_count 输出缓冲区元素容量。
 * @return 转换成功返回 `true`，参数或容量无效时返回 `false`。
 */
bool DequantizeNc1hwc2Int8ToNchwFloat32(const Nc1hwc2Int8View &proto,
                                        float scale, std::int32_t zero_point,
                                        float *output,
                                        std::size_t output_count) noexcept;

/**
 * @brief 将 resize 后的概率 mask 单遍量化并二值化。
 * @details 等价于 `convertTo(CV_8U, 255.0)` 后执行 `THRESH_BINARY`；AArch64
 * 使用 NEON 一次完成量化、饱和窄化、阈值比较和写回。
 * @param probabilities 输入概率 mask，类型为 `CV_32F`。
 * @param threshold_value 量化域阈值，像素满足 `q > threshold_value` 时输出
 * 255。
 * @param[out] binary 输出二值 mask，类型为 `CV_8U`；尺寸匹配时复用内存。
 */
void AssignBinaryMaskFromProbabilityMat(const cv::Mat &probabilities,
                                        int threshold_value, cv::Mat &binary);

} // namespace paddleyolo_rknn::postprocess
