// Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file roi_mask_decoder.cpp
 * @brief YOLOv8 五输出分割模型的 ROI mask 解码内核实现。
 */

#include "postprocess/roi_mask_decoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define PADDLEYOLO_RKNN_HAS_NEON 1
#else
#define PADDLEYOLO_RKNN_HAS_NEON 0
#endif

namespace paddleyolo_rknn::postprocess {
namespace {

/**
 * @brief 校验 proto 形状与 ROI 是否满足内核访存前置条件。
 * @param channels proto 通道数。
 * @param proto_h proto 高度。
 * @param proto_w proto 宽度。
 * @param roi proto 坐标系下的 ROI。
 */
void ValidateShapeAndRoi(const int channels, const int proto_h,
                         const int proto_w, const cv::Rect &roi) {
  CV_Assert(channels > 0);
  CV_Assert(proto_h > 0 && proto_w > 0);
  CV_Assert(roi.x >= 0 && roi.y >= 0);
  CV_Assert(roi.width >= 0 && roi.height >= 0);
  CV_Assert(roi.x <= proto_w && roi.y <= proto_h);
  CV_Assert(roi.width <= proto_w - roi.x);
  CV_Assert(roi.height <= proto_h - roi.y);
}

/**
 * @brief 校验原生 `NC1HWC2` proto 视图与 ROI。
 * @param proto 原生 proto 视图。
 * @param roi proto 坐标系下的 ROI。
 */
void ValidateNc1hwc2View(const Nc1hwc2Int8View &proto, const cv::Rect &roi) {
  ValidateShapeAndRoi(proto.channels, proto.height, proto.width, roi);
  CV_Assert(proto.channel_blocks > 0);
  CV_Assert(proto.block_size == 8 || proto.block_size == 16);
  CV_Assert(proto.width_stride >= proto.width);
  CV_Assert(proto.channels <= proto.channel_blocks * proto.block_size);
}

/**
 * @brief 计算与 NEON 路径一致的标量 sigmoid。
 * @param value FP32 logit。
 * @return sigmoid 概率。
 */
float SigmoidScalar(const float value) {
  const float clipped = std::clamp(value, -88.0F, 88.0F);
  return 1.0F / (1.0F + std::exp(-clipped));
}

#if PADDLEYOLO_RKNN_HAS_NEON
/**
 * @brief 计算四路快速指数。
 * @param value 四路 FP32 输入。
 * @return 四路指数近似值。
 */
float32x4_t ExpFast4(float32x4_t value) {
  value = vminq_f32(vmaxq_f32(value, vdupq_n_f32(-88.0F)), vdupq_n_f32(88.0F));
  const float32x4_t base2 = vmulq_n_f32(value, 1.4426950408889634F);
  const int32x4_t exponent = vcvtmq_s32_f32(base2);
  const float32x4_t fraction = vsubq_f32(base2, vcvtq_f32_s32(exponent));
  float32x4_t polynomial =
      vmlaq_n_f32(vdupq_n_f32(0.0096181291F), fraction, 0.0013333558F);
  polynomial = vmlaq_f32(vdupq_n_f32(0.0555041087F), polynomial, fraction);
  polynomial = vmlaq_f32(vdupq_n_f32(0.2402265070F), polynomial, fraction);
  polynomial = vmlaq_f32(vdupq_n_f32(0.6931471805F), polynomial, fraction);
  polynomial = vmlaq_f32(vdupq_n_f32(1.0F), polynomial, fraction);
  const int32x4_t exponent_bits =
      vshlq_n_s32(vaddq_s32(exponent, vdupq_n_s32(127)), 23);
  return vmulq_f32(polynomial, vreinterpretq_f32_s32(exponent_bits));
}

/**
 * @brief 计算四路 sigmoid。
 * @param value 四路 FP32 logits。
 * @return 四路 sigmoid 概率。
 */
float32x4_t Sigmoid4(const float32x4_t value) {
  const float32x4_t one = vdupq_n_f32(1.0F);
  return vdivq_f32(one, vaddq_f32(one, ExpFast4(vnegq_f32(value))));
}

/**
 * @brief 将 8 行乘 8 列 INT8 方块转置为 8 个连续列向量。
 * @param rows 输入的 8 个行向量。
 * @param[out] columns 输出的 8 个列向量。
 */
void Transpose8x8Int8(const int8x8_t rows[8], int8x8_t columns[8]) {
  const int8x8x2_t bytes01 = vtrn_s8(rows[0], rows[1]);
  const int8x8x2_t bytes23 = vtrn_s8(rows[2], rows[3]);
  const int8x8x2_t bytes45 = vtrn_s8(rows[4], rows[5]);
  const int8x8x2_t bytes67 = vtrn_s8(rows[6], rows[7]);
  const int16x4x2_t words02 = vtrn_s16(vreinterpret_s16_s8(bytes01.val[0]),
                                       vreinterpret_s16_s8(bytes23.val[0]));
  const int16x4x2_t words13 = vtrn_s16(vreinterpret_s16_s8(bytes01.val[1]),
                                       vreinterpret_s16_s8(bytes23.val[1]));
  const int16x4x2_t words46 = vtrn_s16(vreinterpret_s16_s8(bytes45.val[0]),
                                       vreinterpret_s16_s8(bytes67.val[0]));
  const int16x4x2_t words57 = vtrn_s16(vreinterpret_s16_s8(bytes45.val[1]),
                                       vreinterpret_s16_s8(bytes67.val[1]));
  const int32x2x2_t lanes04 = vtrn_s32(vreinterpret_s32_s16(words02.val[0]),
                                       vreinterpret_s32_s16(words46.val[0]));
  const int32x2x2_t lanes15 = vtrn_s32(vreinterpret_s32_s16(words13.val[0]),
                                       vreinterpret_s32_s16(words57.val[0]));
  const int32x2x2_t lanes26 = vtrn_s32(vreinterpret_s32_s16(words02.val[1]),
                                       vreinterpret_s32_s16(words46.val[1]));
  const int32x2x2_t lanes37 = vtrn_s32(vreinterpret_s32_s16(words13.val[1]),
                                       vreinterpret_s32_s16(words57.val[1]));
  columns[0] = vreinterpret_s8_s32(lanes04.val[0]);
  columns[1] = vreinterpret_s8_s32(lanes15.val[0]);
  columns[2] = vreinterpret_s8_s32(lanes26.val[0]);
  columns[3] = vreinterpret_s8_s32(lanes37.val[0]);
  columns[4] = vreinterpret_s8_s32(lanes04.val[1]);
  columns[5] = vreinterpret_s8_s32(lanes15.val[1]);
  columns[6] = vreinterpret_s8_s32(lanes26.val[1]);
  columns[7] = vreinterpret_s8_s32(lanes37.val[1]);
}

/**
 * @brief 载入 8 个相邻像素并按通道转置一个 C2 block。
 * @param source 第一个像素的通道块首地址。
 * @param block_size C2 block 宽度，仅支持 8 或 16。
 * @param[out] low 前 8 个通道对应的 8 像素向量。
 * @param[out] high 后 8 个通道对应的 8 像素向量；C2=8 时不写入。
 */
void LoadTranspose8Pixels(const std::int8_t *source, const int block_size,
                          int8x8_t low[8], int8x8_t high[8]) {
  if (block_size == 8) {
    int8x8_t rows[8];
    for (int pixel = 0; pixel < 8; ++pixel) {
      rows[pixel] = vld1_s8(source + pixel * block_size);
    }
    Transpose8x8Int8(rows, low);
    return;
  }
  int8x8_t low_rows[8];
  int8x8_t high_rows[8];
  for (int pixel = 0; pixel < 8; ++pixel) {
    const int8x16_t row = vld1q_s8(source + pixel * block_size);
    low_rows[pixel] = vget_low_s8(row);
    high_rows[pixel] = vget_high_s8(row);
  }
  Transpose8x8Int8(low_rows, low);
  Transpose8x8Int8(high_rows, high);
}
#endif

} // namespace

RoiMaskDecodePath
SelectRoiMaskDecodePath(const std::uint64_t total_roi_area) noexcept {
  return total_roi_area <= kAutoRoiInt8MaxArea ? RoiMaskDecodePath::kInt8
                                               : RoiMaskDecodePath::kFloat32;
}

void ComputeRoiMaskFloat32(const float *proto, const int channels,
                           const int proto_h, const int proto_w,
                           const float *coeff, const cv::Rect &roi,
                           cv::Mat &logits,
                           const RoiMaskActivation activation) {
  ValidateShapeAndRoi(channels, proto_h, proto_w, roi);
  if (roi.empty()) {
    logits.release();
    return;
  }
  CV_Assert(proto != nullptr);
  CV_Assert(coeff != nullptr);

  logits.create(roi.height, roi.width, CV_32F);
  logits.setTo(0.0F);
  const std::size_t plane_size = static_cast<std::size_t>(proto_h) * proto_w;
  for (int channel = 0; channel < channels; ++channel) {
    const float coefficient = coeff[channel];
    const float *plane = proto + static_cast<std::size_t>(channel) * plane_size;
#if PADDLEYOLO_RKNN_HAS_NEON
    const float32x4_t coefficient_v = vdupq_n_f32(coefficient);
#endif
    for (int y = 0; y < roi.height; ++y) {
      const float *source =
          plane + static_cast<std::size_t>(roi.y + y) * proto_w + roi.x;
      float *destination = logits.ptr<float>(y);
      int x = 0;
#if PADDLEYOLO_RKNN_HAS_NEON
      for (; x + 4 <= roi.width; x += 4) {
        float32x4_t accumulated = vld1q_f32(destination + x);
        accumulated =
            vmlaq_f32(accumulated, coefficient_v, vld1q_f32(source + x));
        if (activation == RoiMaskActivation::kSigmoid &&
            channel + 1 == channels) {
          accumulated = Sigmoid4(accumulated);
        }
        vst1q_f32(destination + x, accumulated);
      }
#endif
      for (; x < roi.width; ++x) {
        destination[x] = std::fma(coefficient, source[x], destination[x]);
        if (activation == RoiMaskActivation::kSigmoid &&
            channel + 1 == channels) {
          destination[x] = SigmoidScalar(destination[x]);
        }
      }
    }
  }
}

void ComputeRoiMaskInt8Nc1hwc2(const Nc1hwc2Int8View &proto, const float *coeff,
                               const float scale, const std::int32_t zero_point,
                               const cv::Rect &roi, cv::Mat &logits,
                               const RoiMaskActivation activation) {
  ValidateNc1hwc2View(proto, roi);
  if (roi.empty()) {
    logits.release();
    return;
  }
  CV_Assert(proto.data != nullptr);
  CV_Assert(coeff != nullptr);

  logits.create(roi.height, roi.width, CV_32F);
  const std::size_t block_plane = static_cast<std::size_t>(proto.height) *
                                  proto.width_stride * proto.block_size;
  for (int y = 0; y < roi.height; ++y) {
    const int source_y = roi.y + y;
    float *destination = logits.ptr<float>(y);
    int x = 0;
#if PADDLEYOLO_RKNN_HAS_NEON
    const int32x4_t zero_point_v = vdupq_n_s32(zero_point);
    const float32x4_t scale_v = vdupq_n_f32(scale);
    for (; x + 8 <= roi.width; x += 8) {
      float32x4_t accumulated_low = vdupq_n_f32(0.0F);
      float32x4_t accumulated_high = vdupq_n_f32(0.0F);
      for (int block = 0; block < proto.channel_blocks; ++block) {
        const int first_channel = block * proto.block_size;
        const int valid_channels =
            std::min(proto.block_size, proto.channels - first_channel);
        if (valid_channels <= 0) {
          break;
        }
        const std::size_t source_offset =
            static_cast<std::size_t>(block) * block_plane +
            (static_cast<std::size_t>(source_y) * proto.width_stride + roi.x +
             x) *
                proto.block_size;
        int8x8_t low_channels[8];
        int8x8_t high_channels[8];
        LoadTranspose8Pixels(proto.data + source_offset, proto.block_size,
                             low_channels, high_channels);
        for (int lane = 0; lane < valid_channels; ++lane) {
          const int8x8_t quantized =
              lane < 8 ? low_channels[lane] : high_channels[lane - 8];
          const int16x8_t quantized_i16 = vmovl_s8(quantized);
          const int32x4_t centered_low =
              vsubq_s32(vmovl_s16(vget_low_s16(quantized_i16)), zero_point_v);
          const int32x4_t centered_high =
              vsubq_s32(vmovl_s16(vget_high_s16(quantized_i16)), zero_point_v);
          const float32x4_t value_low =
              vmulq_f32(vcvtq_f32_s32(centered_low), scale_v);
          const float32x4_t value_high =
              vmulq_f32(vcvtq_f32_s32(centered_high), scale_v);
          const float32x4_t coefficient_v =
              vdupq_n_f32(coeff[first_channel + lane]);
          accumulated_low =
              vmlaq_f32(accumulated_low, coefficient_v, value_low);
          accumulated_high =
              vmlaq_f32(accumulated_high, coefficient_v, value_high);
        }
      }
      if (activation == RoiMaskActivation::kSigmoid) {
        accumulated_low = Sigmoid4(accumulated_low);
        accumulated_high = Sigmoid4(accumulated_high);
      }
      vst1q_f32(destination + x, accumulated_low);
      vst1q_f32(destination + x + 4, accumulated_high);
    }
#endif
    for (; x < roi.width; ++x) {
      const int source_x = roi.x + x;
      float accumulated = 0.0F;
      for (int channel = 0; channel < proto.channels; ++channel) {
        const int block = channel / proto.block_size;
        const int lane = channel % proto.block_size;
        const std::size_t source_offset =
            static_cast<std::size_t>(block) * block_plane +
            (static_cast<std::size_t>(source_y) * proto.width_stride +
             source_x) *
                proto.block_size +
            lane;
        const std::int32_t centered =
            static_cast<std::int32_t>(proto.data[source_offset]) - zero_point;
        accumulated = std::fma(
            coeff[channel], static_cast<float>(centered) * scale, accumulated);
      }
      destination[x] = activation == RoiMaskActivation::kSigmoid
                           ? SigmoidScalar(accumulated)
                           : accumulated;
    }
  }
}

bool DequantizeNc1hwc2Int8ToNchwFloat32(
    const Nc1hwc2Int8View &proto, const float scale,
    const std::int32_t zero_point, float *output,
    const std::size_t output_count) noexcept {
  if (proto.data == nullptr || output == nullptr || proto.channels <= 0 ||
      proto.channel_blocks <= 0 || proto.height <= 0 || proto.width <= 0 ||
      proto.width_stride < proto.width ||
      (proto.block_size != 8 && proto.block_size != 16) ||
      proto.channels > proto.channel_blocks * proto.block_size) {
    return false;
  }
  const std::size_t plane_size =
      static_cast<std::size_t>(proto.height) * proto.width;
  const std::size_t required =
      static_cast<std::size_t>(proto.channels) * plane_size;
  if (output_count < required) {
    return false;
  }
  const std::size_t block_plane = static_cast<std::size_t>(proto.height) *
                                  proto.width_stride * proto.block_size;
  for (int y = 0; y < proto.height; ++y) {
    int x = 0;
#if PADDLEYOLO_RKNN_HAS_NEON
    const int32x4_t zero_point_v = vdupq_n_s32(zero_point);
    const float32x4_t scale_v = vdupq_n_f32(scale);
    for (; x + 8 <= proto.width; x += 8) {
      for (int block = 0; block < proto.channel_blocks; ++block) {
        const int first_channel = block * proto.block_size;
        const int valid_channels =
            std::min(proto.block_size, proto.channels - first_channel);
        if (valid_channels <= 0) {
          break;
        }
        const std::size_t source_offset =
            static_cast<std::size_t>(block) * block_plane +
            (static_cast<std::size_t>(y) * proto.width_stride + x) *
                proto.block_size;
        int8x8_t low_channels[8];
        int8x8_t high_channels[8];
        LoadTranspose8Pixels(proto.data + source_offset, proto.block_size,
                             low_channels, high_channels);
        for (int lane = 0; lane < valid_channels; ++lane) {
          const int8x8_t quantized =
              lane < 8 ? low_channels[lane] : high_channels[lane - 8];
          const int16x8_t quantized_i16 = vmovl_s8(quantized);
          const int32x4_t centered_low =
              vsubq_s32(vmovl_s16(vget_low_s16(quantized_i16)), zero_point_v);
          const int32x4_t centered_high =
              vsubq_s32(vmovl_s16(vget_high_s16(quantized_i16)), zero_point_v);
          float *destination =
              output +
              static_cast<std::size_t>(first_channel + lane) * plane_size +
              static_cast<std::size_t>(y) * proto.width + x;
          vst1q_f32(destination,
                    vmulq_f32(vcvtq_f32_s32(centered_low), scale_v));
          vst1q_f32(destination + 4,
                    vmulq_f32(vcvtq_f32_s32(centered_high), scale_v));
        }
      }
    }
#endif
    for (; x < proto.width; ++x) {
      for (int channel = 0; channel < proto.channels; ++channel) {
        const int block = channel / proto.block_size;
        const int lane = channel % proto.block_size;
        const std::size_t source_offset =
            static_cast<std::size_t>(block) * block_plane +
            (static_cast<std::size_t>(y) * proto.width_stride + x) *
                proto.block_size +
            lane;
        const std::int32_t centered =
            static_cast<std::int32_t>(proto.data[source_offset]) - zero_point;
        output[static_cast<std::size_t>(channel) * plane_size +
               static_cast<std::size_t>(y) * proto.width + x] =
            static_cast<float>(centered) * scale;
      }
    }
  }
  return true;
}

void AssignBinaryMaskFromProbabilityMat(const cv::Mat &probabilities,
                                        const int threshold_value,
                                        cv::Mat &binary) {
  CV_Assert(probabilities.type() == CV_32F);
  binary.create(probabilities.rows, probabilities.cols, CV_8U);
  if (probabilities.empty()) {
    return;
  }
  if (threshold_value < 0) {
    binary.setTo(255);
    return;
  }
  if (threshold_value >= 255) {
    binary.setTo(0);
    return;
  }

  for (int row = 0; row < probabilities.rows; ++row) {
    const float *source = probabilities.ptr<float>(row);
    std::uint8_t *destination = binary.ptr<std::uint8_t>(row);
    int column = 0;
#if PADDLEYOLO_RKNN_HAS_NEON
    const float32x4_t scale = vdupq_n_f32(255.0F);
    const uint8x16_t threshold =
        vdupq_n_u8(static_cast<std::uint8_t>(threshold_value));
    for (; column + 16 <= probabilities.cols; column += 16) {
      const int32x4_t q0 =
          vcvtnq_s32_f32(vmulq_f32(vld1q_f32(source + column), scale));
      const int32x4_t q1 =
          vcvtnq_s32_f32(vmulq_f32(vld1q_f32(source + column + 4), scale));
      const int32x4_t q2 =
          vcvtnq_s32_f32(vmulq_f32(vld1q_f32(source + column + 8), scale));
      const int32x4_t q3 =
          vcvtnq_s32_f32(vmulq_f32(vld1q_f32(source + column + 12), scale));
      const uint16x8_t q01 = vcombine_u16(vqmovun_s32(q0), vqmovun_s32(q1));
      const uint16x8_t q23 = vcombine_u16(vqmovun_s32(q2), vqmovun_s32(q3));
      const uint8x16_t quantized =
          vcombine_u8(vqmovn_u16(q01), vqmovn_u16(q23));
      vst1q_u8(destination + column, vcgtq_u8(quantized, threshold));
    }
#endif
    for (; column < probabilities.cols; ++column) {
      const std::uint8_t quantized =
          cv::saturate_cast<std::uint8_t>(source[column] * 255.0F);
      destination[column] =
          quantized > threshold_value ? UINT8_C(255) : UINT8_C(0);
    }
  }
}

} // namespace paddleyolo_rknn::postprocess
