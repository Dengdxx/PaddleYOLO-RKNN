// Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "rknn_api.h"

namespace paddleyolo_rknn::postprocess {

/** @brief 五输出张量编号。 */
enum class FiveOutputTensor : std::uint8_t {
  kBox = 0,
  kClass = 1,
  kMaskCoeff = 2,
  kProto = 3,
  kScoreSum = 4,
};

/** @brief RKNN 原生 NC1HWC2 INT8 proto 只读视图。 */
struct Nc1hwc2Int8View {
  const std::int8_t *data{nullptr};
  int channels{0};
  int channel_blocks{0};
  int height{0};
  int width{0};
  int width_stride{0};
  int block_size{0};
};

/** @brief 单帧按需同步统计。 */
struct FiveOutputSyncStats {
  std::uint8_t ready_mask{0};  ///< 已同步并准备完成的输出位图。
  std::size_t native_bytes{0}; ///< 已同步的原生 DMA 缓冲区字节数。
  double sync_ms{0.0};         ///< DMA 同步与必要布局恢复耗时。
};

/**
 * @brief 五输出 RKNN full-IO zero-copy 绑定与按需同步运行时。
 * @details 输入和五个输出统一使用 `rknn_set_io_mem()`；输出仅在 CPU 后处理
 * 真正需要时执行 `RKNN_MEMORY_SYNC_FROM_DEVICE`。
 */
class FiveOutputRuntime {
public:
  /** @brief 构造空运行时。 */
  FiveOutputRuntime() = default;

  /** @brief 析构并释放全部 RKNN tensor memory。 */
  ~FiveOutputRuntime();

  FiveOutputRuntime(const FiveOutputRuntime &) = delete;
  FiveOutputRuntime &operator=(const FiveOutputRuntime &) = delete;

  /**
   * @brief 创建并绑定输入与五个原生输出缓冲区。
   * @param context 已初始化的 RKNN context。
   * @param logical_input 逻辑输入属性。
   * @param logical_outputs 五个逻辑输出属性。
   * @param output_count 输出数量，必须为 5。
   * @return 成功返回 true。
   */
  bool Initialize(rknn_context context, const rknn_tensor_attr &logical_input,
                  const rknn_tensor_attr *logical_outputs,
                  std::uint32_t output_count);

  /**
   * @brief 将固定评测输入复制到 zero-copy 输入缓冲区。
   * @param data HWC UINT8 输入数据。
   * @param bytes 输入字节数，必须与模型逻辑输入大小完全一致。
   * @return 字节数严格匹配时返回 true。
   */
  bool SetInput(const void *data, std::size_t bytes);

  /**
   * @brief 同步输入并执行一次 RKNN 推理。
   * @return RKNN 返回码。
   */
  int Run();

  /**
   * @brief 将指定输出同步到 CPU 并准备为后处理可读布局。
   * @param tensor 输出编号。
   * @return RKNN 错误码；布局不支持时返回负值。
   */
  int Prepare(FiveOutputTensor tensor);

  /** @brief 开始新帧并清空 ready mask/同步统计。 */
  void BeginFrame() noexcept;

  /** @brief 获取五个后处理输出描述。 */
  rknn_output *Outputs() noexcept { return outputs_.data(); }

  /** @brief 获取原生 proto 视图。 */
  const Nc1hwc2Int8View &ProtoView() const noexcept { return proto_view_; }

  /** @brief 获取当前帧同步统计。 */
  const FiveOutputSyncStats &Stats() const noexcept { return stats_; }

  /** @brief 查询运行时是否完成初始化。 */
  bool IsInitialized() const noexcept { return initialized_; }

  /** @brief 释放全部绑定资源。 */
  void Release() noexcept;

private:
  bool PrepareLogicalOutput(std::size_t index);
  bool BuildProtoView();

  rknn_context context_{0};
  rknn_tensor_mem *input_mem_{nullptr};
  std::size_t input_capacity_{0};
  std::size_t logical_input_bytes_{0};
  std::array<rknn_tensor_mem *, 5> output_mems_{};
  std::array<rknn_tensor_attr, 5> logical_attrs_{};
  std::array<rknn_tensor_attr, 5> native_attrs_{};
  std::array<rknn_output, 5> outputs_{};
  std::array<std::vector<std::uint8_t>, 5> logical_buffers_{};
  Nc1hwc2Int8View proto_view_{};
  FiveOutputSyncStats stats_{};
  bool initialized_{false};
};

} // namespace paddleyolo_rknn::postprocess
