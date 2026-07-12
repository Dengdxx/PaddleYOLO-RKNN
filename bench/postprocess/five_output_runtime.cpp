// Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
// SPDX-License-Identifier: AGPL-3.0-only

#include "postprocess/five_output_runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>

namespace paddleyolo_rknn::postprocess {
namespace {

/** @brief 返回单调时钟的毫秒差。 */
double ElapsedMs(const std::chrono::steady_clock::time_point begin,
                 const std::chrono::steady_clock::time_point end) noexcept {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

/** @brief 判断原生输出布局是否可直接读取或安全恢复。 */
bool IsSupportedLayout(const rknn_tensor_attr &native_attr,
                       const rknn_tensor_attr &logical_attr) noexcept {
  if (native_attr.type != logical_attr.type)
    return false;
  if (native_attr.fmt == RKNN_TENSOR_NC1HWC2) {
    return logical_attr.fmt == RKNN_TENSOR_NCHW ||
           logical_attr.fmt == RKNN_TENSOR_UNDEFINED;
  }
  if (native_attr.fmt == RKNN_TENSOR_NCHW) {
    return logical_attr.fmt == RKNN_TENSOR_NCHW ||
           logical_attr.fmt == RKNN_TENSOR_UNDEFINED;
  }
  if (native_attr.fmt == RKNN_TENSOR_NHWC) {
    return logical_attr.fmt == RKNN_TENSOR_NHWC;
  }
  return native_attr.fmt == RKNN_TENSOR_UNDEFINED &&
         logical_attr.fmt == RKNN_TENSOR_UNDEFINED;
}

/** @brief 将 NC1HWC2 原生输出恢复为逻辑 NCHW。 */
bool UnpackNc1hwc2(const void *source, const rknn_tensor_attr &native_attr,
                   const rknn_tensor_attr &logical_attr,
                   std::vector<std::uint8_t> *destination) {
  if (source == nullptr || destination == nullptr || native_attr.n_dims != 5 ||
      logical_attr.n_dims < 2 || logical_attr.n_elems == 0 ||
      logical_attr.size == 0 || logical_attr.size % logical_attr.n_elems != 0 ||
      native_attr.type != logical_attr.type) {
    return false;
  }
  const std::size_t element_size = logical_attr.size / logical_attr.n_elems;
  const std::size_t batch = logical_attr.dims[0];
  const std::size_t channels = logical_attr.dims[1];
  std::size_t spatial = 1;
  for (std::uint32_t dimension = 2; dimension < logical_attr.n_dims;
       ++dimension) {
    spatial *= logical_attr.dims[dimension];
  }
  const std::size_t native_batch = native_attr.dims[0];
  const std::size_t channel_blocks = native_attr.dims[1];
  const std::size_t height = native_attr.dims[2];
  const std::size_t width = native_attr.dims[3];
  const std::size_t block_size = native_attr.dims[4];
  const std::size_t width_stride =
      native_attr.w_stride > 0 ? native_attr.w_stride : width;
  if (batch != native_batch || spatial != height * width || block_size == 0 ||
      channels > channel_blocks * block_size || width_stride < width) {
    return false;
  }
  const std::size_t required_native_bytes = batch * channel_blocks * height *
                                            width_stride * block_size *
                                            element_size;
  if (required_native_bytes > native_attr.size_with_stride)
    return false;

  destination->resize(logical_attr.size);
  const auto *source_bytes = static_cast<const std::uint8_t *>(source);
  for (std::size_t n = 0; n < batch; ++n) {
    for (std::size_t channel = 0; channel < channels; ++channel) {
      const std::size_t c1 = channel / block_size;
      const std::size_t c2 = channel % block_size;
      for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
          const std::size_t source_element =
              ((((n * channel_blocks + c1) * height + y) * width_stride + x) *
               block_size) +
              c2;
          const std::size_t destination_element =
              (n * channels + channel) * spatial + y * width + x;
          std::memcpy(destination->data() + destination_element * element_size,
                      source_bytes + source_element * element_size,
                      element_size);
        }
      }
    }
  }
  return true;
}

/** @brief 按 NCHW/NHWC/未指定逻辑布局去除原生张量的宽度 stride padding。 */
bool UnpackStrided(const void *source, const rknn_tensor_attr &native_attr,
                   const rknn_tensor_attr &logical_attr,
                   std::vector<std::uint8_t> *destination) {
  if (source == nullptr || destination == nullptr || logical_attr.n_dims == 0 ||
      logical_attr.n_elems == 0 || logical_attr.size == 0 ||
      logical_attr.size % logical_attr.n_elems != 0 ||
      native_attr.type != logical_attr.type || native_attr.w_stride == 0) {
    return false;
  }
  const std::size_t element_size = logical_attr.size / logical_attr.n_elems;
  std::size_t width = 0;
  std::size_t row_elements = 0;
  std::size_t source_row_elements = 0;
  if (logical_attr.fmt == RKNN_TENSOR_NCHW ||
      logical_attr.fmt == RKNN_TENSOR_UNDEFINED) {
    width = logical_attr.dims[logical_attr.n_dims - 1];
    row_elements = width;
    source_row_elements = native_attr.w_stride;
  } else if (logical_attr.fmt == RKNN_TENSOR_NHWC && logical_attr.n_dims >= 2) {
    width = logical_attr.dims[logical_attr.n_dims - 2];
    const std::size_t channels = logical_attr.dims[logical_attr.n_dims - 1];
    if (channels == 0 ||
        width > std::numeric_limits<std::size_t>::max() / channels ||
        native_attr.w_stride >
            std::numeric_limits<std::size_t>::max() / channels) {
      return false;
    }
    row_elements = width * channels;
    source_row_elements = native_attr.w_stride * channels;
  } else {
    return false;
  }
  if (width == 0 || row_elements == 0 || native_attr.w_stride < width ||
      logical_attr.n_elems % row_elements != 0) {
    return false;
  }
  const std::size_t rows = logical_attr.n_elems / row_elements;
  if (source_row_elements >
          std::numeric_limits<std::size_t>::max() / element_size ||
      rows > std::numeric_limits<std::size_t>::max() /
                 (source_row_elements * element_size) ||
      rows * source_row_elements * element_size >
          native_attr.size_with_stride) {
    return false;
  }
  const auto *source_bytes = static_cast<const std::uint8_t *>(source);
  destination->resize(logical_attr.size);
  for (std::size_t row = 0; row < rows; ++row) {
    std::memcpy(destination->data() + row * row_elements * element_size,
                source_bytes + row * source_row_elements * element_size,
                row_elements * element_size);
  }
  return true;
}

} // namespace

FiveOutputRuntime::~FiveOutputRuntime() { Release(); }

bool FiveOutputRuntime::Initialize(rknn_context context,
                                   const rknn_tensor_attr &logical_input,
                                   const rknn_tensor_attr *logical_outputs,
                                   const std::uint32_t output_count) {
  Release();
  if (context == 0 || logical_outputs == nullptr || output_count != 5U)
    return false;
  context_ = context;

  rknn_tensor_attr native_input{};
  native_input.index = 0;
  int result = rknn_query(context_, RKNN_QUERY_NATIVE_INPUT_ATTR, &native_input,
                          sizeof(native_input));
  if (result != 0) {
    Release();
    return false;
  }
  input_capacity_ = std::max<std::size_t>(logical_input.size_with_stride,
                                          native_input.size_with_stride);
  logical_input_bytes_ = logical_input.size;
  if (logical_input_bytes_ == 0 || input_capacity_ < logical_input_bytes_ ||
      input_capacity_ > std::numeric_limits<std::uint32_t>::max()) {
    Release();
    return false;
  }
  input_mem_ =
      rknn_create_mem(context_, static_cast<std::uint32_t>(input_capacity_));
  if (input_mem_ == nullptr || input_mem_->virt_addr == nullptr) {
    Release();
    return false;
  }
  native_input.type = RKNN_TENSOR_UINT8;
  native_input.fmt = RKNN_TENSOR_NHWC;
  if (rknn_set_io_mem(context_, input_mem_, &native_input) != 0) {
    Release();
    return false;
  }

  for (std::size_t index = 0; index < output_mems_.size(); ++index) {
    logical_attrs_[index] = logical_outputs[index];
    native_attrs_[index] = {};
    native_attrs_[index].index = static_cast<std::uint32_t>(index);
    result = rknn_query(context_, RKNN_QUERY_NATIVE_OUTPUT_ATTR,
                        &native_attrs_[index], sizeof(native_attrs_[index]));
    if (result != 0 || native_attrs_[index].size_with_stride == 0 ||
        !IsSupportedLayout(native_attrs_[index], logical_attrs_[index])) {
      Release();
      return false;
    }
    if (native_attrs_[index].qnt_type != RKNN_TENSOR_QNT_NONE &&
        (native_attrs_[index].qnt_type != RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC ||
         native_attrs_[index].zp != logical_attrs_[index].zp ||
         native_attrs_[index].scale != logical_attrs_[index].scale)) {
      Release();
      return false;
    }
    output_mems_[index] =
        rknn_create_mem(context_, native_attrs_[index].size_with_stride);
    if (output_mems_[index] == nullptr ||
        output_mems_[index]->virt_addr == nullptr ||
        rknn_set_io_mem(context_, output_mems_[index], &native_attrs_[index]) !=
            0) {
      Release();
      return false;
    }
    outputs_[index] = {};
    outputs_[index].index = static_cast<std::uint32_t>(index);
    outputs_[index].want_float = 0;
    outputs_[index].is_prealloc = 1;
    outputs_[index].size = logical_attrs_[index].size;
  }
  if (!BuildProtoView()) {
    Release();
    return false;
  }
  initialized_ = true;
  BeginFrame();
  return true;
}

bool FiveOutputRuntime::SetInput(const void *data, const std::size_t bytes) {
  if (!initialized_ || data == nullptr || bytes != logical_input_bytes_)
    return false;
  std::memcpy(input_mem_->virt_addr, data, bytes);
  if (bytes < input_capacity_) {
    std::memset(static_cast<std::uint8_t *>(input_mem_->virt_addr) + bytes, 0,
                input_capacity_ - bytes);
  }
  return true;
}

int FiveOutputRuntime::Run() {
  if (!initialized_)
    return -1;
  BeginFrame();
  int result = rknn_mem_sync(context_, input_mem_, RKNN_MEMORY_SYNC_TO_DEVICE);
  if (result != 0)
    return result;
  return rknn_run(context_, nullptr);
}

int FiveOutputRuntime::Prepare(const FiveOutputTensor tensor) {
  if (!initialized_)
    return -1;
  const std::size_t index = static_cast<std::size_t>(tensor);
  if (index >= output_mems_.size())
    return -1;
  const std::uint8_t bit = static_cast<std::uint8_t>(1U << index);
  if ((stats_.ready_mask & bit) != 0U)
    return 0;

  const auto begin = std::chrono::steady_clock::now();
  const int result = rknn_mem_sync(context_, output_mems_[index],
                                   RKNN_MEMORY_SYNC_FROM_DEVICE);
  if (result != 0) {
    stats_.sync_ms += ElapsedMs(begin, std::chrono::steady_clock::now());
    return result;
  }
  if (!PrepareLogicalOutput(index)) {
    stats_.sync_ms += ElapsedMs(begin, std::chrono::steady_clock::now());
    return -2;
  }
  stats_.ready_mask = static_cast<std::uint8_t>(stats_.ready_mask | bit);
  stats_.native_bytes += native_attrs_[index].size_with_stride;
  stats_.sync_ms += ElapsedMs(begin, std::chrono::steady_clock::now());
  return 0;
}

void FiveOutputRuntime::BeginFrame() noexcept { stats_ = {}; }

void FiveOutputRuntime::Release() noexcept {
  for (auto *&memory : output_mems_) {
    if (memory != nullptr && context_ != 0)
      rknn_destroy_mem(context_, memory);
    memory = nullptr;
  }
  if (input_mem_ != nullptr && context_ != 0)
    rknn_destroy_mem(context_, input_mem_);
  input_mem_ = nullptr;
  input_capacity_ = 0;
  logical_input_bytes_ = 0;
  proto_view_ = {};
  stats_ = {};
  initialized_ = false;
  context_ = 0;
  for (auto &buffer : logical_buffers_)
    buffer.clear();
  for (auto &output : outputs_)
    output = {};
}

bool FiveOutputRuntime::PrepareLogicalOutput(const std::size_t index) {
  const auto &native_attr = native_attrs_[index];
  const auto &logical_attr = logical_attrs_[index];
  if (index == static_cast<std::size_t>(FiveOutputTensor::kProto)) {
    outputs_[index].buf = output_mems_[index]->virt_addr;
    return proto_view_.data != nullptr;
  }
  if (native_attr.fmt == RKNN_TENSOR_NC1HWC2) {
    if (!UnpackNc1hwc2(output_mems_[index]->virt_addr, native_attr,
                       logical_attr, &logical_buffers_[index])) {
      return false;
    }
    outputs_[index].buf = logical_buffers_[index].data();
    return true;
  }
  std::uint32_t logical_width = 0U;
  if (logical_attr.n_dims > 0 && logical_attr.fmt == RKNN_TENSOR_NCHW) {
    logical_width = logical_attr.dims[logical_attr.n_dims - 1];
  } else if (logical_attr.n_dims >= 2 && logical_attr.fmt == RKNN_TENSOR_NHWC) {
    logical_width = logical_attr.dims[logical_attr.n_dims - 2];
  } else if (logical_attr.n_dims > 0 &&
             logical_attr.fmt == RKNN_TENSOR_UNDEFINED) {
    logical_width = logical_attr.dims[logical_attr.n_dims - 1];
  }
  if (native_attr.w_stride > logical_width && logical_width > 0U) {
    if (!UnpackStrided(output_mems_[index]->virt_addr, native_attr,
                       logical_attr, &logical_buffers_[index])) {
      return false;
    }
    outputs_[index].buf = logical_buffers_[index].data();
    return true;
  }
  if (logical_attr.size > native_attr.size_with_stride)
    return false;
  outputs_[index].buf = output_mems_[index]->virt_addr;
  return true;
}

bool FiveOutputRuntime::BuildProtoView() {
  constexpr std::size_t kProtoIndex =
      static_cast<std::size_t>(FiveOutputTensor::kProto);
  const auto &native_attr = native_attrs_[kProtoIndex];
  const auto &logical_attr = logical_attrs_[kProtoIndex];
  if (output_mems_[kProtoIndex] == nullptr ||
      native_attr.fmt != RKNN_TENSOR_NC1HWC2 ||
      native_attr.type != RKNN_TENSOR_INT8 ||
      logical_attr.type != RKNN_TENSOR_INT8 || native_attr.n_dims != 5 ||
      logical_attr.n_dims != 4 || native_attr.dims[0] != 1 ||
      logical_attr.dims[0] != 1 ||
      logical_attr.dims[2] != native_attr.dims[2] ||
      logical_attr.dims[3] != native_attr.dims[3]) {
    return false;
  }
  const int channels = static_cast<int>(logical_attr.dims[1]);
  const int channel_blocks = static_cast<int>(native_attr.dims[1]);
  const int height = static_cast<int>(native_attr.dims[2]);
  const int width = static_cast<int>(native_attr.dims[3]);
  const int block_size = static_cast<int>(native_attr.dims[4]);
  const int width_stride =
      native_attr.w_stride > 0 ? static_cast<int>(native_attr.w_stride) : width;
  if (channels <= 0 || channel_blocks <= 0 || height <= 0 || width <= 0 ||
      (block_size != 8 && block_size != 16) || width_stride < width ||
      channel_blocks != (channels + block_size - 1) / block_size) {
    return false;
  }
  const std::size_t required = static_cast<std::size_t>(channel_blocks) *
                               height * width_stride * block_size;
  if (required > native_attr.size_with_stride)
    return false;
  proto_view_ = {
      static_cast<const std::int8_t *>(output_mems_[kProtoIndex]->virt_addr),
      channels,
      channel_blocks,
      height,
      width,
      width_stride,
      block_size};
  return true;
}

} // namespace paddleyolo_rknn::postprocess
