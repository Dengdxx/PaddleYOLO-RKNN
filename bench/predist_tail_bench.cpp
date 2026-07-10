// Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file predist_tail_bench.cpp
 * @brief YOLO26 `predist` 后处理尾部的正确性验证与微基准。
 *
 * @details
 * 本工具读取 Python 测试生成的 fixture（`meta.json`、`expected.json`、
 * `ltrb.bin`、`scores.bin`），复现 YOLO26 `reg_max=1` 检测路线的尾部逻辑：
 *
 * 1. 按 anchor 聚合每个位置的最大类别分数；
 * 2. 取 top-k anchor；
 * 3. 在 top-k anchor 内按类别再次排序；
 * 4. 应用置信度阈值；
 * 5. 使用与导出/评测一致的半精度语义解码 box。
 *
 * 该文件不依赖 RKNN Runtime，用于验证 AArch64 NEON 优化实现
 * 是否与 Python 参考一致，并粗略测量筛选和解码阶段耗时。
 */
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

#if !defined(__aarch64__) || !defined(__ARM_NEON)
#error "predist_tail_bench requires AArch64 NEON"
#endif

#include <arm_neon.h>

namespace {

/**
 * @brief 单个 fixture 的模型和后处理元信息。
 */
struct Meta {
  int imgsz = 0;
  int max_det = 0;
  int nc = 0;
  int n = 0;
  float conf_thresh = 0.0f;
  std::vector<int> strides;
};

/**
 * @brief Python 参考实现写出的期望结果。
 */
struct Expected {
  std::string status;
  std::vector<int> final_idx;
  std::vector<int> class_ids;
  std::vector<float> scores;
  std::vector<float> boxes;  // flattened [k, 4]
};

/**
 * @brief C++ 后处理实现的输出结果。
 */
struct TailResult {
  std::vector<int> final_idx;
  std::vector<int> class_ids;
  std::vector<float> scores;
  std::vector<float> boxes;  // flattened [k, 4]
};

/**
 * @brief YOLO 多尺度 anchor 网格及每个 anchor 对应的 stride。
 */
struct AnchorGrid {
  std::vector<float> anchors;      // flattened [n, 2]
  std::vector<float> stride_vals;  // [n]
};

using half_float_t = __fp16;

/**
 * @brief 读取完整文本文件。
 * @param path 输入文件路径。
 * @return 文件全文。
 * @throw std::runtime_error 文件不存在或无法打开。
 */
std::string read_text(const fs::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open " + path.string());
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

/**
 * @brief 确认 fixture 必需文件存在。
 * @param path 待检查路径。
 * @throw std::runtime_error 路径不是普通文件。
 */
void require_file(const fs::path& path) {
  if (!fs::is_regular_file(path)) {
    throw std::runtime_error("missing required file: " + path.string());
  }
}

/**
 * @brief 从简单 JSON 文本中提取指定 key 的原始 value 字符串。
 *
 * @details
 * 这里有意使用很小的解析器而不是引入 JSON 依赖。fixture 由测试代码生成，
 * 字段结构固定，只需要支持字符串、数字和一维数组。
 *
 * @param json JSON 文本。
 * @param key 字段名。
 * @return 去掉 key 和冒号后的 value 内容；字符串会去掉双引号。
 * @throw std::runtime_error key 缺失或 value 格式不完整。
 */
std::string extract_value(const std::string& json, const std::string& key) {
  const std::string quoted = "\"" + key + "\"";
  const auto key_pos = json.find(quoted);
  if (key_pos == std::string::npos) {
    throw std::runtime_error("missing key: " + key);
  }

  const auto colon_pos = json.find(':', key_pos + quoted.size());
  if (colon_pos == std::string::npos) {
    throw std::runtime_error("missing colon for key: " + key);
  }

  std::size_t pos = colon_pos + 1;
  while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
    ++pos;
  }
  if (pos >= json.size()) {
    throw std::runtime_error("missing value for key: " + key);
  }

  const char c = json[pos];
  if (c == '"') {
    const auto end = json.find('"', pos + 1);
    if (end == std::string::npos) {
      throw std::runtime_error("unterminated string for key: " + key);
    }
    return json.substr(pos + 1, end - pos - 1);
  }
  if (c == '[') {
    int depth = 0;
    for (std::size_t i = pos; i < json.size(); ++i) {
      if (json[i] == '[') {
        ++depth;
      } else if (json[i] == ']') {
        --depth;
        if (depth == 0) {
          return json.substr(pos, i - pos + 1);
        }
      }
    }
    throw std::runtime_error("unterminated array for key: " + key);
  }

  std::size_t end = pos;
  while (end < json.size() && json[end] != ',' && json[end] != '}') {
    ++end;
  }
  return json.substr(pos, end - pos);
}

/**
 * @brief 从文本片段中解析浮点数列表。
 * @param text 形如 `[1.0, 2.0]` 的数组文本，也可包含空白和分隔符。
 * @return 按出现顺序解析出的浮点数。
 */
std::vector<float> parse_number_list(const std::string& text) {
  std::vector<float> values;
  std::size_t pos = 0;
  while (pos < text.size()) {
    while (pos < text.size() &&
           !(std::isdigit(static_cast<unsigned char>(text[pos])) ||
             text[pos] == '-' || text[pos] == '+' || text[pos] == '.')) {
      ++pos;
    }
    if (pos >= text.size()) {
      break;
    }
    std::size_t consumed = 0;
    const float value = std::stof(text.substr(pos), &consumed);
    values.push_back(value);
    pos += consumed;
  }
  return values;
}

/**
 * @brief 从文本片段中解析整数列表。
 * @param text 数组文本。
 * @return 四舍五入后的整数列表。
 */
std::vector<int> parse_int_list(const std::string& text) {
  const auto floats = parse_number_list(text);
  std::vector<int> ints;
  ints.reserve(floats.size());
  for (float value : floats) {
    ints.push_back(static_cast<int>(std::lround(value)));
  }
  return ints;
}

/**
 * @brief 读取 `meta.json`。
 * @param path `meta.json` 路径。
 * @return 模型输入尺寸、类别数、anchor 数、阈值和 strides。
 */
Meta load_meta(const fs::path& path) {
  const std::string text = read_text(path);
  Meta meta;
  meta.imgsz = std::stoi(extract_value(text, "imgsz"));
  meta.max_det = std::stoi(extract_value(text, "max_det"));
  meta.nc = std::stoi(extract_value(text, "nc"));
  meta.n = std::stoi(extract_value(text, "n"));
  meta.conf_thresh = std::stof(extract_value(text, "conf_thresh"));
  meta.strides = parse_int_list(extract_value(text, "strides"));
  return meta;
}

/**
 * @brief 读取 `expected.json`。
 * @param path `expected.json` 路径。
 * @return Python 参考实现输出。
 */
Expected load_expected(const fs::path& path) {
  const std::string text = read_text(path);
  Expected expected;
  expected.status = extract_value(text, "status");
  if (text.find("\"final_idx\"") != std::string::npos) {
    expected.final_idx = parse_int_list(extract_value(text, "final_idx"));
  }
  if (text.find("\"class_ids\"") != std::string::npos) {
    expected.class_ids = parse_int_list(extract_value(text, "class_ids"));
  }
  if (text.find("\"scores\"") != std::string::npos) {
    expected.scores = parse_number_list(extract_value(text, "scores"));
  }
  if (text.find("\"boxes\"") != std::string::npos) {
    expected.boxes = parse_number_list(extract_value(text, "boxes"));
  }
  return expected;
}

/**
 * @brief 读取固定数量的 FP32 二进制数据。
 * @param path 二进制文件路径。
 * @param expected_count 期望 float 元素数量。
 * @return 长度为 `expected_count` 的 float 数组。
 * @throw std::runtime_error 文件无法打开或字节数不匹配。
 */
std::vector<float> read_binary_f32(const fs::path& path, std::size_t expected_count) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open " + path.string());
  }
  std::vector<float> data(expected_count);
  in.read(reinterpret_cast<char*>(data.data()),
          static_cast<std::streamsize>(expected_count * sizeof(float)));
  if (in.gcount() != static_cast<std::streamsize>(expected_count * sizeof(float))) {
    throw std::runtime_error("unexpected size for " + path.string());
  }
  return data;
}

/**
 * @brief 按 YOLO stride 生成 anchor 网格。
 *
 * @details
 * anchor 坐标采用 `(x + 0.5, y + 0.5)`，顺序与 Paddle/ONNX 导出后
 * `predist` 输出中的 anchor 维度一致。
 *
 * @param meta fixture 元信息。
 * @return 展平后的 anchor 坐标和每个 anchor 的 stride。
 * @throw std::runtime_error 生成出的 anchor 数与 `meta.n` 不一致。
 */
AnchorGrid make_anchor_grid(const Meta& meta) {
  AnchorGrid grid;
  grid.anchors.reserve(static_cast<std::size_t>(meta.n) * 2);
  grid.stride_vals.reserve(static_cast<std::size_t>(meta.n));

  for (int stride : meta.strides) {
    const int h = meta.imgsz / stride;
    const int w = meta.imgsz / stride;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        grid.anchors.push_back(static_cast<float>(x) + 0.5f);
        grid.anchors.push_back(static_cast<float>(y) + 0.5f);
        grid.stride_vals.push_back(static_cast<float>(stride));
      }
    }
  }

  if (static_cast<int>(grid.stride_vals.size()) != meta.n) {
    throw std::runtime_error("anchor grid size mismatch with meta.n");
  }
  return grid;
}

/**
 * @brief 数值稳定的 sigmoid。
 * @param x logit。
 * @return sigmoid(x)。
 */
float sigmoid(float x) {
  const float clipped = std::clamp(x, -88.0f, 88.0f);
  return 1.0f / (1.0f + std::exp(-clipped));
}

/**
 * @brief 使用半精度中间值复现导出路径中的 box 坐标解码。
 *
 * @details
 * Python/ONNX/RKNN 链路中部分算子会经历 FP16 语义。这里显式把
 * anchor、delta、stride 和中间结果转成 half，避免 C++ 校验因 FP32
 * 中间精度更高而与参考输出出现无意义差异。
 *
 * @param anchor anchor 中心坐标（网格单位）。
 * @param delta l/t/r/b 距离预测。
 * @param stride 当前 anchor 对应 stride。
 * @param add true 表示右/下边界，false 表示左/上边界。
 * @return 输入图像尺度下的坐标。
 */
float decode_coord_f16(float anchor, float delta, float stride, bool add) {
  const half_float_t h_anchor = static_cast<half_float_t>(anchor);
  const half_float_t h_delta = static_cast<half_float_t>(delta);
  const half_float_t h_stride = static_cast<half_float_t>(stride);
  const half_float_t h_pos = add
      ? static_cast<half_float_t>(h_anchor + h_delta)
      : static_cast<half_float_t>(h_anchor - h_delta);
  const half_float_t h_out = static_cast<half_float_t>(h_pos * h_stride);
  return static_cast<float>(h_out);
}

/**
 * @brief 解码筛选后的 l/t/r/b box。
 * @param meta fixture 元信息。
 * @param grid anchor 网格。
 * @param raw_ltrb 距离预测，布局为 `[4, n]`。
 * @param final_idx 需要解码的 anchor 原始下标。
 * @return 展平 `[k, 4]` 的 xyxy 坐标。
 */
std::vector<float> decode_boxes(const Meta& meta,
                                const AnchorGrid& grid,
                                const std::vector<float>& raw_ltrb,
                                const std::vector<int>& final_idx) {
  std::vector<float> boxes;
  boxes.reserve(final_idx.size() * 4);

  for (int idx : final_idx) {
    const float ax = grid.anchors[static_cast<std::size_t>(idx) * 2];
    const float ay = grid.anchors[static_cast<std::size_t>(idx) * 2 + 1];
    const float stride = grid.stride_vals[static_cast<std::size_t>(idx)];
    const float l = raw_ltrb[static_cast<std::size_t>(idx)];
    const float t = raw_ltrb[static_cast<std::size_t>(meta.n) + idx];
    const float r = raw_ltrb[static_cast<std::size_t>(meta.n) * 2 + idx];
    const float b = raw_ltrb[static_cast<std::size_t>(meta.n) * 3 + idx];
    boxes.push_back(decode_coord_f16(ax, l, stride, false));
    boxes.push_back(decode_coord_f16(ay, t, stride, false));
    boxes.push_back(decode_coord_f16(ax, r, stride, true));
    boxes.push_back(decode_coord_f16(ay, b, stride, true));
  }

  return boxes;
}

/**
 * @brief NEON 加速的 logit 阈值选择逻辑。
 *
 * @details
 * 该实现用 `vmaxq_f32` 加速“每个 anchor 跨类别取最大 logit”的阶段。
 * 后续 top-k 排序、类别展开排序和阈值过滤保持导出语义。
 *
 * @param meta fixture 元信息。
 * @param logits 类别 logits，布局为 `[nc, n]`。
 * @return 通过阈值后的 anchor 下标、类别和概率；不含 box。
 */
TailResult select_exact_optimized(const Meta& meta,
                                  const std::vector<float>& logits) {
  TailResult result;
  std::vector<float> max_logits(static_cast<std::size_t>(meta.n),
                                -std::numeric_limits<float>::infinity());

  for (int cls = 0; cls < meta.nc; ++cls) {
    const float* cls_ptr = logits.data() + static_cast<std::size_t>(cls) * meta.n;
    int anchor = 0;
    for (; anchor + 4 <= meta.n; anchor += 4) {
      const float32x4_t prev = vld1q_f32(max_logits.data() + anchor);
      const float32x4_t cur = vld1q_f32(cls_ptr + anchor);
      vst1q_f32(max_logits.data() + anchor, vmaxq_f32(prev, cur));
    }
    for (; anchor < meta.n; ++anchor) {
      if (cls_ptr[anchor] > max_logits[static_cast<std::size_t>(anchor)]) {
        max_logits[static_cast<std::size_t>(anchor)] = cls_ptr[anchor];
      }
    }
  }

  std::vector<std::pair<float, int>> anchor_max;
  anchor_max.reserve(meta.n);
  for (int anchor = 0; anchor < meta.n; ++anchor) {
    anchor_max.emplace_back(max_logits[static_cast<std::size_t>(anchor)], anchor);
  }
  std::stable_sort(anchor_max.begin(), anchor_max.end(),
                   [](const auto& a, const auto& b) { return a.first > b.first; });

  const int k = std::min(meta.max_det, meta.n);
  std::vector<int> ori_index;
  ori_index.reserve(k);
  for (int i = 0; i < k; ++i) {
    ori_index.push_back(anchor_max[static_cast<std::size_t>(i)].second);
  }

  std::vector<std::pair<float, int>> flat_logits;
  flat_logits.reserve(static_cast<std::size_t>(k) * meta.nc);
  for (int gather_pos = 0; gather_pos < k; ++gather_pos) {
    const int anchor = ori_index[static_cast<std::size_t>(gather_pos)];
    for (int cls = 0; cls < meta.nc; ++cls) {
      const float logit = logits[static_cast<std::size_t>(cls) * meta.n + anchor];
      flat_logits.emplace_back(logit, gather_pos * meta.nc + cls);
    }
  }
  std::stable_sort(flat_logits.begin(), flat_logits.end(),
                   [](const auto& a, const auto& b) { return a.first > b.first; });

  const float conf = std::clamp(meta.conf_thresh, 1e-9f, 1.0f - 1e-9f);
  const float logit_thresh = std::log(conf / (1.0f - conf));

  for (int i = 0; i < k; ++i) {
    const float logit = flat_logits[static_cast<std::size_t>(i)].first;
    if (!(logit > logit_thresh)) {
      continue;
    }
    const int flat_index = flat_logits[static_cast<std::size_t>(i)].second;
    const int anchor_idx = flat_index / meta.nc;
    const int class_idx = flat_index % meta.nc;
    const int final_idx = ori_index[static_cast<std::size_t>(anchor_idx)];

    result.final_idx.push_back(final_idx);
    result.class_ids.push_back(class_idx);
    result.scores.push_back(sigmoid(logit));
  }

  return result;
}

/**
 * @brief NEON logit 阈值版完整路径：筛选 + box 解码。
 */
TailResult run_exact_optimized(const Meta& meta,
                               const AnchorGrid& grid,
                               const std::vector<float>& raw_ltrb,
                               const std::vector<float>& logits) {
  TailResult result = select_exact_optimized(meta, logits);
  result.boxes = decode_boxes(meta, grid, raw_ltrb, result.final_idx);
  return result;
}

/**
 * @brief 带容差的浮点比较。
 * @param a 左值。
 * @param b 右值。
 * @param tol 绝对误差容差。
 * @return `|a-b| <= tol` 时返回 true。
 */
bool almost_equal(float a, float b, float tol = 1e-3f) {
  return std::fabs(a - b) <= tol;
}

/**
 * @brief 将 C++ 结果与 Python 参考结果逐项比较。
 *
 * @details
 * `final_idx`、`class_ids` 必须完全一致；score 使用较小 FP32 容差；
 * box 解码经历 half 语义，使用稍宽的坐标容差。
 *
 * @param result C++ 实现输出。
 * @param expected Python fixture 中的参考输出。
 * @throw std::runtime_error 任一字段不一致。
 */
void verify_exact_match(const TailResult& result, const Expected& expected) {
  if (expected.final_idx.empty()) {
    return;
  }
  if (result.final_idx != expected.final_idx) {
    throw std::runtime_error("final_idx mismatch");
  }
  if (result.class_ids != expected.class_ids) {
    throw std::runtime_error("class_ids mismatch");
  }
  if (result.scores.size() != expected.scores.size()) {
    throw std::runtime_error("score size mismatch");
  }
  if (result.boxes.size() != expected.boxes.size()) {
    throw std::runtime_error("box size mismatch");
  }

  for (std::size_t i = 0; i < result.scores.size(); ++i) {
    if (!almost_equal(result.scores[i], expected.scores[i])) {
      throw std::runtime_error("score mismatch");
    }
  }
  for (std::size_t i = 0; i < result.boxes.size(); ++i) {
    if (!almost_equal(result.boxes[i], expected.boxes[i], 1e-2f)) {
      throw std::runtime_error("box mismatch");
    }
  }
}

/**
 * @brief 校验模式入口。
 *
 * @details
 * 读取 fixture 后运行 NEON 优化实现，并与 `expected.json` 中的 Python 参考结果比较。
 * 成功时输出 `fixture_loaded`；若 fixture 含完整期望结果，还会输出
 * `exact_match`。
 *
 * @param fixture_dir fixture 目录。
 * @return 进程退出码，成功为 0。
 */
int run_verify(const fs::path& fixture_dir) {
  require_file(fixture_dir / "meta.json");
  require_file(fixture_dir / "expected.json");
  require_file(fixture_dir / "ltrb.bin");
  require_file(fixture_dir / "scores.bin");

  const Meta meta = load_meta(fixture_dir / "meta.json");
  const Expected expected = load_expected(fixture_dir / "expected.json");
  if (expected.status != "fixture_loaded") {
    throw std::runtime_error("unexpected fixture status");
  }

  const auto raw_ltrb = read_binary_f32(fixture_dir / "ltrb.bin", static_cast<std::size_t>(4) * meta.n);
  const auto logits = read_binary_f32(fixture_dir / "scores.bin", static_cast<std::size_t>(meta.nc) * meta.n);
  const AnchorGrid grid = make_anchor_grid(meta);
  const TailResult result = run_exact_optimized(meta, grid, raw_ltrb, logits);

  std::cout << "fixture_loaded" << std::endl;
  verify_exact_match(result, expected);
  if (!expected.final_idx.empty()) {
    std::cout << "exact_match" << std::endl;
  }
  return 0;
}

/**
 * @brief 微基准模式入口。
 *
 * @details
 * 重复运行 NEON 优化实现，并分别统计筛选阶段和 box 解码阶段的平均耗时。
 * 该模式使用固定 fixture，不包含 RKNN 推理、输入预处理或 NMS。
 *
 * @param fixture_dir fixture 目录。
 * @param iters 迭代次数。
 * @return 进程退出码，成功为 0。
 */
int run_bench(const fs::path& fixture_dir, int iters) {
  if (iters <= 0) {
    throw std::runtime_error("iters must be > 0");
  }

  const Meta meta = load_meta(fixture_dir / "meta.json");
  const auto raw_ltrb = read_binary_f32(fixture_dir / "ltrb.bin", static_cast<std::size_t>(4) * meta.n);
  const auto logits = read_binary_f32(fixture_dir / "scores.bin", static_cast<std::size_t>(meta.nc) * meta.n);
  const AnchorGrid grid = make_anchor_grid(meta);

  TailResult last_result;
  double select_ms = 0.0;
  double decode_ms = 0.0;

  for (int i = 0; i < iters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    last_result = select_exact_optimized(meta, logits);
    const auto t1 = std::chrono::steady_clock::now();
    last_result.boxes = decode_boxes(meta, grid, raw_ltrb, last_result.final_idx);
    const auto t2 = std::chrono::steady_clock::now();

    select_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    decode_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
  }

  const double total_ms = select_ms + decode_ms;
  const double avg_total_ms = total_ms / static_cast<double>(iters);
  const double fps = avg_total_ms > 0.0 ? 1000.0 / avg_total_ms : 0.0;

  std::cout << std::fixed << std::setprecision(4)
            << "stage_select_ms=" << (select_ms / static_cast<double>(iters))
            << " stage_decode_ms=" << (decode_ms / static_cast<double>(iters))
            << " total_ms=" << avg_total_ms
            << " fps=" << fps
            << " dets=" << last_result.final_idx.size()
            << std::endl;
  return 0;
}

}  // namespace

/**
 * @brief 命令行入口。
 *
 * @details
 * 支持两种模式：
 *
 * - `--verify <fixture_dir>`：与 Python 参考结果比对；
 * - `--bench <fixture_dir> --iters <n>`：输出平均阶段耗时。
 *
 * @param argc 参数数量。
 * @param argv 参数数组。
 * @return 0 表示成功，1 表示运行错误，2 表示参数错误。
 */
int main(int argc, char** argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "--verify") {
      return run_verify(argv[2]);
    }
    if (argc == 5 && std::string(argv[1]) == "--bench" &&
        std::string(argv[3]) == "--iters") {
      return run_bench(argv[2], std::stoi(argv[4]));
    }
    std::cerr << "usage: " << argv[0]
              << " --verify <fixture_dir> | --bench <fixture_dir> --iters <n>"
              << std::endl;
      return 2;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
}
