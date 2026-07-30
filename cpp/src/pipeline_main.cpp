#include "solidattention/attention_reference.hpp"
#include "solidattention/selection.hpp"
#include "solidattention/trace.hpp"
#include "solidattention/uring_reader.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

using Clock = std::chrono::steady_clock;

void cuda_check(cudaError_t value, const char* operation) {
  if (value != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(value));
  }
}

void driver_check(CUresult value, const char* operation) {
  if (value != CUDA_SUCCESS) {
    const char* message = nullptr;
    cuGetErrorString(value, &message);
    throw std::runtime_error(std::string(operation) + ": " +
                             (message ? message : "CUDA driver error"));
  }
}

void nvrtc_check(nvrtcResult value, const char* operation) {
  if (value != NVRTC_SUCCESS) {
    throw std::runtime_error(std::string(operation) + ": " +
                             nvrtcGetErrorString(value));
  }
}

double milliseconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

std::uint64_t trace_us(const solidattention::Trace& trace) {
  return trace.now_us();
}

constexpr const char* kKernels = R"(
__device__ float half_to_float(unsigned short value) {
  unsigned int sign = ((unsigned int)value & 0x8000U) << 16;
  unsigned int exponent = (value >> 10) & 0x1fU;
  unsigned int mantissa = value & 0x3ffU;
  unsigned int bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      int shift = 0;
      while ((mantissa & 0x400U) == 0) { mantissa <<= 1; ++shift; }
      mantissa &= 0x3ffU;
      bits = sign | ((127 - 14 - shift) << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000U | (mantissa << 13);
  } else {
    bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
  }
  return __uint_as_float(bits);
}

extern "C" __global__ void sparse_attention_parallel(
    const unsigned short* kv, const float* query, float* output,
    int tokens, int query_heads, int kv_heads, int head_dim, float scale) {
  int query_head = blockIdx.x;
  int lane = threadIdx.x;
  if (query_head >= query_heads || tokens > 128 || head_dim > 128) return;
  int kv_head = query_head / (query_heads / kv_heads);
  __shared__ float logits[128];
  __shared__ float reduction[128];
  float dot = -3.402823466e+38F;
  if (lane < tokens) {
    unsigned long long base =
        (unsigned long long)lane * 2 * kv_heads * head_dim +
        kv_head * head_dim;
    dot = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
      dot += query[query_head * head_dim + dim] *
             half_to_float(kv[base + dim]);
    }
    dot *= scale;
  }
  reduction[lane] = dot;
  __syncthreads();
  for (int stride = 64; stride > 0; stride >>= 1) {
    if (lane < stride) reduction[lane] =
        fmaxf(reduction[lane], reduction[lane + stride]);
    __syncthreads();
  }
  float probability = lane < tokens ? expf(dot - reduction[0]) : 0.0f;
  logits[lane] = probability;
  reduction[lane] = probability;
  __syncthreads();
  for (int stride = 64; stride > 0; stride >>= 1) {
    if (lane < stride) reduction[lane] += reduction[lane + stride];
    __syncthreads();
  }
  float denominator = reduction[0];
  if (lane < head_dim) {
    float value = 0.0f;
    for (int token = 0; token < tokens; ++token) {
      unsigned long long base =
          (unsigned long long)token * 2 * kv_heads * head_dim +
          kv_heads * head_dim + kv_head * head_dim;
      value += logits[token] / denominator * half_to_float(kv[base + lane]);
    }
    output[query_head * head_dim + lane] = value;
  }
}

// P0 uses this arithmetic kernel only as the scheduling window occupied by the
// current layer FFN. P1 replaces it with the actual quantized model FFN.
extern "C" __global__ void synthetic_ffn_window(
    float* values, int elements, int iterations) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  float value = values[index];
  for (int iteration = 0; iteration < iterations; ++iteration) {
    value = fmaf(value, 1.0000001192f, 0.0000000596f);
    value = value / (1.0f + fabsf(value) * 0.0000001f);
  }
  values[index] = value;
}
)";

class CudaPipeline {
 public:
  CudaPipeline(std::size_t kv_bytes, int ffn_iterations)
      : kv_bytes_(kv_bytes), ffn_iterations_(ffn_iterations) {
    driver_check(cuInit(0), "cuInit");
    cuda_check(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking),
               "create copy stream");
    cuda_check(cudaStreamCreateWithFlags(&compute_stream_, cudaStreamNonBlocking),
               "create compute stream");
    for (auto& pointer : host_) {
      cuda_check(cudaHostAlloc(&pointer, kv_bytes_, cudaHostAllocDefault),
                 "allocate pinned KV buffer");
    }
    for (auto& pointer : device_kv_) {
      cuda_check(cudaMalloc(&pointer, kv_bytes_), "allocate VRAM KV slot");
    }
    cuda_check(cudaEventCreate(&qkv_begin_), "create QKV begin event");
    cuda_check(cudaEventCreate(&qkv_end_), "create QKV end event");
    constexpr int kQueryElements = 16 * 128;
    std::vector<float> query(kQueryElements);
    for (int index = 0; index < kQueryElements; ++index) {
      query[index] = std::sin(static_cast<float>(index) * 0.013f) * 0.125f;
    }
    cuda_check(cudaMalloc(&device_query_, query.size() * sizeof(float)),
               "allocate query");
    cuda_check(cudaMemcpy(device_query_, query.data(),
                          query.size() * sizeof(float), cudaMemcpyHostToDevice),
               "copy query");
    cuda_check(cudaMalloc(&device_output_, query.size() * sizeof(float)),
               "allocate output");
    compile();
  }

  ~CudaPipeline() {
    if (module_) cuModuleUnload(module_);
    if (qkv_begin_) cudaEventDestroy(qkv_begin_);
    if (qkv_end_) cudaEventDestroy(qkv_end_);
    if (device_output_) cudaFree(device_output_);
    if (device_query_) cudaFree(device_query_);
    for (void* pointer : device_kv_) if (pointer) cudaFree(pointer);
    for (void* pointer : host_) if (pointer) cudaFreeHost(pointer);
    if (copy_stream_) cudaStreamDestroy(copy_stream_);
    if (compute_stream_) cudaStreamDestroy(compute_stream_);
  }

  std::vector<void*> host_buffers() const {
    return {host_[0], host_[1], host_[2]};
  }

  void update_query(const std::vector<float>& query) {
    if (query.size() != static_cast<std::size_t>(query_heads_ * head_dim_)) {
      throw std::runtime_error("query update has wrong shape");
    }
    cuda_check(cudaMemcpyAsync(device_query_, query.data(),
                               query.size() * sizeof(float),
                               cudaMemcpyHostToDevice, compute_stream_),
               "update attention query");
  }

  double stage_sync(std::size_t slot) {
    cudaEvent_t begin{}, end{};
    cuda_check(cudaEventCreate(&begin), "stage begin event");
    cuda_check(cudaEventCreate(&end), "stage end event");
    cuda_check(cudaEventRecord(begin, copy_stream_), "record stage begin");
    cuda_check(cudaMemcpyAsync(device_kv_[slot], host_[slot], kv_bytes_,
                               cudaMemcpyHostToDevice, copy_stream_),
               "stage KV");
    cuda_check(cudaEventRecord(end, copy_stream_), "record stage end");
    cuda_check(cudaEventSynchronize(end), "wait stage");
    float elapsed = 0;
    cuda_check(cudaEventElapsedTime(&elapsed, begin, end), "time stage");
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return elapsed;
  }

  double attention_sync(std::size_t slot) {
    cudaEvent_t begin{}, end{};
    cuda_check(cudaEventCreate(&begin), "attention begin event");
    cuda_check(cudaEventCreate(&end), "attention end event");
    cuda_check(cudaEventRecord(begin, compute_stream_), "record attention begin");
    void* arguments[] = {&device_kv_[slot], &device_query_, &device_output_,
                         &tokens_, &query_heads_, &kv_heads_, &head_dim_,
                         &scale_};
    driver_check(cuLaunchKernel(attention_, query_heads_, 1, 1, 128, 1, 1, 0,
                                reinterpret_cast<CUstream>(compute_stream_),
                                arguments, nullptr),
                 "launch attention");
    cuda_check(cudaEventRecord(end, compute_stream_), "record attention end");
    cuda_check(cudaEventSynchronize(end), "wait attention");
    float elapsed = 0;
    cuda_check(cudaEventElapsedTime(&elapsed, begin, end), "time attention");
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return elapsed;
  }

  double ffn_sync() {
    cudaEvent_t begin{}, end{};
    cuda_check(cudaEventCreate(&begin), "FFN begin event");
    cuda_check(cudaEventCreate(&end), "FFN end event");
    cuda_check(cudaEventRecord(begin, compute_stream_), "record FFN begin");
    launch_ffn();
    cuda_check(cudaEventRecord(end, compute_stream_), "record FFN end");
    cuda_check(cudaEventSynchronize(end), "wait FFN");
    float elapsed = 0;
    cuda_check(cudaEventElapsedTime(&elapsed, begin, end), "time FFN");
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return elapsed;
  }

  std::pair<double, double> overlap_ffn_and_stage(std::size_t next_slot) {
    cudaEvent_t ffn_begin{}, ffn_end{}, copy_begin{}, copy_end{};
    cudaEventCreate(&ffn_begin);
    cudaEventCreate(&ffn_end);
    cudaEventCreate(&copy_begin);
    cudaEventCreate(&copy_end);
    cudaEventRecord(ffn_begin, compute_stream_);
    launch_ffn();
    cudaEventRecord(ffn_end, compute_stream_);
    cudaEventRecord(copy_begin, copy_stream_);
    cuda_check(cudaMemcpyAsync(device_kv_[next_slot], host_[next_slot],
                               kv_bytes_, cudaMemcpyHostToDevice, copy_stream_),
               "overlapped stage KV");
    cudaEventRecord(copy_end, copy_stream_);
    cudaEventSynchronize(ffn_end);
    cudaEventSynchronize(copy_end);
    float ffn_ms = 0;
    float copy_ms = 0;
    cudaEventElapsedTime(&ffn_ms, ffn_begin, ffn_end);
    cudaEventElapsedTime(&copy_ms, copy_begin, copy_end);
    cudaEventDestroy(ffn_begin);
    cudaEventDestroy(ffn_end);
    cudaEventDestroy(copy_begin);
    cudaEventDestroy(copy_end);
    return {ffn_ms, copy_ms};
  }

  void launch_qkv_window_async() {
    cudaEventRecord(qkv_begin_, compute_stream_);
    launch_ffn();
    cudaEventRecord(qkv_end_, compute_stream_);
  }

  struct CorrectionTiming {
    double qkv_ms{};
    double h2d_ms{};
  };

  CorrectionTiming correct_and_wait(
      std::size_t slot, const std::vector<std::size_t>& replacement_slots,
      std::size_t block_bytes) {
    cudaEvent_t copy_begin{}, copy_end{};
    cudaEventCreate(&copy_begin);
    cudaEventCreate(&copy_end);
    cudaEventRecord(copy_begin, copy_stream_);
    for (std::size_t miss = 0; miss < replacement_slots.size(); ++miss) {
      auto* destination = static_cast<std::uint8_t*>(device_kv_[slot]) +
                          replacement_slots[miss] * block_bytes;
      const auto* source = static_cast<const std::uint8_t*>(host_[2]) +
                           miss * block_bytes;
      cuda_check(cudaMemcpyAsync(destination, source, block_bytes,
                                 cudaMemcpyHostToDevice, copy_stream_),
                 "copy correction block");
    }
    cudaEventRecord(copy_end, copy_stream_);
    cudaEventSynchronize(qkv_end_);
    cudaEventSynchronize(copy_end);
    float qkv_ms = 0.0f;
    float h2d_ms = 0.0f;
    cudaEventElapsedTime(&qkv_ms, qkv_begin_, qkv_end_);
    cudaEventElapsedTime(&h2d_ms, copy_begin, copy_end);
    cudaEventDestroy(copy_begin);
    cudaEventDestroy(copy_end);
    return {qkv_ms, h2d_ms};
  }

  std::vector<float> output() const {
    std::vector<float> values(query_heads_ * head_dim_);
    cuda_check(cudaMemcpy(values.data(), device_output_,
                          values.size() * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "copy pipeline output");
    return values;
  }

 private:
  void compile() {
    nvrtcProgram program{};
    nvrtc_check(nvrtcCreateProgram(&program, kKernels, "pipeline.cu", 0,
                                   nullptr, nullptr),
                "create NVRTC program");
    const char* options[] = {"--std=c++14", "--gpu-architecture=compute_89"};
    const auto result = nvrtcCompileProgram(program, 2, options);
    if (result != NVRTC_SUCCESS) {
      std::size_t size = 0;
      nvrtcGetProgramLogSize(program, &size);
      std::string log(size, '\0');
      nvrtcGetProgramLog(program, log.data());
      nvrtcDestroyProgram(&program);
      throw std::runtime_error("NVRTC compile failed: " + log);
    }
    std::size_t ptx_size = 0;
    nvrtc_check(nvrtcGetPTXSize(program, &ptx_size), "get PTX size");
    std::vector<char> ptx(ptx_size);
    nvrtc_check(nvrtcGetPTX(program, ptx.data()), "get PTX");
    nvrtcDestroyProgram(&program);
    driver_check(cuModuleLoadData(&module_, ptx.data()), "load pipeline module");
    driver_check(cuModuleGetFunction(&attention_, module_,
                                     "sparse_attention_parallel"),
                 "get attention kernel");
    driver_check(cuModuleGetFunction(&ffn_, module_, "synthetic_ffn_window"),
                 "get FFN window kernel");
  }

  void launch_ffn() {
    const int elements = query_heads_ * head_dim_;
    void* arguments[] = {&device_output_, const_cast<int*>(&elements),
                         &ffn_iterations_};
    driver_check(cuLaunchKernel(ffn_, (elements + 255) / 256, 1, 1,
                                256, 1, 1, 0,
                                reinterpret_cast<CUstream>(compute_stream_),
                                arguments, nullptr),
                 "launch FFN window");
  }

  std::size_t kv_bytes_{};
  int ffn_iterations_{};
  std::array<void*, 3> host_{};
  std::array<void*, 2> device_kv_{};
  void* device_query_{};
  void* device_output_{};
  cudaStream_t copy_stream_{};
  cudaStream_t compute_stream_{};
  CUmodule module_{};
  CUfunction attention_{};
  CUfunction ffn_{};
  cudaEvent_t qkv_begin_{};
  cudaEvent_t qkv_end_{};
  int tokens_{128};
  int query_heads_{16};
  int kv_heads_{8};
  int head_dim_{128};
  float scale_{1.0f / std::sqrt(128.0f)};
};

struct Totals {
  double wall_ms{};
  double read_ms{};
  double exposed_read_wait_ms{};
  double h2d_ms{};
  double attention_ms{};
  double ffn_ms{};
  double estimated_ssd_attention_overlap_ms{};
  double estimated_h2d_ffn_overlap_ms{};
  double correction_read_ms{};
  double correction_h2d_ms{};
  double qkv_window_ms{};
  double selection_ms{};
  std::size_t predicted_blocks{};
  std::size_t hit_blocks{};
  std::size_t miss_blocks{};
  std::vector<float> output;
};

using BlockSet = std::array<std::size_t, 4>;

BlockSet oracle_blocks(std::size_t step, std::size_t layer) {
  const std::size_t phase = step / 2;
  std::size_t first = 1 + (layer * 3 + phase) % 13;
  std::size_t second = 1 + (layer * 7 + phase * 2 + 5) % 13;
  if (second == first) second = 1 + second % 13;
  BlockSet blocks{0, first, second, 15};
  std::sort(blocks.begin(), blocks.end());
  return blocks;
}

struct SelectionContext {
  bool infllm{};
  std::vector<float> prompt_queries;
  std::vector<float> representatives;
  double representative_build_ms{};
  double dense_selected_attention_mass{};
  double dense_oracle_block_recall{};

  std::vector<float> query(std::size_t step, std::size_t layer) const {
    constexpr std::size_t tokens = 512;
    constexpr std::size_t query_heads = 16;
    constexpr std::size_t head_dim = 128;
    if (!infllm) {
      std::vector<float> result(query_heads * head_dim);
      for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] =
            std::sin(static_cast<float>(index) * 0.013f) * 0.125f;
      }
      return result;
    }
    const std::size_t position =
        tokens - 1 - ((step * 17 + layer * 11) % 256);
    std::vector<float> result(query_heads * head_dim);
    for (std::size_t head = 0; head < query_heads; ++head) {
      const auto* source =
          prompt_queries.data() + (head * tokens + position) * head_dim;
      std::copy_n(source, head_dim, result.data() + head * head_dim);
    }
    return result;
  }

  BlockSet blocks(std::size_t step, std::size_t layer,
                  double* selection_ms) const {
    if (!infllm) return oracle_blocks(step, layer);
    constexpr std::size_t query_heads = 16;
    constexpr std::size_t head_dim = 128;
    constexpr std::size_t blocks = 16;
    const auto current_query = query(step, layer);
    const auto begin = Clock::now();
    const auto selected = solidattention::select_shared_blocks(
        current_query.data(), representatives.data(), query_heads, blocks, head_dim,
        4, 1, 1);
    if (selection_ms) *selection_ms += milliseconds(begin, Clock::now());
    if (selected.block_ids.size() != 4) {
      throw std::runtime_error("InfLLM selector returned wrong block count");
    }
    BlockSet result{};
    std::copy(selected.block_ids.begin(), selected.block_ids.end(),
              result.begin());
    return result;
  }
};

void audit_dense_selection(SelectionContext& selection,
                           const std::vector<std::uint16_t>& kv,
                           std::size_t steps, std::size_t layers) {
  constexpr std::size_t tokens = 512;
  constexpr std::size_t blocks = 16;
  constexpr std::size_t block_tokens = 32;
  constexpr std::size_t query_heads = 16;
  constexpr std::size_t kv_heads = 8;
  constexpr std::size_t head_dim = 128;
  constexpr std::size_t groups = query_heads / kv_heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  double total_mass = 0.0;
  double total_recall = 0.0;
  std::size_t audits = 0;
  std::vector<float> logits(tokens);
  for (std::size_t step = 0; step < steps; ++step) {
    for (std::size_t layer = 0; layer < layers; ++layer) {
      const auto query = selection.query(step, layer);
      const auto selected = selection.blocks(step, layer, nullptr);
      std::array<double, blocks> block_mass{};
      for (std::size_t head = 0; head < query_heads; ++head) {
        const std::size_t kv_head = head / groups;
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t token = 0; token < tokens; ++token) {
          const std::size_t base =
              token * 2 * kv_heads * head_dim + kv_head * head_dim;
          float dot = 0.0f;
          for (std::size_t dim = 0; dim < head_dim; ++dim) {
            dot += query[head * head_dim + dim] *
                   solidattention::half_to_float(kv[base + dim]);
          }
          logits[token] = dot * scale;
          maximum = std::max(maximum, logits[token]);
        }
        double denominator = 0.0;
        for (float& logit : logits) {
          logit = std::exp(logit - maximum);
          denominator += logit;
        }
        for (std::size_t token = 0; token < tokens; ++token) {
          block_mass[token / block_tokens] +=
              logits[token] / denominator / query_heads;
        }
      }
      double selected_mass = 0.0;
      for (const auto block : selected) selected_mass += block_mass[block];
      std::array<bool, blocks> oracle_chosen{};
      oracle_chosen[0] = true;
      oracle_chosen[15] = true;
      std::array<std::size_t, blocks> ranked{};
      std::iota(ranked.begin(), ranked.end(), 0);
      std::stable_sort(ranked.begin(), ranked.end(),
                       [&](std::size_t left, std::size_t right) {
                         return block_mass[left] > block_mass[right];
                       });
      std::size_t count = 2;
      for (const auto block : ranked) {
        if (count == 4) break;
        if (!oracle_chosen[block]) {
          oracle_chosen[block] = true;
          ++count;
        }
      }
      std::size_t hits = 0;
      for (const auto block : selected) hits += oracle_chosen[block] ? 1 : 0;
      total_mass += selected_mass;
      total_recall += static_cast<double>(hits) / 4.0;
      ++audits;
    }
  }
  selection.dense_selected_attention_mass = total_mass / audits;
  selection.dense_oracle_block_recall = total_recall / audits;
}

struct Fixture {
  std::vector<std::uint16_t> kv;
  SelectionContext selection;
};

Fixture build_infllm_fixture() {
  constexpr std::size_t tokens = 512;
  constexpr std::size_t query_heads = 16;
  constexpr std::size_t kv_heads = 8;
  constexpr std::size_t head_dim = 128;
  constexpr std::size_t groups = query_heads / kv_heads;
  Fixture fixture;
  fixture.kv.resize(tokens * 2 * kv_heads * head_dim);
  for (std::size_t token = 0; token < tokens; ++token) {
    for (std::size_t half = 0; half < 2; ++half) {
      for (std::size_t head = 0; head < kv_heads; ++head) {
        for (std::size_t dim = 0; dim < head_dim; ++dim) {
          const auto index =
              token * 2 * kv_heads * head_dim +
              half * kv_heads * head_dim + head * head_dim + dim;
          const float value =
              std::sin(static_cast<float>(token * 11 + head * 17 +
                                          dim * 5 + half * 23) *
                       0.007f) *
              0.2f;
          fixture.kv[index] = solidattention::float_to_half(value);
        }
      }
    }
  }
  fixture.selection.infllm = true;
  fixture.selection.prompt_queries.resize(query_heads * tokens * head_dim);
  for (std::size_t query_head = 0; query_head < query_heads; ++query_head) {
    const auto kv_head = query_head / groups;
    for (std::size_t token = 0; token < tokens; ++token) {
      for (std::size_t dim = 0; dim < head_dim; ++dim) {
        const auto key =
            token * 2 * kv_heads * head_dim + kv_head * head_dim + dim;
        fixture.selection.prompt_queries[
            (query_head * tokens + token) * head_dim + dim] =
            solidattention::half_to_float(fixture.kv[key]) * 0.85f +
            std::cos(static_cast<float>(query_head * 19 + token * 7 +
                                        dim * 3) *
                     0.011f) *
                0.03f;
      }
    }
  }
  const auto begin = Clock::now();
  auto representatives = solidattention::build_local_causal_representatives(
      fixture.selection.prompt_queries.data(), fixture.kv.data(), tokens,
      query_heads, kv_heads, head_dim, 32, 32, 4);
  fixture.selection.representative_build_ms =
      milliseconds(begin, Clock::now());
  fixture.selection.representatives = std::move(representatives.values);
  return fixture;
}

void write_layer_store(const std::string& path,
                       const std::vector<std::uint16_t>& layer,
                       std::size_t layers) {
  const std::size_t bytes = layer.size() * sizeof(std::uint16_t);
  void* aligned = nullptr;
  if (posix_memalign(&aligned, 4096, bytes) != 0) throw std::bad_alloc();
  std::memcpy(aligned, layer.data(), bytes);
  const int fd = ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_DIRECT,
                        S_IRUSR | S_IWUSR);
  if (fd < 0) {
    std::free(aligned);
    throw std::runtime_error("cannot create InfLLM KV store");
  }
  for (std::size_t layer_index = 0; layer_index < layers; ++layer_index) {
    if (::pwrite(fd, aligned, bytes, layer_index * bytes) !=
        static_cast<ssize_t>(bytes)) {
      ::close(fd);
      std::free(aligned);
      throw std::runtime_error("short InfLLM KV store write");
    }
  }
  ::fsync(fd);
  ::close(fd);
  std::free(aligned);
}

std::vector<std::uint64_t> offsets(
    std::size_t layer, std::size_t blocks_per_layer,
    std::size_t block_bytes, const BlockSet& selected) {
  std::vector<std::uint64_t> result;
  for (const auto block : selected) {
    result.push_back((layer * blocks_per_layer + block) * block_bytes);
  }
  return result;
}

std::vector<std::uint64_t> miss_offsets(
    std::size_t layer, std::size_t blocks_per_layer,
    std::size_t block_bytes, const std::vector<std::size_t>& misses) {
  std::vector<std::uint64_t> result;
  for (const auto block : misses) {
    result.push_back((layer * blocks_per_layer + block) * block_bytes);
  }
  return result;
}

struct Correction {
  std::vector<std::size_t> misses;
  std::vector<std::size_t> replacement_slots;
  std::size_t hits{};
};

Correction compare_prediction(const BlockSet& predicted,
                              const BlockSet& oracle) {
  Correction correction;
  std::vector<std::size_t> evicted_slots;
  for (std::size_t slot = 0; slot < predicted.size(); ++slot) {
    if (std::find(oracle.begin(), oracle.end(), predicted[slot]) !=
        oracle.end()) {
      ++correction.hits;
    } else {
      evicted_slots.push_back(slot);
    }
  }
  for (const auto block : oracle) {
    if (std::find(predicted.begin(), predicted.end(), block) ==
        predicted.end()) {
      correction.misses.push_back(block);
    }
  }
  if (correction.misses.size() != evicted_slots.size()) {
    throw std::runtime_error("prediction correction cardinality mismatch");
  }
  correction.replacement_slots = std::move(evicted_slots);
  BlockSet corrected = predicted;
  for (std::size_t miss = 0; miss < correction.misses.size(); ++miss) {
    corrected[correction.replacement_slots[miss]] = correction.misses[miss];
  }
  std::sort(corrected.begin(), corrected.end());
  if (corrected != oracle) {
    throw std::runtime_error("corrected resident block set differs from oracle");
  }
  return correction;
}

Totals run_serial(CudaPipeline& gpu, solidattention::UringReader& reader,
                  std::size_t steps, std::size_t layers,
                  std::size_t blocks_per_layer, std::size_t block_bytes,
                  bool history_correction,
                  const SelectionContext& selection) {
  Totals totals;
  const auto wall_begin = Clock::now();
  for (std::size_t step = 0; step < steps; ++step) {
    for (std::size_t layer = 0; layer < layers; ++layer) {
      const std::size_t slot = layer % 2;
      const auto selected =
          selection.blocks(step, layer, &totals.selection_ms);
      if (selection.infllm) {
        gpu.update_query(selection.query(step, layer));
      }
      totals.read_ms += reader.read_blocks_fixed(
          slot, offsets(layer, blocks_per_layer, block_bytes, selected),
          block_bytes);
      totals.h2d_ms += gpu.stage_sync(slot);
      if (history_correction) {
        gpu.launch_qkv_window_async();
        const auto qkv = gpu.correct_and_wait(slot, {}, block_bytes);
        totals.qkv_window_ms += qkv.qkv_ms;
      }
      totals.attention_ms += gpu.attention_sync(slot);
      totals.ffn_ms += gpu.ffn_sync();
    }
  }
  totals.wall_ms = milliseconds(wall_begin, Clock::now());
  totals.output = gpu.output();
  return totals;
}

Totals run_pipeline(CudaPipeline& gpu, solidattention::UringReader& reader,
                    solidattention::Trace& trace, std::size_t steps,
                    std::size_t layers, std::size_t blocks_per_layer,
                    std::size_t block_bytes, bool history_correction,
                    const SelectionContext& selection) {
  Totals totals;
  const std::size_t kv_bytes = 4 * block_bytes;
  std::vector<BlockSet> history(layers);
  for (std::size_t layer = 0; layer < layers; ++layer) {
    history[layer] = selection.blocks(0, layer, nullptr);
  }
  const auto wall_begin = Clock::now();
  for (std::size_t step = 0; step < steps; ++step) {
    const BlockSet first_prediction =
        history_correction ? history[0]
                           : selection.blocks(step, 0, &totals.selection_ms);
    double first_read = reader.read_blocks_fixed(
        0, offsets(0, blocks_per_layer, block_bytes, first_prediction),
        block_bytes);
    totals.read_ms += first_read;
    const auto first_stage_begin = trace_us(trace);
    const double first_h2d = gpu.stage_sync(0);
    totals.h2d_ms += first_h2d;
    trace.add({"startup H2D", "h2d", first_stage_begin,
               static_cast<std::uint64_t>(first_h2d * 1000), 2, step, 0,
               kv_bytes});

    for (std::size_t layer = 0; layer < layers; ++layer) {
      const std::size_t slot = layer % 2;
      const BlockSet predicted =
          history_correction
              ? history[layer]
              : selection.blocks(step, layer, &totals.selection_ms);
      const BlockSet oracle =
          selection.blocks(step, layer, &totals.selection_ms);
      const Correction correction = compare_prediction(predicted, oracle);
      if (selection.infllm) {
        gpu.update_query(selection.query(step, layer));
      }
      totals.predicted_blocks += predicted.size();
      totals.hit_blocks += correction.hits;
      totals.miss_blocks += correction.misses.size();

      if (history_correction) {
        const auto correction_begin_us = trace_us(trace);
        gpu.launch_qkv_window_async();
        double correction_read_ms = 0.0;
        if (!correction.misses.empty()) {
          correction_read_ms = reader.read_blocks_fixed(
              2, miss_offsets(layer, blocks_per_layer, block_bytes,
                              correction.misses),
              block_bytes);
        }
        const auto correction_timing = gpu.correct_and_wait(
            slot, correction.replacement_slots, block_bytes);
        const auto correction_end_us = trace_us(trace);
        totals.correction_read_ms += correction_read_ms;
        totals.correction_h2d_ms += correction_timing.h2d_ms;
        totals.qkv_window_ms += correction_timing.qkv_ms;
        if (!correction.misses.empty()) {
          trace.add({"selection miss SSD read", "correction",
                     correction_begin_us,
                     static_cast<std::uint64_t>(correction_read_ms * 1000), 1,
                     step, layer, correction.misses.size() * block_bytes});
          trace.add({"selection correction H2D overwrite", "correction",
                     correction_begin_us +
                         static_cast<std::uint64_t>(correction_read_ms * 1000),
                     static_cast<std::uint64_t>(
                         correction_timing.h2d_ms * 1000),
                     2, step, layer,
                     correction.misses.size() * block_bytes});
        }
        trace.add({"QKV projection window", "compute", correction_begin_us,
                   static_cast<std::uint64_t>(
                       correction_timing.qkv_ms * 1000),
                   3, step, layer, 0});
        trace.add({"selection correction barrier", "synchronization",
                   correction_begin_us,
                   correction_end_us - correction_begin_us, 0, step, layer,
                   0});
      }
      history[layer] = oracle;

      std::uint64_t read_begin_us = 0;
      if (layer + 1 < layers) {
        const std::size_t next = layer + 1;
        const BlockSet next_prediction =
            history_correction
                ? history[next]
                : selection.blocks(step, next, &totals.selection_ms);
        read_begin_us = trace_us(trace);
        reader.submit_blocks_fixed(
            next % 2,
            offsets(next, blocks_per_layer, block_bytes, next_prediction),
            block_bytes);
      }

      const auto attention_begin_us = trace_us(trace);
      const double attention_ms = gpu.attention_sync(slot);
      const auto attention_end_us = trace_us(trace);
      totals.attention_ms += attention_ms;
      trace.add({"sparse attention", "compute", attention_begin_us,
                 static_cast<std::uint64_t>(attention_ms * 1000), 3, step,
                 layer, kv_bytes});

      if (layer + 1 < layers) {
        const auto wait_begin = Clock::now();
        const double read_ms = reader.wait_blocks_fixed();
        const std::uint64_t read_end_us = trace_us(trace);
        const double wait_ms = milliseconds(wait_begin, Clock::now());
        totals.read_ms += read_ms;
        totals.exposed_read_wait_ms += wait_ms;
        const std::uint64_t overlap_begin =
            std::max(read_begin_us, attention_begin_us);
        const std::uint64_t overlap_end =
            std::min(read_end_us, attention_end_us);
        if (overlap_end > overlap_begin) {
          totals.estimated_ssd_attention_overlap_ms +=
              static_cast<double>(overlap_end - overlap_begin) / 1000.0;
        }
        trace.add({"next-layer KV read", "ssd", read_begin_us,
                   read_end_us - read_begin_us, 1, step, layer + 1, kv_bytes});

        const auto overlap_begin_us = trace_us(trace);
        const auto [ffn_ms, h2d_ms] =
            gpu.overlap_ffn_and_stage((layer + 1) % 2);
        const auto overlap_end_us = trace_us(trace);
        totals.ffn_ms += ffn_ms;
        totals.h2d_ms += h2d_ms;
        totals.estimated_h2d_ffn_overlap_ms += std::min(ffn_ms, h2d_ms);
        trace.add({"current-layer FFN window", "compute", overlap_begin_us,
                   static_cast<std::uint64_t>(ffn_ms * 1000), 3, step, layer,
                   0});
        trace.add({"next-layer H2D", "h2d", overlap_begin_us,
                   static_cast<std::uint64_t>(h2d_ms * 1000), 2, step,
                   layer + 1, kv_bytes});
        trace.add({"overlap barrier", "synchronization", overlap_begin_us,
                   overlap_end_us - overlap_begin_us, 0, step, layer, 0});
      } else {
        const auto ffn_begin_us = trace_us(trace);
        const double ffn_ms = gpu.ffn_sync();
        totals.ffn_ms += ffn_ms;
        trace.add({"final-layer FFN window", "compute", ffn_begin_us,
                   static_cast<std::uint64_t>(ffn_ms * 1000), 3, step, layer,
                   0});
      }
    }
  }
  totals.wall_ms = milliseconds(wall_begin, Clock::now());
  totals.output = gpu.output();
  return totals;
}

void write_metrics(const std::string& path, const Totals& serial,
                   const Totals& pipeline, std::size_t steps,
                   std::size_t layers, int ffn_iterations,
                   float maximum_error, bool history_correction,
                   const SelectionContext& selection) {
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot write metrics");
  const double executions = static_cast<double>(steps * layers);
  output << std::fixed << std::setprecision(6)
         << "{\n"
         << "  \"version\": \""
         << (selection.infllm
                 ? "P1.1-infllm-history-correction"
                 : (history_correction ? "P1.0-history-correction"
                                       : "P0-cuda-dual-pipeline"))
         << "\",\n"
         << "  \"scope\": \"synthetic 28-layer scheduler; real sparse attention; "
         << (history_correction ? "synthetic QKV/FFN windows"
                                : "synthetic FFN window")
         << "\",\n"
         << "  \"steps\": " << steps << ",\n"
         << "  \"layers\": " << layers << ",\n"
         << "  \"ffn_iterations\": " << ffn_iterations << ",\n"
         << "  \"representative_build_ms\": "
         << selection.representative_build_ms << ",\n"
         << "  \"pipeline_selection_ms\": " << pipeline.selection_ms << ",\n"
         << "  \"dense_selected_attention_mass\": "
         << selection.dense_selected_attention_mass << ",\n"
         << "  \"dense_oracle_block_recall\": "
         << selection.dense_oracle_block_recall << ",\n"
         << "  \"serial_wall_ms\": " << serial.wall_ms << ",\n"
         << "  \"pipeline_wall_ms\": " << pipeline.wall_ms << ",\n"
         << "  \"serial_layer_ms\": " << serial.wall_ms / executions << ",\n"
         << "  \"pipeline_layer_ms\": " << pipeline.wall_ms / executions << ",\n"
         << "  \"speedup\": " << serial.wall_ms / pipeline.wall_ms << ",\n"
         << "  \"pipeline_read_ms\": " << pipeline.read_ms << ",\n"
         << "  \"pipeline_exposed_read_wait_ms\": "
         << pipeline.exposed_read_wait_ms << ",\n"
         << "  \"pipeline_h2d_ms\": " << pipeline.h2d_ms << ",\n"
         << "  \"pipeline_attention_ms\": " << pipeline.attention_ms << ",\n"
         << "  \"pipeline_ffn_ms\": " << pipeline.ffn_ms << ",\n"
         << "  \"estimated_ssd_attention_overlap_ms\": "
         << pipeline.estimated_ssd_attention_overlap_ms << ",\n"
         << "  \"estimated_h2d_ffn_overlap_ms\": "
         << pipeline.estimated_h2d_ffn_overlap_ms << ",\n"
         << "  \"history_hit_rate\": "
         << (pipeline.predicted_blocks == 0
                 ? 1.0
                 : static_cast<double>(pipeline.hit_blocks) /
                       pipeline.predicted_blocks)
         << ",\n"
         << "  \"predicted_blocks\": " << pipeline.predicted_blocks << ",\n"
         << "  \"hit_blocks\": " << pipeline.hit_blocks << ",\n"
         << "  \"miss_blocks\": " << pipeline.miss_blocks << ",\n"
         << "  \"corrected_oracle_block_recall\": 1.000000,\n"
         << "  \"correction_read_ms\": " << pipeline.correction_read_ms
         << ",\n"
         << "  \"correction_h2d_ms\": " << pipeline.correction_h2d_ms
         << ",\n"
         << "  \"qkv_window_ms\": " << pipeline.qkv_window_ms << ",\n"
         << "  \"serial_pipeline_max_abs_error\": " << maximum_error << "\n"
         << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string output_dir = "artifacts/cpp-p0";
    std::size_t steps = 8;
    int ffn_iterations = 512;
    bool history_correction = false;
    bool infllm_selection = false;
    for (int index = 1; index < argc; ++index) {
      const std::string argument = argv[index];
      if (argument == "--output" && index + 1 < argc) {
        output_dir = argv[++index];
      } else if (argument == "--steps" && index + 1 < argc) {
        steps = std::stoul(argv[++index]);
      } else if (argument == "--ffn-iterations" && index + 1 < argc) {
        ffn_iterations = std::stoi(argv[++index]);
      } else if (argument == "--history-correction") {
        history_correction = true;
      } else if (argument == "--infllm-selection") {
        history_correction = true;
        infllm_selection = true;
      } else {
        throw std::runtime_error("unknown argument: " + argument);
      }
    }

    constexpr std::size_t kLayers = 28;
    constexpr std::size_t kBlocksPerLayer = 16;
    constexpr std::size_t kBlockBytes = 128 * 1024;
    constexpr std::size_t kPackedBytes = 4 * kBlockBytes;
    std::filesystem::create_directories(output_dir);
    SelectionContext selection;
    std::string store = output_dir + "/pipeline-kv.bin";
    if (infllm_selection) {
      Fixture fixture = build_infllm_fixture();
      audit_dense_selection(fixture.selection, fixture.kv, steps, kLayers);
      selection = std::move(fixture.selection);
      store = output_dir + "/pipeline-infllm-kv.bin";
      if (!std::filesystem::exists(store) ||
          std::filesystem::file_size(store) !=
              kLayers * kBlocksPerLayer * kBlockBytes) {
        write_layer_store(store, fixture.kv, kLayers);
      }
    } else if (!std::filesystem::exists(store) ||
               std::filesystem::file_size(store) !=
                   kLayers * kBlocksPerLayer * kBlockBytes) {
        solidattention::create_deterministic_store(
            store, kLayers * kBlocksPerLayer, kBlockBytes);
    }

    CudaPipeline gpu(kPackedBytes, ffn_iterations);
    solidattention::UringReader reader(store, gpu.host_buffers(), kPackedBytes);
    const Totals serial = run_serial(gpu, reader, steps, kLayers,
                                     kBlocksPerLayer, kBlockBytes,
                                     history_correction, selection);
    solidattention::Trace trace;
    const Totals pipeline = run_pipeline(gpu, reader, trace, steps, kLayers,
                                         kBlocksPerLayer, kBlockBytes,
                                         history_correction, selection);
    float maximum_error = 0.0f;
    for (std::size_t index = 0; index < serial.output.size(); ++index) {
      maximum_error = std::max(
          maximum_error, std::abs(serial.output[index] - pipeline.output[index]));
    }
    trace.write(output_dir + "/pipeline-trace.json");
    write_metrics(output_dir + "/pipeline-metrics.json", serial, pipeline,
                  steps, kLayers, ffn_iterations, maximum_error,
                  history_correction, selection);
    std::cout << std::fixed << std::setprecision(6)
              << "version="
              << (selection.infllm
                      ? "P1.1-infllm-history-correction"
                      : (history_correction ? "P1.0-history-correction"
                                            : "P0-cuda-dual-pipeline"))
              << '\n'
              << "serial_wall_ms=" << serial.wall_ms << '\n'
              << "pipeline_wall_ms=" << pipeline.wall_ms << '\n'
              << "speedup=" << serial.wall_ms / pipeline.wall_ms << '\n'
              << "exposed_ssd_wait_ms=" << pipeline.exposed_read_wait_ms << '\n'
              << "ssd_attention_overlap_ms="
              << pipeline.estimated_ssd_attention_overlap_ms << '\n'
              << "h2d_ffn_overlap_ms="
              << pipeline.estimated_h2d_ffn_overlap_ms << '\n'
              << "history_hit_rate="
              << (pipeline.predicted_blocks == 0
                      ? 1.0
                      : static_cast<double>(pipeline.hit_blocks) /
                            pipeline.predicted_blocks)
              << '\n'
              << "miss_blocks=" << pipeline.miss_blocks << '\n'
              << "correction_read_ms=" << pipeline.correction_read_ms << '\n'
              << "correction_h2d_ms=" << pipeline.correction_h2d_ms << '\n'
              << "representative_build_ms="
              << selection.representative_build_ms << '\n'
              << "selection_ms=" << pipeline.selection_ms << '\n'
              << "dense_selected_attention_mass="
              << selection.dense_selected_attention_mass << '\n'
              << "dense_oracle_block_recall="
              << selection.dense_oracle_block_recall << '\n'
              << "serial_pipeline_max_abs_error=" << maximum_error << '\n'
              << "trace=" << output_dir << "/pipeline-trace.json\n";
    return maximum_error == 0.0f ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
