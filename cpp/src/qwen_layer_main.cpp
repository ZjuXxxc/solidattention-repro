#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>
#include "solidattention/uring_reader.hpp"
#include "solidattention/trace.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

extern "C" {
using cublasHandle_t = void*;
using cublasStatus_t = int;
enum cublasOperation_t { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1 };
int cublasCreate_v2(cublasHandle_t*);
int cublasDestroy_v2(cublasHandle_t);
int cublasSetStream_v2(cublasHandle_t, cudaStream_t);
int cublasSgemm_v2(cublasHandle_t, cublasOperation_t, cublasOperation_t,
                   int, int, int, const float*, const float*, int,
                   const float*, int, const float*, float*, int);
}

namespace {

constexpr cublasStatus_t CUBLAS_STATUS_SUCCESS = 0;

void cuda_check(cudaError_t value, const char* operation) {
  if (value != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(value));
  }
}
void cublas_check(cublasStatus_t value, const char* operation) {
  if (value != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + ": cuBLAS failure");
  }
}
void driver_check(CUresult value, const char* operation) {
  if (value != CUDA_SUCCESS) {
    const char* message = nullptr;
    cuGetErrorString(value, &message);
    throw std::runtime_error(std::string(operation) + ": " +
                             (message ? message : "driver failure"));
  }
}
void nvrtc_check(nvrtcResult value, const char* operation) {
  if (value != NVRTC_SUCCESS) {
    throw std::runtime_error(std::string(operation) + ": " +
                             nvrtcGetErrorString(value));
  }
}

std::vector<float> load(const std::filesystem::path& directory,
                        const std::string& name, std::size_t elements) {
  std::vector<float> result(elements);
  std::ifstream input(directory / (name + ".f32"), std::ios::binary);
  if (!input ||
      !input.read(reinterpret_cast<char*>(result.data()),
                  result.size() * sizeof(float))) {
    throw std::runtime_error("cannot read tensor " + name);
  }
  return result;
}

struct Device {
  void* pointer{};
  Device() = default;
  explicit Device(std::size_t bytes) { cuda_check(cudaMalloc(&pointer, bytes), "cudaMalloc"); }
  ~Device() { if (pointer) cudaFree(pointer); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  float* f32() const { return static_cast<float*>(pointer); }
};

struct Error {
  double maximum{};
  double cosine{};
};

Error compare(const std::vector<float>& actual,
              const std::vector<float>& expected) {
  if (actual.size() != expected.size()) throw std::runtime_error("shape mismatch");
  double dot = 0.0, left = 0.0, right = 0.0, maximum = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    maximum = std::max(maximum,
                       std::abs(static_cast<double>(actual[index]) -
                                expected[index]));
    dot += static_cast<double>(actual[index]) * expected[index];
    left += static_cast<double>(actual[index]) * actual[index];
    right += static_cast<double>(expected[index]) * expected[index];
  }
  return {maximum, dot / std::sqrt(left * right)};
}

constexpr const char* kSource = R"(
__device__ float half_to_float(unsigned short value) {
  unsigned int sign = ((unsigned int)value & 0x8000U) << 16;
  unsigned int exponent = (value >> 10) & 0x1fU;
  unsigned int mantissa = value & 0x3ffU;
  unsigned int bits;
  if (exponent == 0) {
    if (mantissa == 0) bits = sign;
    else {
      int shift = 0;
      while ((mantissa & 0x400U) == 0) { mantissa <<= 1; ++shift; }
      mantissa &= 0x3ffU;
      bits = sign | ((127 - 14 - shift) << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) bits = sign | 0x7f800000U | (mantissa << 13);
  else bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
  return __uint_as_float(bits);
}

extern "C" __global__ void rms_norm(
    const float* input, const float* weight, float* output, int width,
    float epsilon) {
  __shared__ float reduction[256];
  float sum = 0.0f;
  for (int index = threadIdx.x; index < width; index += blockDim.x) {
    sum += input[index] * input[index];
  }
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduction[threadIdx.x] +=
        reduction[threadIdx.x + stride];
    __syncthreads();
  }
  float inverse = rsqrtf(reduction[0] / width + epsilon);
  for (int index = threadIdx.x; index < width; index += blockDim.x) {
    output[index] = input[index] * inverse * weight[index];
  }
}

extern "C" __global__ void head_rms(
    float* values, const float* weight, int rows, int width, float epsilon) {
  int row = blockIdx.x;
  if (row >= rows) return;
  __shared__ float reduction[128];
  float value = values[row * width + threadIdx.x];
  reduction[threadIdx.x] = value * value;
  __syncthreads();
  for (int stride = 64; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduction[threadIdx.x] +=
        reduction[threadIdx.x + stride];
    __syncthreads();
  }
  values[row * width + threadIdx.x] =
      value * rsqrtf(reduction[0] / width + epsilon) * weight[threadIdx.x];
}

extern "C" __global__ void rope(
    float* values, int tokens, int heads, int width, int position_start,
    float theta) {
  int row = blockIdx.x;
  int dim = threadIdx.x;
  if (row >= tokens * heads || dim >= width) return;
  __shared__ float original_row[128];
  original_row[dim] = values[row * width + dim];
  __syncthreads();
  int token = row / heads;
  int half = width / 2;
  int paired = dim < half ? dim + half : dim - half;
  int frequency_index = dim < half ? dim : dim - half;
  float frequency = powf(theta, -2.0f * frequency_index / width);
  float angle = (position_start + token) * frequency;
  float original = original_row[dim];
  float partner = original_row[paired];
  float rotated = dim < half ? -partner : partner;
  values[row * width + dim] =
      original * cosf(angle) + rotated * sinf(angle);
}

extern "C" __global__ void decode_attention(
    const float* query, const float* key, const float* value, float* output,
    int tokens, int query_heads, int kv_heads, int width, float scale) {
  int head = blockIdx.x;
  int lane = threadIdx.x;
  int kv_head = head / (query_heads / kv_heads);
  __shared__ float probability[128];
  __shared__ float reduction[128];
  float dot = 0.0f;
  for (int dim = 0; dim < width; ++dim) {
    dot += query[head * width + dim] *
           key[(lane * kv_heads + kv_head) * width + dim];
  }
  dot *= scale;
  reduction[lane] = dot;
  __syncthreads();
  for (int stride = 64; stride > 0; stride >>= 1) {
    if (lane < stride) reduction[lane] =
        fmaxf(reduction[lane], reduction[lane + stride]);
    __syncthreads();
  }
  float p = expf(dot - reduction[0]);
  probability[lane] = p;
  reduction[lane] = p;
  __syncthreads();
  for (int stride = 64; stride > 0; stride >>= 1) {
    if (lane < stride) reduction[lane] += reduction[lane + stride];
    __syncthreads();
  }
  if (lane < width) {
    float result = 0.0f;
    for (int token = 0; token < tokens; ++token) {
      result += probability[token] / reduction[0] *
          value[(token * kv_heads + kv_head) * width + lane];
    }
    output[head * width + lane] = result;
  }
}

extern "C" __global__ void sparse_fp16_attention(
    const float* query, const unsigned short* kv, float* output,
    int tokens, int query_heads, int kv_heads, int width, float scale) {
  int head = blockIdx.x;
  int lane = threadIdx.x;
  int kv_head = head / (query_heads / kv_heads);
  __shared__ float probability[128];
  __shared__ float reduction[128];
  unsigned long long base =
      (unsigned long long)lane * 2 * kv_heads * width + kv_head * width;
  float dot = 0.0f;
  for (int dim = 0; dim < width; ++dim) {
    dot += query[head * width + dim] * half_to_float(kv[base + dim]);
  }
  dot *= scale;
  reduction[lane] = dot;
  __syncthreads();
  for (int stride = 64; stride > 0; stride >>= 1) {
    if (lane < stride) reduction[lane] =
        fmaxf(reduction[lane], reduction[lane + stride]);
    __syncthreads();
  }
  float p = expf(dot - reduction[0]);
  probability[lane] = p;
  reduction[lane] = p;
  __syncthreads();
  for (int stride = 64; stride > 0; stride >>= 1) {
    if (lane < stride) reduction[lane] += reduction[lane + stride];
    __syncthreads();
  }
  if (lane < width) {
    float result = 0.0f;
    for (int token = 0; token < tokens; ++token) {
      unsigned long long value_base =
          (unsigned long long)token * 2 * kv_heads * width +
          kv_heads * width + kv_head * width;
      result += probability[token] / reduction[0] *
          half_to_float(kv[value_base + lane]);
    }
    output[head * width + lane] = result;
  }
}

extern "C" __global__ void add(
    const float* left, const float* right, float* output, int elements) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) output[index] = left[index] + right[index];
}

extern "C" __global__ void silu_multiply(
    const float* gate, const float* up, float* output, int elements) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) {
    float value = gate[index];
    output[index] = value / (1.0f + expf(-value)) * up[index];
  }
}
)";

class Kernels {
 public:
  Kernels() {
    driver_check(cuInit(0), "cuInit");
    nvrtcProgram program{};
    nvrtc_check(nvrtcCreateProgram(&program, kSource, "qwen_layer.cu", 0,
                                   nullptr, nullptr), "nvrtcCreateProgram");
    const char* options[] = {"--std=c++14", "--gpu-architecture=compute_89"};
    auto result = nvrtcCompileProgram(program, 2, options);
    if (result != NVRTC_SUCCESS) {
      std::size_t size = 0;
      nvrtcGetProgramLogSize(program, &size);
      std::string log(size, '\0');
      nvrtcGetProgramLog(program, log.data());
      throw std::runtime_error("NVRTC: " + log);
    }
    std::size_t size = 0;
    nvrtcGetPTXSize(program, &size);
    std::vector<char> ptx(size);
    nvrtcGetPTX(program, ptx.data());
    nvrtcDestroyProgram(&program);
    driver_check(cuModuleLoadData(&module_, ptx.data()), "cuModuleLoadData");
    get(rms_, "rms_norm");
    get(head_rms_, "head_rms");
    get(rope_, "rope");
    get(attention_, "decode_attention");
    get(sparse_attention_, "sparse_fp16_attention");
    get(add_, "add");
    get(silu_, "silu_multiply");
  }
  ~Kernels() { if (module_) cuModuleUnload(module_); }
  void rms(float* input, float* weight, float* output, int width, float eps,
           cudaStream_t stream) {
    void* args[]{&input, &weight, &output, &width, &eps};
    launch(rms_, 1, 256, args, stream);
  }
  void head_norm(float* values, float* weight, int rows, int width, float eps,
                 cudaStream_t stream) {
    void* args[]{&values, &weight, &rows, &width, &eps};
    launch(head_rms_, rows, 128, args, stream);
  }
  void apply_rope(float* values, int tokens, int heads, int width,
                  int position, float theta, cudaStream_t stream) {
    void* args[]{&values, &tokens, &heads, &width, &position, &theta};
    launch(rope_, tokens * heads, 128, args, stream);
  }
  void attention(float* query, float* key, float* value, float* output,
                 int tokens, int q_heads, int kv_heads, int width,
                 cudaStream_t stream) {
    float scale = 1.0f / std::sqrt(static_cast<float>(width));
    void* args[]{&query, &key, &value, &output, &tokens, &q_heads, &kv_heads,
                 &width, &scale};
    launch(attention_, q_heads, 128, args, stream);
  }
  void sparse_attention(float* query, void* kv, float* output, int tokens,
                        int q_heads, int kv_heads, int width,
                        cudaStream_t stream) {
    float scale = 1.0f / std::sqrt(static_cast<float>(width));
    void* args[]{&query, &kv, &output, &tokens, &q_heads, &kv_heads,
                 &width, &scale};
    launch(sparse_attention_, q_heads, 128, args, stream);
  }
  void add(float* left, float* right, float* output, int elements,
           cudaStream_t stream) {
    void* args[]{&left, &right, &output, &elements};
    launch(add_, (elements + 255) / 256, 256, args, stream);
  }
  void silu(float* gate, float* up, float* output, int elements,
            cudaStream_t stream) {
    void* args[]{&gate, &up, &output, &elements};
    launch(silu_, (elements + 255) / 256, 256, args, stream);
  }
 private:
  void get(CUfunction& function, const char* name) {
    driver_check(cuModuleGetFunction(&function, module_, name), name);
  }
  static void launch(CUfunction function, int blocks, int threads, void** args,
                     cudaStream_t stream) {
    driver_check(cuLaunchKernel(function, blocks, 1, 1, threads, 1, 1, 0,
                                reinterpret_cast<CUstream>(stream), args,
                                nullptr), "cuLaunchKernel");
  }
  CUmodule module_{};
  CUfunction rms_{}, head_rms_{}, rope_{}, attention_{}, sparse_attention_{},
      add_{}, silu_{};
};

void upload(Device& device, const std::vector<float>& host) {
  cuda_check(cudaMemcpy(device.pointer, host.data(), host.size() * sizeof(float),
                        cudaMemcpyHostToDevice), "upload tensor");
}
std::vector<float> download(float* pointer, std::size_t elements) {
  std::vector<float> result(elements);
  cuda_check(cudaMemcpy(result.data(), pointer, elements * sizeof(float),
                        cudaMemcpyDeviceToHost), "download tensor");
  return result;
}

}  // namespace

#ifndef SOLIDATTENTION_QWEN_LAYER_LIBRARY
int main(int argc, char** argv) {
  try {
    std::filesystem::path directory = "artifacts/qwen-layer0";
    std::filesystem::path metrics = "artifacts/qwen-layer0/native-metrics.json";
    std::filesystem::path trace_path;
    std::filesystem::path kv_store_override;
    std::uint64_t kv_base_offset = 0;
    int layer_index = 0;
    bool chain_mode = false;
    std::filesystem::path hidden_input_override;
    std::filesystem::path hidden_output_path;
    bool sparse = false;
    int io_repeats = 1;
    bool pipeline_next = false;
    std::vector<std::size_t> selected_blocks;
    for (int index = 1; index < argc; ++index) {
      std::string argument = argv[index];
      if (argument == "--input" && index + 1 < argc) directory = argv[++index];
      else if (argument == "--metrics" && index + 1 < argc) metrics = argv[++index];
      else if (argument == "--trace" && index + 1 < argc) trace_path = argv[++index];
      else if (argument == "--kv-store" && index + 1 < argc) {
        kv_store_override = argv[++index];
      }
      else if (argument == "--kv-offset" && index + 1 < argc) {
        kv_base_offset = std::stoull(argv[++index]);
      }
      else if (argument == "--layer" && index + 1 < argc) {
        layer_index = std::stoi(argv[++index]);
      }
      else if (argument == "--chain") chain_mode = true;
      else if (argument == "--hidden-input" && index + 1 < argc) {
        hidden_input_override = argv[++index];
      }
      else if (argument == "--hidden-output" && index + 1 < argc) {
        hidden_output_path = argv[++index];
      }
      else if (argument == "--sparse") sparse = true;
      else if (argument == "--io-repeats" && index + 1 < argc) {
        io_repeats = std::stoi(argv[++index]);
      }
      else if (argument == "--pipeline-next") pipeline_next = true;
      else if (argument == "--selected" && index + 1 < argc) {
        std::stringstream stream(argv[++index]);
        std::string value;
        while (std::getline(stream, value, ',')) {
          selected_blocks.push_back(std::stoul(value));
        }
      }
      else throw std::runtime_error("unknown argument: " + argument);
    }
    const int tokens = sparse ? 512 : 128;
    const int attention_tokens = sparse ? 128 : tokens;
    constexpr int hidden = 1024, intermediate = 3072;
    constexpr int q_heads = 16, kv_heads = 8, head_dim = 128;
    constexpr int q_width = q_heads * head_dim, kv_width = kv_heads * head_dim;
    constexpr float epsilon = 1e-6f, theta = 1000000.0f;
    const int position_start = sparse ? 0 : 384;
    if (sparse && selected_blocks.size() != 4) {
      throw std::runtime_error("sparse mode requires four --selected blocks");
    }

    std::vector<float> hidden_host;
    if (chain_mode) {
      hidden_host.assign(static_cast<std::size_t>(tokens) * hidden, 0.0f);
      const auto chain_input = hidden_input_override.empty()
          ? load(directory, "chain_input", hidden)
          : load(hidden_input_override.parent_path(),
                 hidden_input_override.stem().string(), hidden);
      std::copy(chain_input.begin(), chain_input.end(),
                hidden_host.end() - hidden);
    } else {
      hidden_host = load(directory, "hidden", tokens * hidden);
    }
    auto input_norm = load(directory, "input_norm_weight", hidden);
    auto q_weight = load(directory, "q_weight", q_width * hidden);
    auto k_weight = load(directory, "k_weight", kv_width * hidden);
    auto v_weight = load(directory, "v_weight", kv_width * hidden);
    auto q_norm = load(directory, "q_norm_weight", head_dim);
    auto k_norm = load(directory, "k_norm_weight", head_dim);
    auto o_weight = load(directory, "o_weight", hidden * q_width);
    auto post_norm = load(directory, "post_norm_weight", hidden);
    auto gate_weight = load(directory, "gate_weight", intermediate * hidden);
    auto up_weight = load(directory, "up_weight", intermediate * hidden);
    auto down_weight = load(directory, "down_weight", hidden * intermediate);

    Device d_hidden(hidden_host.size() * 4), d_input_norm(hidden * 4);
    Device d_q_weight(q_weight.size() * 4), d_k_weight(k_weight.size() * 4);
    Device d_v_weight(v_weight.size() * 4), d_q_norm(head_dim * 4);
    Device d_k_norm(head_dim * 4), d_o_weight(o_weight.size() * 4);
    Device d_post_norm(hidden * 4), d_gate_weight(gate_weight.size() * 4);
    Device d_up_weight(up_weight.size() * 4), d_down_weight(down_weight.size() * 4);
    Device d_normalized(tokens * hidden * 4), d_query(tokens * q_width * 4);
    Device d_key(tokens * kv_width * 4), d_value(tokens * kv_width * 4);
    Device d_attended(q_width * 4), d_attention_output(hidden * 4);
    Device d_attention_residual(hidden * 4), d_post_normalized(hidden * 4);
    Device d_gate(intermediate * 4), d_up(intermediate * 4);
    Device d_activated(intermediate * 4), d_mlp_output(hidden * 4);
    Device d_output(hidden * 4);
    Device d_sparse_kv(sparse ? 4 * 128 * 1024 : 1);
    Device d_next_sparse_kv(sparse ? 4 * 128 * 1024 : 1);
    upload(d_hidden, hidden_host); upload(d_input_norm, input_norm);
    upload(d_q_weight, q_weight); upload(d_k_weight, k_weight);
    upload(d_v_weight, v_weight); upload(d_q_norm, q_norm); upload(d_k_norm, k_norm);
    upload(d_o_weight, o_weight); upload(d_post_norm, post_norm);
    upload(d_gate_weight, gate_weight); upload(d_up_weight, up_weight);
    upload(d_down_weight, down_weight);

    cudaStream_t stream{};
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    cublasHandle_t handle{};
    cublas_check(cublasCreate_v2(&handle), "cublasCreate");
    cublas_check(cublasSetStream_v2(handle, stream), "cublasSetStream");
    Kernels kernels;
    const float one = 1.0f, zero = 0.0f;
    void* pinned_kv = nullptr;
    double ssd_read_ms = 0.0;
    std::vector<double> ssd_reads_ms;
    double sparse_h2d_ms = 0.0;
    double next_read_ms = 0.0;
    double exposed_next_read_wait_ms = 0.0;
    double next_h2d_ms = 0.0;
    double next_h2d_start_offset_ms = 0.0;
    std::vector<std::uint64_t> sparse_offsets;
    std::unique_ptr<solidattention::UringReader> sparse_reader;
    if (sparse) {
      constexpr std::size_t block_bytes = 128 * 1024;
      constexpr std::size_t packed_bytes = 4 * block_bytes;
      cuda_check(cudaHostAlloc(&pinned_kv, packed_bytes, cudaHostAllocDefault),
                 "allocate sparse pinned KV");
      sparse_reader = std::make_unique<solidattention::UringReader>(
          (kv_store_override.empty() ? directory / "kv-store-fp16.bin"
                                     : kv_store_override).string(),
          std::vector<void*>{pinned_kv}, packed_bytes);
      for (const auto block : selected_blocks) {
        sparse_offsets.push_back(kv_base_offset + block * block_bytes);
      }
      for (int repeat = 0; repeat < io_repeats; ++repeat) {
        ssd_reads_ms.push_back(
            sparse_reader->read_blocks_fixed(0, sparse_offsets, block_bytes));
      }
      ssd_read_ms = ssd_reads_ms.front();
      cudaEvent_t begin{}, end{};
      cudaEventCreate(&begin);
      cudaEventCreate(&end);
      cudaEventRecord(begin, stream);
      cudaMemcpyAsync(d_sparse_kv.pointer, pinned_kv, packed_bytes,
                      cudaMemcpyHostToDevice, stream);
      cudaEventRecord(end, stream);
      cudaEventSynchronize(end);
      float elapsed = 0.0f;
      cudaEventElapsedTime(&elapsed, begin, end);
      sparse_h2d_ms = elapsed;
      cudaEventDestroy(begin);
      cudaEventDestroy(end);
      if (pipeline_next) {
        sparse_reader->submit_blocks_fixed(0, sparse_offsets, block_bytes);
      }
    }
    const auto wall_begin = std::chrono::steady_clock::now();
    float* decode_hidden = d_hidden.f32() + (tokens - 1) * hidden;
    if (sparse) {
      kernels.rms(decode_hidden, d_input_norm.f32(), d_normalized.f32(), hidden,
                  epsilon, stream);
    } else {
      // Dense parity constructs context K/V from all 128 real embeddings.
      for (int token = 0; token < tokens; ++token) {
        kernels.rms(d_hidden.f32() + token * hidden, d_input_norm.f32(),
                    d_normalized.f32() + token * hidden, hidden, epsilon,
                    stream);
      }
    }
    auto gemm = [&](float* weight, int output, float* input, int batch,
                    float* result) {
      cublas_check(cublasSgemm_v2(handle, CUBLAS_OP_T, CUBLAS_OP_N, output, batch,
                               hidden, &one, weight, hidden, input, hidden,
                               &zero, result, output), "projection SGEMM");
    };
    if (sparse) {
      gemm(d_q_weight.f32(), q_width, d_normalized.f32(), 1, d_query.f32());
      kernels.head_norm(d_query.f32(), d_q_norm.f32(), q_heads, head_dim,
                        epsilon, stream);
      kernels.apply_rope(d_query.f32(), 1, q_heads, head_dim,
                         position_start + tokens - 1, theta, stream);
    } else {
      gemm(d_q_weight.f32(), q_width, d_normalized.f32(), tokens, d_query.f32());
      gemm(d_k_weight.f32(), kv_width, d_normalized.f32(), tokens, d_key.f32());
      gemm(d_v_weight.f32(), kv_width, d_normalized.f32(), tokens, d_value.f32());
      kernels.head_norm(d_query.f32(), d_q_norm.f32(), tokens * q_heads,
                        head_dim, epsilon, stream);
      kernels.head_norm(d_key.f32(), d_k_norm.f32(), tokens * kv_heads,
                        head_dim, epsilon, stream);
      kernels.apply_rope(d_query.f32(), tokens, q_heads, head_dim,
                         position_start, theta, stream);
      kernels.apply_rope(d_key.f32(), tokens, kv_heads, head_dim,
                         position_start, theta, stream);
    }
    if (sparse) {
      kernels.sparse_attention(d_query.f32(),
                               d_sparse_kv.pointer, d_attended.f32(),
                               attention_tokens, q_heads, kv_heads, head_dim,
                               stream);
    } else {
      kernels.attention(d_query.f32() + (tokens - 1) * q_width, d_key.f32(),
                        d_value.f32(), d_attended.f32(), tokens, q_heads,
                        kv_heads, head_dim, stream);
    }
    cublas_check(cublasSgemm_v2(handle, CUBLAS_OP_T, CUBLAS_OP_N, hidden, 1,
                             q_width, &one, d_o_weight.f32(), q_width,
                             d_attended.f32(), q_width, &zero,
                             d_attention_output.f32(), hidden), "o projection");
    kernels.add(decode_hidden, d_attention_output.f32(),
                d_attention_residual.f32(), hidden, stream);
    kernels.rms(d_attention_residual.f32(), d_post_norm.f32(),
                d_post_normalized.f32(), hidden, epsilon, stream);
    cudaStream_t copy_stream{};
    cudaEvent_t next_copy_begin{}, next_copy_end{};
    if (pipeline_next) {
      cuda_check(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking),
                 "next-layer copy stream");
      const auto wait_begin = std::chrono::steady_clock::now();
      next_read_ms = sparse_reader->wait_blocks_fixed();
      exposed_next_read_wait_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - wait_begin).count();
      next_h2d_start_offset_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - wall_begin).count();
      cudaEventCreate(&next_copy_begin);
      cudaEventCreate(&next_copy_end);
      cudaEventRecord(next_copy_begin, copy_stream);
      cudaMemcpyAsync(d_next_sparse_kv.pointer, pinned_kv, 512 * 1024,
                      cudaMemcpyHostToDevice, copy_stream);
      cudaEventRecord(next_copy_end, copy_stream);
    }
    gemm(d_gate_weight.f32(), intermediate, d_post_normalized.f32(), 1,
         d_gate.f32());
    gemm(d_up_weight.f32(), intermediate, d_post_normalized.f32(), 1,
         d_up.f32());
    kernels.silu(d_gate.f32(), d_up.f32(), d_activated.f32(), intermediate,
                 stream);
    cublas_check(cublasSgemm_v2(handle, CUBLAS_OP_T, CUBLAS_OP_N, hidden, 1,
                             intermediate, &one, d_down_weight.f32(),
                             intermediate, d_activated.f32(), intermediate,
                             &zero, d_mlp_output.f32(), hidden),
                 "down projection");
    kernels.add(d_attention_residual.f32(), d_mlp_output.f32(), d_output.f32(),
                hidden, stream);
    cuda_check(cudaStreamSynchronize(stream), "layer synchronize");
    if (pipeline_next) {
      cudaEventSynchronize(next_copy_end);
      float elapsed = 0.0f;
      cudaEventElapsedTime(&elapsed, next_copy_begin, next_copy_end);
      next_h2d_ms = elapsed;
    }
    const double wall_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wall_begin).count();

    struct Audit { const char* name; float* pointer; std::size_t elements; };
    std::vector<Audit> audits;
    if (sparse) {
      if (chain_mode) {
        audits.push_back({"chain_input", decode_hidden, hidden});
      }
      audits.push_back({chain_mode ? "chain_decode_normalized"
                                   : "decode_normalized",
                        d_normalized.f32(), hidden});
      audits.push_back({chain_mode ? "chain_decode_query" : "decode_query",
                        d_query.f32(), q_width});
    } else {
      audits.push_back({"normalized", d_normalized.f32(),
                        static_cast<std::size_t>(tokens) * hidden});
      audits.push_back(
          {"query", d_query.f32(), static_cast<std::size_t>(tokens) * q_width});
      audits.push_back(
          {"key", d_key.f32(), static_cast<std::size_t>(tokens) * kv_width});
      audits.push_back(
          {"value", d_value.f32(), static_cast<std::size_t>(tokens) * kv_width});
    }
    std::vector<Audit> tail_audits{
        {chain_mode ? "chain_sparse_attended"
                    : (sparse ? "sparse_attended" : "attended"),
         d_attended.f32(), q_width},
        {sparse ? "sparse_attention_output" : "attention_output",
         d_attention_output.f32(), hidden},
        {sparse ? "sparse_attention_residual" : "attention_residual",
         d_attention_residual.f32(), hidden},
        {sparse ? "sparse_post_normalized" : "post_normalized",
         d_post_normalized.f32(), hidden},
        {sparse ? "sparse_gate" : "gate", d_gate.f32(), intermediate},
        {sparse ? "sparse_up" : "up", d_up.f32(), intermediate},
        {sparse ? "sparse_activated" : "activated", d_activated.f32(),
         intermediate},
        {sparse ? "sparse_mlp_output" : "mlp_output", d_mlp_output.f32(),
         hidden},
        {sparse ? "sparse_layer_output" : "layer_output", d_output.f32(),
         hidden},
    };
    if (chain_mode) {
      for (std::size_t index = 1; index < tail_audits.size(); ++index) {
        tail_audits[index].name = nullptr;
      }
      tail_audits[1].name = "chain_sparse_attention_output";
      tail_audits[2].name = "chain_sparse_attention_residual";
      tail_audits[3].name = "chain_sparse_post_normalized";
      tail_audits[4].name = "chain_sparse_gate";
      tail_audits[5].name = "chain_sparse_up";
      tail_audits[6].name = "chain_sparse_activated";
      tail_audits[7].name = "chain_sparse_mlp_output";
      tail_audits[8].name = "chain_sparse_layer_output";
    }
    audits.insert(audits.end(), tail_audits.begin(), tail_audits.end());
    const auto dense_quality = sparse && !chain_mode
        ? compare(download(d_output.f32(), hidden),
                  load(directory, "layer_output", hidden))
        : Error{0.0, 1.0};
    std::filesystem::create_directories(metrics.parent_path());
    std::ofstream report(metrics);
    std::vector<double> steady_reads = ssd_reads_ms.size() > 1
        ? std::vector<double>(ssd_reads_ms.begin() + 1, ssd_reads_ms.end())
        : ssd_reads_ms;
    std::sort(steady_reads.begin(), steady_reads.end());
    const double steady_read_median = steady_reads.empty()
        ? 0.0
        : steady_reads[steady_reads.size() / 2];
    const double steady_read_mean = steady_reads.empty()
        ? 0.0
        : std::accumulate(steady_reads.begin(), steady_reads.end(), 0.0) /
              static_cast<double>(steady_reads.size());
    report << std::fixed << std::setprecision(9)
           << "{\n  \"version\": \""
           << (chain_mode ? "P1.2e-chained-qwen-ssd-sparse"
                          : sparse ? "P1.2b-real-qwen-ssd-sparse"
                      : "P1.2a-real-qwen-layer")
           << "\",\n"
           << "  \"model\": \"Qwen/Qwen3-0.6B\",\n"
           << "  \"layer\": " << layer_index << ",\n"
           << "  \"prompt_tokens\": " << tokens << ",\n"
           << "  \"attention_tokens\": " << attention_tokens << ",\n"
           << "  \"native_wall_ms\": " << wall_ms << ",\n"
           << "  \"ssd_read_ms\": " << ssd_read_ms << ",\n"
           << "  \"io_repeats\": " << io_repeats << ",\n"
           << "  \"persistent_read_median_ms\": " << steady_read_median
           << ",\n"
           << "  \"post_first_read_mean_ms\": " << steady_read_mean
           << ",\n"
           << "  \"all_ssd_reads_ms\": [";
    for (std::size_t index = 0; index < ssd_reads_ms.size(); ++index) {
      if (index) report << ", ";
      report << ssd_reads_ms[index];
    }
    report << "],\n"
           << "  \"sparse_h2d_ms\": " << sparse_h2d_ms << ",\n"
           << "  \"pipeline_next\": " << (pipeline_next ? "true" : "false")
           << ",\n"
           << "  \"next_read_ms\": " << next_read_ms << ",\n"
           << "  \"exposed_next_read_wait_ms\": "
           << exposed_next_read_wait_ms << ",\n"
           << "  \"next_h2d_ms\": " << next_h2d_ms << ",\n"
           << "  \"next_h2d_start_offset_ms\": "
           << next_h2d_start_offset_ms << ",\n"
           << "  \"sparse_vs_dense_layer_max_abs_error\": "
           << dense_quality.maximum << ",\n"
           << "  \"sparse_vs_dense_layer_cosine\": "
           << dense_quality.cosine << ",\n"
           << "  \"audits\": {\n";
    bool pass = true;
    for (std::size_t index = 0; index < audits.size(); ++index) {
      const auto& audit = audits[index];
      const auto result = compare(download(audit.pointer, audit.elements),
                                  load(directory, audit.name, audit.elements));
      const bool final_output =
          std::string(audit.name).find("layer_output") != std::string::npos;
      pass &= result.cosine >= 0.99999;
      if (final_output) pass &= result.maximum <= 2e-3;
      report << "    \"" << audit.name << "\": {\"max_abs_error\": "
             << result.maximum << ", \"cosine\": " << result.cosine << "}";
      report << (index + 1 == audits.size() ? "\n" : ",\n");
      std::cout << audit.name << "_max_error=" << result.maximum
                << " cosine=" << result.cosine << '\n';
    }
    report << "  },\n  \"pass\": " << (pass ? "true" : "false") << "\n}\n";
    if (!hidden_output_path.empty()) {
      const auto output_hidden = download(d_output.f32(), hidden);
      std::ofstream hidden_stream(hidden_output_path, std::ios::binary);
      hidden_stream.write(reinterpret_cast<const char*>(output_hidden.data()),
                          output_hidden.size() * sizeof(float));
    }
    if (!trace_path.empty()) {
      solidattention::Trace trace;
      std::uint64_t cursor = 0;
      trace.add({"selected FP16 KV fixed read", "NVMe SSD to pinned DRAM",
                 cursor, static_cast<std::uint64_t>(ssd_read_ms * 1000), 1,
                 0, 0, sparse ? static_cast<std::size_t>(512 * 1024) : 0});
      cursor += static_cast<std::uint64_t>(ssd_read_ms * 1000);
      trace.add({"selected KV H2D", "pinned DRAM to VRAM", cursor,
                 static_cast<std::uint64_t>(sparse_h2d_ms * 1000), 2, 0, 0,
                 sparse ? static_cast<std::size_t>(512 * 1024) : 0});
      cursor += static_cast<std::uint64_t>(sparse_h2d_ms * 1000);
      trace.add({"real Qwen sparse layer", "CUDA and cuBLAS", cursor,
                 static_cast<std::uint64_t>(wall_ms * 1000), 3, 0, 0, 0});
      if (pipeline_next) {
        trace.add({"next-layer selected KV read", "NVMe SSD to pinned DRAM",
                   cursor, static_cast<std::uint64_t>(next_read_ms * 1000), 1,
                   0, 1, 512 * 1024});
        trace.add({"next-layer KV H2D", "pinned DRAM to second VRAM slot",
                   cursor + static_cast<std::uint64_t>(
                                next_h2d_start_offset_ms * 1000),
                   static_cast<std::uint64_t>(next_h2d_ms * 1000), 2, 0, 1,
                   512 * 1024});
      }
      trace.write(trace_path.string());
    }
    std::cout << "native_wall_ms=" << wall_ms << "\nmetrics=" << metrics << '\n';
    cublasDestroy_v2(handle);
    cudaStreamDestroy(stream);
    if (next_copy_begin) cudaEventDestroy(next_copy_begin);
    if (next_copy_end) cudaEventDestroy(next_copy_end);
    if (copy_stream) cudaStreamDestroy(copy_stream);
    sparse_reader.reset();
    if (pinned_kv) cudaFreeHost(pinned_kv);
    return pass ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
#endif
