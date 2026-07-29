#include "solidattention/backend.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace solidattention {
namespace {

void cuda_check(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

void driver_check(CUresult result, const char* operation) {
  if (result != CUDA_SUCCESS) {
    const char* message = nullptr;
    cuGetErrorString(result, &message);
    throw std::runtime_error(std::string(operation) + ": " +
                             (message ? message : "unknown CUDA driver error"));
  }
}

void nvrtc_check(nvrtcResult result, const char* operation) {
  if (result != NVRTC_SUCCESS) {
    throw std::runtime_error(std::string(operation) + ": " +
                             nvrtcGetErrorString(result));
  }
}

constexpr const char* kKernel = R"(
extern "C" __global__ void kv_transform(
    const unsigned char* input, unsigned char* output,
    unsigned long long bytes, unsigned char mask) {
  unsigned long long index =
      (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (index < bytes) output[index] = input[index] ^ mask;
}

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

extern "C" __global__ void sparse_attention(
    const unsigned short* kv, const float* query, float* output,
    int tokens, int query_heads, int kv_heads, int head_dim, float scale) {
  int query_head = blockIdx.x;
  if (query_head >= query_heads || threadIdx.x != 0 || tokens > 256) return;
  int groups = query_heads / kv_heads;
  int kv_head = query_head / groups;
  float logits[256];
  float maximum = -3.402823466e+38F;
  for (int token = 0; token < tokens; ++token) {
    unsigned long long base =
        (unsigned long long)token * 2 * kv_heads * head_dim +
        kv_head * head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
      dot += query[query_head * head_dim + dim] *
             half_to_float(kv[base + dim]);
    }
    logits[token] = dot * scale;
    maximum = fmaxf(maximum, logits[token]);
  }
  float denominator = 0.0f;
  for (int token = 0; token < tokens; ++token) {
    logits[token] = expf(logits[token] - maximum);
    denominator += logits[token];
  }
  for (int dim = 0; dim < head_dim; ++dim) {
    float value = 0.0f;
    for (int token = 0; token < tokens; ++token) {
      unsigned long long base =
          (unsigned long long)token * 2 * kv_heads * head_dim +
          kv_heads * head_dim + kv_head * head_dim;
      value += (logits[token] / denominator) *
               half_to_float(kv[base + dim]);
    }
    output[query_head * head_dim + dim] = value;
  }
}

extern "C" __global__ void sparse_attention_parallel(
    const unsigned short* kv, const float* query, float* output,
    int tokens, int query_heads, int kv_heads, int head_dim, float scale) {
  int query_head = blockIdx.x;
  int lane = threadIdx.x;
  if (query_head >= query_heads || tokens > 128 || head_dim > 128) return;
  int groups = query_heads / kv_heads;
  int kv_head = query_head / groups;
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
    if (lane < stride) reduction[lane] = fmaxf(reduction[lane], reduction[lane + stride]);
    __syncthreads();
  }
  const float maximum = reduction[0];
  const float probability = lane < tokens ? expf(dot - maximum) : 0.0f;
  logits[lane] = probability;
  reduction[lane] = probability;
  __syncthreads();
  for (int stride = 64; stride > 0; stride >>= 1) {
    if (lane < stride) reduction[lane] += reduction[lane + stride];
    __syncthreads();
  }
  const float denominator = reduction[0];
  if (lane < head_dim) {
    float value = 0.0f;
    for (int token = 0; token < tokens; ++token) {
      unsigned long long base =
          (unsigned long long)token * 2 * kv_heads * head_dim +
          kv_heads * head_dim + kv_head * head_dim;
      value += (logits[token] / denominator) *
               half_to_float(kv[base + lane]);
    }
    output[query_head * head_dim + lane] = value;
  }
}
)";

class CudaBackend final : public AcceleratorBackend {
 public:
  CudaBackend() {
    cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "cudaStreamCreate");
    driver_check(cuInit(0), "cuInit");
    nvrtcProgram program{};
    nvrtc_check(nvrtcCreateProgram(&program, kKernel, "kv_transform.cu", 0,
                                   nullptr, nullptr),
                "nvrtcCreateProgram");
    const char* options[] = {"--std=c++14", "--gpu-architecture=compute_89"};
    const auto compile = nvrtcCompileProgram(program, 2, options);
    if (compile != NVRTC_SUCCESS) {
      std::size_t size = 0;
      nvrtcGetProgramLogSize(program, &size);
      std::string log(size, '\0');
      nvrtcGetProgramLog(program, log.data());
      nvrtcDestroyProgram(&program);
      throw std::runtime_error("NVRTC compile failed: " + log);
    }
    std::size_t ptx_size = 0;
    nvrtc_check(nvrtcGetPTXSize(program, &ptx_size), "nvrtcGetPTXSize");
    std::vector<char> ptx(ptx_size);
    nvrtc_check(nvrtcGetPTX(program, ptx.data()), "nvrtcGetPTX");
    nvrtcDestroyProgram(&program);
    driver_check(cuModuleLoadData(&module_, ptx.data()), "cuModuleLoadData");
    driver_check(cuModuleGetFunction(&kernel_, module_, "kv_transform"),
                 "cuModuleGetFunction");
    driver_check(cuModuleGetFunction(&attention_kernel_, module_,
                                     "sparse_attention"),
                 "cuModuleGetFunction sparse_attention");
    driver_check(cuModuleGetFunction(&attention_parallel_kernel_, module_,
                                     "sparse_attention_parallel"),
                 "cuModuleGetFunction sparse_attention_parallel");
  }

  ~CudaBackend() override {
    if (device_input_) cudaFree(device_input_);
    if (device_output_) cudaFree(device_output_);
    if (attention_kv_) cudaFree(attention_kv_);
    if (attention_query_) cudaFree(attention_query_);
    if (attention_output_) cudaFree(attention_output_);
    if (module_) cuModuleUnload(module_);
    if (stream_) cudaStreamDestroy(stream_);
  }

  std::string name() const override { return "cuda-nvrtc"; }

  void* allocate_host(std::size_t bytes) override {
    void* pointer = nullptr;
    cuda_check(cudaHostAlloc(&pointer, bytes, cudaHostAllocDefault),
               "cudaHostAlloc");
    return pointer;
  }

  void free_host(void* pointer) override { cuda_check(cudaFreeHost(pointer), "cudaFreeHost"); }

  TransferResult execute(void* pinned_input, std::size_t bytes,
                         std::uint8_t mask) override {
    ensure_capacity(bytes);
    cudaEvent_t h2d_start{}, h2d_end{}, kernel_end{}, d2h_end{};
    cuda_check(cudaEventCreate(&h2d_start), "cudaEventCreate");
    cuda_check(cudaEventCreate(&h2d_end), "cudaEventCreate");
    cuda_check(cudaEventCreate(&kernel_end), "cudaEventCreate");
    cuda_check(cudaEventCreate(&d2h_end), "cudaEventCreate");
    std::vector<std::uint8_t> output(bytes);
    cudaEventRecord(h2d_start, stream_);
    cuda_check(cudaMemcpyAsync(device_input_, pinned_input, bytes,
                               cudaMemcpyHostToDevice, stream_),
               "cudaMemcpyAsync H2D");
    cudaEventRecord(h2d_end, stream_);
    const unsigned blocks = static_cast<unsigned>((bytes + 255) / 256);
    void* arguments[] = {&device_input_, &device_output_, &bytes, &mask};
    driver_check(cuLaunchKernel(kernel_, blocks, 1, 1, 256, 1, 1, 0,
                                reinterpret_cast<CUstream>(stream_), arguments,
                                nullptr),
                 "cuLaunchKernel");
    cudaEventRecord(kernel_end, stream_);
    cuda_check(cudaMemcpyAsync(output.data(), device_output_, bytes,
                               cudaMemcpyDeviceToHost, stream_),
               "cudaMemcpyAsync D2H");
    cudaEventRecord(d2h_end, stream_);
    cuda_check(cudaEventSynchronize(d2h_end), "cudaEventSynchronize");
    float h2d = 0, kernel = 0, d2h = 0;
    cudaEventElapsedTime(&h2d, h2d_start, h2d_end);
    cudaEventElapsedTime(&kernel, h2d_end, kernel_end);
    cudaEventElapsedTime(&d2h, kernel_end, d2h_end);
    std::uint64_t checksum = 0;
    for (const auto value : output) checksum += value;
    cudaEventDestroy(h2d_start);
    cudaEventDestroy(h2d_end);
    cudaEventDestroy(kernel_end);
    cudaEventDestroy(d2h_end);
    return {.h2d_ms = h2d, .kernel_ms = kernel, .d2h_ms = d2h,
            .checksum = checksum};
  }

  AttentionResult attention(const AttentionProblem& problem) override {
    return run_attention(problem, attention_kernel_, 1);
  }

  AttentionResult attention_optimized(
      const AttentionProblem& problem) override {
    if (problem.tokens > 128 || problem.head_dim > 128) {
      throw std::runtime_error("C1.1 CUDA kernel supports at most 128x128");
    }
    return run_attention(problem, attention_parallel_kernel_, 128);
  }

 private:
  AttentionResult run_attention(const AttentionProblem& problem,
                                CUfunction function,
                                unsigned threads) {
    if (problem.tokens > 256 ||
        problem.query_heads % problem.kv_heads != 0) {
      throw std::runtime_error("unsupported CUDA attention dimensions");
    }
    const std::size_t kv_bytes =
        problem.kv_elements() * sizeof(std::uint16_t);
    const std::size_t query_bytes =
        problem.query_elements() * sizeof(float);
    const std::size_t output_bytes = query_bytes;
    ensure_attention_capacity(kv_bytes, query_bytes, output_bytes);
    cudaEvent_t start{}, h2d_end{}, kernel_end{}, d2h_end{};
    cuda_check(cudaEventCreate(&start), "cudaEventCreate");
    cuda_check(cudaEventCreate(&h2d_end), "cudaEventCreate");
    cuda_check(cudaEventCreate(&kernel_end), "cudaEventCreate");
    cuda_check(cudaEventCreate(&d2h_end), "cudaEventCreate");
    AttentionResult result;
    result.output.resize(problem.query_elements());
    cudaEventRecord(start, stream_);
    cuda_check(cudaMemcpyAsync(attention_kv_, problem.interleaved_kv, kv_bytes,
                               cudaMemcpyHostToDevice, stream_),
               "attention KV H2D");
    cuda_check(cudaMemcpyAsync(attention_query_, problem.query, query_bytes,
                               cudaMemcpyHostToDevice, stream_),
               "attention query H2D");
    cudaEventRecord(h2d_end, stream_);
    int tokens = static_cast<int>(problem.tokens);
    int query_heads = static_cast<int>(problem.query_heads);
    int kv_heads = static_cast<int>(problem.kv_heads);
    int head_dim = static_cast<int>(problem.head_dim);
    float scale = problem.scale;
    void* arguments[] = {&attention_kv_, &attention_query_, &attention_output_,
                         &tokens, &query_heads, &kv_heads, &head_dim, &scale};
    driver_check(cuLaunchKernel(
                     function, query_heads, 1, 1, threads, 1, 1, 0,
                     reinterpret_cast<CUstream>(stream_), arguments, nullptr),
                 "cuLaunchKernel sparse_attention");
    cudaEventRecord(kernel_end, stream_);
    cuda_check(cudaMemcpyAsync(result.output.data(), attention_output_,
                               output_bytes, cudaMemcpyDeviceToHost, stream_),
               "attention output D2H");
    cudaEventRecord(d2h_end, stream_);
    cuda_check(cudaEventSynchronize(d2h_end), "attention synchronize");
    float h2d = 0, kernel = 0, d2h = 0;
    cudaEventElapsedTime(&h2d, start, h2d_end);
    cudaEventElapsedTime(&kernel, h2d_end, kernel_end);
    cudaEventElapsedTime(&d2h, kernel_end, d2h_end);
    result.h2d_ms = h2d;
    result.kernel_ms = kernel;
    result.d2h_ms = d2h;
    cudaEventDestroy(start);
    cudaEventDestroy(h2d_end);
    cudaEventDestroy(kernel_end);
    cudaEventDestroy(d2h_end);
    return result;
  }

  void ensure_capacity(std::size_t bytes) {
    if (bytes <= capacity_) return;
    if (device_input_) cudaFree(device_input_);
    if (device_output_) cudaFree(device_output_);
    cuda_check(cudaMalloc(&device_input_, bytes), "cudaMalloc input");
    cuda_check(cudaMalloc(&device_output_, bytes), "cudaMalloc output");
    capacity_ = bytes;
  }

  void ensure_attention_capacity(std::size_t kv_bytes,
                                 std::size_t query_bytes,
                                 std::size_t output_bytes) {
    if (kv_bytes > attention_kv_capacity_) {
      if (attention_kv_) cudaFree(attention_kv_);
      cuda_check(cudaMalloc(&attention_kv_, kv_bytes), "cudaMalloc attention KV");
      attention_kv_capacity_ = kv_bytes;
    }
    if (query_bytes > attention_query_capacity_) {
      if (attention_query_) cudaFree(attention_query_);
      cuda_check(cudaMalloc(&attention_query_, query_bytes),
                 "cudaMalloc attention query");
      attention_query_capacity_ = query_bytes;
    }
    if (output_bytes > attention_output_capacity_) {
      if (attention_output_) cudaFree(attention_output_);
      cuda_check(cudaMalloc(&attention_output_, output_bytes),
                 "cudaMalloc attention output");
      attention_output_capacity_ = output_bytes;
    }
  }

  cudaStream_t stream_{};
  CUmodule module_{};
  CUfunction kernel_{};
  CUfunction attention_kernel_{};
  CUfunction attention_parallel_kernel_{};
  void* device_input_{};
  void* device_output_{};
  std::size_t capacity_{};
  void* attention_kv_{};
  void* attention_query_{};
  void* attention_output_{};
  std::size_t attention_kv_capacity_{};
  std::size_t attention_query_capacity_{};
  std::size_t attention_output_capacity_{};
};

}  // namespace

std::unique_ptr<AcceleratorBackend> make_cuda_backend() {
  return std::make_unique<CudaBackend>();
}

}  // namespace solidattention
