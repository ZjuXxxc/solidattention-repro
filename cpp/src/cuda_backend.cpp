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
  }

  ~CudaBackend() override {
    if (device_input_) cudaFree(device_input_);
    if (device_output_) cudaFree(device_output_);
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

 private:
  void ensure_capacity(std::size_t bytes) {
    if (bytes <= capacity_) return;
    if (device_input_) cudaFree(device_input_);
    if (device_output_) cudaFree(device_output_);
    cuda_check(cudaMalloc(&device_input_, bytes), "cudaMalloc input");
    cuda_check(cudaMalloc(&device_output_, bytes), "cudaMalloc output");
    capacity_ = bytes;
  }

  cudaStream_t stream_{};
  CUmodule module_{};
  CUfunction kernel_{};
  void* device_input_{};
  void* device_output_{};
  std::size_t capacity_{};
};

}  // namespace

std::unique_ptr<AcceleratorBackend> make_cuda_backend() {
  return std::make_unique<CudaBackend>();
}

}  // namespace solidattention
