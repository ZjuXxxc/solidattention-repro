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
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

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
    if (device_output_) cudaFree(device_output_);
    if (device_query_) cudaFree(device_query_);
    for (void* pointer : device_kv_) if (pointer) cudaFree(pointer);
    for (void* pointer : host_) if (pointer) cudaFreeHost(pointer);
    if (copy_stream_) cudaStreamDestroy(copy_stream_);
    if (compute_stream_) cudaStreamDestroy(compute_stream_);
  }

  std::vector<void*> host_buffers() const { return {host_[0], host_[1]}; }

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
  std::array<void*, 2> host_{};
  std::array<void*, 2> device_kv_{};
  void* device_query_{};
  void* device_output_{};
  cudaStream_t copy_stream_{};
  cudaStream_t compute_stream_{};
  CUmodule module_{};
  CUfunction attention_{};
  CUfunction ffn_{};
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
  std::vector<float> output;
};

std::vector<std::uint64_t> offsets(std::size_t layer,
                                   std::size_t blocks_per_layer,
                                   std::size_t block_bytes) {
  constexpr std::array<std::size_t, 4> selected{0, 8, 13, 15};
  std::vector<std::uint64_t> result;
  for (const auto block : selected) {
    result.push_back((layer * blocks_per_layer + block) * block_bytes);
  }
  return result;
}

Totals run_serial(CudaPipeline& gpu, solidattention::UringReader& reader,
                  std::size_t steps, std::size_t layers,
                  std::size_t blocks_per_layer, std::size_t block_bytes) {
  Totals totals;
  const auto wall_begin = Clock::now();
  for (std::size_t step = 0; step < steps; ++step) {
    for (std::size_t layer = 0; layer < layers; ++layer) {
      const std::size_t slot = layer % 2;
      totals.read_ms += reader.read_blocks_fixed(
          slot, offsets(layer, blocks_per_layer, block_bytes), block_bytes);
      totals.h2d_ms += gpu.stage_sync(slot);
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
                    std::size_t block_bytes) {
  Totals totals;
  const std::size_t kv_bytes = 4 * block_bytes;
  const auto wall_begin = Clock::now();
  for (std::size_t step = 0; step < steps; ++step) {
    double first_read = reader.read_blocks_fixed(
        0, offsets(0, blocks_per_layer, block_bytes), block_bytes);
    totals.read_ms += first_read;
    const auto first_stage_begin = trace_us(trace);
    const double first_h2d = gpu.stage_sync(0);
    totals.h2d_ms += first_h2d;
    trace.add({"startup H2D", "h2d", first_stage_begin,
               static_cast<std::uint64_t>(first_h2d * 1000), 2, step, 0,
               kv_bytes});

    for (std::size_t layer = 0; layer < layers; ++layer) {
      const std::size_t slot = layer % 2;
      std::uint64_t read_begin_us = 0;
      if (layer + 1 < layers) {
        const std::size_t next = layer + 1;
        read_begin_us = trace_us(trace);
        reader.submit_blocks_fixed(
            next % 2, offsets(next, blocks_per_layer, block_bytes),
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
                   float maximum_error) {
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot write metrics");
  const double executions = static_cast<double>(steps * layers);
  output << std::fixed << std::setprecision(6)
         << "{\n"
         << "  \"version\": \"P0-cuda-dual-pipeline\",\n"
         << "  \"scope\": \"synthetic 28-layer scheduler; real sparse attention; "
            "synthetic FFN window\",\n"
         << "  \"steps\": " << steps << ",\n"
         << "  \"layers\": " << layers << ",\n"
         << "  \"ffn_iterations\": " << ffn_iterations << ",\n"
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
         << "  \"serial_pipeline_max_abs_error\": " << maximum_error << "\n"
         << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string output_dir = "artifacts/cpp-p0";
    std::size_t steps = 8;
    int ffn_iterations = 512;
    for (int index = 1; index < argc; ++index) {
      const std::string argument = argv[index];
      if (argument == "--output" && index + 1 < argc) {
        output_dir = argv[++index];
      } else if (argument == "--steps" && index + 1 < argc) {
        steps = std::stoul(argv[++index]);
      } else if (argument == "--ffn-iterations" && index + 1 < argc) {
        ffn_iterations = std::stoi(argv[++index]);
      } else {
        throw std::runtime_error("unknown argument: " + argument);
      }
    }

    constexpr std::size_t kLayers = 28;
    constexpr std::size_t kBlocksPerLayer = 16;
    constexpr std::size_t kBlockBytes = 128 * 1024;
    constexpr std::size_t kPackedBytes = 4 * kBlockBytes;
    std::filesystem::create_directories(output_dir);
    const std::string store = output_dir + "/pipeline-kv.bin";
    if (!std::filesystem::exists(store) ||
        std::filesystem::file_size(store) !=
            kLayers * kBlocksPerLayer * kBlockBytes) {
      solidattention::create_deterministic_store(
          store, kLayers * kBlocksPerLayer, kBlockBytes);
    }

    CudaPipeline gpu(kPackedBytes, ffn_iterations);
    solidattention::UringReader reader(store, gpu.host_buffers(), kPackedBytes);
    const Totals serial = run_serial(gpu, reader, steps, kLayers,
                                     kBlocksPerLayer, kBlockBytes);
    solidattention::Trace trace;
    const Totals pipeline = run_pipeline(gpu, reader, trace, steps, kLayers,
                                         kBlocksPerLayer, kBlockBytes);
    float maximum_error = 0.0f;
    for (std::size_t index = 0; index < serial.output.size(); ++index) {
      maximum_error = std::max(
          maximum_error, std::abs(serial.output[index] - pipeline.output[index]));
    }
    trace.write(output_dir + "/pipeline-trace.json");
    write_metrics(output_dir + "/pipeline-metrics.json", serial, pipeline,
                  steps, kLayers, ffn_iterations, maximum_error);
    std::cout << std::fixed << std::setprecision(6)
              << "version=P0-cuda-dual-pipeline\n"
              << "serial_wall_ms=" << serial.wall_ms << '\n'
              << "pipeline_wall_ms=" << pipeline.wall_ms << '\n'
              << "speedup=" << serial.wall_ms / pipeline.wall_ms << '\n'
              << "exposed_ssd_wait_ms=" << pipeline.exposed_read_wait_ms << '\n'
              << "ssd_attention_overlap_ms="
              << pipeline.estimated_ssd_attention_overlap_ms << '\n'
              << "h2d_ffn_overlap_ms="
              << pipeline.estimated_h2d_ffn_overlap_ms << '\n'
              << "serial_pipeline_max_abs_error=" << maximum_error << '\n'
              << "trace=" << output_dir << "/pipeline-trace.json\n";
    return maximum_error == 0.0f ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
