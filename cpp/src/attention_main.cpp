#include "solidattention/attention_reference.hpp"
#include "solidattention/backend.hpp"
#include "solidattention/trace.hpp"
#include "solidattention/uring_reader.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace solidattention {
#ifdef SOLIDATTENTION_ENABLE_CUDA
std::unique_ptr<AcceleratorBackend> make_cuda_backend();
#endif
#ifdef SOLIDATTENTION_ENABLE_SYCL
std::unique_ptr<AcceleratorBackend> make_sycl_backend();
#endif
}  // namespace solidattention

namespace {

struct HostBufferGuard {
  solidattention::AcceleratorBackend& backend;
  void* pointer;
  ~HostBufferGuard() { backend.free_host(pointer); }
};

double mean(const std::vector<double>& values) {
  return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

void write_store(const std::string& path, const void* data, std::size_t bytes) {
  void* aligned = nullptr;
  if (posix_memalign(&aligned, 4096, bytes) != 0) throw std::bad_alloc();
  std::memcpy(aligned, data, bytes);
  const int fd = ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_DIRECT,
                        S_IRUSR | S_IWUSR);
  if (fd < 0 || ::pwrite(fd, aligned, bytes, 0) !=
                    static_cast<ssize_t>(bytes)) {
    std::free(aligned);
    if (fd >= 0) ::close(fd);
    throw std::runtime_error("failed to create C1 O_DIRECT KV store");
  }
  ::fsync(fd);
  ::close(fd);
  std::free(aligned);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string backend_name = "cuda";
    std::string output_dir = "artifacts/cpp-c1";
    std::size_t operations = 64;
    bool optimized = false;
    bool persistent = false;
    bool resident = false;
    std::size_t audit_every = 16;
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      if (argument == "--backend" && i + 1 < argc) backend_name = argv[++i];
      else if (argument == "--output" && i + 1 < argc) output_dir = argv[++i];
      else if (argument == "--operations" && i + 1 < argc)
        operations = std::stoull(argv[++i]);
      else if (argument == "--optimized")
        optimized = true;
      else if (argument == "--persistent") {
        optimized = true;
        persistent = true;
      }
      else if (argument == "--resident") {
        optimized = true;
        persistent = true;
        resident = true;
      }
      else if (argument == "--audit-every" && i + 1 < argc)
        audit_every = std::stoull(argv[++i]);
      else throw std::runtime_error("unknown/incomplete argument: " + argument);
    }
    if (audit_every == 0) throw std::runtime_error("--audit-every must be positive");
    std::unique_ptr<solidattention::AcceleratorBackend> backend;
    if (backend_name == "cuda") {
#ifdef SOLIDATTENTION_ENABLE_CUDA
      backend = solidattention::make_cuda_backend();
#else
      throw std::runtime_error("CUDA backend was not built");
#endif
    }
#ifdef SOLIDATTENTION_ENABLE_SYCL
    else if (backend_name == "sycl") {
      backend = solidattention::make_sycl_backend();
    }
#endif
    else {
      throw std::runtime_error("backend was not built: " + backend_name);
    }

    constexpr std::size_t tokens = 128;
    constexpr std::size_t query_heads = 16;
    constexpr std::size_t kv_heads = 8;
    constexpr std::size_t head_dim = 128;
    constexpr float scale = 0.08838834764831845f;  // 1/sqrt(128)
    constexpr std::size_t kv_elements =
        tokens * 2 * kv_heads * head_dim;
    constexpr std::size_t kv_bytes = kv_elements * sizeof(std::uint16_t);
    static_assert(kv_bytes == 4 * 128 * 1024);
    std::filesystem::create_directories(output_dir);
    void* pinned_kv = backend->allocate_host(kv_bytes);
    HostBufferGuard kv_guard{*backend, pinned_kv};
    auto* kv = static_cast<std::uint16_t*>(pinned_kv);
    for (std::size_t token = 0; token < tokens; ++token) {
      for (std::size_t half = 0; half < 2; ++half) {
        for (std::size_t head = 0; head < kv_heads; ++head) {
          for (std::size_t dim = 0; dim < head_dim; ++dim) {
            const auto index =
                token * 2 * kv_heads * head_dim +
                half * kv_heads * head_dim + head * head_dim + dim;
            const float value =
                std::sin(static_cast<float>(token * 17 + head * 13 +
                                            dim * 3 + half * 19) *
                         0.013f) *
                0.25f;
            kv[index] = solidattention::float_to_half(value);
          }
        }
      }
    }
    const std::string store_path = output_dir + "/c1-attention-kv.bin";
    write_store(store_path, pinned_kv, kv_bytes);
    std::memset(pinned_kv, 0, kv_bytes);
    std::vector<void*> registered_buffers{pinned_kv};
    solidattention::UringReader reader(store_path, registered_buffers, kv_bytes);
    std::vector<float> query(query_heads * head_dim);
    for (std::size_t index = 0; index < query.size(); ++index) {
      query[index] = std::cos(static_cast<float>(index * 7) * 0.009f) * 0.2f;
    }
    solidattention::AttentionProblem problem{
        kv, query.data(), tokens, query_heads, kv_heads, head_dim, scale};
    const double reference_read_ms = reader.read_fixed(0, 0);
    const auto reference = solidattention::attention_reference(problem);
    const auto warmup =
        resident ? backend->attention_resident(problem, false)
        : persistent ? backend->attention_persistent(problem)
        : optimized ? backend->attention_optimized(problem)
                    : backend->attention(problem);
    if (!resident && warmup.output.size() != reference.size()) {
      throw std::runtime_error("attention warmup returned wrong output size");
    }

    solidattention::Trace trace;
    std::vector<double> reads, h2ds, kernels, d2hs, device_calls;
    double maximum_absolute_error = 0.0;
    double dot = 0.0, reference_norm = 0.0, output_norm = 0.0;
    std::size_t audit_operations = 0;
    for (std::size_t operation = 0; operation < operations; ++operation) {
      const auto read_start = trace.now_us();
      const double read_ms = reader.read_fixed(0, 0);
      trace.add({"C1 four-block fixed read", "NVMe SSD → pinned DRAM",
                 read_start, static_cast<std::uint64_t>(read_ms * 1000), 1,
                 operation, 0, kv_bytes});
      const auto device_start = trace.now_us();
      const auto wall_start = std::chrono::steady_clock::now();
      const bool audit_output =
          !resident || operation % audit_every == 0 ||
          operation + 1 == operations;
      const auto result =
          resident ? backend->attention_resident(problem, audit_output)
          : persistent ? backend->attention_persistent(problem)
          : optimized ? backend->attention_optimized(problem)
                      : backend->attention(problem);
      const auto wall_end = std::chrono::steady_clock::now();
      const double device_call_ms =
          std::chrono::duration<double, std::milli>(wall_end - wall_start)
              .count();
      trace.add({"C1 FP16 KV + FP32 query H2D", "pinned DRAM → device",
                 device_start,
                 static_cast<std::uint64_t>(result.h2d_ms * 1000), 2,
                 operation, 0, kv_bytes + query.size() * sizeof(float)});
      trace.add({resident ? "C1.3 device-resident GQA attention"
                 : persistent ? "C1.2 persistent parallel GQA attention"
                 : optimized ? "C1.1 parallel GQA sparse attention"
                           : "C1 serial GQA sparse attention",
                 "device kernel",
                 device_start +
                     static_cast<std::uint64_t>(result.h2d_ms * 1000),
                 static_cast<std::uint64_t>(result.kernel_ms * 1000), 3,
                 operation, 0, kv_bytes});
      if (audit_output) {
        trace.add({"C1.3 sampled output audit D2H", "device → host",
                   device_start + static_cast<std::uint64_t>(
                                      (result.h2d_ms + result.kernel_ms) * 1000),
                   static_cast<std::uint64_t>(result.d2h_ms * 1000), 4,
                   operation, 0, result.output.size() * sizeof(float)});
      }
      reads.push_back(read_ms);
      h2ds.push_back(result.h2d_ms);
      kernels.push_back(result.kernel_ms);
      d2hs.push_back(result.d2h_ms);
      device_calls.push_back(device_call_ms);
      if (audit_output) {
        ++audit_operations;
        dot = reference_norm = output_norm = 0.0;
        for (std::size_t index = 0; index < reference.size(); ++index) {
          maximum_absolute_error =
              std::max(maximum_absolute_error,
                       std::abs(static_cast<double>(reference[index]) -
                                result.output[index]));
          dot += static_cast<double>(reference[index]) * result.output[index];
          reference_norm +=
              static_cast<double>(reference[index]) * reference[index];
          output_norm +=
              static_cast<double>(result.output[index]) * result.output[index];
        }
      }
    }
    const double cosine = dot / std::sqrt(reference_norm * output_norm);
    if (!std::isfinite(cosine) || maximum_absolute_error > 2e-5 ||
        cosine < 0.999999) {
      throw std::runtime_error("C1 device attention differs from CPU reference");
    }
    trace.write(output_dir + "/c1-trace.json");
    std::ofstream metrics(output_dir + "/c1-metrics.json");
    metrics << "{\n"
            << "  \"version\": \""
            << (resident ? "C1.3"
                : persistent ? "C1.2"
                : optimized ? "C1.1" : "C1") << "\",\n"
            << "  \"backend\": \"" << backend->name() << "\",\n"
            << "  \"io_backend\": \"liburing-registered-fixed-buffer\",\n"
            << "  \"kv_dtype\": \"fp16\",\n"
            << "  \"query_accumulation_dtype\": \"fp32\",\n"
            << "  \"tokens\": " << tokens << ",\n"
            << "  \"query_heads\": " << query_heads << ",\n"
            << "  \"kv_heads\": " << kv_heads << ",\n"
            << "  \"head_dim\": " << head_dim << ",\n"
            << "  \"operations\": " << operations << ",\n"
            << "  \"warmup_operations\": 1,\n"
            << "  \"parallel_kernel\": "
            << (optimized ? "true" : "false") << ",\n"
            << "  \"persistent_device_resources\": "
            << (persistent ? "true" : "false") << ",\n"
            << "  \"device_resident_output\": "
            << (resident ? "true" : "false") << ",\n"
            << "  \"audit_every\": " << (resident ? audit_every : 1) << ",\n"
            << "  \"audit_operations\": " << audit_operations << ",\n"
            << "  \"kv_bytes\": " << kv_bytes << ",\n"
            << "  \"reference_read_ms\": " << reference_read_ms << ",\n"
            << "  \"mean_read_ms\": " << mean(reads) << ",\n"
            << "  \"mean_h2d_ms\": " << mean(h2ds) << ",\n"
            << "  \"mean_kernel_ms\": " << mean(kernels) << ",\n"
            << "  \"mean_d2h_ms\": " << mean(d2hs) << ",\n"
            << "  \"mean_device_call_wall_ms\": " << mean(device_calls)
            << ",\n"
            << "  \"max_absolute_error\": " << maximum_absolute_error << ",\n"
            << "  \"cosine_vs_cpu\": " << cosine << "\n}\n";
    std::cout << (resident ? "C1.3 "
                  : persistent ? "C1.2 "
                  : optimized ? "C1.1 " : "C1 ")
              << backend->name()
              << " GQA attention: max error "
              << maximum_absolute_error << ", cosine " << cosine
              << ", mean kernel " << mean(kernels) << " ms\n";
  } catch (const std::exception& error) {
    std::cerr << "fatal: " << error.what() << '\n';
    return 1;
  }
}
