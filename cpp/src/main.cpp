#include "solidattention/backend.hpp"
#include "solidattention/trace.hpp"
#include "solidattention/uring_reader.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

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

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string backend_name = "cuda";
    std::string output_dir = "artifacts/cpp";
    std::size_t operations = 64;
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      if (argument == "--backend" && i + 1 < argc) backend_name = argv[++i];
      else if (argument == "--output" && i + 1 < argc) output_dir = argv[++i];
      else if (argument == "--operations" && i + 1 < argc)
        operations = std::stoull(argv[++i]);
      else throw std::runtime_error("unknown/incomplete argument: " + argument);
    }
    std::unique_ptr<solidattention::AcceleratorBackend> backend;
    if (backend_name == "cuda") {
#ifdef SOLIDATTENTION_ENABLE_CUDA
      backend = solidattention::make_cuda_backend();
#else
      throw std::runtime_error("CUDA backend was not built");
#endif
    }
#ifdef SOLIDATTENTION_ENABLE_SYCL
    else if (backend_name == "sycl") backend = solidattention::make_sycl_backend();
#endif
    else throw std::runtime_error("backend was not built: " + backend_name);

    constexpr std::size_t block_bytes = 128 * 1024;
    constexpr std::size_t store_blocks = 128;
    std::filesystem::create_directories(output_dir);
    const std::string store_path = output_dir + "/c0-kv-store.bin";
    solidattention::create_deterministic_store(store_path, store_blocks, block_bytes);
    void* first = backend->allocate_host(block_bytes);
    void* second = backend->allocate_host(block_bytes);
    HostBufferGuard first_guard{*backend, first};
    HostBufferGuard second_guard{*backend, second};
    std::vector<void*> buffers{first, second};
    solidattention::UringReader reader(store_path, buffers, block_bytes);
    solidattention::Trace trace;
    std::vector<double> reads, h2ds, kernels, d2hs;
    std::uint64_t checksum = 0;
    for (std::size_t operation = 0; operation < operations; ++operation) {
      const std::size_t slot = operation % buffers.size();
      const std::size_t block = (operation * 37) % store_blocks;
      const auto read_start = trace.now_us();
      const double read_ms = reader.read_fixed(slot, block * block_bytes);
      trace.add({"io_uring fixed read", "NVMe SSD → pinned DRAM", read_start,
                 static_cast<std::uint64_t>(read_ms * 1000), 1, operation, 0,
                 block_bytes});
      const auto gpu_start = trace.now_us();
      std::uint64_t expected_checksum = 0;
      const auto mask = static_cast<std::uint8_t>(operation);
      const auto* input_bytes = static_cast<const std::uint8_t*>(buffers[slot]);
      for (std::size_t index = 0; index < block_bytes; ++index) {
        expected_checksum += static_cast<std::uint8_t>(input_bytes[index] ^ mask);
      }
      auto result = backend->execute(buffers[slot], block_bytes,
                                     mask);
      if (result.checksum != expected_checksum) {
        throw std::runtime_error("accelerator output differs from CPU reference");
      }
      trace.add({"CUDA/SYCL H2D", "pinned DRAM → VRAM", gpu_start,
                 static_cast<std::uint64_t>(result.h2d_ms * 1000), 2,
                 operation, 0, block_bytes});
      trace.add({"device KV transform", "GPU kernel",
                 gpu_start + static_cast<std::uint64_t>(result.h2d_ms * 1000),
                 static_cast<std::uint64_t>(result.kernel_ms * 1000), 3,
                 operation, 0, block_bytes});
      trace.add({"CUDA/SYCL D2H audit", "VRAM → host reference",
                 gpu_start + static_cast<std::uint64_t>(
                                 (result.h2d_ms + result.kernel_ms) * 1000),
                 static_cast<std::uint64_t>(result.d2h_ms * 1000), 4,
                 operation, 0, block_bytes});
      reads.push_back(read_ms);
      h2ds.push_back(result.h2d_ms);
      kernels.push_back(result.kernel_ms);
      d2hs.push_back(result.d2h_ms);
      checksum ^= result.checksum;
    }
    trace.write(output_dir + "/c0-trace.json");
    std::ofstream metrics(output_dir + "/c0-metrics.json");
    metrics << "{\n"
            << "  \"version\": \"C0\",\n"
            << "  \"backend\": \"" << backend->name() << "\",\n"
            << "  \"io_backend\": \"liburing-registered-fixed-buffer\",\n"
            << "  \"operations\": " << operations << ",\n"
            << "  \"block_bytes\": " << block_bytes << ",\n"
            << "  \"mean_read_ms\": " << mean(reads) << ",\n"
            << "  \"mean_h2d_ms\": " << mean(h2ds) << ",\n"
            << "  \"mean_kernel_ms\": " << mean(kernels) << ",\n"
            << "  \"mean_d2h_ms\": " << mean(d2hs) << ",\n"
            << "  \"cpu_reference_exact\": true,\n"
            << "  \"checksum\": " << checksum << "\n}\n";
    std::cout << "C0 " << backend->name() << " + liburing: "
              << operations << " operations, mean read " << mean(reads)
              << " ms, H2D " << mean(h2ds) << " ms, kernel "
              << mean(kernels) << " ms\n";
  } catch (const std::exception& error) {
    std::cerr << "fatal: " << error.what() << '\n';
    return 1;
  }
}
