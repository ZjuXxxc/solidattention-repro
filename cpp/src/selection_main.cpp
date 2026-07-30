#include "solidattention/attention_reference.hpp"
#include "solidattention/backend.hpp"
#include "solidattention/selection.hpp"
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

void write_store(const std::string& path, const void* data, std::size_t bytes) {
  void* aligned = nullptr;
  if (posix_memalign(&aligned, 4096, bytes) != 0) throw std::bad_alloc();
  std::memcpy(aligned, data, bytes);
  const int fd = ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_DIRECT,
                        S_IRUSR | S_IWUSR);
  if (fd < 0 ||
      ::pwrite(fd, aligned, bytes, 0) != static_cast<ssize_t>(bytes)) {
    std::free(aligned);
    if (fd >= 0) ::close(fd);
    throw std::runtime_error("failed to create C2 KV store");
  }
  ::fsync(fd);
  ::close(fd);
  std::free(aligned);
}

double cosine(const std::vector<float>& left,
              const std::vector<float>& right) {
  double dot = 0.0, left_norm = 0.0, right_norm = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    dot += static_cast<double>(left[index]) * right[index];
    left_norm += static_cast<double>(left[index]) * left[index];
    right_norm += static_cast<double>(right[index]) * right[index];
  }
  return dot / std::sqrt(left_norm * right_norm);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string backend_name = "cuda";
    std::string output_dir = "artifacts/cpp-c2";
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      if (argument == "--backend" && i + 1 < argc) backend_name = argv[++i];
      else if (argument == "--output" && i + 1 < argc) output_dir = argv[++i];
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
    else if (backend_name == "sycl") {
      backend = solidattention::make_sycl_backend();
    }
#endif
    else {
      throw std::runtime_error("backend was not built: " + backend_name);
    }

    constexpr std::size_t block_tokens = 32;
    constexpr std::size_t blocks = 16;
    constexpr std::size_t budget_blocks = 4;
    constexpr std::size_t tokens = block_tokens * blocks;
    constexpr std::size_t selected_tokens = block_tokens * budget_blocks;
    constexpr std::size_t query_heads = 16;
    constexpr std::size_t kv_heads = 8;
    constexpr std::size_t head_dim = 128;
    constexpr std::size_t groups = query_heads / kv_heads;
    constexpr std::size_t bytes_per_token =
        2 * kv_heads * head_dim * sizeof(std::uint16_t);
    constexpr std::size_t block_bytes = block_tokens * bytes_per_token;
    constexpr std::size_t selected_bytes = budget_blocks * block_bytes;
    static_assert(block_bytes == 128 * 1024);

    std::filesystem::create_directories(output_dir);
    std::vector<std::uint16_t> full_kv(
        tokens * 2 * kv_heads * head_dim);
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
            full_kv[index] = solidattention::float_to_half(value);
          }
        }
      }
    }
    std::vector<float> prompt_queries(query_heads * tokens * head_dim);
    for (std::size_t query_head = 0; query_head < query_heads; ++query_head) {
      const auto kv_head = query_head / groups;
      for (std::size_t token = 0; token < tokens; ++token) {
        for (std::size_t dim = 0; dim < head_dim; ++dim) {
          const auto key_offset =
              token * 2 * kv_heads * head_dim + kv_head * head_dim + dim;
          prompt_queries[(query_head * tokens + token) * head_dim + dim] =
              solidattention::half_to_float(full_kv[key_offset]) * 0.85f +
              std::cos(static_cast<float>(query_head * 19 + token * 7 +
                                          dim * 3) *
                       0.011f) *
                  0.03f;
        }
      }
    }
    std::vector<float> query(query_heads * head_dim);
    for (std::size_t head = 0; head < query_heads; ++head) {
      std::memcpy(query.data() + head * head_dim,
                  prompt_queries.data() +
                      (head * tokens + tokens - 1) * head_dim,
                  head_dim * sizeof(float));
    }

    solidattention::Trace trace;
    const auto representative_start = trace.now_us();
    const auto representative_host_start = std::chrono::steady_clock::now();
    const auto representative_result =
        solidattention::build_local_causal_representatives(
            prompt_queries.data(), full_kv.data(), tokens, query_heads,
            kv_heads, head_dim, block_tokens, 32, 4);
    const auto representative_host_end = std::chrono::steady_clock::now();
    const double representative_build_ms =
        std::chrono::duration<double, std::milli>(
            representative_host_end - representative_host_start).count();
    std::uint64_t representative_offset_hash = 1469598103934665603ULL;
    for (const auto offset : representative_result.chosen_token_offsets) {
      representative_offset_hash ^= static_cast<std::uint64_t>(offset);
      representative_offset_hash *= 1099511628211ULL;
    }
    trace.add({"C2 InfLLM local-causal representative build", "CPU selection",
               representative_start,
               static_cast<std::uint64_t>(representative_build_ms * 1000),
               0, 0, 0, 0});
    const auto selection_start = trace.now_us();
    const auto host_selection_start = std::chrono::steady_clock::now();
    const auto selection = solidattention::select_shared_blocks(
        query.data(), representative_result.values.data(),
        query_heads, blocks, head_dim,
        budget_blocks, 1, 1);
    const auto host_selection_end = std::chrono::steady_clock::now();
    const double selection_ms =
        std::chrono::duration<double, std::milli>(
            host_selection_end - host_selection_start).count();
    trace.add({"C2 representative selection", "CPU selection",
               selection_start,
               static_cast<std::uint64_t>(selection_ms * 1000), 0, 0, 0, 0});

    const std::string store_path = output_dir + "/c2-kv-store.bin";
    write_store(store_path, full_kv.data(), full_kv.size() * sizeof(std::uint16_t));
    void* pinned = backend->allocate_host(selected_bytes);
    HostBufferGuard pinned_guard{*backend, pinned};
    std::vector<void*> buffers{pinned};
    solidattention::UringReader reader(store_path, buffers, selected_bytes);
    std::vector<std::uint64_t> offsets;
    for (const auto block : selection.block_ids) {
      offsets.push_back(block * block_bytes);
    }
    const auto read_start = trace.now_us();
    const double batch_read_ms =
        reader.read_blocks_fixed(0, offsets, block_bytes);
    trace.add({"C2 batched selected-block read",
               "NVMe SSD → packed pinned DRAM", read_start,
               static_cast<std::uint64_t>(batch_read_ms * 1000), 1, 0, 0,
               selected_bytes});

    const auto* packed = static_cast<const std::uint16_t*>(pinned);
    bool packed_exact = true;
    for (std::size_t slot = 0; slot < selection.block_ids.size(); ++slot) {
      const auto source = full_kv.data() +
                          selection.block_ids[slot] * block_bytes /
                              sizeof(std::uint16_t);
      const auto destination =
          packed + slot * block_bytes / sizeof(std::uint16_t);
      packed_exact &= std::memcmp(source, destination, block_bytes) == 0;
    }
    if (!packed_exact) throw std::runtime_error("C2 packed KV differs from store");

    solidattention::AttentionProblem sparse_problem{
        packed, query.data(), selected_tokens, query_heads, kv_heads, head_dim,
        1.0f / std::sqrt(static_cast<float>(head_dim))};
    const auto sparse_reference =
        solidattention::attention_reference(sparse_problem);
    const auto warmup = backend->attention_persistent(sparse_problem);
    if (warmup.output.size() != sparse_reference.size()) {
      throw std::runtime_error("C2 accelerator warmup returned wrong shape");
    }
    const auto device_start = trace.now_us();
    const auto device = backend->attention_persistent(sparse_problem);
    trace.add({"C2 selected KV H2D", "pinned DRAM → device", device_start,
               static_cast<std::uint64_t>(device.h2d_ms * 1000), 2, 0, 0,
               selected_bytes});
    trace.add({"C2 selected sparse attention", "device kernel",
               device_start +
                   static_cast<std::uint64_t>(device.h2d_ms * 1000),
               static_cast<std::uint64_t>(device.kernel_ms * 1000), 3, 0, 0,
               selected_bytes});
    double max_error = 0.0;
    for (std::size_t index = 0; index < sparse_reference.size(); ++index) {
      max_error = std::max(
          max_error, std::abs(static_cast<double>(sparse_reference[index]) -
                              device.output[index]));
    }
    const double device_cosine = cosine(sparse_reference, device.output);
    solidattention::AttentionProblem dense_problem{
        full_kv.data(), query.data(), tokens, query_heads, kv_heads, head_dim,
        sparse_problem.scale};
    const auto dense_reference =
        solidattention::attention_reference(dense_problem);
    const double selected_dense_cosine =
        cosine(dense_reference, sparse_reference);
    if (max_error > 2e-5 || device_cosine < 0.999999) {
      throw std::runtime_error("C2 device attention differs from sparse oracle");
    }

    trace.write(output_dir + "/c2-trace.json");
    std::ofstream metrics(output_dir + "/c2-metrics.json");
    metrics << "{\n"
            << "  \"version\": \"C2.1\",\n"
            << "  \"backend\": \"" << backend->name() << "\",\n"
            << "  \"representative_source\": \"infllm-local-causal-synthetic-prompt\",\n"
            << "  \"representative_topk\": 4,\n"
            << "  \"representative_local_window\": 32,\n"
            << "  \"representative_build_ms\": " << representative_build_ms
            << ",\n"
            << "  \"representative_token_offset_hash\": "
            << representative_offset_hash << ",\n"
            << "  \"selection_policy\": \"mean-query-head-score+init1+local1\",\n"
            << "  \"blocks\": " << blocks << ",\n"
            << "  \"block_tokens\": " << block_tokens << ",\n"
            << "  \"budget_blocks\": " << budget_blocks << ",\n"
            << "  \"selected_block_ids\": [";
    for (std::size_t index = 0; index < selection.block_ids.size(); ++index) {
      if (index) metrics << ',';
      metrics << selection.block_ids[index];
    }
    metrics << "],\n"
            << "  \"selection_ms\": " << selection_ms << ",\n"
            << "  \"batch_read_ms\": " << batch_read_ms << ",\n"
            << "  \"read_requests\": " << selection.block_ids.size() << ",\n"
            << "  \"packed_bytes\": " << selected_bytes << ",\n"
            << "  \"packed_kv_exact\": true,\n"
            << "  \"accelerator_warmup_operations\": 1,\n"
            << "  \"h2d_ms\": " << device.h2d_ms << ",\n"
            << "  \"attention_kernel_ms\": " << device.kernel_ms << ",\n"
            << "  \"max_error_vs_sparse_cpu\": " << max_error << ",\n"
            << "  \"cosine_vs_sparse_cpu\": " << device_cosine << ",\n"
            << "  \"selected_vs_dense_cosine\": " << selected_dense_cosine
            << "\n}\n";
    std::cout << "C2.1 " << backend->name() << " selected blocks:";
    for (const auto block : selection.block_ids) std::cout << ' ' << block;
    std::cout << ", packed exact, sparse oracle cosine " << device_cosine
              << ", selected-vs-dense cosine " << selected_dense_cosine
              << '\n';
  } catch (const std::exception& error) {
    std::cerr << "fatal: " << error.what() << '\n';
    return 1;
  }
}
