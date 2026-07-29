#include "solidattention/backend.hpp"

#include <sycl/sycl.hpp>

#include <memory>
#include <stdexcept>
#include <vector>

namespace solidattention {
namespace {

double event_ms(const sycl::event& event) {
  const auto start =
      event.get_profiling_info<sycl::info::event_profiling::command_start>();
  const auto end =
      event.get_profiling_info<sycl::info::event_profiling::command_end>();
  return static_cast<double>(end - start) / 1e6;
}

}  // namespace

class SyclBackend final : public AcceleratorBackend {
 public:
  SyclBackend()
      : queue_(sycl::default_selector_v,
               sycl::property::queue::enable_profiling{}) {}
  ~SyclBackend() override {
    if (persistent_kv_) sycl::free(persistent_kv_, queue_);
    if (persistent_query_) sycl::free(persistent_query_, queue_);
    if (persistent_output_) sycl::free(persistent_output_, queue_);
  }
  std::string name() const override {
    return "oneapi-sycl:" +
           queue_.get_device().get_info<sycl::info::device::name>();
  }
  void* allocate_host(std::size_t bytes) override {
    auto* pointer = sycl::malloc_host<std::uint8_t>(bytes, queue_);
    if (!pointer) throw std::bad_alloc();
    return pointer;
  }
  void free_host(void* pointer) override { sycl::free(pointer, queue_); }
  TransferResult execute(void* pinned_input, std::size_t bytes,
                         std::uint8_t mask) override {
    auto* input = sycl::malloc_device<std::uint8_t>(bytes, queue_);
    auto* output = sycl::malloc_device<std::uint8_t>(bytes, queue_);
    std::vector<std::uint8_t> host_output(bytes);
    auto h2d = queue_.memcpy(input, pinned_input, bytes);
    auto kernel = queue_.submit([&](sycl::handler& handler) {
      handler.depends_on(h2d);
      handler.parallel_for(sycl::range<1>(bytes), [=](sycl::id<1> id) {
        const auto index = id[0];
        output[index] = input[index] ^ mask;
      });
    });
    auto d2h = queue_.submit([&](sycl::handler& handler) {
      handler.depends_on(kernel);
      handler.memcpy(host_output.data(), output, bytes);
    });
    d2h.wait();
    std::uint64_t checksum = 0;
    for (const auto value : host_output) checksum += value;
    sycl::free(input, queue_);
    sycl::free(output, queue_);
    return {.h2d_ms = event_ms(h2d), .kernel_ms = event_ms(kernel),
            .d2h_ms = event_ms(d2h), .checksum = checksum};
  }

  AttentionResult attention(const AttentionProblem& problem) override {
    if (problem.tokens > 256 ||
        problem.query_heads % problem.kv_heads != 0) {
      throw std::runtime_error("unsupported SYCL attention dimensions");
    }
    const auto kv_elements = problem.kv_elements();
    const auto query_elements = problem.query_elements();
    auto* kv = sycl::malloc_device<std::uint16_t>(kv_elements, queue_);
    auto* query = sycl::malloc_device<float>(query_elements, queue_);
    auto* output = sycl::malloc_device<float>(query_elements, queue_);
    if (!kv || !query || !output) throw std::bad_alloc();
    AttentionResult result;
    result.output.resize(query_elements);
    auto kv_h2d = queue_.memcpy(kv, problem.interleaved_kv,
                                kv_elements * sizeof(std::uint16_t));
    auto query_h2d = queue_.memcpy(query, problem.query,
                                   query_elements * sizeof(float));
    const int tokens = static_cast<int>(problem.tokens);
    const int query_heads = static_cast<int>(problem.query_heads);
    const int kv_heads = static_cast<int>(problem.kv_heads);
    const int head_dim = static_cast<int>(problem.head_dim);
    const float scale = problem.scale;
    auto kernel = queue_.submit([&](sycl::handler& handler) {
      handler.depends_on({kv_h2d, query_h2d});
      handler.parallel_for(sycl::range<1>(query_heads), [=](sycl::id<1> id) {
        const int query_head = static_cast<int>(id[0]);
        const int groups = query_heads / kv_heads;
        const int kv_head = query_head / groups;
        float logits[256];
        float maximum = -3.402823466e+38F;
        for (int token = 0; token < tokens; ++token) {
          const std::size_t base =
              static_cast<std::size_t>(token) * 2 * kv_heads * head_dim +
              kv_head * head_dim;
          float dot = 0.0f;
          for (int dim = 0; dim < head_dim; ++dim) {
            const auto key = sycl::bit_cast<sycl::half>(kv[base + dim]);
            dot += query[query_head * head_dim + dim] *
                   static_cast<float>(key);
          }
          logits[token] = dot * scale;
          maximum = sycl::fmax(maximum, logits[token]);
        }
        float denominator = 0.0f;
        for (int token = 0; token < tokens; ++token) {
          logits[token] = sycl::exp(logits[token] - maximum);
          denominator += logits[token];
        }
        for (int dim = 0; dim < head_dim; ++dim) {
          float value = 0.0f;
          for (int token = 0; token < tokens; ++token) {
            const std::size_t base =
                static_cast<std::size_t>(token) * 2 * kv_heads * head_dim +
                kv_heads * head_dim + kv_head * head_dim;
            const auto item = sycl::bit_cast<sycl::half>(kv[base + dim]);
            value += (logits[token] / denominator) *
                     static_cast<float>(item);
          }
          output[query_head * head_dim + dim] = value;
        }
      });
    });
    auto d2h = queue_.submit([&](sycl::handler& handler) {
      handler.depends_on(kernel);
      handler.memcpy(result.output.data(), output,
                     query_elements * sizeof(float));
    });
    d2h.wait();
    result.h2d_ms = event_ms(kv_h2d) + event_ms(query_h2d);
    result.kernel_ms = event_ms(kernel);
    result.d2h_ms = event_ms(d2h);
    sycl::free(kv, queue_);
    sycl::free(query, queue_);
    sycl::free(output, queue_);
    return result;
  }

  AttentionResult attention_optimized(
      const AttentionProblem& problem) override {
    return run_optimized(problem, false);
  }

  AttentionResult attention_persistent(
      const AttentionProblem& problem) override {
    return run_optimized(problem, true);
  }

 private:
  AttentionResult run_optimized(const AttentionProblem& problem,
                                bool persistent) {
    if (problem.tokens > 128 || problem.head_dim > 128 ||
        problem.query_heads % problem.kv_heads != 0) {
      throw std::runtime_error("unsupported C1.1 SYCL attention dimensions");
    }
    const auto kv_elements = problem.kv_elements();
    const auto query_elements = problem.query_elements();
    std::uint16_t* kv = nullptr;
    float* query = nullptr;
    float* output = nullptr;
    if (persistent) {
      ensure_persistent(kv_elements, query_elements);
      kv = persistent_kv_;
      query = persistent_query_;
      output = persistent_output_;
    } else {
      kv = sycl::malloc_device<std::uint16_t>(kv_elements, queue_);
      query = sycl::malloc_device<float>(query_elements, queue_);
      output = sycl::malloc_device<float>(query_elements, queue_);
    }
    if (!kv || !query || !output) throw std::bad_alloc();
    AttentionResult result;
    result.output.resize(query_elements);
    auto kv_h2d = queue_.memcpy(kv, problem.interleaved_kv,
                                kv_elements * sizeof(std::uint16_t));
    auto query_h2d = queue_.memcpy(query, problem.query,
                                   query_elements * sizeof(float));
    const int tokens = static_cast<int>(problem.tokens);
    const int query_heads = static_cast<int>(problem.query_heads);
    const int kv_heads = static_cast<int>(problem.kv_heads);
    const int head_dim = static_cast<int>(problem.head_dim);
    const float scale = problem.scale;
    auto kernel = queue_.submit([&](sycl::handler& handler) {
      handler.depends_on({kv_h2d, query_h2d});
      sycl::local_accessor<float, 1> logits(sycl::range<1>(128), handler);
      handler.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(query_heads * 128),
                            sycl::range<1>(128)),
          [=](sycl::nd_item<1> item) {
            const int query_head = static_cast<int>(item.get_group(0));
            const int lane = static_cast<int>(item.get_local_id(0));
            const int groups = query_heads / kv_heads;
            const int kv_head = query_head / groups;
            float dot = -3.402823466e+38F;
            if (lane < tokens) {
              const std::size_t base =
                  static_cast<std::size_t>(lane) * 2 * kv_heads * head_dim +
                  kv_head * head_dim;
              dot = 0.0f;
              for (int dim = 0; dim < head_dim; ++dim) {
                const auto key =
                    sycl::bit_cast<sycl::half>(kv[base + dim]);
                dot += query[query_head * head_dim + dim] *
                       static_cast<float>(key);
              }
              dot *= scale;
            }
            const float maximum = sycl::reduce_over_group(
                item.get_group(), dot, sycl::maximum<float>());
            const float probability =
                lane < tokens ? sycl::exp(dot - maximum) : 0.0f;
            logits[lane] = probability;
            const float denominator = sycl::reduce_over_group(
                item.get_group(), probability, sycl::plus<float>());
            item.barrier(sycl::access::fence_space::local_space);
            if (lane < head_dim) {
              float value = 0.0f;
              for (int token = 0; token < tokens; ++token) {
                const std::size_t base =
                    static_cast<std::size_t>(token) * 2 * kv_heads * head_dim +
                    kv_heads * head_dim + kv_head * head_dim;
                const auto element =
                    sycl::bit_cast<sycl::half>(kv[base + lane]);
                value += (logits[token] / denominator) *
                         static_cast<float>(element);
              }
              output[query_head * head_dim + lane] = value;
            }
          });
    });
    auto d2h = queue_.submit([&](sycl::handler& handler) {
      handler.depends_on(kernel);
      handler.memcpy(result.output.data(), output,
                     query_elements * sizeof(float));
    });
    d2h.wait();
    result.h2d_ms = event_ms(kv_h2d) + event_ms(query_h2d);
    result.kernel_ms = event_ms(kernel);
    result.d2h_ms = event_ms(d2h);
    if (!persistent) {
      sycl::free(kv, queue_);
      sycl::free(query, queue_);
      sycl::free(output, queue_);
    }
    return result;
  }

  void ensure_persistent(std::size_t kv_elements,
                         std::size_t query_elements) {
    if (kv_elements > persistent_kv_elements_) {
      if (persistent_kv_) sycl::free(persistent_kv_, queue_);
      persistent_kv_ =
          sycl::malloc_device<std::uint16_t>(kv_elements, queue_);
      persistent_kv_elements_ = kv_elements;
    }
    if (query_elements > persistent_query_elements_) {
      if (persistent_query_) sycl::free(persistent_query_, queue_);
      if (persistent_output_) sycl::free(persistent_output_, queue_);
      persistent_query_ = sycl::malloc_device<float>(query_elements, queue_);
      persistent_output_ = sycl::malloc_device<float>(query_elements, queue_);
      persistent_query_elements_ = query_elements;
    }
    if (!persistent_kv_ || !persistent_query_ || !persistent_output_) {
      throw std::bad_alloc();
    }
  }

  sycl::queue queue_;
  std::uint16_t* persistent_kv_{};
  float* persistent_query_{};
  float* persistent_output_{};
  std::size_t persistent_kv_elements_{};
  std::size_t persistent_query_elements_{};
};

std::unique_ptr<AcceleratorBackend> make_sycl_backend() {
  return std::make_unique<SyclBackend>();
}

}  // namespace solidattention
