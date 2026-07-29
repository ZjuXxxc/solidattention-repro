#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace solidattention {

struct TransferResult {
  double read_ms{};
  double h2d_ms{};
  double kernel_ms{};
  double d2h_ms{};
  std::uint64_t checksum{};
};

struct AttentionProblem {
  const std::uint16_t* interleaved_kv{};
  const float* query{};
  std::size_t tokens{};
  std::size_t query_heads{};
  std::size_t kv_heads{};
  std::size_t head_dim{};
  float scale{};

  std::size_t kv_elements() const {
    return tokens * 2 * kv_heads * head_dim;
  }
  std::size_t query_elements() const { return query_heads * head_dim; }
};

struct AttentionResult {
  double h2d_ms{};
  double kernel_ms{};
  double d2h_ms{};
  std::vector<float> output;
};

class AcceleratorBackend {
 public:
  virtual ~AcceleratorBackend() = default;
  virtual std::string name() const = 0;
  virtual void* allocate_host(std::size_t bytes) = 0;
  virtual void free_host(void* pointer) = 0;
  virtual TransferResult execute(void* pinned_input, std::size_t bytes,
                                 std::uint8_t mask) = 0;
  virtual AttentionResult attention(const AttentionProblem& problem) = 0;
  virtual AttentionResult attention_optimized(
      const AttentionProblem& problem) = 0;
  virtual AttentionResult attention_persistent(
      const AttentionProblem& problem) = 0;
  virtual AttentionResult attention_resident(
      const AttentionProblem& problem, bool audit_output) = 0;
};

}  // namespace solidattention
