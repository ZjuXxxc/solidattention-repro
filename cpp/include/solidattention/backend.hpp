#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace solidattention {

struct TransferResult {
  double read_ms{};
  double h2d_ms{};
  double kernel_ms{};
  double d2h_ms{};
  std::uint64_t checksum{};
};

class AcceleratorBackend {
 public:
  virtual ~AcceleratorBackend() = default;
  virtual std::string name() const = 0;
  virtual void* allocate_host(std::size_t bytes) = 0;
  virtual void free_host(void* pointer) = 0;
  virtual TransferResult execute(void* pinned_input, std::size_t bytes,
                                 std::uint8_t mask) = 0;
};

}  // namespace solidattention
