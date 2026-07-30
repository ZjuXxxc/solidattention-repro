#pragma once

#include <cstddef>
#include <cstdint>
#include <chrono>
#include <liburing.h>
#include <string>
#include <vector>

namespace solidattention {

class UringReader {
 public:
  UringReader(const std::string& path, const std::vector<void*>& buffers,
              std::size_t buffer_bytes);
  ~UringReader();
  UringReader(const UringReader&) = delete;
  UringReader& operator=(const UringReader&) = delete;

  double read_fixed(std::size_t buffer_index, std::uint64_t offset);
  double read_blocks_fixed(std::size_t buffer_index,
                           const std::vector<std::uint64_t>& offsets,
                           std::size_t block_bytes);
  void submit_blocks_fixed(std::size_t buffer_index,
                           const std::vector<std::uint64_t>& offsets,
                           std::size_t block_bytes);
  double wait_blocks_fixed();
  std::size_t buffer_bytes() const { return buffer_bytes_; }

 private:
  int fd_{-1};
  io_uring ring_{};
  std::size_t buffer_bytes_{};
  std::vector<void*> buffers_;
  std::size_t outstanding_requests_{};
  std::size_t outstanding_block_bytes_{};
  std::chrono::steady_clock::time_point outstanding_start_{};
};

void create_deterministic_store(const std::string& path, std::size_t blocks,
                                std::size_t block_bytes);

}  // namespace solidattention
