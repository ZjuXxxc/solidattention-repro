#include "solidattention/uring_reader.hpp"

#include <chrono>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace solidattention {
namespace {

void check(int result, const char* operation) {
  if (result < 0) {
    throw std::runtime_error(std::string(operation) + ": " +
                             std::strerror(-result));
  }
}

}  // namespace

UringReader::UringReader(const std::string& path,
                         const std::vector<void*>& buffers,
                         std::size_t buffer_bytes)
    : buffer_bytes_(buffer_bytes) {
  buffers_ = buffers;
  fd_ = ::open(path.c_str(), O_RDONLY | O_DIRECT);
  if (fd_ < 0) {
    throw std::runtime_error("open O_DIRECT: " + std::string(std::strerror(errno)));
  }
  check(io_uring_queue_init(32, &ring_, 0), "io_uring_queue_init");
  std::vector<iovec> vectors;
  vectors.reserve(buffers.size());
  for (void* buffer : buffers) {
    vectors.push_back({buffer, buffer_bytes_});
  }
  check(io_uring_register_buffers(&ring_, vectors.data(), vectors.size()),
        "io_uring_register_buffers");
  check(io_uring_register_files(&ring_, &fd_, 1), "io_uring_register_files");
}

UringReader::~UringReader() {
  io_uring_unregister_files(&ring_);
  io_uring_unregister_buffers(&ring_);
  io_uring_queue_exit(&ring_);
  if (fd_ >= 0) ::close(fd_);
}

double UringReader::read_fixed(std::size_t buffer_index, std::uint64_t offset) {
  auto* sqe = io_uring_get_sqe(&ring_);
  if (sqe == nullptr) throw std::runtime_error("io_uring SQ is full");
  io_uring_prep_read_fixed(sqe, 0, buffers_.at(buffer_index), buffer_bytes_,
                           offset, buffer_index);
  sqe->flags |= IOSQE_FIXED_FILE;
  const auto start = std::chrono::steady_clock::now();
  check(io_uring_submit(&ring_), "io_uring_submit");
  io_uring_cqe* cqe = nullptr;
  check(io_uring_wait_cqe(&ring_, &cqe), "io_uring_wait_cqe");
  const int result = cqe->res;
  io_uring_cqe_seen(&ring_, cqe);
  check(result, "fixed-buffer read");
  if (static_cast<std::size_t>(result) != buffer_bytes_) {
    throw std::runtime_error("short fixed-buffer read");
  }
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

double UringReader::read_blocks_fixed(
    std::size_t buffer_index, const std::vector<std::uint64_t>& offsets,
    std::size_t block_bytes) {
  submit_blocks_fixed(buffer_index, offsets, block_bytes);
  return wait_blocks_fixed();
}

void UringReader::submit_blocks_fixed(
    std::size_t buffer_index, const std::vector<std::uint64_t>& offsets,
    std::size_t block_bytes) {
  if (outstanding_requests_ != 0) {
    throw std::runtime_error("an io_uring block batch is already outstanding");
  }
  if (offsets.empty() || offsets.size() * block_bytes > buffer_bytes_) {
    throw std::runtime_error("invalid fixed-buffer block batch");
  }
  auto* destination = static_cast<std::uint8_t*>(buffers_.at(buffer_index));
  for (std::size_t request = 0; request < offsets.size(); ++request) {
    auto* sqe = io_uring_get_sqe(&ring_);
    if (sqe == nullptr) throw std::runtime_error("io_uring SQ is full");
    io_uring_prep_read_fixed(sqe, 0, destination + request * block_bytes,
                             block_bytes, offsets[request], buffer_index);
    sqe->flags |= IOSQE_FIXED_FILE;
    io_uring_sqe_set_data64(sqe, request);
  }
  outstanding_requests_ = offsets.size();
  outstanding_block_bytes_ = block_bytes;
  outstanding_start_ = std::chrono::steady_clock::now();
  check(io_uring_submit(&ring_), "io_uring_submit block batch");
}

double UringReader::wait_blocks_fixed() {
  if (outstanding_requests_ == 0) {
    throw std::runtime_error("no io_uring block batch is outstanding");
  }
  for (std::size_t completed = 0; completed < outstanding_requests_; ++completed) {
    io_uring_cqe* cqe = nullptr;
    check(io_uring_wait_cqe(&ring_, &cqe), "io_uring_wait_cqe block batch");
    const int result = cqe->res;
    io_uring_cqe_seen(&ring_, cqe);
    check(result, "fixed-buffer block read");
    if (static_cast<std::size_t>(result) != outstanding_block_bytes_) {
      throw std::runtime_error("short fixed-buffer block read");
    }
  }
  const auto end = std::chrono::steady_clock::now();
  const double elapsed =
      std::chrono::duration<double, std::milli>(end - outstanding_start_).count();
  outstanding_requests_ = 0;
  outstanding_block_bytes_ = 0;
  return elapsed;
}

void create_deterministic_store(const std::string& path, std::size_t blocks,
                                std::size_t block_bytes) {
  const int fd = ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_DIRECT,
                        S_IRUSR | S_IWUSR);
  if (fd < 0) {
    throw std::runtime_error("create store: " + std::string(std::strerror(errno)));
  }
  void* raw = nullptr;
  if (posix_memalign(&raw, 4096, block_bytes) != 0) {
    ::close(fd);
    throw std::bad_alloc();
  }
  auto* bytes = static_cast<std::uint8_t*>(raw);
  for (std::size_t block = 0; block < blocks; ++block) {
    for (std::size_t index = 0; index < block_bytes; ++index) {
      bytes[index] = static_cast<std::uint8_t>((block * 17 + index * 13) & 0xff);
    }
    const auto written = ::pwrite(fd, bytes, block_bytes, block * block_bytes);
    if (written != static_cast<ssize_t>(block_bytes)) {
      std::free(raw);
      ::close(fd);
      throw std::runtime_error("short O_DIRECT write");
    }
  }
  ::fsync(fd);
  std::free(raw);
  ::close(fd);
}

}  // namespace solidattention
