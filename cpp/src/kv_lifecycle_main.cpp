#include "solidattention/uring_reader.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

namespace {
constexpr std::size_t layers = 28, q_heads = 16, kv_heads = 8, head_dim = 128;
constexpr std::size_t block_tokens = 32, local_tokens = 32;
constexpr std::size_t token_values = 2 * kv_heads * head_dim;
constexpr std::size_t block_bytes = block_tokens * token_values * sizeof(std::uint16_t);

struct Tail {
  std::vector<std::uint16_t> kv;
  std::array<std::vector<float>, q_heads> score;
  std::size_t tokens{};
};

void append(Tail& tail, std::size_t layer, std::size_t token) {
  for (std::size_t value = 0; value < token_values; ++value)
    tail.kv.push_back(static_cast<std::uint16_t>((layer * 1009 + token * 37 + value * 13) & 0xffff));
  ++tail.tokens;
  for (std::size_t head = 0; head < q_heads; ++head) {
    tail.score[head].push_back(0.0f);
    for (std::size_t old = 0; old < tail.tokens; ++old)
      tail.score[head][old] += static_cast<float>(((head + 3) * (old + 5) + token) % 97) / 97.0f;
  }
}
}

int main(int argc, char** argv) {
  try {
    std::filesystem::path output = "artifacts/cpp-p1-3b0";
    std::size_t decode_tokens = 96;
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--output" && i + 1 < argc) output = argv[++i];
      else if (arg == "--tokens" && i + 1 < argc) decode_tokens = std::stoul(argv[++i]);
      else throw std::runtime_error("unknown argument: " + arg);
    }
    std::filesystem::create_directories(output);
    const std::size_t initial_blocks = 16;
    const std::size_t capacity_blocks = initial_blocks + (decode_tokens + 31) / 32 + 1;
    const auto store = output / "main-kv-store.bin";
    int fd = ::open(store.c_str(), O_CREAT | O_TRUNC | O_RDWR | O_DIRECT, 0644);
    if (fd < 0) throw std::runtime_error("open main store");
    if (::posix_fallocate(fd, 0, layers * capacity_blocks * block_bytes) != 0)
      throw std::runtime_error("fallocate main store");
    void* aligned = nullptr;
    if (::posix_memalign(&aligned, 4096, block_bytes) != 0) throw std::bad_alloc();
    std::vector<Tail> tails(layers);
    std::vector<std::size_t> sealed(layers), generation(layers);
    std::size_t writes = 0, representative_audits = 0;
    const auto begin = std::chrono::steady_clock::now();
    for (std::size_t layer = 0; layer < layers; ++layer)
      for (std::size_t token = 0; token < local_tokens; ++token) append(tails[layer], layer, token);
    for (std::size_t step = 0; step < decode_tokens; ++step) {
      for (std::size_t layer = 0; layer < layers; ++layer) {
        auto& tail = tails[layer];
        append(tail, layer, local_tokens + step);
        while (tail.tokens >= local_tokens + block_tokens) {
          for (std::size_t head = 0; head < q_heads; ++head) {
            std::array<std::size_t, 4> top{};
            for (std::size_t rank = 0; rank < 4; ++rank) {
              std::size_t best = block_tokens;
              for (std::size_t token = 0; token < block_tokens; ++token) {
                if (std::find(top.begin(), top.begin() + rank, token) != top.begin() + rank) continue;
                if (best == block_tokens || tail.score[head][token] > tail.score[head][best]) best = token;
              }
              top[rank] = best;
            }
            representative_audits += top[0] < block_tokens;
          }
          std::memcpy(aligned, tail.kv.data(), block_bytes);
          const off_t offset = (layer * capacity_blocks + initial_blocks + sealed[layer]) * block_bytes;
          if (::pwrite(fd, aligned, block_bytes, offset) != static_cast<ssize_t>(block_bytes))
            throw std::runtime_error("short main-store write");
          tail.kv.erase(tail.kv.begin(), tail.kv.begin() + block_tokens * token_values);
          for (auto& scores : tail.score) scores.erase(scores.begin(), scores.begin() + block_tokens);
          tail.tokens -= block_tokens;
          ++sealed[layer]; ++generation[layer]; ++writes;
        }
      }
    }
    ::fsync(fd); ::close(fd);
    void* readback = nullptr;
    if (::posix_memalign(&readback, 4096, block_bytes) != 0) throw std::bad_alloc();
    auto reader = std::make_unique<solidattention::UringReader>(
        store.string(), std::vector<void*>{readback}, block_bytes);
    std::size_t verified = 0;
    for (std::size_t layer = 0; layer < layers; ++layer) {
      for (std::size_t block = 0; block < sealed[layer]; ++block) {
        const auto offset = (layer * capacity_blocks + initial_blocks + block) * block_bytes;
        reader->read_fixed(0, offset);
        auto* values = static_cast<std::uint16_t*>(aligned);
        const std::size_t first_token = block * block_tokens;
        for (std::size_t token = 0; token < block_tokens; ++token)
          for (std::size_t value = 0; value < token_values; ++value)
            values[token * token_values + value] = static_cast<std::uint16_t>(
                (layer * 1009 + (first_token + token) * 37 + value * 13) & 0xffff);
        if (std::memcmp(readback, aligned, block_bytes) != 0)
          throw std::runtime_error("sealed block readback mismatch");
        ++verified;
      }
      if (tails[layer].tokens < local_tokens || tails[layer].tokens >= local_tokens + block_tokens)
        throw std::runtime_error("unbounded local tail");
    }
    const double wall = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    std::ofstream metrics(output / "metrics.json");
    metrics << std::fixed << std::setprecision(6)
      << "{\n  \"version\": \"P1.3b.0-native-main-store-lifecycle\",\n"
      << "  \"scope\": \"synthetic FP16 KV bytes; physical native lifecycle\",\n"
      << "  \"layers\": " << layers << ",\n  \"decode_tokens\": " << decode_tokens
      << ",\n  \"sealed_blocks\": " << writes << ",\n  \"verified_main_store_reads\": " << verified
      << ",\n  \"representative_head_audits\": " << representative_audits
      << ",\n  \"resident_tail_tokens_per_layer\": " << tails[0].tokens
      << ",\n  \"selection_generation_per_layer\": " << generation[0]
      << ",\n  \"store_bytes\": " << layers * capacity_blocks * block_bytes
      << ",\n  \"wall_ms\": " << wall << "\n}\n";
    reader.reset();
    std::free(readback); std::free(aligned);
    std::cout << "sealed_blocks=" << writes << " verified=" << verified
              << " tail=" << tails[0].tokens << " generation=" << generation[0] << '\n';
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n'; return 1;
  }
}
