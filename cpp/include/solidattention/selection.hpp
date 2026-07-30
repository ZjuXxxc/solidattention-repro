#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace solidattention {

struct SelectionResult {
  std::vector<std::size_t> block_ids;
  std::vector<float> mean_head_scores;
};

struct RepresentativeResult {
  std::vector<float> values;
  std::vector<std::size_t> chosen_token_offsets;
};

RepresentativeResult build_local_causal_representatives(
    const float* prompt_queries, const std::uint16_t* interleaved_kv,
    std::size_t tokens, std::size_t query_heads, std::size_t kv_heads,
    std::size_t head_dim, std::size_t block_tokens,
    std::size_t local_window, std::size_t representative_topk);

SelectionResult select_shared_blocks(
    const float* query, const float* representatives,
    std::size_t query_heads, std::size_t blocks, std::size_t head_dim,
    std::size_t budget_blocks, std::size_t init_blocks,
    std::size_t local_blocks);

}  // namespace solidattention
