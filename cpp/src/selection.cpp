#include "solidattention/selection.hpp"
#include "solidattention/attention_reference.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace solidattention {

RepresentativeResult build_local_causal_representatives(
    const float* prompt_queries, const std::uint16_t* interleaved_kv,
    std::size_t tokens, std::size_t query_heads, std::size_t kv_heads,
    std::size_t head_dim, std::size_t block_tokens,
    std::size_t local_window, std::size_t representative_topk) {
  if (!prompt_queries || !interleaved_kv || tokens == 0 ||
      query_heads % kv_heads != 0 || tokens % block_tokens != 0 ||
      local_window == 0 || representative_topk == 0 ||
      representative_topk > block_tokens) {
    throw std::runtime_error("invalid local-causal representative dimensions");
  }
  const auto groups = query_heads / kv_heads;
  const auto blocks = tokens / block_tokens;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  std::vector<float> received(query_heads * tokens, 0.0f);
  std::vector<float> logits(local_window);
  for (std::size_t head = 0; head < query_heads; ++head) {
    const auto kv_head = head / groups;
    for (std::size_t query_position = 0; query_position < tokens;
         ++query_position) {
      const auto start =
          query_position + 1 > local_window
              ? query_position + 1 - local_window
              : 0;
      const auto count = query_position - start + 1;
      float maximum = -std::numeric_limits<float>::infinity();
      for (std::size_t within = 0; within < count; ++within) {
        const auto key_position = start + within;
        const auto key_base =
            key_position * 2 * kv_heads * head_dim + kv_head * head_dim;
        float dot = 0.0f;
        for (std::size_t dim = 0; dim < head_dim; ++dim) {
          dot += prompt_queries[
                     (head * tokens + query_position) * head_dim + dim] *
                 half_to_float(interleaved_kv[key_base + dim]);
        }
        logits[within] = dot * scale;
        maximum = std::max(maximum, logits[within]);
      }
      float denominator = 0.0f;
      for (std::size_t within = 0; within < count; ++within) {
        logits[within] = std::exp(logits[within] - maximum);
        denominator += logits[within];
      }
      for (std::size_t within = 0; within < count; ++within) {
        received[head * tokens + start + within] +=
            logits[within] / denominator;
      }
    }
  }
  RepresentativeResult result;
  result.values.resize(query_heads * blocks * head_dim);
  result.chosen_token_offsets.reserve(
      query_heads * blocks * representative_topk);
  for (std::size_t head = 0; head < query_heads; ++head) {
    const auto kv_head = head / groups;
    for (std::size_t block = 0; block < blocks; ++block) {
      std::vector<std::size_t> within(block_tokens);
      std::iota(within.begin(), within.end(), 0);
      std::stable_sort(
          within.begin(), within.end(),
          [&](std::size_t left, std::size_t right) {
            return received[head * tokens + block * block_tokens + left] >
                   received[head * tokens + block * block_tokens + right];
          });
      for (std::size_t rank = 0; rank < representative_topk; ++rank) {
        result.chosen_token_offsets.push_back(within[rank]);
      }
      for (std::size_t dim = 0; dim < head_dim; ++dim) {
        float value = 0.0f;
        for (std::size_t rank = 0; rank < representative_topk; ++rank) {
          const auto token = block * block_tokens + within[rank];
          const auto key_base =
              token * 2 * kv_heads * head_dim + kv_head * head_dim;
          value += half_to_float(interleaved_kv[key_base + dim]);
        }
        result.values[(head * blocks + block) * head_dim + dim] =
            value / static_cast<float>(representative_topk);
      }
    }
  }
  return result;
}

SelectionResult select_shared_blocks(
    const float* query, const float* representatives,
    std::size_t query_heads, std::size_t blocks, std::size_t head_dim,
    std::size_t budget_blocks, std::size_t init_blocks,
    std::size_t local_blocks) {
  if (!query || !representatives || query_heads == 0 || blocks == 0 ||
      head_dim == 0 || budget_blocks == 0 || budget_blocks > blocks ||
      init_blocks + local_blocks > budget_blocks) {
    throw std::runtime_error("invalid representative selection dimensions");
  }
  SelectionResult result;
  result.mean_head_scores.assign(blocks, 0.0f);
  for (std::size_t head = 0; head < query_heads; ++head) {
    for (std::size_t block = 0; block < blocks; ++block) {
      float dot = 0.0f;
      const auto representative =
          representatives + (head * blocks + block) * head_dim;
      for (std::size_t dim = 0; dim < head_dim; ++dim) {
        dot += query[head * head_dim + dim] * representative[dim];
      }
      result.mean_head_scores[block] +=
          dot / static_cast<float>(query_heads);
    }
  }
  std::vector<bool> chosen(blocks, false);
  for (std::size_t block = 0; block < std::min(init_blocks, blocks); ++block) {
    chosen[block] = true;
  }
  for (std::size_t block = blocks - std::min(local_blocks, blocks);
       block < blocks; ++block) {
    chosen[block] = true;
  }
  std::vector<std::size_t> ranked(blocks);
  std::iota(ranked.begin(), ranked.end(), 0);
  std::stable_sort(ranked.begin(), ranked.end(),
                   [&](std::size_t left, std::size_t right) {
    return result.mean_head_scores[left] > result.mean_head_scores[right];
  });
  std::size_t chosen_count =
      static_cast<std::size_t>(std::count(chosen.begin(), chosen.end(), true));
  for (const auto block : ranked) {
    if (chosen_count == budget_blocks) break;
    if (!chosen[block]) {
      chosen[block] = true;
      ++chosen_count;
    }
  }
  for (std::size_t block = 0; block < blocks; ++block) {
    if (chosen[block]) result.block_ids.push_back(block);
  }
  return result;
}

}  // namespace solidattention
