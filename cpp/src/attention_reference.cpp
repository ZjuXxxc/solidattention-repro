#include "solidattention/attention_reference.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace solidattention {

std::uint16_t float_to_half(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t sign = (bits >> 16) & 0x8000;
  int exponent = static_cast<int>((bits >> 23) & 0xff) - 127 + 15;
  std::uint32_t mantissa = bits & 0x7fffff;
  if (exponent <= 0) {
    if (exponent < -10) return static_cast<std::uint16_t>(sign);
    mantissa = (mantissa | 0x800000) >> (1 - exponent);
    return static_cast<std::uint16_t>(
        sign | ((mantissa + 0x1000) >> 13));
  }
  if (exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7c00);
  }
  return static_cast<std::uint16_t>(
      sign | (static_cast<std::uint32_t>(exponent) << 10) |
      ((mantissa + 0x1000) >> 13));
}

float half_to_float(std::uint16_t value) {
  const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000) << 16;
  std::uint32_t exponent = (value >> 10) & 0x1f;
  std::uint32_t mantissa = value & 0x3ff;
  std::uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      int shift = 0;
      while ((mantissa & 0x400) == 0) {
        mantissa <<= 1;
        ++shift;
      }
      mantissa &= 0x3ff;
      bits = sign | ((127 - 14 - shift) << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000 | (mantissa << 13);
  } else {
    bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
  }
  return std::bit_cast<float>(bits);
}

std::vector<float> attention_reference(const AttentionProblem& problem) {
  if (problem.query_heads % problem.kv_heads != 0 || problem.tokens == 0) {
    throw std::runtime_error("invalid GQA attention dimensions");
  }
  const auto groups = problem.query_heads / problem.kv_heads;
  std::vector<float> output(problem.query_elements(), 0.0f);
  std::vector<float> logits(problem.tokens);
  for (std::size_t query_head = 0; query_head < problem.query_heads;
       ++query_head) {
    const auto kv_head = query_head / groups;
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::size_t token = 0; token < problem.tokens; ++token) {
      const auto base =
          token * 2 * problem.kv_heads * problem.head_dim +
          kv_head * problem.head_dim;
      float dot = 0.0f;
      for (std::size_t dim = 0; dim < problem.head_dim; ++dim) {
        dot += problem.query[query_head * problem.head_dim + dim] *
               half_to_float(problem.interleaved_kv[base + dim]);
      }
      logits[token] = dot * problem.scale;
      maximum = std::max(maximum, logits[token]);
    }
    float denominator = 0.0f;
    for (float& logit : logits) {
      logit = std::exp(logit - maximum);
      denominator += logit;
    }
    for (std::size_t token = 0; token < problem.tokens; ++token) {
      const auto value_base =
          token * 2 * problem.kv_heads * problem.head_dim +
          problem.kv_heads * problem.head_dim +
          kv_head * problem.head_dim;
      const float weight = logits[token] / denominator;
      for (std::size_t dim = 0; dim < problem.head_dim; ++dim) {
        output[query_head * problem.head_dim + dim] +=
            weight * half_to_float(problem.interleaved_kv[value_base + dim]);
      }
    }
  }
  return output;
}

}  // namespace solidattention
