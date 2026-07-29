#pragma once

#include "solidattention/backend.hpp"

#include <cstdint>
#include <vector>

namespace solidattention {

std::uint16_t float_to_half(float value);
float half_to_float(std::uint16_t value);
std::vector<float> attention_reference(const AttentionProblem& problem);

}  // namespace solidattention
