#pragma once
#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::cpu
{
void attention(
    std::byte* attn_val, std::byte* q, std::byte* k, std::byte* v,
    float scale,
    std::vector<size_t> q_shape,
    std::vector<size_t> k_shape,
    std::vector<size_t> v_shape,
    llaisysDataType_t type
);
}