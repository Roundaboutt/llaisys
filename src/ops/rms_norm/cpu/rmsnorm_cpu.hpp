#pragma once
#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::cpu
{
void rmsnorm(
    std::byte* out, 
    std::byte* in, 
    std::byte* weight, 
    float eps, 
    llaisysDataType_t type, 
    std::vector<size_t> input_shape);
}