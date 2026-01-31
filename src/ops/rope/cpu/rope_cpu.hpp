#pragma once
#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::cpu
{
void rope(std::byte* out, 
    std::byte* in, 
    std::byte* pos_id, 
    float theta, 
    llaisysDataType_t type, 
    std::vector<size_t> shape);    
}