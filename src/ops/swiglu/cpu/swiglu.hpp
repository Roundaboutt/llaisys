#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu
{
void swiglu(std::byte* out, std::byte* gate, std::byte* up, size_t numel, llaisysDataType_t type);
}