#pragma once
#include "llaisys.h"

#include <cstddef>
#include <vector>
namespace llaisys::ops::cpu
{
void embedding(std::byte *out, std::byte *index, std::byte *weight, llaisysDataType_t type, size_t size_index, std::vector<size_t> shape);
}