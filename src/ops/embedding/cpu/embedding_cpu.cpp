#include "embedding_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <vector>

template <typename T>
void embedding_(T* out, long* index, T* weight, size_t size_index, std::vector<size_t> shape)
{
    size_t rows = shape[0];
    size_t cols = shape[1];
    for (long i = 0; i < (long)size_index; ++i)
    {
        long row = index[i];
        for (size_t col = 0; col < cols; ++col)
        {
            out[col] = weight[row * rows + col];
        }
    }
}

namespace llaisys::ops::cpu
{
void embedding(std::byte *out, std::byte *index, std::byte *weight, llaisysDataType_t type, size_t size_index, std::vector<size_t> shape)
{
    switch(type)
    {
        case LLAISYS_DTYPE_BF16:
            return embedding_(reinterpret_cast<llaisys::bf16_t*>(out),
             reinterpret_cast<long*>(index),
             reinterpret_cast<llaisys::bf16_t*>(weight),
             size_index,
             shape);
        case LLAISYS_DTYPE_F16:
            return embedding_(reinterpret_cast<llaisys::fp16_t*>(out),
             reinterpret_cast<long*>(index),
             reinterpret_cast<llaisys::fp16_t*>(weight),
             size_index,
             shape);
        case LLAISYS_DTYPE_F32:
            return embedding_(reinterpret_cast<float*>(out),
             reinterpret_cast<long*>(index),
             reinterpret_cast<float*>(weight),
             size_index,
             shape);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);           
    }
}    
}
