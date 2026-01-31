#include "swiglu.hpp"
#include "../../../utils.hpp"
#include <vector>
#include <cmath>


template<typename T>
void swiglu_(T* out, T* gate, T* up, size_t numel)
{

    for (size_t i = 0; i < numel; ++i)
    {
        float g = llaisys::utils::cast<float>(gate[i]);
        float u = llaisys::utils::cast<float>(up[i]);

        float swish_gate = g / (1.0f + std::exp(-g));
        float res = swish_gate * u;
        out[i] = llaisys::utils::cast<T>(res);
    }
}


namespace llaisys::ops::cpu
{
void swiglu(std::byte* out, std::byte* gate, std::byte* up, size_t numel, llaisysDataType_t type)
{
    switch(type)
    {
        case LLAISYS_DTYPE_BF16:
            return swiglu_(
                reinterpret_cast<llaisys::bf16_t*>(out),
                reinterpret_cast<llaisys::bf16_t*>(gate),
                reinterpret_cast<llaisys::bf16_t*>(up),
                numel
            );
        case LLAISYS_DTYPE_F16:
            return swiglu_(
                reinterpret_cast<llaisys::fp16_t*>(out),
                reinterpret_cast<llaisys::fp16_t*>(gate),
                reinterpret_cast<llaisys::fp16_t*>(up),
                numel
            );    
        case LLAISYS_DTYPE_F32:
            return swiglu_(
                reinterpret_cast<float*>(out),
                reinterpret_cast<float*>(gate),
                reinterpret_cast<float*>(up),
                numel
            );   
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);          
    }
}
}