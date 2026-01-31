#include "linear_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <vector>


template<typename T>
void linear_(T* out, T* in, T* weight, T* bias, std::vector<size_t> in_shape, std::vector<size_t> weight_shape)
{
    size_t M = in_shape[0];
    size_t K = in_shape[1];
    size_t N = weight_shape[0];
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>)
    {
        for (size_t m = 0; m < M; m++)
        {
            for (size_t n = 0; n < N; n++)
            {
                float sum = 0;
                for (size_t k = 0; k < K; k++)
                {
                    sum += llaisys::utils::cast<float>(in[m * K + k]) * llaisys::utils::cast<float>(weight[n * K + k]);
                }
                out[m * N + n] = llaisys::utils::cast<T>(sum + llaisys::utils::cast<float>(bias[n]));
            }
        }        
    }
    else
    {
        for (size_t m = 0; m < M; m++)
        {
            for (size_t n = 0; n < N; n++)
            { 
                T sum = 0;
                for (size_t k = 0; k < K; k++)
                {
                    sum += in[m * K + k] * weight[n * K + k];
                }
                out[m * N + n] = sum + bias[n];
            }
        }        
    }

}


namespace llaisys::ops::cpu
{
void linear(
    std::byte* out, 
    std::byte* in, 
    std::byte* weight, 
    std::byte* bias, 
    llaisysDataType_t type, 
    std::vector<size_t> in_shape,
    std::vector<size_t> weight_shape
)
{
    switch (type)
    {
        case LLAISYS_DTYPE_BF16:
            return linear_(
                reinterpret_cast<llaisys::bf16_t*>(out), 
                reinterpret_cast<llaisys::bf16_t*>(in),
                reinterpret_cast<llaisys::bf16_t*>(weight),
                reinterpret_cast<llaisys::bf16_t*>(bias),
                in_shape,
                weight_shape
            );
        case LLAISYS_DTYPE_F16:
            return linear_(
                reinterpret_cast<llaisys::fp16_t*>(out), 
                reinterpret_cast<llaisys::fp16_t*>(in),
                reinterpret_cast<llaisys::fp16_t*>(weight),
                reinterpret_cast<llaisys::fp16_t*>(bias),
                in_shape,
                weight_shape
            );   
        case LLAISYS_DTYPE_F32:
            return linear_(
                reinterpret_cast<float*>(out), 
                reinterpret_cast<float*>(in),
                reinterpret_cast<float*>(weight),
                reinterpret_cast<float*>(bias),
                in_shape,
                weight_shape
            );
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);                             
    }
}    
}