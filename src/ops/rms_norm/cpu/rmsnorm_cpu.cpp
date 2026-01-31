#include "rmsnorm_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <vector>

template <typename T>
void rmsnorm_(T* out, T* in, T* weight, float eps, std::vector<size_t> input_shape)
{
    size_t rows = input_shape[0];
    size_t cols = input_shape[1];
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>)
    {
        for (size_t r = 0; r < rows; ++r)
        {
            float sum = 0;
            for (size_t c = 0; c < cols; ++c)
            {
                sum += llaisys::utils::cast<float>(in[r * cols + c]) * llaisys::utils::cast<float>(in[r * cols + c]);
            }
            sum = 1 / (std::sqrt(sum / static_cast<float>(cols)) + eps);

            for (size_t c = 0; c < cols; ++c)
            {
                out[r * cols + c] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(weight[c]) * llaisys::utils::cast<float>(in[r * cols + c]) * sum);
            }
        }
    }
    else
    {
        for (size_t r = 0; r < rows; ++r)
        {
            T sum = 0;
            for (size_t c = 0; c < cols; ++c)
            {
                sum += in[r * cols + c] * in[r * cols + c];
            }
            sum = 1 / (std::sqrt(sum / static_cast<float>(cols)) + eps);

            for (size_t c = 0; c < cols; ++c)
            {
                out[r * cols + c] = weight[c] * in[r * cols + c] * sum;
            }
        }
    }

}


namespace llaisys::ops::cpu
{
void rmsnorm(
    std::byte* out, 
    std::byte* in, 
    std::byte* weight, 
    float eps, 
    llaisysDataType_t type, 
    std::vector<size_t> input_shape)
{
    switch (type)
    {
        case LLAISYS_DTYPE_BF16:
            return rmsnorm_(
                reinterpret_cast<llaisys::bf16_t*>(out),
                reinterpret_cast<llaisys::bf16_t*>(in),
                reinterpret_cast<llaisys::bf16_t*>(weight),
                eps,
                input_shape
            );
        case LLAISYS_DTYPE_F16:
            return rmsnorm_(
                reinterpret_cast<llaisys::fp16_t*>(out),
                reinterpret_cast<llaisys::fp16_t*>(in),
                reinterpret_cast<llaisys::fp16_t*>(weight),
                eps,
                input_shape
            );
        case LLAISYS_DTYPE_F32:
            return rmsnorm_(
                reinterpret_cast<float*>(out),
                reinterpret_cast<float*>(in),
                reinterpret_cast<float*>(weight),
                eps,
                input_shape
            );
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);                                  
    }
}
}