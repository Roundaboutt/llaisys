#include "rope_cpu.hpp"
#include "../../../utils.hpp"
#include <vector>
#include <cmath>

template <typename T>
void rope_(T* out, T* in, long* pos_id, float theta, std::vector<size_t> shape)
{
    size_t seq_len = shape[0];
    size_t n_heads = shape[1];
    size_t d = shape[2];
    size_t half_d = d / 2;

    for (size_t s = 0; s < seq_len; ++s)
    {
        float position = static_cast<float>(pos_id[s]);
        for (size_t h = 0; h < n_heads; ++h)
        {
            for (size_t i = 0; i < half_d; ++i)
            {

                float freq = position / std::pow(theta, 2.f * i / static_cast<float>(d));
                float sin_v = std::sin(freq);
                float cos_v = std::cos(freq);

                size_t base_idx = s * n_heads * d + h * d;
                size_t idx_a = base_idx + i;
                size_t idx_b = idx_a + half_d;

                float x_a = llaisys::utils::cast<float>(in[idx_a]);
                float x_b = llaisys::utils::cast<float>(in[idx_b]);

                out[idx_a] = llaisys::utils::cast<T>(x_a * cos_v - x_b * sin_v);
                out[idx_b] = llaisys::utils::cast<T>(x_b * cos_v + x_a * sin_v);
            }
        }
    }

}

namespace llaisys::ops::cpu
{
void rope(std::byte* out, std::byte* in, std::byte* pos_id, float theta, llaisysDataType_t type, std::vector<size_t> shape)
{
    switch (type)
    {
        case LLAISYS_DTYPE_BF16:
            return rope_(
                reinterpret_cast<llaisys::bf16_t*>(out),
                reinterpret_cast<llaisys::bf16_t*>(in),
                reinterpret_cast<long*>(pos_id),
                theta,
                shape
            );
        case LLAISYS_DTYPE_F16:
            return rope_(
                reinterpret_cast<llaisys::fp16_t*>(out),
                reinterpret_cast<llaisys::fp16_t*>(in),
                reinterpret_cast<long*>(pos_id),
                theta,
                shape
            );   
        case LLAISYS_DTYPE_F32:
            return rope_(
                reinterpret_cast<float*>(out),
                reinterpret_cast<float*>(in),
                reinterpret_cast<long*>(pos_id),
                theta,
                shape
            );  
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}    
}