#include "attention.hpp"
#include "../../../utils.hpp"
#include <vector>
#include <cmath>

template<typename T>
void attention_(
    T* attn_val, T* q, T* k, T* v, float scale, 
    std::vector<size_t> q_shape,
    std::vector<size_t> k_shape,
    std::vector<size_t> v_shape
)
{
    // q: [seqlen, nhead, d]
    // k: [total_len, nkvhead, d]
    // v: [total_len, nkvhead, dv]
    size_t seq_len = q_shape[0];     
    size_t n_heads = q_shape[1];
    size_t d = q_shape[2];
    size_t total_len = k_shape[0];
    size_t n_kvheads = k_shape[1];
    size_t dv = v_shape[2];

    size_t group_size = n_heads / n_kvheads;
    // 用于计算因果掩码的偏移量
    int64_t mask_offset = static_cast<int64_t>(total_len) - static_cast<int64_t>(seq_len);

    for (size_t h = 0; h < n_heads; ++h) {
        size_t kv_h = h / group_size;

        for (size_t i = 0; i < seq_len; ++i) {
            std::vector<float> scores(total_len);
            float max_score = -std::numeric_limits<float>::infinity();

            for (size_t j = 0; j < total_len; ++j) {
                // 因果掩码逻辑：如果 j > i + mask_offset，则该位置不可见
                if (static_cast<int64_t>(j) > static_cast<int64_t>(i) + mask_offset) {
                    scores[j] = -std::numeric_limits<float>::infinity();
                    continue;
                }

                float sum = 0.0f;
                for (size_t k_idx = 0; k_idx < d; ++k_idx) {
                    float q_val = llaisys::utils::cast<float>(q[i * n_heads * d + h * d + k_idx]);
                    float k_val = llaisys::utils::cast<float>(k[j * n_kvheads * d + kv_h * d + k_idx]);
                    sum += q_val * k_val;
                }
                scores[j] = sum * scale;
                if (scores[j] > max_score) max_score = scores[j];
            }

            // Softmax
            float exp_sum = 0.0f;
            for (size_t j = 0; j < total_len; ++j) {
                if (scores[j] == -std::numeric_limits<float>::infinity()) {
                    scores[j] = 0.0f;
                } else {
                    scores[j] = std::exp(scores[j] - max_score);
                    exp_sum += scores[j];
                }
            }
            
            if (exp_sum > 0) {
                for (size_t j = 0; j < total_len; ++j) scores[j] /= exp_sum;
            }

            // Scores * V
            for (size_t d_idx = 0; d_idx < dv; ++d_idx) {
                float out_val = 0.0f;
                for (size_t j = 0; j < total_len; ++j) {
                    if (scores[j] == 0.0f) continue;
                    float v_val = llaisys::utils::cast<float>(v[j * n_kvheads * dv + kv_h * dv + d_idx]);
                    out_val += scores[j] * v_val;
                }
                attn_val[i * n_heads * dv + h * dv + d_idx] = llaisys::utils::cast<T>(out_val);
            }
        }
    }
}


namespace llaisys::ops::cpu
{
void attention(
    std::byte* attn_val, std::byte* q, std::byte* k, std::byte* v,
    float scale,
    std::vector<size_t> q_shape,
    std::vector<size_t> k_shape,
    std::vector<size_t> v_shape,
    llaisysDataType_t type
)    
{
    switch (type)
    {
        case LLAISYS_DTYPE_BF16:
            return attention_(
                reinterpret_cast<llaisys::bf16_t*>(attn_val),
                reinterpret_cast<llaisys::bf16_t*>(q),
                reinterpret_cast<llaisys::bf16_t*>(k),
                reinterpret_cast<llaisys::bf16_t*>(v),
                scale,
                q_shape,
                k_shape,
                v_shape
            );
        case LLAISYS_DTYPE_F16:
            return attention_(
                reinterpret_cast<llaisys::fp16_t*>(attn_val),
                reinterpret_cast<llaisys::fp16_t*>(q),
                reinterpret_cast<llaisys::fp16_t*>(k),
                reinterpret_cast<llaisys::fp16_t*>(v),
                scale,
                q_shape,
                k_shape,
                v_shape
            ); 
        case LLAISYS_DTYPE_F32:
            return attention_(
                reinterpret_cast<float*>(attn_val),
                reinterpret_cast<float*>(q),
                reinterpret_cast<float*>(k),
                reinterpret_cast<float*>(v),
                scale,
                q_shape,
                k_shape,
                v_shape
            );
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);            
    }
}
}