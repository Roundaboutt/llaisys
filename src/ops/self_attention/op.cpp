#include "op.hpp"
#include "cpu/attention.hpp"


namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    switch (q->deviceType())
    {
        case LLAISYS_DEVICE_CPU:
            return cpu::attention(
                attn_val->data(),
                q->data(),
                k->data(),
                v->data(),
                scale,
                q->shape(),
                k->shape(),
                v->shape(),
                q->dtype()
            );
#ifdef ENABLE_NVIDIA_API
        case LLAISYS_DEVICE_NVIDIA:
            TO_BE_IMPLEMENTED();
            return;
#endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
