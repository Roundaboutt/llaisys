#include "op.hpp"
#include "cpu/rmsnorm_cpu.hpp"


namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "");
    ASSERT(in->shape()[1] == weight->shape()[0], "");
    switch (weight->deviceType())
    {
        case LLAISYS_DEVICE_CPU:
            return cpu::rmsnorm(
                out->data(),
                in->data(),
                weight->data(),
                eps,
                in->dtype(),
                in->shape()
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
