#include "op.hpp"
#include "cpu/linear_cpu.hpp"


namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight, bias);

    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous() && bias->isContiguous(), "Linear: all tensors must be contiguous");
    ASSERT((in->shape()[1] == weight->shape()[1]) && (bias->shape()[0] == weight->shape()[0]), "Linear: 维度不匹配");
    switch (in->deviceType())
    {
        case LLAISYS_DEVICE_CPU:
            return cpu::linear(out->data(), in->data(), weight->data(), bias->data(), in->dtype(), in->shape(), weight->shape());
#ifdef ENABLE_NVIDIA_API
        case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();    
#endif        
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
