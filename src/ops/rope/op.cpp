#include "op.hpp"
#include "cpu/rope_cpu.hpp"


namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    ASSERT(out->shape() == in->shape(), "");
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "");
    ASSERT(pos_ids->shape()[0] == in->shape()[0], "");

    switch (in->deviceType())
    {
        case LLAISYS_DEVICE_CPU:
            return cpu::rope(out->data(), in->data(), pos_ids->data(), theta, in->dtype(),in->shape());
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
