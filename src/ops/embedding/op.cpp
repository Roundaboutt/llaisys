#include "op.hpp"
#include "cpu/embedding_cpu.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // TO_BE_IMPLEMENTED();
    CHECK_SAME_DEVICE(out, index, weight);
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), 
    "Embedding: all tensors must be contiguous.");
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index's type must be int64");


    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());

    switch (weight->deviceType())
    {
        case LLAISYS_DEVICE_CPU:
            return cpu::embedding(out->data(), index->data(), weight->data(), weight->dtype(), index->numel(), weight->shape());
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
