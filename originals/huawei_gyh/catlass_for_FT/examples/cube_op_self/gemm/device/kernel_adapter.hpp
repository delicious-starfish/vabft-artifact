#ifndef CATLASS_GEMM_DEVICE_KERNEL_ADAPTER_HPP_SELF
#define CATLASS_GEMM_DEVICE_KERNEL_ADAPTER_HPP_SELF

#include "catlass/catlass.hpp"

#if defined(ENABLE_ASCENDC_DUMP)
#include "catlass/debug.hpp"
#endif

namespace CubeSelf{

/// Generic Catlass kernel template
template <class Operator>
CATLASS_GLOBAL void KernelAdapter(typename Operator::Params params, GM_ADDR ptrDump = nullptr)
{
    Operator op;
#if defined(ENABLE_ASCENDC_DUMP)
    AscendC::InitDump(false, ptrDump, ALL_DUMPSIZE);
#endif
    op(params);
}

template <class Operator>
CATLASS_GLOBAL void KernelAdapter(typename Operator::Params params, uint64_t fftsAddr, GM_ADDR ptrDump = nullptr)
{
    AscendC::SetSyncBaseAddr(fftsAddr);
    Operator op;
#if defined(ENABLE_ASCENDC_DUMP)
    AscendC::InitDump(false, ptrDump, ALL_DUMPSIZE);
#endif
    op(params);
}
} // namespace CubeSelf
#endif