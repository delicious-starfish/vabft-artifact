#ifndef CATLASS_GEMM_DEVICE_DEVICE_GEMM_HPP_SELF
#define CATLASS_GEMM_DEVICE_DEVICE_GEMM_HPP_SELF

#include <acl/acl.h>
#include "catlass/catlass.hpp"
#include "catlass/status.hpp"
#include "gemm/device/kernel_adapter.hpp"

// catlass/

#if defined(ENABLE_ASCENDC_DUMP)
#include "catlass/debug.hpp"
#endif

namespace CubeSelf::Gemm::Device {

template <class GemmKernel>
class DeviceGemm{
public:
    /// Argument structure: User API
    using Arguments = typename GemmKernel::Arguments;
    /// Argument structure: Kernel API
    using Params = typename GemmKernel::Params;
private:
    /// kernel API parameters object
    Params params_;
public:
    DeviceGemm() {}
    ~DeviceGemm() {}

    /// Access the Params structure
    Params const &params() const
    {
        return params_;
    }

    /// Determines whether the GEMM can execute the given problem.
    Catlass::Status CanImplement(Arguments const &args)
    {
        if(GemmKernel::CanImplement(args)) {
            return Catlass::Status::kSuccess;
        }else{
            return Catlass::Status::kInvalid;
        }
    }

    /// Gets the workspace size
    static size_t GetWorkspaceSize(Arguments const &args)
    {
        size_t workspace_bytes = 0;
        workspace_bytes += GemmKernel::GetWorkspaceSize(args);
        return workspace_bytes;
    }

    /// Initializes GEMM state from arguments
    Catlass::Status Initialize(Arguments const &args, uint8_t *workspace=nullptr, aclrtStream stream=nullptr)
    {
        // Initialize the Params structure
        params_ = GemmKernel::ToUnderlyingArguments(args,workspace);

        return Catlass::Status::kSuccess;
    }

    /// Primary run() entry point API that is static allowing users to create and manage their own params.
    /// Supplied params struct must be construct by calling matmul Kernel::to_underling arguments
    inline Catlass::Status Run(aclrtStream stream, uint32_t blockDim, uint64_t fftsAddr)
    {
#if defined(ENABLE_ASCENDC_DUMP)
        uint8_t *ptrDump{nullptr};
        aclCheck(aclrtMalloc(reinterpret_cast<void **>(&ptrDump), ALL_DUMPSIZE, ACL_MEM_MALLOC_HUGE_FIRST));
        if(fftsAddr == 0){
            CubeSelf::KernelAdapter<GemmKernel><<<blockDim, nullptr, stream>>>(params_, ptrDump);
        }else{
            CubeSelf::KernelAdapter<GemmKernel><<<blockDim, nullptr, stream>>>(params_, fftsAddr, ptrDump);
        }
        aclCheck(aclrtSynchronizeStream(stream));
        Adx::AdumpPrintWorkSpace(ptrDump, ALL_DUMPSIZE, stream, "device_gemm");
        aclCheck(aclrtFree(ptrDump));
#else
        if (fftsAddr == 0) {
            CubeSelf::KernelAdapter<GemmKernel><<<blockDim, nullptr, stream>>>(params_);
        } else {
            CubeSelf::KernelAdapter<GemmKernel><<<blockDim, nullptr, stream>>>(params_, fftsAddr);
        }
#endif
        return Catlass::Status::kSuccess;
    }

    /// Runs the kernel using initialized state
    inline Catlass::Status operator()(aclrtStream stream, uint32_t blockDim)
    {
        return Run(stream, blockDim, 0);
    }

    inline Catlass::Status operator()(aclrtStream stream, uint32_t blockDim, uint64_t fftsAddr)
    {
        return Run(stream, blockDim, fftsAddr);
    }
};
///////////////////////////////////////////////////////////////////////////////////

} // namespace CubeSelf::Gemm::Device
#endif