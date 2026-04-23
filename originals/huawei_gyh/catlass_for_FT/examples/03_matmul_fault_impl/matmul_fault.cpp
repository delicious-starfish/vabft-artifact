/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// By setting the K_MAX_SHAPE_DIM macro, the dimension of the AscendC Tensor's ShapeInfo is configured to 0, 
// optimizing stack space. If you need to use the ShapeInfo of the AscendC Tensor, please undefine this macro.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <iostream>
#include <vector>
#include <iomanip>

#include "helper.hpp"
#include "golden.hpp"
#include "fp16_t.h"

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/matmul_fault_tolerance.hpp"
#include "catlass/gemm/kernel/grouped_matmul_slice_m.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemv/tile/tile_fault_copy.hpp"
#include "catlass/gemv/tile/tile_fault_vmad.hpp"
#include "catlass/gemv/tile/tile_fault_sum.hpp"
#include "catlass/gemv/tile/tile_vmuls.hpp"
#include "catlass/gemv/block/block_gemv.hpp"
#include "catlass/gemv/kernel/kernel_gemv_aiv.hpp"
#include "catlass/layout/layout.hpp"

#include "catlass/status.hpp"
#include "catlass/gemm/device/device_gemm.hpp"

using namespace Catlass;
using fp16_t = op::fp16_t;

struct Options {
    const std::string HELPER = "03_matmul_fault m n k check_sum_type [device_id]";

    GemmCoord problemShape{4096, 4096, 4096};
    int32_t checkSumType = 2;
    int32_t deviceId{0};

    Options() = default;

    int Parse(int argc, const char **argv)
    {
        enum ArgsIndex {
            M_INDEX = 1,
            N_INDEX,
            K_INDEX,
            CHECKSUMTYPE_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        if (argc > ARGS_MAX || argc <= CHECKSUMTYPE_INDEX) {
            std::cerr << HELPER << std::endl;
            return -1;
        }

        problemShape.m() = std::atoi(argv[M_INDEX]);
        problemShape.n() = std::atoi(argv[N_INDEX]);
        problemShape.k() = std::atoi(argv[K_INDEX]);
        checkSumType = std::atoi(argv[CHECKSUMTYPE_INDEX]);
        if (argc == ARGS_MAX) {
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        }
        return 0;
    }
};

void Run(Options const &options)
{
    std::cout << std::fixed;
    std::cout << std::setprecision(6);

    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();

    // Compute the length of each matrix and the size of each buffer
    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenC = static_cast<size_t>(m) * n;
    size_t lenD = lenC;

    size_t sizeA = lenA * sizeof(float);
    size_t sizeB = lenB * sizeof(float);
    size_t sizeC = lenC * sizeof(float);
    size_t sizeD = lenD * sizeof(float);

    // Define the layout of each matrix
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;       // must be rowmajor
    using LayoutX = layout::VectorLayout;
    using LayoutY = layout::VectorLayout;
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m, n};

    // Prepare input data A, B, and X
    std::vector<float> hostA(lenA);
    std::vector<float> hostB(lenB);
    std::vector<float> hostC(lenC); 
    std::vector<float> hostD(lenC, 0.0f);
    golden::FillRandomData<float>(hostA, -5.0f, 5.0f);
    golden::FillRandomData<float>(hostB, -5.0f, 5.0f);
    // golden::FillNormalData<float>(hostA, 0.0f, 10.0f);
    // golden::FillNormalData<float>(hostB, 0.0f, 10.0f);

    // Allocate device memory and copy data from host to device
    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *deviceD{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceD), sizeD, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceD, sizeD, hostD.data(), sizeD, ACL_MEMCPY_HOST_TO_DEVICE));

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    // Define ArchTag
    using ArchTag = Arch::AtlasA2;

    // Block level, define BlockMmad
    constexpr bool enableUnitFlag = false;
    using MmadDispatchPolicy = Gemm::MmadAtlasA2Pingpong<enableUnitFlag>;
    using L1TileShape = GemmShape<128, 128, 128>;
    using L0TileShape = GemmShape<128, 128, 64>;
    using AType = Gemm::GemmType<float, LayoutA>;
    using BType = Gemm::GemmType<float, LayoutB>;
    using CType = Gemm::GemmType<float, LayoutC>;
    using BlockMmad = Gemm::Block::BlockMmadFault<MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

    // Block level, define sum and Gemv
    using GemvDispatchPolicy = Gemm::GemvAtlasA2;
    using XType = Gemm::GemmType<float, LayoutX>;
    using YType = Gemm::GemmType<float, LayoutY>;
    using TileVmuls = Gemv::Tile::TileVmuls<ArchTag, XType>;

    using UBTileShape = GemvShape<L1TileShape::M, L1TileShape::K>;
    using TileFaultCopy = Gemv::Tile::TileFaultCopyGemvAiv<ArchTag, CType, XType, YType>;
    using TileFaultVmad = Gemv::Tile::TileFaultVmad<ArchTag, CType, XType, YType>;
    using TileFaultSum = Gemv::Tile::TileFaultSum<ArchTag, CType, YType>;
    using BlockSumGemv = Gemv::Block::BlockSumGemv<GemvDispatchPolicy, UBTileShape, CType, XType, YType, void, TileFaultCopy, TileFaultVmad, TileFaultSum, TileVmuls>;

    if (m > n) {
        // Define BlockScheduler
        // Swizzle offset is 3 and direction is 0.
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
        // Kernel level
        using MatmulKernel = Gemm::Kernel::MatmulFaultTolerance<BlockMmad, BlockSumGemv, BlockScheduler>;
        // Prepare params
        typename MatmulKernel::Arguments arguments{
            options.problemShape, deviceA, deviceB, deviceC, deviceD, options.checkSumType, aicCoreNum};
        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        MatmulAdapter matmul_op;
        size_t sizeWorkspace = matmul_op.GetWorkspaceSize(arguments);
        uint8_t *deviceWorkspace{nullptr};
        if (sizeWorkspace > 0) {
            ACL_CHECK(
                aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        matmul_op.Initialize(arguments, deviceWorkspace);

        matmul_op(stream, aicCoreNum, fftsAddr);

        ACL_CHECK(aclrtSynchronizeStream(stream));
        if (sizeWorkspace > 0) {
            ACL_CHECK(aclrtFree(deviceWorkspace));
        }

        // Copy the result from device to host
        ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));
    } else {
        // Define BlockScheduler
        // Swizzle offset is 3 and direction is 1.
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
        // Kernel level
        using MatmulKernel = Gemm::Kernel::MatmulFaultTolerance<BlockMmad, BlockSumGemv, BlockScheduler>;
        // Prepare params
        typename MatmulKernel::Arguments arguments{
            options.problemShape, deviceA, deviceB, deviceC, deviceD, options.checkSumType, aicCoreNum};
        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        MatmulAdapter matmul_op;
        size_t sizeWorkspace = matmul_op.GetWorkspaceSize(arguments);
        uint8_t *deviceWorkspace{nullptr};
        if (sizeWorkspace > 0) {
            ACL_CHECK(
                aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace,ACL_MEM_MALLOC_HUGE_FIRST));
        }
        matmul_op.Initialize(arguments, deviceWorkspace);

        matmul_op(stream, aicCoreNum, fftsAddr);
        ACL_CHECK(aclrtSynchronizeStream(stream));
        // Copy the result from device to host
        ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_CHECK(aclrtMemcpy(hostD.data(), sizeC, deviceD, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

        // std::cout << "host A" << std::endl;
        // PrintMatrix(hostA.data(), m, k);
        // std::cout << "host B" << std::endl;
        // PrintMatrix(hostB.data(), k, n);
        // std::cout << "host C" << std::endl;
        // PrintMatrix(hostC.data(), m, n);

        uint32_t loopsM = CeilDiv(m, L1TileShape::M);
        uint32_t loopsN = CeilDiv(n, L1TileShape::N);
        uint32_t loopsK = CeilDiv(k, L1TileShape::K);
        uint32_t UBKRound = RoundUp(k, Catlass::Gemv::helper::UBAlignHelper<float>::ALIGN);

        int eTA_index = 0, Be_index = 0, eTAB_index = 0, ABe_index = 0, eTC_index = 0, Ce_index = 0;
        if(options.checkSumType == 1) {  // ROW_CHECKSUM
            eTA_index = 0;
            Be_index =  eTA_index + loopsN * UBKRound;
            eTAB_index = Be_index;
            ABe_index = eTAB_index + (loopsM * loopsN * loopsK) * L1TileShape::M;
            eTC_index = ABe_index;
            Ce_index = eTC_index + (loopsM * loopsN * loopsK) * L1TileShape::M;


            for(int i = 0; i < ABe_index - eTAB_index; ++i) {
                uint32_t numSum = i / L1TileShape::M;
                uint32_t offset = i % L1TileShape::M;
                uint32_t numCore = numSum / loopsK;
                uint32_t numK = numSum % loopsK;

                // if(offset == 0) {
                //     std::cout << "numCore: " << numCore << ", numK: " << numK << " " << static_cast<float>(hostD[eTAB_index + i]) << " " << static_cast<float>(hostD[eTC_index + i]) << std::endl;
                // }

                if(abs(static_cast<float>(hostD[eTAB_index + i]) - static_cast<float>(hostD[eTC_index + i])) > 0.02f && offset < 5) {
                    std::cout << "hostABE[" << numCore << ", " << numK << ", " << offset << "] = " << static_cast<float>(hostD[eTAB_index + i]) << ", hostCE[" << numCore << ", " << numK << ", " << offset << "] = " << static_cast<float>(hostD[eTC_index + i]) << std::endl;
                }
            }

            // std::cout << "Be" << std::endl;
            // PrintMatrix(hostD.data() + eTA_index, loopsN, UBKRound);
            // std::cout << "ABe" << std::endl;
            // PrintMatrix(hostD.data() + eTAB_index, (loopsM * loopsN), L1TileShape::M);
            // std::cout << "Ce" << std::endl;
            // PrintMatrix(hostD.data() + eTC_index, (loopsM * loopsN), L1TileShape::M);
        }

        std::cout << "checkSumType: " << options.checkSumType << ", eTA_index: " << eTA_index << ", Be_index: " << Be_index << ", eTAB_index: " << eTAB_index << ", ABe_index: " << ABe_index << ", eTC_index: " << eTC_index << ", Ce_index: " << Ce_index << std::endl;

        for (int i = 0; i < 100; ++i) {
            ACL_CHECK(aclrtSynchronizeStream(stream));
            matmul_op(stream, aicCoreNum, fftsAddr);
        }

        int num_repeat = 10000;

        aclrtEvent start, stop;
        float temp_time = 0;
        float time = 0;
        ACL_CHECK(aclrtCreateEvent(&start));
        ACL_CHECK(aclrtCreateEvent(&stop));

        for (int i = 0; i < num_repeat; ++i) {
            ACL_CHECK(aclrtSynchronizeStream(stream));
            ACL_CHECK(aclrtRecordEvent(start, stream));

            matmul_op(stream, aicCoreNum, fftsAddr);
            
            ACL_CHECK(aclrtSynchronizeStream(stream));
            ACL_CHECK(aclrtRecordEvent(stop, stream));
            ACL_CHECK(aclrtSynchronizeEvent(stop));
            ACL_CHECK(aclrtEventElapsedTime(&temp_time, start, stop));
            time += temp_time;
        }

        std::cout << "m: " << m << ", n: " << n << ", k: " << k << ", " << (float)2 * m * n * k / (time / num_repeat * 1e-3) / 1e12 << " TFLOPS, " << (time / num_repeat) << " ms, repeat: " << num_repeat << std::endl;

        ACL_CHECK(aclrtSynchronizeStream(stream));
        if (sizeWorkspace > 0) {
            ACL_CHECK(aclrtFree(deviceWorkspace));
        }
    }


    // Compute and compare the result
    std::vector<uint64_t> errorIndices = golden::ComputeAndCompareMatmul(options.problemShape, hostA, layoutA, hostB, layoutB, hostC, layoutC);

    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed." << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char **argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    Run(options);
    return 0;
}
