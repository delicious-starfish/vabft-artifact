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

#include "helper.hpp"
#include "golden.hpp"
#include "fp16_t.h"
#include "bfloat16.h"

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemv/tile/tile_copy.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_add.hpp"
#include "gemm/block/block_mmad.hpp" // catlass/
#include "gemm/block/block_mmad.hpp" // catlass/
#include "gemm/block/block_swizzle.hpp" // catlass/
#include "gemm/dispatch_policy.hpp" // catlass/
#include "gemm/kernel/matmul_epilogue.hpp" // catlass/ catlass/
#include "gemm/kernel/matmul_no_epilogue.hpp" // catlass/ catlass/
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

#include "catlass/status.hpp"
#include "gemm/device/device_gemm.hpp" // catlass/

using namespace Catlass;
using fp16_t = op::fp16_t;
using op_bfloat16 = op::bfloat16;



struct Options {
    const std::string HELPER = "03_matmul_add m n k [device_id, make_golden]";

    GemmCoord problemShape{4096, 4096, 4096};
    int32_t deviceId{0};
    int32_t make_golden{0};

    Options() = default;

    int Parse(int argc, const char **argv)
    {
        enum ArgsIndex {
            M_INDEX = 1,
            N_INDEX,
            K_INDEX,
            DEVICE_ID_INDEX,
            GOLDEN,
            ARGS_MAX
        };

        if (argc > ARGS_MAX || argc <= K_INDEX) {
            std::cerr << HELPER << std::endl;
            return -1;
        }

        problemShape.m() = std::atoi(argv[M_INDEX]);
        problemShape.n() = std::atoi(argv[N_INDEX]);
        problemShape.k() = std::atoi(argv[K_INDEX]);
        if (argc >= GOLDEN) {
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        }
        if (argc == ARGS_MAX){
            make_golden = std::atoi(argv[GOLDEN]);
        }
        return 0;
    }
};

void Run(Options const &options)
{
    using GemmInTypeC = op_bfloat16;
    // op_bfloat16;
    // op_bfloat16;
    using GemmInTypeN = bfloat16_t;
    // bfloat16_t;
    using GemmOutTypeC = float;
    using GemmOutTypeN = float;
    // using GemmOutTypeC = op_bfloat16;
    // using GemmOutTypeN = bfloat16_t;
    
    using ScalarTypeC = float;
    using ScalarTypeN = float;

    std::cout<<"Device ID: "<<options.deviceId<<std::endl;
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();
    int32_t make_golden = options.make_golden;

    // Compute the length of each matrix and the size of each buffer
    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenD = static_cast<size_t>(m) * n;
    size_t lenX = lenD;

    size_t sizeA = lenA * sizeof(GemmInTypeC);
    size_t sizeB = lenB * sizeof(GemmInTypeC);
    size_t sizeD = lenD * sizeof(GemmOutTypeC);

    // Define the layout of each matrix
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;

    
    
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutD{m, n};

    // Prepare input data A, B, and X
    std::vector<GemmInTypeC> hostA(lenA);
    std::vector<GemmInTypeC> hostB(lenB);
    std::vector<GemmOutTypeC> hostX(lenX);
    golden::FillRandomData<GemmInTypeC>(hostA, -1.0f, 1.0f);
    golden::FillRandomData<GemmInTypeC>(hostB, -1.0f, 1.0f);
    golden::FillRandomData<GemmOutTypeC>(hostX, 0.0f, 0.0f);

    // Allocate device memory and copy data from host to device
    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    // The data of X is stored on deviceD to save storage space
    uint8_t *deviceD{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceD), sizeD, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceD, sizeD, hostX.data(), sizeD, ACL_MEMCPY_HOST_TO_DEVICE));

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    // Define ArchTag
    using ArchTag = Arch::AtlasA2;

    // Block level, define BlockMmad
    constexpr bool enableUnitFlag = true;
    // CubeSelf
    using MmadDispatchPolicy = CubeSelf::Gemm::MmadAtlasA2Pingpong<enableUnitFlag>;
    using L1TileShape = GemmShape<128, 256, 256>;
    using L0TileShape = GemmShape<128, 256, 64>;
    // using L1TileShape = GemmShape<128, 256, 256>;
    // using L0TileShape = GemmShape<128, 256, 64>;
    using AType = Gemm::GemmType<GemmInTypeN, LayoutA>;
    using BType = Gemm::GemmType<GemmInTypeN, LayoutB>;
    using CType = Gemm::GemmType<GemmOutTypeN, LayoutC>;
    using BlockMmad = CubeSelf::Gemm::Block::BlockMmad<MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

    // Block level, define BlockEpilogue
    // using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2ElemWiseOneSource;
    // using XType = CType;
    // using DType = CType;
    // using ComputeType = CType;
    // constexpr uint32_t computeLength = 16384;
    // using TileElemWiseEpilogue = Epilogue::Tile::TileElemWiseAdd<ArchTag, ComputeType, computeLength>;
    // using EpilogueTileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, XType, DType>;
    // using BlockEpilogue = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy, CType, XType, DType,
    //     TileElemWiseEpilogue, EpilogueTileCopy>;
    std::vector<GemmOutTypeC> hostD(lenD);
    if (m > n) {
        // Define BlockScheduler
        // Swizzle offset is 3 and direction is 0.
        // CubeSelf::
        using BlockScheduler = typename CubeSelf::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
        // Kernel level
        // CubeSelf:: CubeSelf::
        // using MatmulKernel = CubeSelf::Gemm::Kernel::MatmulNoEpilogue<BlockMmad, BlockEpilogue, BlockScheduler>;
        // BlockEpilogue, 
        using MatmulKernel = CubeSelf::Gemm::Kernel::MatmulNoEpilogue<BlockMmad, BlockScheduler>;
        // Prepare params
        typename MatmulKernel::Arguments arguments{
            options.problemShape, sizeof(GemmOutTypeC), deviceA, deviceB, deviceD};
        // CubeSelf::
        using MatmulAdapter = CubeSelf::Gemm::Device::DeviceGemm<MatmulKernel>;
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
        if (make_golden > 0){
            // Copy the result from device to host
            ACL_CHECK(aclrtMemcpy(hostD.data(), sizeD, deviceD, sizeD, ACL_MEMCPY_DEVICE_TO_HOST));

            // Compute the golden result
            std::vector<GemmOutTypeC> hostGolden(lenD);
            golden::ComputeMatmulElemWiseAdd(options.problemShape, hostA, layoutA, hostB, layoutB, hostX, hostGolden, layoutD);

            // Compare the result
            std::vector<uint64_t> errorIndices = golden::CompareData(hostD, hostGolden, k);
            if (errorIndices.empty()) {
                std::cout << "Compare success." << std::endl;
            } else {
                std::cout << "Compare failed. Error count: " << errorIndices.size() << std::endl;
            }
        }
        
        for (int i = 0; i < 100; ++i) {
            ACL_CHECK(aclrtSynchronizeStream(stream));
            matmul_op(stream, aicCoreNum, fftsAddr);
        }

        int num_repeat = 1000;

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
        // if (sizeWorkspace > 0) {
        //     ACL_CHECK(aclrtFree(deviceWorkspace));
        // }

        if (sizeWorkspace > 0) {
            ACL_CHECK(aclrtFree(deviceWorkspace));
        }

        // Copy the result from device to host
        ACL_CHECK(aclrtMemcpy(hostD.data(), sizeD, deviceD, sizeD, ACL_MEMCPY_DEVICE_TO_HOST));
    } else {
        // Define BlockScheduler
        // Swizzle offset is 3 and direction is 1.
        // CubeSelf::
        using BlockScheduler = typename CubeSelf::Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
        // Kernel level
        // CubeSelf::
        // using MatmulKernel = CubeSelf::Gemm::Kernel::MatmulEpilogue<BlockMmad, BlockEpilogue, BlockScheduler>;
        // BlockEpilogue, 
        using MatmulKernel = CubeSelf::Gemm::Kernel::MatmulNoEpilogue<BlockMmad, BlockScheduler>;
        // Prepare params
        // Prepare params
        // CubeSelf::
        typename MatmulKernel::Arguments arguments{
            options.problemShape, sizeof(GemmOutTypeC), deviceA, deviceB, deviceD};
        using MatmulAdapter = CubeSelf::Gemm::Device::DeviceGemm<MatmulKernel>;
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

        if (make_golden > 0){
            // Copy the result from device to host
            ACL_CHECK(aclrtMemcpy(hostD.data(), sizeD, deviceD, sizeD, ACL_MEMCPY_DEVICE_TO_HOST));

            // Compute the golden result
            std::vector<GemmOutTypeC> hostGolden(lenD);
            golden::ComputeMatmulElemWiseAdd(options.problemShape, hostA, layoutA, hostB, layoutB, hostX, hostGolden, layoutD);

            // Compare the result
            std::vector<uint64_t> errorIndices = golden::CompareData(hostD, hostGolden, k);
            if (errorIndices.empty()) {
                std::cout << "Compare success." << std::endl;
            } else {
                std::cout << "Compare failed. Error count: " << errorIndices.size() << std::endl;
            }
        }

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

        // if (sizeWorkspace > 0) {
        //     ACL_CHECK(aclrtFree(deviceWorkspace));
        // }

        if (sizeWorkspace > 0) {
            ACL_CHECK(aclrtFree(deviceWorkspace));
        }

        // Copy the result from device to host
        ACL_CHECK(aclrtMemcpy(hostD.data(), sizeD, deviceD, sizeD, ACL_MEMCPY_DEVICE_TO_HOST));
    }


    

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceD));

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
