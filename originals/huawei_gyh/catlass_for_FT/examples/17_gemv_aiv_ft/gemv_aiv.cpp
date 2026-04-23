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
#include "catlass/gemv/block/block_gemv_encoding_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/matmul_fault_tolerance_pingpong.hpp"
#include "catlass/gemm/kernel/grouped_matmul_slice_m.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemv/tile/tile_fault_copy.hpp"
#include "catlass/gemv/tile/tile_fault_vmad.hpp"
#include "catlass/gemv/tile/tile_fault_sum.hpp"
#include "catlass/gemv/tile/tile_vmuls.hpp"
#include "catlass/gemv/tile/tile_vmad.hpp"
#include "catlass/gemv/tile/tile_copy.hpp"
#include "catlass/gemv/block/block_gemv.hpp"
#include "catlass/gemv/kernel/kernel_gemv_aiv.hpp"
#include "catlass/layout/layout.hpp"

#include "catlass/status.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemv/kernel/kernel_gemv_aiv_FT_double.hpp"

using namespace Catlass;
using fp16_t = op::fp16_t;
using ScalarType = float;


struct Options {
    const std::string HELPER = "17_gemv_fault m n k check_sum_type [device_id]";

    GemmCoord problemGemmShape{128, 128, 128};
    GemvCoord problemShape{128, 128};
    int32_t deviceId{0};

    Options() = default;

    int Parse(int argc, const char **argv)
    {
        enum ArgsIndex {
            M_INDEX = 1,
            N_INDEX,
            K_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        if (argc > ARGS_MAX || argc <= K_INDEX) {
            std::cerr << HELPER << std::endl;
            return -1;
        }

        problemGemmShape.m() = std::atoi(argv[M_INDEX]);
        problemGemmShape.n() = std::atoi(argv[N_INDEX]);
        problemGemmShape.k() = std::atoi(argv[K_INDEX]);
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

    uint32_t m = options.problemGemmShape.m();
    uint32_t n = options.problemGemmShape.n();
    uint32_t k = options.problemGemmShape.k();

    GemvCoord problemShape{m,n};

    // Compute the length of each matrix and the size of each buffer
    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenC = static_cast<size_t>(n);
    size_t lenDRow = static_cast<size_t>(m);
    size_t lenDCol = static_cast<size_t>(n);
    size_t lenX = static_cast<size_t>(n);
    size_t lenY = static_cast<size_t>(k);
    size_t lenZ = static_cast<size_t>(m);

    size_t sizeA = lenA * sizeof(float);
    size_t sizeB = lenB * sizeof(float);
    size_t sizeC = lenC * sizeof(float);
    size_t sizeDRow = lenDRow * sizeof(float);
    size_t sizeDCol = lenDCol * sizeof(float);
    size_t sizeX = lenX * sizeof(float);
    size_t sizeY = lenY * sizeof(float);
    size_t sizeZ = lenZ * sizeof(float);

    // Define the layout of each matrix
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;       // must be rowmajor
    using LayoutACol = layout::ColumnMajor;
    using LayoutBCol = layout::ColumnMajor;
    using LayoutX = layout::VectorLayout;
    using LayoutY = layout::VectorLayout;
    using LayoutZ = layout::VectorLayout;
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m, n};
    LayoutACol layoutACol{k, m};
    LayoutBCol layoutBCol{n, k};
    LayoutX layoutX{n};
    LayoutY layoutY{k};
    LayoutZ layoutZ{m};

    // Prepare input data A, B, and X
    std::vector<float> hostA(lenA);
    std::vector<float> hostB(lenB);
    std::vector<float> hostC(lenC, 1.0f);
    std::vector<float> hostDRow(lenDRow, 0.0f);
    std::vector<float> hostDCol(lenDCol, 0.0f);
    std::vector<float> hostX(lenX);
    std::vector<float> hostY(lenY, 0.0f);
    std::vector<float> hostZ(lenZ);
    golden::FillRandomData<float>(hostA, -5.0f, 5.0f);
    golden::FillRandomData<float>(hostB, -5.0f, 5.0f);
    golden::FillRandomData<float>(hostC, 1.0f, 1.0f);
    golden::FillRandomData<float>(hostX, 1.0f, 1.0f);
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
    ACL_CHECK(aclrtMemcpy(deviceC, sizeC, hostC.data(), sizeC, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceDRow{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceDRow), sizeDRow, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceDRow, sizeDRow, hostDRow.data(), sizeDRow, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceDCol{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceDCol), sizeDCol, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceDCol, sizeDCol, hostDCol.data(), sizeDCol, ACL_MEMCPY_HOST_TO_DEVICE));

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
    using L1TileShape = GemmShape<64, 64, 64>;
    using L0TileShape = GemmShape<64, 64, 64>;
    using AType = Gemm::GemmType<float, LayoutA>;
    using BType = Gemm::GemmType<float, LayoutB>;
    using CType = Gemm::GemmType<float, LayoutC>;
    using BlockMmad = Gemm::Block::BlockMmadFault<MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

    using FT_ENC_TYPE = Gemv::helper::FT_ENC_TYPE;

    // Block level, define sum and Gemv
    using GemvDispatchPolicy = Gemm::GemvAtlasA2;
    using XType = Gemm::GemmType<float, LayoutX>;
    using YType = Gemm::GemmType<float, LayoutY>;
    using TileVmuls = Gemv::Tile::TileVmuls<ArchTag, XType>;

    using UBTileShape = GemmShape<L1TileShape::M, L1TileShape::N, L1TileShape::K>;
    using TileFaultCopy = Gemv::Tile::TileCopyGemvAiv<ArchTag, CType, XType, YType>;
    using TileFaultVmad = Gemv::Tile::TileVmad<ArchTag, CType, XType, YType>;
    using TileFaultSum = Gemv::Tile::TileFaultSum<ArchTag, CType, YType>;
    using BlockSumGemv = Gemv::Block::BlockSumGemv<GemvDispatchPolicy, UBTileShape, CType, XType, YType, void, TileFaultCopy, TileFaultVmad, TileVmuls>;
    

    // Define BlockScheduler
    // Swizzle offset is 3 and direction is 1.
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
    using BlockEncodeScheduler = typename Gemv::Block::GemvIdentityBlockSwizzle<3,1>;
    // Kernel level
    //using MatmulKernel = Gemm::Kernel::GemvKernelFTAiv<BlockMmad, BlockSumGemv, BlockScheduler, BlockEncodeScheduler>;
    using MatmulKernel = Gemm::Kernel::GemvKernelFTAiv<BlockSumGemv>;
    // Prepare params
/*
    struct Arguments {
        GemmCoord problemGemmShape;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
        GM_ADDR ptrDRow;
        GM_ADDR ptrDCol;
        uint32_t blockNum;
        FT_ENC_TYPE enc_type;
    };*/

    typename MatmulKernel::Arguments arguments{
        options.problemGemmShape, deviceA, deviceB, deviceC, deviceDRow, deviceDCol,
        aicCoreNum,
        FT_ENC_TYPE::BOTHC};
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
    ACL_CHECK(aclrtMemcpy(hostDRow.data(), sizeDRow, deviceDRow, sizeDRow, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostDCol.data(), sizeDCol, deviceDCol, sizeDCol, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGoldenRow(lenZ, 0.0f);
    std::vector<float> hostGoldenCol(lenZ, 0.0f);

    ScalarType alpha{1.0};
    ScalarType beta{0.0};

    /*golden::ComputeMatmul(options.problemGemmShape, hostA, layoutA, hostB, layoutB, hostC, layoutC);
    //golden::ComputeGemm(options.problemGemmShape, alpha, beta, hostA, layoutA, hostB, layoutB, hostC, layoutC, hostC, layoutC);


    golden::ComputeGemv(problemShape, alpha, beta, hostC, layoutC, hostX, layoutX, hostY, layoutY, hostGolden, layoutY);
*/
    golden::ComputeGemv(problemShape, alpha, beta, hostB, layoutB, hostX, layoutX, hostY, layoutY, hostY, layoutY);
    golden::ComputeGemv(problemShape, alpha, beta, hostA, layoutA, hostY, layoutY, hostGoldenRow, layoutZ, hostGoldenRow, layoutZ);

    golden::FillRandomData<float>(hostY, 0.0f, 0.0f);
    golden::ComputeGemv(problemShape, alpha, beta, hostA, layoutACol, hostX, layoutX, hostY, layoutY, hostY, layoutY);
    golden::ComputeGemv(problemShape, alpha, beta, hostB, layoutBCol, hostY, layoutY, hostGoldenCol, layoutZ, hostGoldenCol, layoutZ);

    // Compare the result

    std::vector<uint64_t> errorIndices = golden::CompareData(hostDRow, hostGoldenRow, m);

    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed." << std::endl;
    }

    errorIndices = golden::CompareData(hostDCol, hostGoldenCol, n);

    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed." << std::endl;
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

    std::cout << "m: " << m << ", n: " << n << ", " << (float)4 * m * n / (time / num_repeat * 1e-3) / 1e9 << " GFLOPS, " << (time / num_repeat) << " ms, repeat: " << num_repeat << std::endl;


    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    ACL_CHECK(aclrtFree(deviceDRow));
    ACL_CHECK(aclrtFree(deviceDCol));

    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }

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
