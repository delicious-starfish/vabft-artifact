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

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemv/block/block_gemv.hpp"

#include "catlass/gemv/kernel/kernel_gemv_aic_FT.hpp"
#include "catlass/gemv/kernel/kernel_gemv_aic_FT_double.hpp"
#include "catlass/gemv/tile/tile_copy.hpp"
#include "catlass/gemv/helper.hpp"

#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"

#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_add.hpp"
#include "catlass/epilogue/tile/tile_elemwise_muls.hpp"

#include "catlass/layout/layout.hpp"
#include "catlass/gemv/device/device_gemv.hpp"
#include "catlass/status.hpp"

using namespace Catlass;

using ScalarType = float;


struct Options {
    const std::string HELPER = "18_gemv_aic m n [device_id]";

    GemvCoord problemShape{128, 128};
    int32_t deviceId{1};

    Options() = default;

    int Parse(int argc, const char** argv) {
        enum ArgsIndex {
            M_INDEX = 1,
            N_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };
        if (argc > ARGS_MAX || argc < N_INDEX) {
            std::cerr << HELPER << std::endl;
            return -1;
        }
        problemShape.m() = std::atoi(argv[M_INDEX]);
        problemShape.n() = std::atoi(argv[N_INDEX]);
        if (argc == ARGS_MAX) {
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        }
        return 0;
    }
};

template <class Adapter>
void RunAdapter(Adapter gemv_op, typename Adapter::Arguments args, aclrtStream stream,
    uint32_t aicCoreNum, uint64_t fftsAddr)
{
    size_t sizeWorkspace = gemv_op.GetWorkspaceSize(args);
    uint8_t *deviceWorkspace = nullptr;
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    gemv_op.Initialize(args, deviceWorkspace);
    gemv_op(stream, aicCoreNum, fftsAddr);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }
}

template<class ElementRandom>
void FillRandomScalarData(ElementRandom &scalarData, ElementRandom low, ElementRandom high)
{
    scalarData = static_cast<ElementRandom>(low + (static_cast<ElementRandom>(rand()) / static_cast<ElementRandom>(RAND_MAX)) * (high - low));
}

void Run(Options options) {
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();

    GemvCoord problemShapeCol{n, m};


    size_t lenA = static_cast<size_t>(n) * m;
    size_t lenX = (static_cast<size_t>(m) + static_cast<size_t>(n)) * 1;
    size_t lenYRow = (static_cast<size_t>(m)) * 1;
    size_t lenYCol = (static_cast<size_t>(n)) * 1;
    size_t lenZRow = static_cast<size_t>(m) * 1;
    size_t lenZCol = static_cast<size_t>(n) * 1;

    size_t sizeA = lenA * sizeof(float);
    size_t sizeX = lenX * sizeof(float);

    size_t sizeZRow = lenZRow * sizeof(float);
    size_t sizeYRow = lenYRow * sizeof(float);
    size_t sizeZCol = lenZCol * sizeof(float);
    size_t sizeYCol = lenYCol * sizeof(float);

    size_t sizeWorkspace;

    using LayoutX = layout::VectorLayout;
    using LayoutA = layout::RowMajor;
    using LayoutACol = layout::ColumnMajor;

    using LayoutZ = layout::VectorLayout;

    LayoutX layoutXRow{n};
    LayoutX layoutXCol{m};

    LayoutA layoutA{m, n};
    LayoutACol layoutACol{n,m};

    LayoutZ layoutZ{m};
    LayoutZ layoutZCol{n};

    ScalarType alpha{1.0};
    ScalarType beta{0.0};
    // FillRandomScalarData(alpha, -1.0f, 1.0f);
    // FillRandomScalarData(beta, -1.0f, 1.0f);

    float sum_base = 1.0;
    std::vector<float> hostA(lenA);
    std::vector<float> hostX(lenX,sum_base);
    std::vector<float> hostYRow(lenYRow);
    std::vector<float> hostYCol(lenYCol,0.0f);
    golden::FillRandomData(hostA, -1.0f, 1.0f);
    // golden::FillRandomData(hostX, -1.0f, 1.0f);
    golden::FillRandomData(hostYRow, -1.0f, 1.0f);
    // golden::FillRandomData(hostYCol, -1.0f, 1.0f);
    uint8_t* deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceX{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceX), sizeX, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceX, sizeX, hostX.data(), sizeX, ACL_MEMCPY_HOST_TO_DEVICE));


    uint8_t* deviceZRow{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceZRow), sizeZRow, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceZRow, sizeZRow, hostYRow.data(), sizeZRow, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceZCol{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceZCol), sizeZCol, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceZCol, sizeZCol, hostYCol.data(), sizeZCol, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceWorkspace{nullptr};

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    using ArchTag = Arch::AtlasA2;
    using LayoutC = layout::RowMajor;

    // Block level, define BlockGemv
    // static_cast<CheckSumType>(checkSumType_)
    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;
    using DispatchPolicy = Gemm::MmadAtlasA2Preload<enableUnitFlag, enableShuffleK>;
    using FT_ENC_TYPE = Gemv::helper::FT_ENC_TYPE;
    using L1TileShape = GemvShape<32, 512>;
    using L0TileShape = GemvShape<32, 256>;
    using AType = Gemm::GemmType<float, LayoutA>;
    using XType = Gemm::GemmType<float, LayoutX>;
    using CType = Gemm::GemmType<float, LayoutC>;
    using BiasType = void;
    using TileCopy = Gemv::Tile::TileCopyGemvAic<typename DispatchPolicy::ArchTag, AType, XType, CType, BiasType>;
    using TileMmad = Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, XType, AType, BiasType>;

    using BlockGemv = Gemv::Block::BlockFTGemvDouble<DispatchPolicy, FT_ENC_TYPE::BOTHC, L1TileShape, L0TileShape, AType, XType, CType, BiasType, TileCopy, TileMmad>;

    // Block level, define BlockEpilogue
    using EpilogueBlockDispatchPolicy = Epilogue::EpilogueAtlasA2Gemv;
    using YType = Gemm::GemmType<float, LayoutZ>;
    using ZType = Gemm::GemmType<float, LayoutZ>;
    using AXType = Gemm::GemmType<float, LayoutZ>;

    using ComputeType = AXType;
    constexpr uint32_t computeLength = 8192;
    

    using TileElemWiseAddGemv = Epilogue::Tile::TileElemWiseAdd<ArchTag, ComputeType, computeLength>;
    using TileElemWiseMulsGemv = Epilogue::Tile::TileElemWiseMuls<ArchTag, ComputeType, computeLength>;

    using EpilogueTileCopy = Epilogue::Tile::TileCopy<ArchTag, YType, AXType, ZType>;

    using BlockEpilogue = Epilogue::Block::BlockEpilogue<EpilogueBlockDispatchPolicy, AXType, YType, ZType, TileElemWiseAddGemv, TileElemWiseMulsGemv, EpilogueTileCopy>;

    // kernle levels
    using GemvKernel = Gemv::Kernel::KernelGemvFTDoubleAic<BlockGemv, BlockEpilogue>;

    // TODO:  use adapter to activate the kernel
    using GemvAdapter = Gemv::Device::DeviceGemv<GemvKernel>;

    /*
    struct Arguments {
        GemvCoord problemShape;
        ElementY alpha;
        ElementY beta;
        size_t elementSize;
        GM_ADDR ptrX;
        GM_ADDR ptrA;
        GM_ADDR ptrZRow;
        GM_ADDR ptrZCol;
        FT_ENC_TYPE enc_type;
    };
    */

    GemvKernel::Arguments arguments{options.problemShape, alpha, beta, sizeof(float), deviceX, deviceA, deviceZRow, deviceZCol, FT_ENC_TYPE::BOTHC};
    GemvAdapter gemv_op;
    RunAdapter(gemv_op, arguments, stream, aicCoreNum, fftsAddr);

    std::vector<float> hostResRow(lenZRow);
    std::vector<float> hostResCol(lenZCol);

    ACL_CHECK(aclrtMemcpy(hostResRow.data(), sizeZRow, deviceZRow, sizeZRow, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostResCol.data(), sizeZCol, deviceZCol, sizeZCol, ACL_MEMCPY_DEVICE_TO_HOST));
    
    std::vector<float> hostGoldenRow(lenZRow);
    std::vector<float> hostGoldenCol(lenZCol);
    // Host
    golden::ComputeGemv(problemShapeCol, alpha, beta, hostA, layoutACol, hostX, layoutXCol, hostYCol, layoutZCol, hostGoldenCol, layoutZCol);
    golden::ComputeGemv(options.problemShape, alpha, beta, hostA, layoutA, hostX, layoutXRow, hostYRow, layoutZ, hostGoldenRow, layoutZ);
    float aim_data = 0.0;

    std::vector<uint64_t> errorIndices = golden::CompareData(hostResCol, hostGoldenCol, n);
    for(uint32_t j=0; j < m; j++){
        printf("%f ",hostA[j*n+560]);
        aim_data += hostA[j*n+560];
    }
    std::cout<<std::endl;
    printf("real: %f \n", aim_data);
    printf("golden: %f \n", hostGoldenCol[560]);
    printf("computed: %f ",hostResCol[560]);
    // std::cout<<hostResCol[0]<<std::endl;
    
    if (errorIndices.empty()) {
        std::cout << "ColSum Compare success." << std::endl;
    } else {
        std::cerr << "ColSum Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    errorIndices = golden::CompareData(hostResRow, hostGoldenRow, m);

    if (errorIndices.empty()) {
        std::cout << "RowSum Compare success." << std::endl;
    } else {
        std::cerr << "RowSum Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    for (int i = 0; i < 100; ++i) {
        ACL_CHECK(aclrtSynchronizeStream(stream));
        RunAdapter(gemv_op, arguments, stream, aicCoreNum, fftsAddr);
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

        RunAdapter(gemv_op, arguments, stream, aicCoreNum, fftsAddr);
            
        ACL_CHECK(aclrtSynchronizeStream(stream));
        ACL_CHECK(aclrtRecordEvent(stop, stream));
        ACL_CHECK(aclrtSynchronizeEvent(stop));
        ACL_CHECK(aclrtEventElapsedTime(&temp_time, start, stop));
        time += temp_time;
    }

    std::cout << "m: " << m << ", n: " << n << ", " << (float)4 * m * n / (time / num_repeat * 1e-3) / 1e9 << " GFLOPS, " << (time / num_repeat) << " ms, repeat: " << num_repeat << std::endl;

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceX));
    ACL_CHECK(aclrtFree(deviceZRow));
    ACL_CHECK(aclrtFree(deviceZCol));

    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char** argv) {
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    Run(options);
    return 0;
}