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
#include <cmath>

#include "helper.hpp"
#include "golden.hpp"

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemv/kernel/kernel_threshold_calc_aiv.hpp"
// #include "catlass/gemv/kernel/kernel_gemv_aiv.hpp"
#include "catlass/gemv/block/block_gemv.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemv/tile/tile_copy.hpp"
#include "catlass/gemv/tile/tile_vmad.hpp"
#include "catlass/gemv/tile/tile_vmuls.hpp"
// include/catlass/gemv/tile/tile_threshold_compute.hpp
#include "catlass/gemv/tile/tile_threshold_compute.hpp"
#include "catlass/gemv/device/device_gemv.hpp"
#include "catlass/status.hpp"
#include "catlass/gemv/helper.hpp"

using namespace Catlass;

using ScalarType = float;

struct Options{
    const std::string HELPER = "17_gemv_aiv m n rt beta [device_id]";

    uint32_t M = 32;
    uint32_t N = 32;
    uint32_t deviceId{0};

    float round_exp{0.0f};
    float beta{1.0f};


    Options() = default;
    
    GemvCoord problemShape{M, N};

    int Parse(int argc, const char **argv)
    {
        enum ArgsIndex {
            M_INDEX = 1,
            N_INDEX,
            RT_INDEX,
            BETA_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };
        if (argc > ARGS_MAX || argc <= N_INDEX) 
        {
            std::cerr << HELPER << std::endl;
            return -1;
        }
        problemShape.m() = std::atoi(argv[M_INDEX]);
        problemShape.n() = std::atoi(argv[N_INDEX]);
        round_exp = static_cast<float>(std::stof(argv[RT_INDEX]));
        beta = static_cast<float>(std::stof(argv[BETA_INDEX]));
        if (argc == ARGS_MAX)
        {
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        }
        return 0;
    }
};

uint32_t getSplictNum(bool trans, uint32_t M, uint32_t N, uint32_t M1, uint32_t N1, uint32_t maxSplict)
{
    uint32_t CORENUM = 20;
    uint32_t splitNum = 1;
    uint32_t maxOccupancy = 0; 
    uint32_t blockNum = (M - 1) / M1 + 1;
    if (!trans)
    {
        splitNum = 1;
    }
    else{
        uint32_t splitNum1 = 1, splitNum2 = 1;
        for (uint32_t i = 1; i <= maxSplict; i += 1)
        {
            uint32_t occupancy = (i * blockNum) % (CORENUM * 2);
            if (!occupancy)
                occupancy = (CORENUM * 2);
            if (occupancy > maxOccupancy)
            {
                maxOccupancy = occupancy;
                splitNum1 = i;
            }
        }
        maxOccupancy = 0;
        for (uint32_t i = 1; i <= maxSplict; i <<= 1)
        {
            uint32_t occupancy = (i * blockNum) % (CORENUM * 2);
            if (!occupancy)
                occupancy = (CORENUM * 2);
            if (occupancy > maxOccupancy)
            {
                maxOccupancy = occupancy;
                splitNum2 = i;
            }
        }
        splitNum = (splitNum1 - splitNum2) > 4 ? splitNum1 : splitNum2;
    }
    return splitNum;
}

template <class Adapter>
void RunAdapter(Adapter threshold_op, typename Adapter::Arguments args, aclrtStream stream,
    uint32_t aicCoreNum)
{
    size_t sizeWorkspace = threshold_op.GetWorkspaceSize(args);
    uint8_t *deviceWorkspace = nullptr;
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    threshold_op.Initialize(args, deviceWorkspace);
    threshold_op(stream, aicCoreNum);
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

void Run(Options options){
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    using UBTileShape = GemvShape<32,512>;

    uint32_t maxSplict = 20;
    uint32_t const split = getSplictNum(false, m, n, UBTileShape::M, UBTileShape::N, maxSplict);

    size_t lenA = static_cast<size_t>(m) * n;
    size_t lenX = static_cast<size_t>(n) * 1;
    size_t lenY = static_cast<size_t>(m) * 1;
    size_t scalarLen = 1;

    size_t sizeA = lenA * sizeof(float);
    size_t sizeX = lenX * sizeof(float);
    size_t sizeY = lenY * sizeof(float);

    using LayoutA = layout::RowMajor;
    using LayoutX = layout::VectorLayout;
    using LayoutY = layout::VectorLayout;
    
    LayoutA layoutA{m, n};
    LayoutX layoutX{n};
    LayoutY layoutY{m};

    // ScalarType alpha{0};
    ScalarType beta = options.beta;
    ScalarType round_exp = options.round_exp;
    // FillRandomScalarData(alpha, -1.0f, 1.0f);
    // FillRandomScalarData(beta, -1.0f, 1.0f);

    std::vector<float> hostA(lenA);
    // std::vector<float> hostX(lenX);
    std::vector<float> hostY(lenY, 0.0f);
    golden::FillRandomData(hostA,  -10.0f, 10.0f);
    // golden::FillRandomData(hostX,  -1.0f, 1.0f);
    // golden::FillRandomData(hostY,  -1.0f, 1.0f);

    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    // uint8_t *deviceX{nullptr};
    // ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceX), sizeX, ACL_MEM_MALLOC_HUGE_FIRST));
    // ACL_CHECK(aclrtMemcpy(deviceX, sizeX, hostX.data(), sizeX, ACL_MEMCPY_HOST_TO_DEVICE));

    // uint8_t *deviceY{nullptr};
    // ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceY), sizeY, ACL_MEM_MALLOC_HUGE_FIRST));
    // ACL_CHECK(aclrtMemcpy(deviceY, sizeY, hostY.data(), sizeY, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceZ{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceZ), sizeY, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceZ, sizeY, hostY.data(), sizeY, ACL_MEMCPY_HOST_TO_DEVICE));
    
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::GemvAtlasA2;

    /*
    Gemm::GemvAtlasA2,
    Gemv::helper::FT_ENC_TYPE::RCE,
    */

    using AType = Gemm::GemmType<float, LayoutA>;
    using XType = Gemm::GemmType<float, LayoutX>;
    using YType = Gemm::GemmType<float, LayoutY>;
    using BiasType = void;

    /*
    template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    /// MatmulType type for X vector operand
    class XType,
    /// MatmulType type for Y vector operand
    class YType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
    >
    struct TileCopyMatrixReduceAiv
    */

    using TileCopy = Gemv::Tile::TileCopyMatrixReduceAiv<typename DispatchPolicy::ArchTag, AType, XType, YType, BiasType>;
    // using TileVmad = Gemv::Tile::TileVmad<typename DispatchPolicy::ArchTag, AType, XType, YType, BiasType>;
    using TileVmuls = Gemv::Tile::TileVmuls<typename DispatchPolicy::ArchTag, YType>;
    /*
    template <
    /// Tag indicating architecture
    class ArchTag,
    class AType,
    class XType,
    class YType,
    class BiasType = void
    >
    struct TileThreCalc
    */
    using TileThreCalc = Gemv::Tile::TileThreCalc<typename DispatchPolicy::ArchTag, AType, XType, YType, BiasType>; 

    /*
    template <
    class UBTileShape_,
    class AType_,
    class XType_,
    class YType_,
    class BiasType_,
    class TileCopy_,
    class TileThreCalc_,
    class TileVmuls_
    >
    struct BlockThresholdCalc
    */

    using BlockThresholdCalc = Gemv::Block::BlockThresholdCalc<DispatchPolicy, Gemv::helper::FT_ENC_TYPE::RCE, UBTileShape, AType, XType, YType, BiasType, TileCopy, TileThreCalc, TileVmuls>;
    using BlockEpilogue = void;

    // kernel level
    /*
    struct Arguments {
        GemvCoord problemShape;
        GM_ADDR ptrA;
        GM_ADDR ptrZ;
        float rounding_exponent;
        float beta;   
    };
    // deviceX, deviceY, 
    alpha, beta, split
    */
    using ThreCalcKernel = Gemv::Kernel::KernelThreCalcAiv<BlockThresholdCalc, BlockEpilogue>;
    typename ThreCalcKernel::Arguments arguments{options.problemShape, deviceA, deviceZ, options.round_exp, options.beta};
    
    using ThreCalAdapter = Gemv::Device::DeviceGemv<ThreCalcKernel>;
    ThreCalAdapter threshold_op;
    threshold_op.CanImplement(arguments);
    RunAdapter(threshold_op, arguments, stream, aicCoreNum);

    std::vector<float> hostRes(lenY);
    ACL_CHECK(aclrtMemcpy(hostRes.data(), sizeY, deviceZ, sizeY, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenY);
    /*
    void ComputeThresholds(
    const Catlass::GemvCoord &problemShape,
    Element round_exp, Element beta,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataGolden, const LayoutGolden &layoutGolden
    )
    */
    golden::ComputeThresholds(options.problemShape, options.round_exp, options.beta,hostA, layoutA, hostGolden, layoutY);
    // golden::ComputeGemv(options.problemShape, alpha, beta, hostA, layoutA, hostX, layoutX, hostY, layoutY, hostGolden, layoutY);
    std::vector<uint64_t> errorIndices = golden::CompareData(hostRes, hostGolden, m);
    for(int jj = 0; jj < 10; jj++){
        printf("computed: %f real: %f\n", hostRes[jj],hostGolden[jj]);
        // printf("real:")
    }
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    printf("Total Error Idx len: %d\n", static_cast<int>(totalErrorIdxRow.size()));
    printf("Total Error Data len: %d\n", static_cast<int>(totalErrorDataRow.size()));

    for (int i = 0; i < 100; ++i) {
        ACL_CHECK(aclrtSynchronizeStream(stream));
        RunAdapter(threshold_op, arguments, stream, aicCoreNum);
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

        RunAdapter(threshold_op, arguments, stream, aicCoreNum);
            
        ACL_CHECK(aclrtSynchronizeStream(stream));
        ACL_CHECK(aclrtRecordEvent(stop, stream));
        ACL_CHECK(aclrtSynchronizeEvent(stop));
        ACL_CHECK(aclrtEventElapsedTime(&temp_time, start, stop));
        time += temp_time;
    }

    std::cout << "m: " << m << ", n: " << n << ", " << (float)2 * m * n / (time / num_repeat * 1e-3) / 1e9 << " GFLOPS, " << (time / num_repeat) << " ms, repeat: " << num_repeat << std::endl;

    ACL_CHECK(aclrtFree(deviceA));
    // ACL_CHECK(aclrtFree(deviceX));
    // ACL_CHECK(aclrtFree(deviceY));
    ACL_CHECK(aclrtFree(deviceZ));

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char **argv){
    Options options;
    if(options.Parse(argc, argv) != 0){
        return -1;
    }
    Run(options);
    return 0;
}