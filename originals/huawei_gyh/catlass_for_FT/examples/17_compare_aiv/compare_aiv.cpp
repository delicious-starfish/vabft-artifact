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
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemv/kernel/kernel_large_local_compare.hpp"
#include "catlass/gemv/block/block_large_local_compare.hpp"
#include "catlass/gemv/kernel/kernel_gemv_aiv.hpp"
#include "catlass/gemv/block/block_gemv.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemv/tile/tile_copy.hpp"
#include "catlass/gemv/tile/tile_vmad.hpp"
#include "catlass/gemv/tile/tile_vmuls.hpp"
#include "catlass/gemv/device/device_gemv.hpp"
#include "catlass/status.hpp"

using namespace Catlass;

using ScalarType = float;

struct Options{
    const std::string HELPER = "17_compare_aiv m n [device_id]";

    uint32_t M = 32;
    uint32_t N = 32;
    uint32_t deviceId{0};

    Options() = default;
    
    GemvCoord problemShape{M, N};

    int Parse(int argc, const char **argv)
    {
        enum ArgsIndex {
            M_INDEX = 1,
            N_INDEX,
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
        if (argc == ARGS_MAX)
        {
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        }
        return 0;
    }
};

// uint32_t getSplictNum(bool trans, uint32_t M, uint32_t N, uint32_t M1, uint32_t N1, uint32_t maxSplict)
// {
//     uint32_t CORENUM = 20;
//     uint32_t splitNum = 1;
//     uint32_t maxOccupancy = 0; 
//     uint32_t blockNum = (M - 1) / M1 + 1;
//     if (!trans)
//     {
//         splitNum = 1;
//     }
//     else{
//         uint32_t splitNum1 = 1, splitNum2 = 1;
//         for (uint32_t i = 1; i <= maxSplict; i += 1)
//         {
//             uint32_t occupancy = (i * blockNum) % (CORENUM * 2);
//             if (!occupancy)
//                 occupancy = (CORENUM * 2);
//             if (occupancy > maxOccupancy)
//             {
//                 maxOccupancy = occupancy;
//                 splitNum1 = i;
//             }
//         }
//         maxOccupancy = 0;
//         for (uint32_t i = 1; i <= maxSplict; i <<= 1)
//         {
//             uint32_t occupancy = (i * blockNum) % (CORENUM * 2);
//             if (!occupancy)
//                 occupancy = (CORENUM * 2);
//             if (occupancy > maxOccupancy)
//             {
//                 maxOccupancy = occupancy;
//                 splitNum2 = i;
//             }
//         }
//         splitNum = (splitNum1 - splitNum2) > 4 ? splitNum1 : splitNum2;
//     }
//     return splitNum;
// }

// , std::vector<ElementWork>* hostWorkspace
// , class ElementWork
// , std::vector<ElementWork>* hostWorkspace
// , class ElementWork
template <class Adapter>
void RunAdapter(Adapter compare_op, typename Adapter::Arguments args, aclrtStream stream, uint32_t aicCoreNum)
{
    size_t sizeWorkspace = compare_op.GetWorkspaceSize(args);
    uint8_t *deviceWorkspace = nullptr;
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    compare_op.Initialize(args, deviceWorkspace);

    compare_op(stream, aicCoreNum);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    // if(sizeWorkspace > 0 && args.OutputWorkspace){
    //     ACL_CHECK(aclrtMemcpy((*hostWorkspace).data(), sizeWorkspace, deviceWorkspace, sizeWorkspace, ACL_MEMCPY_DEVICE_TO_HOST));
    // }

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
    using UBTileShapeCOMP = GemvShape<1,8192>;

    size_t lenX = static_cast<size_t>(m) * n;
    size_t lenY = static_cast<size_t>(m) * n;
    
    size_t lenZ = (lenX + 8 -1)/8;

    size_t sizeX = lenX * sizeof(float);
    size_t sizeY = lenY * sizeof(float);
    size_t sizeZ = lenZ * sizeof(uint8_t);

    using LayoutX = layout::VectorLayout;
    using LayoutY = layout::VectorLayout;
    using LayoutZ = layout::VectorLayout;
    
    LayoutX layoutX{m*n};
    LayoutY layoutY{m*n};
    LayoutZ layoutZ{(m*n + 8 - 1)/8};


    ScalarType threshold{0.0002f};
    
    // FillRandomScalarData(alpha, -1.0f, 1.0f);
    // FillRandomScalarData(beta, -1.0f, 1.0f);

    std::vector<float> hostX(lenX, 2.0f);
    //  2.0003f 2.00003f
    std::vector<float> hostY(lenY, 2.0003f);
    std::vector<uint8_t> hostZ(lenZ);
    // golden::FillRandomData(hostA,  -1.0f, 1.0f);
    // golden::FillRandomData(hostX,  -1.0f, 1.0f);
    // golden::FillRandomData(hostY,  -1.0f, 1.0f);

    uint8_t *deviceX{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceX), sizeX, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceX, sizeX, hostX.data(), sizeX, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceY{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceY), sizeY, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceY, sizeY, hostY.data(), sizeY, ACL_MEMCPY_HOST_TO_DEVICE));


    uint8_t *deviceZ{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceZ), sizeZ, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceZ, sizeZ, hostZ.data(), sizeZ, ACL_MEMCPY_HOST_TO_DEVICE));
    
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    using ArchTag = Arch::AtlasA2;
    using COMPDispatchPolicy = Gemm::GemvAtlasA2;

    using XType = Gemm::GemmType<float, LayoutX>;
    using YType = Gemm::GemmType<float, LayoutY>;
    using ZType = Gemm::GemmType<uint8_t, LayoutZ>;
    using BiasType = void;
    /*
    template <
    Catlass::Gemv::helper::FT_COMP_TYPE COMP_TYPE_,
    /// Tag indicating architecture
    class ArchTag,
    /// Output Result operand namely vector Z
    class ZType,
    /// MatmulType type for X vector operand
    class XType,
    /// MatmulType type for Y vector operand
    class YType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
    >
    struct TileCopyCompareAiv
    */
    using FT_COMP_TYPE = Catlass::Gemv::helper::FT_COMP_TYPE;

    using TileCompareCopy = Gemv::Tile::TileCopyCompareAiv<FT_COMP_TYPE::XOR, typename COMPDispatchPolicy::ArchTag, ZType, XType, YType, BiasType>;

    /*
    template <
    Catlass::Gemv::helper::FT_COMP_TYPE COMP_TYPE_,
    class UBTileShape_,
    class ZType_,
    class XType_,
    class YType_,
    class TileCopy_
    >

    struct BlockCompare <
        Gemm::GemvAtlasA2,
        COMP_TYPE_,
        UBTileShape_,
        ZType_,
        XType_,
        YType_,
        TileCopy_,
    >
    */
    using CompareBlock = Gemv::Block::BlockCompare<COMPDispatchPolicy,FT_COMP_TYPE::XOR, UBTileShapeCOMP, ZType, XType, YType, TileCompareCopy>;
    using BlockEpilogue = void;

    // kernel level
    using ElementWork = typename std::conditional<
        (CompareBlock::COMP_TYPE == FT_COMP_TYPE::XOR),
        uint16_t,
        typename std::conditional<(CompareBlock::COMP_TYPE == FT_COMP_TYPE::COMPARE), int32_t, float>::type>::type;

    size_t sizeW = RoundUp(static_cast<uint32_t>(lenX*sizeof(float)),static_cast<uint32_t>(sizeof(ElementWork)));
    size_t lenW = sizeW / sizeof(ElementWork);

    printf("workspace len: %zu\n", lenW);
    std::vector<ElementWork> hostWorkspace(lenW);

    using CompareKernel = Gemv::Kernel::KernelCompareAiv<CompareBlock, BlockEpilogue>;
    /*
    struct Arguments {
        GemvCoord problemShape;
        GM_ADDR ptrInputX;
        GM_ADDR ptrInputY;
        GM_ADDR ptrOutputZ;
        uint32_t UbNum;
        bool OutputWorkspace;
        ElementX threshold;
    };
    */

    uint32_t UbNum = 2;
    bool OutputWorkspace = true;
    /*
    struct Arguments {
        GemvCoord problemShape;
        GM_ADDR ptrInputX;
        GM_ADDR ptrInputY;
        GM_ADDR ptrOutputZ;
        uint32_t UbNum;
        bool OutputWorkspace;
        ElementX threshold;
    };
    */

    typename CompareKernel::Arguments arguments{options.problemShape, deviceX, deviceY, deviceZ, UbNum, false, threshold};
    typename CompareKernel::Arguments arguments_test{options.problemShape, deviceX, deviceY, deviceZ, UbNum, false, threshold};
    
    using CompareAdapter = Gemv::Device::DeviceGemv<CompareKernel>;
    CompareAdapter compare_op;
    compare_op.CanImplement(arguments);
    /*
    RunAdapter(Adapter compare_op, typename Adapter::Arguments args, aclrtStream stream,
    uint32_t aicCoreNum, std::vector<ElementWork> & hostWorkspace)
    */
    // , &hostWorkspace
    // , &hostWorkspace
    
    RunAdapter(compare_op, arguments, stream, aicCoreNum);

    std::vector<uint8_t> hostRes(lenZ);
    ACL_CHECK(aclrtMemcpy(hostRes.data(), sizeZ, deviceZ, sizeZ, ACL_MEMCPY_DEVICE_TO_HOST));
    printf("%f\n", hostX[0]);
    printf("%f\n", hostY[0]);
    printf("%u\n",hostWorkspace[0]);
    printf("%hhu\n",hostRes[0]);
    printf("%hhu\n",hostRes[lenZ - 1]);


    std::vector<uint8_t> hostGolden(lenZ,255);
    // golden::ComputeGemv(options.problemShape, alpha, beta, hostA, layoutA, hostX, layoutX, hostY, layoutY, hostGolden, layoutY);
    std::vector<uint64_t> errorIndices = golden::CompareData(hostRes, hostGolden, lenZ);
    printf("Method: XOR\n");
    if (errorIndices.empty()) {
        std::cout << "compare success." << std::endl;
    } else {
        std::cerr << "compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    for (int i = 0; i < 100; ++i) {
        ACL_CHECK(aclrtSynchronizeStream(stream));
        // , &hostWorkspace
        RunAdapter(compare_op, arguments_test, stream, aicCoreNum);
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

        // , &hostWorkspace
        RunAdapter(compare_op, arguments_test, stream, aicCoreNum);
            
        ACL_CHECK(aclrtSynchronizeStream(stream));
        ACL_CHECK(aclrtRecordEvent(stop, stream));
        ACL_CHECK(aclrtSynchronizeEvent(stop));
        ACL_CHECK(aclrtEventElapsedTime(&temp_time, start, stop));
        time += temp_time;
    }

    std::cout << "m: " << m << ", n: " << n << ", " << (time / num_repeat) << " ms, repeat: " << num_repeat << std::endl;

    ACL_CHECK(aclrtFree(deviceX));
    ACL_CHECK(aclrtFree(deviceY));
    ACL_CHECK(aclrtFree(deviceZ));
    // ACL_CHECK(aclrtFree(deviceZ));

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