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

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp" //catlass/
#include "gemm/dispatch_policy.hpp" // catlass/
// examples/cube_op_self/gemm/kernel/matmul_epilogue_splitk_preload.hpp
#include "gemm/kernel/matmul_epilogue_splitk_preload.hpp"
// #include "catlass/gemm/kernel/splitk_matmul.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

#include "catlass/status.hpp"
#include "catlass/gemm/device/device_gemm.hpp" // catlass/
// #include "catlass/gemm/device/device_gemm.hpp"

using namespace Catlass;
using fp16_t = op::fp16_t;


struct Options {
    const std::string HELPER = "09_splitk_matmul m n k e_max sliceKUnit splitKNumLimit [device_id]";

    GemmCoord problemGemmShape{1024, 1024, 1024};
    GemvCoord problemShape{128, 128};
    int32_t deviceId{0};

    uint32_t SliceKUnit{1024};
    uint32_t SplitKNumLimit{1};

    float e_max{0.003F};

    Options() = default;

    int Parse(int argc, const char **argv)
    {
        enum ArgsIndex {
            M_INDEX = 1,
            N_INDEX,
            K_INDEX,
            E_MAX_INDEX,
            SLICE_UNIT_INDEX,
            SPLIT_NUM_LIMIT_INDEX,
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

        problemShape.m() = std::atoi(argv[M_INDEX]);
        problemShape.n() = std::atoi(argv[N_INDEX]);

        SliceKUnit = std::atoi(argv[SLICE_UNIT_INDEX]);
        SplitKNumLimit = std::atoi(argv[SPLIT_NUM_LIMIT_INDEX]);

        e_max = static_cast<float>(std::stof(argv[E_MAX_INDEX]));

        if (argc == ARGS_MAX) {
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        }

        return 0;
    }
};


void Run(Options const &options)
{
    using       GemmInTypeC             =               half;
    using       GemmInTypeN             =               half;

    using       GemmOutTypeC            =               float;
    using       GemmOutTypeN            =               float;

    using       ScalarTypeC             =               float;
    using       ScalarTypeN             =               float;
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    // std::cout << "Device id: " << options.deviceId << std::endl;

    using L1TileShape = GemmShape<128, 256, 256>;
    using L0TileShape = GemmShape<128, 256, 64>;

    static constexpr uint32_t BYTE_FOR_EACH_BLK = 32;
    static constexpr uint32_t ELE_WORK_FOR_EACH_BLK = BYTE_FOR_EACH_BLK / sizeof(GemmOutTypeC);

    uint32_t m = options.problemGemmShape.m();
    uint32_t n = options.problemGemmShape.n();
    uint32_t k = options.problemGemmShape.k();

    GemvCoord problemShapeBE{k,n};
    GemvCoord problemShapeABE{m,k};

    
    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenC = static_cast<size_t>(m) * n;

    size_t sizeA = lenA * sizeof(GemmInTypeC);
    size_t sizeB = lenB * sizeof(GemmInTypeC);
    size_t sizeC = lenC * sizeof(GemmOutTypeC);

    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m, n};

    std::vector<GemmInTypeC> hostA(lenA);
    std::vector<GemmInTypeC> hostB(lenB);

    float sum_base = 1.0;
    size_t lenX =  static_cast<size_t>(n) * 1;
    size_t lenXV = static_cast<size_t>(n) * 1;

    uint32_t splitNnum = (n + L1TileShape::N - 1) / L1TileShape::N;

    uint32_t split_block_num = (splitNnum + ELE_WORK_FOR_EACH_BLK - 1) / ELE_WORK_FOR_EACH_BLK;

    uint32_t lenBMean = split_block_num * ELE_WORK_FOR_EACH_BLK + ELE_WORK_FOR_EACH_BLK;
    uint32_t lenBMax = lenBMean;


    size_t lenZRow = static_cast<size_t>(m) * splitNnum;
    size_t lenZCol = static_cast<size_t>(n) * 1;

    size_t lenThre = static_cast<size_t>(m) * splitNnum;

    std::vector<GemmInTypeC> hostX(lenX,(GemmInTypeC)sum_base);
    std::vector<GemmOutTypeC> hostXV(lenXV,(GemmOutTypeC)sum_base);

    size_t sizeX = lenX * sizeof(GemmInTypeC);
    size_t sizeXV = lenXV * sizeof(GemmOutTypeC);

    size_t lenDRow = (static_cast<size_t>(m)) * splitNnum;
    size_t lenDCol = (static_cast<size_t>(n)) * 1;

    size_t sizeZRow = lenZRow * sizeof(GemmOutTypeC);
    size_t sizeDRow = lenDRow * sizeof(GemmOutTypeC);

    size_t sizeBMean = lenBMean * sizeof(GemmOutTypeC);
    size_t sizeBMax = lenBMax * sizeof(GemmOutTypeC);

    std::vector<GemmOutTypeC> hostDRow(lenZRow,0.0f);
    std::vector<GemmOutTypeC> hostDCol(lenZCol,0.0f);

    std::vector<GemmOutTypeC> hostZRow(lenZRow,0.0f);
    std::vector<GemmOutTypeC> hostZCol(lenZCol,0.0f);

    golden::FillRandomData<GemmInTypeC>(hostA, -1.0f, 1.0f);
    golden::FillRandomData<GemmInTypeC>(hostB, -1.0f, 1.0f);

    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    using ArchTag = Arch::AtlasA2;
    constexpr bool enableUnitFlag = false;
    using MmadDispatchPolicy = CubeSelf::Gemm::MmadAtlasA2Pingpong<enableUnitFlag>;
    // using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;
    

    using LayoutVC = layout::VectorLayout;

    using AType = Gemm::GemmType<GemmInTypeN, LayoutA>;
    using BType = Gemm::GemmType<GemmInTypeN, LayoutB>;
    using CType = Gemm::GemmType<GemmOutTypeN, LayoutC>;

    // using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    // Preload
    using BlockMmad = CubeSelf::Gemm::Block::BlockMmadPreload<
        MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    using BlockEpilogue = void;

    // After the Matmul computation is completed, launch the ReduceAdd kernel to accumulate the partial sums.
    constexpr uint32_t computeLength = 32 * 1024 / sizeof(GemmOutTypeC);
    using ReduceAdd = CubeSelf::Gemm::Kernel::ReduceAdd<ArchTag, GemmOutTypeN, GemmOutTypeN, computeLength>;
    // Catlass
    // Swizzle offset is 3 and direction is 0.
    using BlockScheduler = typename Catlass::Gemm::Block::SplitkGemmIdentityBlockSwizzle<3, 0>;

    // kernel level
    /*
    template <
        class BlockMmad_,
        class BlockEpilogue_,
        class BlockScheduler_,
        class ReduceAdd_
    >
    class MatmulAsVarSplitKPreload
    */
    using MatmulKernel = CubeSelf::Gemm::Kernel::MatmulAsVarSplitKPreload<BlockMmad, BlockEpilogue, BlockScheduler, ReduceAdd>;

    using MatmulAdapter = Catlass::Gemm::Device::DeviceGemm<MatmulKernel>;
    // MatmulKernel::Arguments arguments{options.problemGemmShape,
    //     aicCoreNum,
    //     sizeof(GemmOutTypeC),
    //     deviceA,
    //     deviceB,
    //     deviceC};
    MatmulKernel::Arguments arguments{options.problemGemmShape,
        aicCoreNum,
        sizeof(GemmOutTypeC),
        deviceA,
        deviceB,
        deviceC,
        options.SliceKUnit,
        options.SplitKNumLimit
    };
    MatmulAdapter matmul_op;
    matmul_op.CanImplement(arguments);

    size_t sizeWorkspace = matmul_op.GetWorkspaceSize(arguments);
    uint8_t *deviceWorkspace = nullptr;
    if (sizeWorkspace > 0) {
        ACL_CHECK(
            aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST)
        );
    }
    matmul_op.Initialize(arguments, deviceWorkspace);

    // 跳过正确性检验

    matmul_op(stream, aicCoreNum, fftsAddr);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<GemmOutTypeC> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));
    printf("hostC[0]: %f\n",hostC[0]);

    LayoutVC layoutZHost{m * splitNnum};
    LayoutVC layoutXRow{n};
    LayoutVC layoutXCol{m};
    GemvCoord BESliceShape{1, L1TileShape::N};

    float alpha{1.0};
    float beta{0.0};

    golden::ComputeGemvSlice(options.problemShape, BESliceShape, alpha, beta,
         hostC, layoutC, hostXV, layoutXRow, hostDRow, layoutZHost, hostDRow);

    std::vector<GemmInTypeC> hostYForAB(static_cast<uint32_t>(k * splitNnum), (GemmInTypeC)0.0f);
    LayoutVC layoutYForAB{static_cast<uint32_t>(k * splitNnum)};

    golden::ComputeGemvSlice(problemShapeBE, BESliceShape, 
        alpha, beta, hostB, layoutB, hostX, layoutXRow, 
        hostYForAB, layoutYForAB, hostYForAB);

    GemvCoord ABESliceShape{static_cast<uint32_t>(splitNnum), problemShapeABE.n()};
    golden::ComputeGemvSlice(problemShapeABE, ABESliceShape, alpha, beta,
         hostA, layoutA, hostYForAB, layoutYForAB, 
         hostZRow, layoutZHost, hostZRow);

    std::vector<GemmOutTypeC> hostBMeanGolden(lenBMean,(GemmOutTypeC)0.0f);
    std::vector<GemmOutTypeC> hostBMaxGolden(lenBMax, (GemmOutTypeC)0.0f);

    GemvCoord BReduceSliceShape{problemShapeBE.m(), L1TileShape::N};
    golden::ComputeMeanMaxSlice(problemShapeBE, BReduceSliceShape,
        hostB, layoutB, hostBMeanGolden, hostBMaxGolden);

    std::vector<GemmOutTypeC> hostThreGolden(lenThre);

    GemvCoord ThreSliceShape{L1TileShape::M, L1TileShape::N};


    std::vector<GemmOutTypeC> hostAMaxGolden(lenThre);
    std::vector<GemmOutTypeC> hostAMeanGolden(lenThre);
    std::vector<GemmOutTypeC> hostAStdGolden(lenThre);

    Gemv::helper::FT_RCE_THRE_TYPE rce_thre_type = Gemv::helper::FT_RCE_THRE_TYPE::ROUND;

    golden::ComputeThresholdsASVARTSlice(problemShapeABE, splitNnum,
         hostBMeanGolden, hostBMaxGolden, options.problemShape.n(),
         L1TileShape::N, hostA, layoutA, hostAMeanGolden, hostAMaxGolden,
         hostAStdGolden, hostThreGolden, options.e_max, rce_thre_type);


    // // errorIndices = golden::CompareData(hostThre, hostThreGolden, lenThre);
    // errorIndices = golden::CompareData(hostThreGolden, hostThreGolden, lenThre);
    // if (errorIndices.empty()) {
    //     std::cout << "Threshold Compare success." << std::endl;
    // } else {
    //     std::cerr << "Threshold Compare failed. Error count: " << errorIndices.size() << std::endl;
    // }
    
    std::vector<uint8_t> hostGoldenCOMPRow(lenZRow,255);

    std::vector<uint64_t> totalErrorIdxRow;
    std::vector<uint64_t> totalErrorIdxRow_m;
    std::vector<uint64_t> totalErrorIdxRow_n;
    std::vector<float> totalErrorDataRow;
    std::vector<float> totalFailThresholds;
    
    
    printf("%f\n", hostZRow[0]);
    printf("%f\n", hostDRow[0]);
    printf("Method: Verify with Computed Threshold\n");

    /*
    template<class ElementData>
    std::vector<uint64_t> CompareDataAndIndexSliceWithThreshold(
        const Catlass::GemvCoord &problemShape,
        const std::vector<ElementData> &actualdata, 
        const std::vector<ElementData> &expectdata,
        const std::vector<ElementData> &thresholddata, 
        uint32_t computeNum, const char* IdNameAct, const char* IdNameExp,
        std::vector<uint64_t>& total_error_idies,
        std::vector<uint64_t>& total_error_idies_m,
        std::vector<uint64_t>& total_error_idies_n, 
        std::vector<ElementData>& total_error_data,
        std::vector<ElementData>& total_fail_threshold_data)
    */
    std::vector<uint64_t> errorIndices = golden::CompareDataAndIndexSliceWithThreshold(
        options.problemShape, hostDRow, hostZRow, hostThreGolden, 
        lenZRow, "CE", "ABE", totalErrorIdxRow, totalErrorIdxRow_m, totalErrorIdxRow_n,
        totalErrorDataRow, totalFailThresholds
    );
    
    if (errorIndices.empty()) {
        std::cout << "Row COMP OP compare success." << std::endl;
    } else {
        std::cerr << "Row COMP OP compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    printf("Total Error Idx len: %d\n", static_cast<int>(totalErrorIdxRow.size()));
    printf("Total Error Data len: %d\n", static_cast<int>(totalErrorDataRow.size()));

    // std::vector<float> hostGolden(lenC);
    // golden::ComputeMatmul(options.problemGemmShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC);

    // std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k);
    // if (errorIndices.empty()) {
        // std::cout << "Compare success." << std::endl;
    // } else {
        // std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    // }

    // 正确性检验完成，开始测试速度

    // 热身暖机
    for(int i = 0;i < 100; ++ i){
        ACL_CHECK(aclrtSynchronizeStream(stream));
        matmul_op(stream, aicCoreNum, fftsAddr);
    }

    // 运行10000遍测试结果
    int                 num_repeat                  =                   10000;
    float               time                        =                   0;
    float               temp_time                   =                   0;
    aclrtEvent          start;
    aclrtEvent          stop;

    ACL_CHECK(aclrtCreateEvent(&start));
    ACL_CHECK(aclrtCreateEvent(&stop));

    for(int i = 0;i < num_repeat; ++ i){
        ACL_CHECK(aclrtSynchronizeStream(stream));
        ACL_CHECK(aclrtRecordEvent(start, stream));

        matmul_op(stream, aicCoreNum, fftsAddr);

        ACL_CHECK(aclrtSynchronizeStream(stream));
        ACL_CHECK(aclrtRecordEvent(stop, stream));
        ACL_CHECK(aclrtSynchronizeEvent(stop));
        ACL_CHECK(aclrtEventElapsedTime(&temp_time, start, stop));
        time += temp_time;
    }

    float               average_time                =                   time / num_repeat;

    std::cout << "m: " << m << ", n: " << n << ", k: " << k << ", " << "time_usage = " << average_time << " ms" << std::endl;

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
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
