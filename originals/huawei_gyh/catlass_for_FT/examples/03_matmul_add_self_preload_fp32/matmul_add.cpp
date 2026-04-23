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
// #include "gemm/block/block_mmad.hpp" // catlass/
#include "gemm/block/block_mmad.hpp" // catlass/
#include "gemm/block/block_swizzle.hpp" // catlass/
#include "gemm/dispatch_policy.hpp" // catlass/
#include "gemm/kernel/matmul_epilogue.hpp" // catlass/ catlass/
#include "gemm/kernel/matmul_no_epilogue.hpp" // catlass/ catlass/
// examples/cube_op_self/gemm/kernel/matmul_epilogue_preload.hpp
#include "gemm/kernel/matmul_epilogue_preload.hpp" // L1-level prefecthing optimization
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
    GemvCoord problemShapeGemv{4096, 4096};
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

        problemShapeGemv.m() = std::atoi(argv[M_INDEX]);
        problemShapeGemv.n() = std::atoi(argv[N_INDEX]);
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
    using GemmInTypeC = float;
    // op_bfloat16;
    // op_bfloat16;
    using GemmInTypeN = float;
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

    GemvCoord problemShapeBE{k,n};
    GemvCoord problemShapeABE{m,k};

    // Compute the length of each matrix and the size of each buffer
    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenC = static_cast<size_t>(m) * n;
    size_t lenD = static_cast<size_t>(m) * n;
    size_t lenXRaw = lenD;

    size_t sizeA = lenA * sizeof(GemmInTypeC);
    size_t sizeB = lenB * sizeof(GemmInTypeC);
    size_t sizeC = lenC * sizeof(GemmOutTypeC);
    size_t sizeD = lenD * sizeof(GemmOutTypeC);

    // Define the layout of each matrix
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    using LayoutVC = layout::VectorLayout;

    using L1TileShape = GemmShape<128, 256, 128>;
    using L0TileShape = GemmShape<128, 256, 32>;

    static constexpr uint32_t BYTE_FOR_EACH_BLK = 32;
    static constexpr uint32_t ELE_WORK_FOR_EACH_BLK = BYTE_FOR_EACH_BLK / sizeof(GemmOutTypeC);
    
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutD{m, n};
    LayoutC layoutC{m, n};

    // Prepare input data A, B, and X
    std::vector<GemmInTypeC> hostA(lenA);
    std::vector<GemmInTypeC> hostB(lenB);
    std::vector<GemmOutTypeC> hostXRaw(lenXRaw);

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
    golden::FillRandomData<GemmOutTypeC>(hostXRaw, 0.0f, 0.0f);

    // Allocate device memory and copy data from host to device
    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    // The data of X is stored on deviceD to save storage space
    uint8_t *deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceC, sizeC, hostXRaw.data(), sizeC, ACL_MEMCPY_HOST_TO_DEVICE));

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
    
    // using L1TileShape = GemmShape<128, 256, 256>;
    // using L0TileShape = GemmShape<128, 256, 64>;
    using AType = Gemm::GemmType<GemmInTypeN, LayoutA>;
    using BType = Gemm::GemmType<GemmInTypeN, LayoutB>;
    using CType = Gemm::GemmType<GemmOutTypeN, LayoutC>;
    /*
    template<
    class DispatchPolicy,
    class L1TileShape,
    class L0TileShape,
    class AType,
    class BType,
    class CType,
    class BiasType = void,
    class TileCopy = CubeSelf::Gemm::Tile::TileCopy<typename DispatchPolicy::ArchTag, AType, BType, CType, BiasType>,
    class TileMmad = CubeSelf::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>
    >
    struct BlockMmadPreload
    */
    using BlockMmadPreload = CubeSelf::Gemm::Block::BlockMmadPreload<
        MmadDispatchPolicy, 
        L1TileShape, 
        L0TileShape, 
        AType, 
        BType, 
        CType
    >;

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

    std::vector<GemmOutTypeC> hostC(lenC);
    if (m >= n) {
        // Define BlockScheduler
        // Swizzle offset is 3 and direction is 0.
        // CubeSelf::
        using BlockScheduler = typename CubeSelf::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
        // Kernel level
        // CubeSelf:: CubeSelf::
        // using MatmulKernel = CubeSelf::Gemm::Kernel::MatmulNoEpilogue<BlockMmad, BlockEpilogue, BlockScheduler>;
        // BlockEpilogue, 
        /*
        template <
            class BlockMmad_,
            // class BlockEpilogue_,
            class BlockScheduler_
        >
        class MatmulEpiloguePreload{
        */
        // BlockEpilogue,
        using MatmulKernel = CubeSelf::Gemm::Kernel::MatmulEpiloguePreload<
            BlockMmadPreload, BlockScheduler>;
        // Prepare params
        /*
        struct Arguments {
            Catlass::GemmCoord problemShape;
            size_t elementSize;
            GM_ADDR ptrA;
            GM_ADDR ptrB;
            GM_ADDR ptrC;
        };
        */
        typename MatmulKernel::Arguments arguments{
            options.problemShape, sizeof(GemmOutTypeC), deviceA, deviceB, deviceC};
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
            // Completely validate the correctness of each element in D, which may be time-consuming for large matrices. 
            // Please set make_golden to 0 for performance testing without correctness validation.
            // Copy the result from device to host
            ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

            // Compute the golden result
            std::vector<GemmOutTypeC> hostGolden(lenC);
            golden::ComputeMatmulElemWiseAdd(options.problemShape, hostA, layoutA, hostB, layoutB, hostXRaw, hostGolden, layoutC);

            // Compare the result
            std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k);
            if (errorIndices.empty()) {
                std::cout << "Compare success." << std::endl;
            } else {
                std::cout << "Compare failed. Error count: " << errorIndices.size() << std::endl;
            }
        }else{
            // validate the correctness of the checksum vectors

            ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));
            printf("hostC[0]: %f\n",hostC[0]);

            LayoutVC layoutZHost{m * splitNnum};
            LayoutVC layoutXRow{n};
            LayoutVC layoutXCol{m};
            
            GemvCoord BESliceShape{1, L1TileShape::N};

            float alpha{1.0};
            float beta{0.0};
            float e_max = 16.0f / 1000000.0f; // 0.000016, which is 2 times of the maximum error of FP16, to make the threshold more reasonable.
            if(k <= 1024){
                e_max = e_max * 1.0f;
            }else{
                e_max = e_max * std::sqrt(((k*1.0f / 1024*1.0f)*1.0f));
            }
            

            golden::ComputeGemvSlice(options.problemShapeGemv, BESliceShape, alpha, beta,
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
                hostBMeanGolden, hostBMaxGolden, options.problemShapeGemv.n(),
                L1TileShape::N, hostA, layoutA, hostAMeanGolden, hostAMaxGolden,
                hostAStdGolden, hostThreGolden, e_max, rce_thre_type);
    
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
                options.problemShapeGemv, hostDRow, hostZRow, hostThreGolden, 
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

        }
        
        for (int i = 0; i < 100; ++i) {
            ACL_CHECK(aclrtSynchronizeStream(stream));
            // RunAdapter(gemv_op, arguments, stream, aicCoreNum);
            // RunAdapter(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
            matmul_op(stream, aicCoreNum, fftsAddr);
            ACL_CHECK(aclrtSynchronizeStream(stream));
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
        ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));
    } else {
        // Define BlockScheduler
        // Swizzle offset is 3 and direction is 1.
        // CubeSelf::
        using BlockScheduler = typename CubeSelf::Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
        /*
        template <
            class BlockMmad_,
            // class BlockEpilogue_,
            class BlockScheduler_
        >
        class MatmulEpiloguePreload{
        */
        // BlockEpilogue,
        using MatmulKernel = CubeSelf::Gemm::Kernel::MatmulEpiloguePreload<
            BlockMmadPreload, BlockScheduler>;
        // Prepare params
        // Prepare params
        // CubeSelf::
        typename MatmulKernel::Arguments arguments{
            options.problemShape, sizeof(GemmOutTypeC), deviceA, deviceB, deviceC};

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
            ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

            // Compute the golden result
            std::vector<GemmOutTypeC> hostGolden(lenC);
            golden::ComputeMatmulElemWiseAdd(options.problemShape, hostA, layoutA, hostB, layoutB, hostXRaw, hostGolden, layoutC);

            // Compare the result
            std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k);
            if (errorIndices.empty()) {
                std::cout << "Compare success." << std::endl;
            } else {
                std::cout << "Compare failed. Error count: " << errorIndices.size() << std::endl;
            }
        }else{
            // validate the correctness of the checksum vectors

            ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));
            printf("hostC[0]: %f\n",hostC[0]);

            LayoutVC layoutZHost{m * splitNnum};
            LayoutVC layoutXRow{n};
            LayoutVC layoutXCol{m};
            
            GemvCoord BESliceShape{1, L1TileShape::N};

            float alpha{1.0};
            float beta{0.0};
            float e_max = 16.0f / 1000000.0f; // 0.000016, which is 2 times of the maximum error of FP16, to make the threshold more reasonable.
            if(k <= 1024){
                e_max = e_max * 1.0f;
            }else{
                e_max = e_max * std::sqrt(((k*1.0f / 1024*1.0f)*1.0f));
            }
            // * std::sqrt(((k / 1024)*1.0f));

            golden::ComputeGemvSlice(options.problemShapeGemv, BESliceShape, alpha, beta,
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
                hostBMeanGolden, hostBMaxGolden, options.problemShapeGemv.n(),
                L1TileShape::N, hostA, layoutA, hostAMeanGolden, hostAMaxGolden,
                hostAStdGolden, hostThreGolden, e_max, rce_thre_type);
    
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
                options.problemShapeGemv, hostDRow, hostZRow, hostThreGolden, 
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

        }

        for (int i = 0; i < 100; ++i) {
            ACL_CHECK(aclrtSynchronizeStream(stream));
            // RunAdapter(gemv_op, arguments, stream, aicCoreNum);
            // RunAdapter(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
            matmul_op(stream, aicCoreNum, fftsAddr);
            ACL_CHECK(aclrtSynchronizeStream(stream));
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
        ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));
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
