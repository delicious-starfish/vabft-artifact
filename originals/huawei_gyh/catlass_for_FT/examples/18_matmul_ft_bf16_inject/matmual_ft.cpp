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
#include "catlass/gemv/block/block_large_local_compare.hpp"
#include "catlass/gemv/block/block_threshold_compare_fused.hpp"
#include "catlass/gemv/tile/tile_std_estimate.hpp"
// examples/cube_op_self/gemm/kernel/matmul_epilogue_double_FT.hpp
// #include "catlass/gemv/kernel/kernel_gemv_FT_double_total_aiv.hpp"

// #include "catlass/gemv/kernel/kernel_gemv_aic_FT.hpp"
// #include "catlass/gemv/kernel/kernel_gemv_FT_double.hpp"
#include "catlass/gemv/tile/tile_copy.hpp"
#include "catlass/gemv/helper.hpp"

#include "catlass/gemv/tile/tile_threshold.hpp"
#include "catlass/gemv/tile/tile_slice_reduce_sum.hpp"
#include "catlass/gemv/block/block_slice_reduce_sum.hpp"

#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"

#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_add.hpp"
#include "catlass/epilogue/tile/tile_elemwise_muls.hpp"

#include "catlass/gemv/tile/tile_fault_copy.hpp"
#include "catlass/gemv/tile/tile_fault_vmad.hpp"
#include "catlass/gemv/tile/tile_fault_sum.hpp"
#include "catlass/gemv/tile/tile_vmuls.hpp"
#include "catlass/gemv/tile/tile_vmad.hpp"
#include "catlass/gemv/tile/tile_copy.hpp"

#include "catlass/layout/layout.hpp"
#include "catlass/gemv/device/device_gemv.hpp"
#include "catlass/status.hpp"

#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_add.hpp"
#include "gemm/block/block_mmad.hpp" // catlass/
#include "gemm/block/block_swizzle.hpp" // catlass/
#include "gemm/dispatch_policy.hpp" // catlass/
#include "gemm/kernel/matmul_epilogue.hpp" // catlass/
// #include "gemm/kernel/matmul_epilogue_double_FT.hpp"
// #include "gemm/kernel/matmul_epilogue_double_FT_thre_no_splitk.hpp"
// examples/cube_op_self/gemm/kernel/
// #include "gemm/kernel/matmul_epilogue_asvar_abft_FT_thre_no_splitk_aiv_pipe.hpp"
// #include "gemm/kernel/matmul_epilogue_asvar_abft_FT_thre_no_splitk_aiv_pipe_relieve_mixed.hpp"
// #include "catlass/status.hpp"
// #include "gemm/kernel/matmul_epilogue_asvar_thre_abft_no_splitk_aic_aiv_pipe_mixed.hpp"
// #include "gemm/kernel/matmul_epilogue_asvar_thre_abft_no_splitk_aic_aiv_pipe_mixed_spec.hpp"

// #include "gemm/kernel/matmul_epilogue_asvar_thre_abrft_no_splitk_aic_aiv_pipe_mixed_chunk_robust_preload.hpp"
#include "gemm/kernel/matmul_epilogue_half_inject.hpp"
// #include "gemm/kernel/matmul_epilogue_asvar_thre_abft_no_splitk_aic_aiv_pipe_mixed_spec_robust_preload.hpp"
#include "gemm/device/device_gemm.hpp" // catlass/

#include "fp16_t.h"
#include "bfloat16.h"

using namespace Catlass;


using fp16_t = op::fp16_t;
using op_bfloat16 = op::bfloat16;

using GemmInTypeC = op_bfloat16;
// op_bfloat16;
using GemmInTypeN = bfloat16_t;
// bfloat16_t;

using GemmOutTypeC = float;
using GemmOutTypeN = float;

using GemvInTypeCforCE = float;
// op_bfloat16;
using GemvInTypeNforCE = float;
// bfloat16_t;

using GemvInTypeCforAB = op_bfloat16;
// op_bfloat16;
using GemvInTypeNforAB = bfloat16_t;
// bfloat16_t;

using GemvOutTypeC = float;
using GemvOutTypeN = float;

using ScalarTypeC = float;
using ScalarTypeN = float;

using ScalarType = float;


struct Options {
    const std::string HELPER = "18_matmul_ft_bf16_inject m n k rt beta thre_type e_max red_cores split_ks device_id inject_row inject_col inject_bit";

    GemmCoord problemGemmShape{128, 128, 128};
    GemvCoord problemShape{128, 128};

    int32_t deviceId{1};

    float round_exp{0.0f};
    float beta{1.0f};

    int thre_type{0};

    uint32_t reduce_cores{8};
    uint32_t split_ks{1};

    float e_max;

    // Fault injection parameters
    int32_t inject_row{-1};
    int32_t inject_col{-1};
    int32_t inject_bit{-1};
    int32_t skip_gemm{0};

    Options() = default;

    int Parse(int argc, const char** argv) {
        enum ArgsIndex {
            M_INDEX = 1,
            N_INDEX,
            K_INDEX,
            RT_INDEX,
            BETA_INDEX,
            THRE_TYPE_INDEX,
            E_MAX_INDEX,
            RED_CORES_INDEX,
            SPLIT_KS_INDEX,
            DEVICE_ID_INDEX,
            INJECT_ROW_INDEX,
            INJECT_COL_INDEX,
            INJECT_BIT_INDEX,
            ARGS_MAX
        };
        if (argc > ARGS_MAX || argc < N_INDEX) {
            std::cerr << HELPER << std::endl;
            return -1;
        }
        problemGemmShape.m() = std::atoi(argv[M_INDEX]);
        problemGemmShape.n() = std::atoi(argv[N_INDEX]);
        problemGemmShape.k() = std::atoi(argv[K_INDEX]);

        problemShape.m() = std::atoi(argv[M_INDEX]);
        problemShape.n() = std::atoi(argv[N_INDEX]);

        round_exp = static_cast<float>(std::stof(argv[RT_INDEX]));
        beta = static_cast<float>(std::stof(argv[BETA_INDEX]));

        thre_type = std::atoi(argv[THRE_TYPE_INDEX]);

        e_max = static_cast<float>(std::stof(argv[E_MAX_INDEX]));

        reduce_cores = std::atoi(argv[RED_CORES_INDEX]);
        split_ks = std::atoi(argv[SPLIT_KS_INDEX]);

        if (argc > DEVICE_ID_INDEX) {
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        }
        if (argc > INJECT_ROW_INDEX) {
            inject_row = std::atoi(argv[INJECT_ROW_INDEX]);
        }
        if (argc > INJECT_COL_INDEX) {
            inject_col = std::atoi(argv[INJECT_COL_INDEX]);
        }
        if (argc > INJECT_BIT_INDEX) {
            inject_bit = std::atoi(argv[INJECT_BIT_INDEX]);
        }
        return 0;
    }
};

template <class Adapter>
void RunAdapter(Adapter matmul_op, typename Adapter::Arguments args, aclrtStream stream,
    uint32_t aicCoreNum, uint64_t fftsAddr)
{
    size_t sizeWorkspace = matmul_op.GetWorkspaceSize(args);
    uint8_t *deviceWorkspace = nullptr;
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    matmul_op.Initialize(args, deviceWorkspace);
    // printf("Initialized!!!!\n");
    matmul_op(stream, aicCoreNum, fftsAddr);
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

    std::cout<<"Device ID: "<<options.deviceId<<std::endl;

    using L1TileShape = GemmShape<128, 256, 256>;
    using L0TileShape = GemmShape<128, 256, 64>;
    using FTChunkShape = GemvShape<512, 512>;

    

    // 64

    static constexpr uint32_t BYTE_FOR_EACH_BLK = 32;
    static constexpr uint32_t ELE_WORK_FOR_EACH_BLK = BYTE_FOR_EACH_BLK / sizeof(GemvOutTypeC);
    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemGemmShape.k();

    uint32_t Remain_Chunk_N = n % FTChunkShape::N;
    if (Remain_Chunk_N < 1){
        Remain_Chunk_N = FTChunkShape::N;
    }

    float lambda_R2 = 2.821 / (FTChunkShape::N * 1.0f);
    float lambda_R2_remain = 2.821 / (Remain_Chunk_N * 1.0f);

    GemvCoord problemShapeCol{n, m};

    GemvCoord problemShapeBR{k,n};
    GemvCoord problemShapeABR{m,k};

    GemvCoord problemShapeETA{k,m};
    GemvCoord problemShapeETAB{n,k};

    uint32_t splitNnumChunk = (n + FTChunkShape::N - 1) / FTChunkShape::N;
    uint32_t splitNnum = (n + L1TileShape::N - 1) / L1TileShape::N;

    uint32_t split_block_num = (splitNnumChunk + ELE_WORK_FOR_EACH_BLK - 1) / ELE_WORK_FOR_EACH_BLK;

    uint32_t lenBMean = split_block_num * ELE_WORK_FOR_EACH_BLK + ELE_WORK_FOR_EACH_BLK;
    uint32_t lenBMax = lenBMean;
    uint32_t lenBMin = lenBMean;
    uint32_t lenBMeanAbs = lenBMean;
    uint32_t lenBMeanSquare = lenBMean;
    uint32_t lenBVar = lenBMean;
    // uint32_t lenBMax = splitNnum * options.problemGemmShape.k();

    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenC = static_cast<size_t>(n) * m;

    

    // static_cast<size_t>(m) + static_cast<size_t>(n)

    size_t lenX =  static_cast<size_t>(n) * 1;
    size_t lenXR2 = static_cast<size_t>(FTChunkShape::N) * 1;
    size_t lenXR2Tail = static_cast<size_t>(Remain_Chunk_N) * 1;
    size_t lenXV = static_cast<size_t>(FTChunkShape::N) * 2;
    size_t lenXVTail = static_cast<size_t>(Remain_Chunk_N) * 2;
    size_t lenVXforAe = static_cast<size_t>(k) * 1;

    size_t lenDRowR1 = (static_cast<size_t>(m)) * splitNnumChunk;
    size_t lenDRowR2 = (static_cast<size_t>(m)) * splitNnumChunk;

    size_t lenCOMPRow = (((static_cast<size_t>(m)) + 8 - 1) / 8) * splitNnumChunk;

    size_t lenZRowR1 = static_cast<size_t>(m) * splitNnumChunk;
    size_t lenZRowR2 = static_cast<size_t>(m) * splitNnumChunk;
    size_t lenBR = static_cast<size_t>(k) * splitNnumChunk;

    uint32_t ChunkTileNum = (FTChunkShape::N + L1TileShape::N - 1) / L1TileShape::N;

    size_t lenSliceCR1 = (static_cast<size_t>(m)) * splitNnumChunk * ChunkTileNum;
    size_t lenSliceCR2 = (static_cast<size_t>(m)) * splitNnumChunk * ChunkTileNum; 

    size_t lenThre = static_cast<size_t>(m) * splitNnumChunk;

    size_t lenARed = static_cast<size_t>(m) * 1;

    size_t sizeA = lenA * sizeof(GemmInTypeC);
    size_t sizeACast = lenA * sizeof(GemvOutTypeC);
    size_t sizeB = lenB * sizeof(GemmInTypeC);
    size_t sizeC = lenC * sizeof(GemmOutTypeC);

    size_t sizeXR2 = lenXR2 * sizeof(GemvOutTypeC);
    size_t sizeXR2Tail = lenXR2Tail * sizeof(GemvOutTypeC);
    size_t sizeXV = lenXV * sizeof(GemvInTypeCforAB);
    size_t sizeXVTail = lenXVTail * sizeof(GemvInTypeCforAB);
    size_t sizeVXforAe = lenVXforAe * sizeof(GemvInTypeCforAB);
    size_t sizeBRforAIV = lenBR * sizeof(GemvOutTypeC);

    size_t sizeBR1 = lenBR * sizeof(GemvOutTypeC);
    size_t sizeBR2 = lenBR * sizeof(GemvOutTypeC);

    size_t sizeZRowR1 = lenZRowR1 * sizeof(GemvOutTypeC);
    size_t sizeZRowR2 = lenZRowR2 * sizeof(GemvOutTypeC);

    size_t sizeSliceCR1 = lenSliceCR1 * sizeof(GemvOutTypeC);
    size_t sizeSliceCR2 = lenSliceCR2 * sizeof(GemvOutTypeC);

    size_t sizeDRowR1 = lenDRowR1 * sizeof(GemvOutTypeC);
    size_t sizeDRowR2 = lenDRowR2 * sizeof(GemvOutTypeC);

    size_t sizeCOMPRow = lenCOMPRow * sizeof(uint8_t);

    size_t sizeThre = lenThre * sizeof(GemvOutTypeC);

    size_t sizeBMeanAbs = lenBMeanAbs * sizeof(GemvOutTypeC);
    size_t sizeBMeanSquare = lenBMeanSquare * sizeof(GemvOutTypeC);
    size_t sizeBVar = lenBVar * sizeof(GemvOutTypeC);

    size_t sizeAMean = lenARed * sizeof(GemvOutTypeC);
    size_t sizeAMax = lenARed * sizeof(GemvOutTypeC);
    size_t sizeAMin = lenARed * sizeof(GemvOutTypeC);

    using LayoutX = layout::VectorLayout;
    using LayoutY = layout::VectorLayout;
    using LayoutCOMP = layout::VectorLayout;

    using LayoutMY = layout::RowMajor;
    using LayoutMX = layout::RowMajor;
    using LayoutCOMPM = layout::RowMajor;

    using LayoutA = layout::RowMajor;
    using LayoutACol = layout::ColumnMajor;

    using LayoutB = layout::RowMajor;
    using LayoutBCol = layout::ColumnMajor;

    using LayoutC = layout::RowMajor;
    using LayoutCCol = layout::ColumnMajor;

    using LayoutZ = layout::VectorLayout;
    using LayoutMZ = layout::RowMajor;
    using FT_COMP_TYPE = Catlass::Gemv::helper::FT_COMP_TYPE;

    LayoutX layoutXRow{n};
    LayoutX layoutXRowChunk{FTChunkShape::N};
    LayoutX layoutXRowChunkTail{Remain_Chunk_N};

    LayoutC layoutC{m, n};
    LayoutCCol layoutCCol{n, m};

    LayoutA layoutA{m, k};
    LayoutACol layoutACol{k,m};

    LayoutB layoutB{k, n};
    LayoutBCol layoutBCol{n, k};

    LayoutZ layoutZ{m};

    LayoutZ layoutZHost{m * splitNnumChunk};

    LayoutZ layoutThre{m * splitNnumChunk};

    ScalarType alpha{1.0};
    ScalarType beta{0.0};
    // FillRandomScalarData(alpha, -1.0f, 1.0f);
    // FillRandomScalarData(beta, -1.0f, 1.0f);

    float sum_base = 1.0;
    float mean_base = 1.0f / (1.0f * k);

    std::vector<uint8_t> hostCOMPRow(lenCOMPRow,0);

    std::vector<GemmOutTypeC> hostC(lenC,0.0);
    std::vector<GemmInTypeC> hostA(lenA);
    std::vector<GemvOutTypeC> hostACast(lenA, 0.0);
    std::vector<GemmInTypeC> hostB(lenB);

    std::vector<GemvOutTypeC> hostXR2(lenXR2,(GemvOutTypeC)0.0);
    std::vector<GemvOutTypeC> hostXR2Tail(lenXR2Tail,(GemvOutTypeC)0.0);

    std::vector<GemvInTypeCforAB> hostXV(lenXV,(GemvInTypeCforAB)1.0);
    std::vector<GemvInTypeCforAB> hostXVTail(lenXVTail,(GemvInTypeCforAB)1.0);

    std::vector<GemvOutTypeC> hostXVCR(lenXV,(GemvOutTypeC)1.0);
    std::vector<GemvOutTypeC> hostXVTailCR(lenXVTail,(GemvOutTypeC)1.0);

    /*
    float ComputeSinhValue(uint32_t idj, unt32_t ChunkNSize, float lambda_v)
    */

    for(uint32_t jj=0; jj < FTChunkShape::N; jj++){
        uint32_t idj_R2 = FTChunkShape::N + jj;
        hostXR2[jj] = (GemvOutTypeC)golden::ComputeSinhValue(jj, FTChunkShape::N, lambda_R2);
        hostXV[idj_R2] = (GemvInTypeCforAB)golden::ComputeSinhValue(jj, FTChunkShape::N, lambda_R2);
        hostXVCR[idj_R2] = (GemvOutTypeC)golden::ComputeSinhValue(jj, FTChunkShape::N, lambda_R2);
    }

    for(uint32_t jj=0; jj < Remain_Chunk_N; jj++){
        uint32_t idj_R2 = Remain_Chunk_N + jj;
        hostXR2Tail[jj] = (GemvOutTypeC)golden::ComputeSinhValue(jj, Remain_Chunk_N, lambda_R2_remain);
        hostXVTail[idj_R2] = (GemvInTypeCforAB)golden::ComputeSinhValue(jj, Remain_Chunk_N, lambda_R2_remain);
        hostXVTailCR[idj_R2] = (GemvOutTypeC)golden::ComputeSinhValue(jj, Remain_Chunk_N, lambda_R2_remain);
    }

    for(uint32_t jj=0; jj < 10; jj++){
        uint32_t idj_R2 = FTChunkShape::N + jj;
        printf("R1: %f R2: %f\n", (float)hostXV[jj], (float)hostXV[idj_R2]);
    }

    std::vector<GemvInTypeCforAB> hostVXforAe(lenVXforAe, (GemvInTypeCforAB)mean_base);

    printf("VX for Ae: %f\n",(float)hostVXforAe[0]);


    std::vector<GemvOutTypeC> hostBR1(lenBR,(GemvOutTypeC)0.0f);
    std::vector<GemvOutTypeC> hostBR2(lenBR,(GemvOutTypeC)0.0f);

    std::vector<GemvOutTypeC> hostBRforAIV(lenBR,(GemvOutTypeC)0.0f);

    std::vector<GemvInTypeCforCE> hostBMaxSlice(lenBR,(GemvInTypeCforCE)0.0f);
    std::vector<GemvInTypeCforCE> hostBMinSlice(lenBR,(GemvInTypeCforCE)0.0f);

    std::vector<GemvOutTypeC> hostDRowR1(lenZRowR1,0.0f);
    std::vector<GemvOutTypeC> hostDRowR2(lenZRowR2,0.0f);

    std::vector<GemvOutTypeC> hostZRowR1(lenZRowR1,0.0f);
    std::vector<GemvOutTypeC> hostZRowR2(lenZRowR2,0.0f);

    std::vector<GemvOutTypeC> hostSliceCR1(lenSliceCR1,0.0f);
    std::vector<GemvOutTypeC> hostSliceCR2(lenSliceCR2,0.0f);

    std::vector<GemvOutTypeC> hostThre(lenThre,0.0f);
    std::vector<GemvOutTypeC> hostAMean(lenARed, 0.0f);
    std::vector<GemvOutTypeC> hostAMax(lenARed, 0.0f);
    std::vector<GemvOutTypeC> hostAMin(lenARed, 0.0f);

    std::vector<GemvOutTypeC> hostBMeanAbs(lenBMeanAbs, 0.0f);
    std::vector<GemvOutTypeC> hostBMeanSquare(lenBMeanSquare, 0.0f);
    std::vector<GemvOutTypeC> hostBVar(lenBVar, 0.0f);

    // golden::FillRandomData(hostC, 0.0f, 0.0f);
    // golden::FillRandomData(hostA, 0.0f, 0.0f);
    // golden::FillRandomData(hostB, 0.0f, 0.0f);


    // golden::FillRandomData(hostC, -1.0f, 1.0f);
    golden::FillRandomData(hostA, -1.0f, 1.0f);
    golden::FillRandomData(hostB, -1.0f, 1.0f);


    uint8_t* deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceC, sizeC, hostC.data(), sizeC, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceACast{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceACast), sizeACast, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceACast, sizeACast, hostACast.data(), sizeACast, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceXR2{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceXR2), sizeXR2, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceXR2, sizeXR2, hostXR2.data(), sizeXR2, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceXR2Tail{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceXR2Tail), sizeXR2Tail, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceXR2Tail, sizeXR2Tail, hostXR2Tail.data(), sizeXR2Tail, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceXV{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceXV), sizeXV, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceXV, sizeXV, hostXV.data(), sizeXV, ACL_MEMCPY_HOST_TO_DEVICE));

    
    uint8_t* deviceXVTail{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceXVTail), sizeXVTail, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceXVTail, sizeXVTail, hostXVTail.data(), sizeXVTail, ACL_MEMCPY_HOST_TO_DEVICE));

    // hostVXforAe
    uint8_t* deviceVXforAe{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceVXforAe), sizeVXforAe, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceVXforAe, sizeVXforAe, hostVXforAe.data(), sizeVXforAe, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceBR1{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceBR1), sizeBR1, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceBR1, sizeBR1, hostBR1.data(), sizeBR1, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceBR2{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceBR2), sizeBR2, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceBR2, sizeBR2, hostBR2.data(), sizeBR2, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceBRforAIV{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceBRforAIV), sizeBRforAIV, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceBRforAIV, sizeBRforAIV, hostBRforAIV.data(), sizeBRforAIV, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceBMaxSlice{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceBMaxSlice), sizeBRforAIV, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceBMaxSlice, sizeBRforAIV, hostBMaxSlice.data(), sizeBRforAIV, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceBMinSlice{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceBMinSlice), sizeBRforAIV, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceBMinSlice, sizeBRforAIV, hostBMinSlice.data(), sizeBRforAIV, ACL_MEMCPY_HOST_TO_DEVICE));

    printf("size of A: %zu\n", sizeA);
    printf("size of B: %zu\n", sizeB);
    printf("size of C: %zu\n", sizeC);
    printf("size of X: %zu\n", sizeXR2);
    printf("size of Z: %zu\n", sizeZRowR1);
    printf("size of Z: %zu\n", sizeZRowR2);



    uint8_t* deviceZRowR1{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceZRowR1), sizeZRowR1, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceZRowR1, sizeZRowR1, hostZRowR1.data(), sizeZRowR1, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceZRowR2{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceZRowR2), sizeZRowR2, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceZRowR2, sizeZRowR2, hostZRowR2.data(), sizeZRowR2, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceDRowR1{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceDRowR1), sizeZRowR1, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceDRowR1, sizeZRowR1, hostDRowR1.data(), sizeZRowR1, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceDRowR2{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceDRowR2), sizeZRowR2, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceDRowR2, sizeZRowR2, hostDRowR2.data(), sizeZRowR2, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceSliceCR1{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceSliceCR1), sizeSliceCR1, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceSliceCR1, sizeSliceCR1, hostSliceCR1.data(), sizeSliceCR1, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceSliceCR2{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceSliceCR2), sizeSliceCR2, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceSliceCR2, sizeSliceCR2, hostSliceCR2.data(), sizeSliceCR2, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceCOMPRow{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceCOMPRow), sizeCOMPRow, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceCOMPRow, sizeCOMPRow, hostCOMPRow.data(), sizeCOMPRow, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceThre{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceThre), sizeThre, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* deviceBMeanAbs{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceBMeanAbs), sizeBMeanAbs, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* deviceBMeanSquare{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceBMeanSquare), sizeBMeanSquare, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* deviceBVar{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceBVar), sizeBVar, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* deviceAMean{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceAMean), sizeAMean, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* deviceAMax{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceAMax), sizeAMax, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* deviceAMin{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceAMin), sizeAMin, ACL_MEM_MALLOC_HUGE_FIRST));

    // ACL_CHECK(aclrtMemcpy(deviceThre, sizeThre, hostThre.data(), sizeThre, ACL_MEMCPY_HOST_TO_DEVICE));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    using ArchTag = Arch::AtlasA2;
    using FT_REDUCE_TYPE = Gemv::helper::FT_REDUCE_TYPE;

    
    // layout::ColumnMajor;
    // layout::RowMajor;
    // layout::ColumnMajor;

    // constexpr bool enableUnitFlag = true;
    // constexpr bool enableShuffleK = true;

    constexpr bool enableUnitFlag = false;
    constexpr bool enableShuffleK = true;

    
    using FT_ENC_TYPE = Gemv::helper::FT_ENC_TYPE;
    using FT_RCE_THRE_TYPE = Gemv::helper::FT_RCE_THRE_TYPE;
    using FT_L02L1_TYPE = Gemv::helper::FT_L02L1_TYPE;
    using FT_AIC_BE_SCHEME = Gemv::helper::FT_AIC_BE_SCHEME;

    // using L1TileShapeC = GemvShape<32, 512>;
    // using L0TileShapeC = GemvShape<32, 256>;

    using L1TileShapeAB = GemmShape<128, 128, 128>;
    using L0TileShapeAB = GemmShape<128, 128, 128>;

    using L1TileShapeBR = GemvShape<256, 256>;
    using L0TileShapeBR = GemvShape<256, 64>;

    // L1TileShapeBR::N*1

    using UBBlockShapeBR = GemvShape<L1TileShapeBR::M*1, FTChunkShape::N>;
    // using UBBlockShapeBR = GemvShape<L1TileShapeBR::M*2, L1TileShapeBR::N*2>;

    using AType = Gemm::GemmType<GemmInTypeN, LayoutA>;
    using ACastType = Gemm::GemmType<GemvOutTypeN, LayoutA>;
    using BType = Gemm::GemmType<GemmInTypeN, LayoutB>;

    using CType = Gemm::GemmType<GemmOutTypeN, LayoutC>;
    
    using XTypeAIC = Gemm::GemmType<GemvInTypeNforAB, LayoutX>;
    using MXTypeAIC = Gemm::GemmType<GemvInTypeNforAB, LayoutMX>;
    using YTypeBRAIC = Gemm::GemmType<GemvOutTypeN, LayoutMY>;
    using BiasType = void;

    using XTypeAIV = Gemm::GemmType<GemvInTypeNforCE, LayoutX>;

    using GemvDispatchPolicy = Gemm::GemvAtlasA2;
    using COMPDispatchPolicy = Gemm::GemvAtlasA2;
    using BrAICDispatchPolicy = Gemm::MmadAtlasA2Preload<enableUnitFlag, enableShuffleK>;
    using GEMVAICDispatchPolicy = Gemm::MmadAtlasA2Preload<enableUnitFlag, enableShuffleK>;
    using TileCopyGemvAic = Gemv::Tile::TileCopyGemvAicMultiVector<typename BrAICDispatchPolicy::ArchTag, BType, MXTypeAIC, YTypeBRAIC, BiasType>;
    using TileMmadGemvAic = Gemm::Tile::TileMmad<typename  BrAICDispatchPolicy::ArchTag, MXTypeAIC, BType, BiasType>;

    /*
    struct BlockFTGemvBr<
        Gemm::MmadAtlasA2Preload<ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_>,
        Gemv::helper::FT_AIC_BE_SCHEME::ROWCOMPLETE,
        UBBlockShape_,
        L1TileShape_,
        L0TileShape_,
        AType_,
        XType_,
        YType_,
        BiasType_,
        TileCopy_,
        TileMmad_
    > 
    */
    // FT_AIC_BE_SCHEME::COLCOMPLETE,
    // FT_AIC_BE_SCHEME::ROWCOMPLETE,
    using BlockFTGemvBRAIC = Gemv::Block::BlockFTGemvBr<BrAICDispatchPolicy, 
        FT_AIC_BE_SCHEME::ROWCOMPLETE,
        UBBlockShapeBR, L1TileShapeBR, L0TileShapeBR, 
        BType, MXTypeAIC, YTypeBRAIC, BiasType, TileCopyGemvAic, TileMmadGemvAic>;

    using ZType = Gemm::GemmType<GemvOutTypeN, LayoutZ>;
    using MZType = Gemm::GemmType<GemvOutTypeN, LayoutMZ>;

    static constexpr FT_AIC_BE_SCHEME BE_SCHEME = BlockFTGemvBRAIC::BE_SCHEME;

    constexpr uint32_t computeLength = 8192;

    using YType = Gemm::GemmType<GemvInTypeNforCE, LayoutY>;
    using BRedYType = Gemm::GemmType<GemvOutTypeN, LayoutY>;

    using BRedType = Gemm::GemmType<GemvOutTypeN, LayoutB>;
    using TileVmuls = Gemv::Tile::TileVmuls<ArchTag, XTypeAIV>;

    using MmadDispatchPolicy = CubeSelf::Gemm::MmadAtlasA2Pingpong<enableUnitFlag>;
    
    // using L0TileShapeforFT = GemvShape<L1TileShape::M,128>;
    // using L1TileShape = GemmShape<128, 240, 256>;
    // using L0TileShape = GemmShape<128, 240, 64>;

    using L1TileShapeFirst = GemmShape<256,256,128>;
    using L0TileShapeFirst = GemmShape<256,256,32>;

    using L1TileShapeAe = GemmShape<256,256,256>;
    using L0TileShapeAe = GemmShape<256,256,64>;

    using L1TileShapeforFT = GemmShape<128, 16, 256>;
    using L0TileShapeforFT = GemmShape<128, 16, 64>;

    using L1TileShapeforAe = GemmShape<L1TileShapeAe::M, 16, L1TileShapeAe::K>;
    using L0TileShapeforAe = GemmShape<L0TileShapeAe::M, 16, L0TileShapeAe::K>;

    // using L1TileShapeforFT = GemmShape<L1TileShapeFirst::M, 32, L1TileShapeFirst::K>;
    // using L0TileShapeforFT = GemmShape<L0TileShapeFirst::M, 32, L0TileShapeFirst::K>;

    /*
    using LayoutMY = layout::RowMajor;
    using LayoutMX = layout::ColumnMajor;
    */
    using MXType = Gemm::GemmType<GemvOutTypeN, LayoutMX>;
    using MYType = Gemm::GemmType<GemvOutTypeN, LayoutMY>;

    // using LayoutMY = layout::RowMajor;
    // using MYType = Gemm::GemmType<GemvOutTypeN, LayoutMY>;
    /*
    struct BlockMmadSpecAeNoSplitKRobust<
        CubeSelf::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>,
        L1TileShapeforFT_,
        L0TileShapeforFT_,
        AType_,
        BType_,
        CType_,
        XType_,
        YType_,
        BiasType_,
        TileCopyFTABonAic_,
        TileMmad_
    >
    */
    using BlockFTGemvAEAIC = CubeSelf::Gemm::Block::BlockMmadSpecAeNoSplitKRobust<
        MmadDispatchPolicy, 
        L1TileShapeforAe,
        L0TileShapeforAe, 
        AType, BType, CType, MXTypeAIC, MYType, BiasType>;

    /*
    struct BlockMmadSpecABrNoSplitKRobust<
        CubeSelf::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>,
        L1TileShapeforFT_,
        L0TileShapeforFT_,
        AType_,
        BType_,
        CType_,
        XType_,
        YType_,
        BiasType_,
        TileCopyFTABonAic_,
        TileMmad_
    >
    */
    using BlockMmadABr = CubeSelf::Gemm::Block::BlockMmadSpecABrNoSplitKRobust<
        MmadDispatchPolicy, 
        L1TileShapeforFT,
        L0TileShapeforFT, 
        AType, BType, CType, 
        MXType, MYType,
        BiasType>;
    
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
        MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

    using BlockSchedulerFirst = typename CubeSelf::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
    using BlockScheduler = typename CubeSelf::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

    /*
    template <
        /// Tag indicating architecture
        class ArchTag,
        /// MatmulType for A matrix operand
        class AType,
        class BType,
        /// MatmulType type for X vector operand
        class XType,
        /// MatmulType type for Y vector operand
        class YType,
        /// MatmulTpe type for Bias operand
        class BiasType = void
    >
    struct TileCopyFTRedAiv 
    */

    using TileFaultCopyRedAiv = Gemv::Tile::TileCopyFTRedAiv<ArchTag, 
        AType, BType, BRedYType, ZType>;

    using UBTileShapeforB = GemvShape<48, L0TileShape::N>;
    // UBTileShapeforB::N*2
    using UBBlockShapeforB = GemvShape<UBTileShapeforB::M*2, FTChunkShape::N>;

    using UBTileShapeforA = GemvShape<48, 256>;
    using UBTileShapeforACast = GemvShape<48, 128>;

    

    using ARedType = Gemm::GemmType<GemvInTypeNforCE, LayoutA>;
    using TileFaultSum = Gemv::Tile::TileFaultSum<ArchTag, FT_REDUCE_TYPE::MAX_MIN, ARedType, ZType>;

    /*
    struct BlockFTSumNoSplitK <
        Gemm::GemvAtlasA2,
        Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
        Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::A_B_MIXED_CHUNK_BF,
        UBTileShapeforB_,
        UBBlockShapeforB_,
        UBTileShapeforA_,
        UBTileShapeforACast_,
        L1TileShape_,
        AType_,
        BType_,
        XType_,
        YType_,
        BiasType_,
        TileCopy_,
        TileFaultSum_>
    */
    
    using BlockFTSum = Gemv::Block::BlockFTSumNoSplitK<
        GemvDispatchPolicy,
        Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
        Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::A_B_MIXED_CHUNK_BF,
        UBTileShapeforB, UBBlockShapeforB, 
        UBTileShapeforA, UBTileShapeforACast,
        L1TileShape, AType, BType, BRedYType, ZType, void,
        TileFaultCopyRedAiv, TileFaultSum>;

    using SliceSumDispatchPolicy = Gemm::GemvAtlasA2;

    using SliceSumUBTileShape = GemvShape<8, 256>;
    using TileMatrixAddforABRReduce = Gemv::Tile::TileMatmulAdd<
        typename SliceSumDispatchPolicy::ArchTag, MYType, MYType, void>;
    using TileCopyMatrixAddforABRReduce = Gemv::Tile::TileCopyMatrixAddAiv<
        typename SliceSumDispatchPolicy::ArchTag, MYType, MYType, void>; 
    
    using BlockSliceSum = Gemv::Block::BlockSliceKMNSum<SliceSumDispatchPolicy,
        Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::ABE_FUSED_THRE,
        SliceSumUBTileShape, MYType, MYType, void, TileCopyMatrixAddforABRReduce, TileMatrixAddforABRReduce>;

    /*
    struct BlockSliceKMNSum <
        Gemm::GemvAtlasA2,
        Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::A_B_ROBUST,
        UBTileShapeforA_,
        UBTileShapeforB_,
        AType_,
        BType_,
        XType_,
        YType_,
        BiasType_,
        TileCopyforA_,
        TileCopyforB_,
        TileMatrixAdd_,
        TileFaultSum_,
        TileVmuls_
    >
    */
    using MeanMaxTileVmuls = Gemv::Tile::TileVmuls<typename GemvDispatchPolicy::ArchTag, ZType>;
    using UBTileShapeforBRed = GemvShape<8,256>;
    using TileFaultSumBReduce = Gemv::Tile::TileFaultSum<ArchTag, FT_REDUCE_TYPE::SUM_MAX, BRedType, ZType>;
    using TileFaultCopyBReduce = Gemv::Tile::TileCopyGemvAiv<ArchTag, BRedType, BRedYType, ZType>;
    
    using BlockSliceRed = Gemv::Block::BlockSliceKMNSum<
        SliceSumDispatchPolicy,
        Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::A_B_ROBUST,
        SliceSumUBTileShape,
        UBTileShapeforBRed,
        MYType, BRedType, ZType,
        MYType, void,
        TileCopyMatrixAddforABRReduce, 
        TileFaultCopyBReduce,
        TileMatrixAddforABRReduce,
        TileFaultSumBReduce,
        MeanMaxTileVmuls>;

    using UBTileShapeCE = GemvShape<64, L1TileShape::N>;
    using UBBlockShapeCE = GemvShape<L1TileShape::M, UBTileShapeCE::N>;
    using UBTileShapeACast = GemvShape<64, 128>;

    using COMPZType = Gemm::GemmType<uint8_t, LayoutZ>;
    /*
    template <
        /// Tag indicating architecture
        class ArchTag,
        /// MatmulType for A matrix operand
        class AType,
        /// MatmulType for C matric operand
        class CType,
        /// MatmulType type for X vector operand
        class XType,
        /// MatmulType type for Y vector operand
        class YType,
        /// MatmulTpe type for Bias operand
        class BiasType = void
    >
    struct TileCopyGemvCastFusedAiv
    */
    // using TileFaultCopyABE = Gemv::Tile::TileCopyGemvAiv<ArchTag, AType, XType, YType>;
    using TileFaultCopyCR = Gemv::Tile::TileCopyGemvCastFusedAiv<ArchTag, 
        AType, CType, BRedYType, ZType, void>;

    /*
    struct BlockFTGemvCENoSplitK <
        Gemm::GemvAtlasA2,
        Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
        Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::THRE_FUSED,
        Gemv::helper::FT_ENC_TYPE::RCE,
        Gemv::helper::FT_COMP_TYPE::RSUB,
        Gemv::helper::FT_ABE_TYPE::CENTRAL_BLOCK_CHUNK_16,
        UBTileShapeforC_,
        UBTileShapeforA_,
        UBBlockShape_,
        L1TileShape_,
        AType_,
        CType_,
        XType_,
        YType_,
        BiasType_,
        TileCopy_,
        TileFaultSumVmad_
    >
    */

    using TileFaultSumVmad = Gemv::Tile::TileFaultSumVmad<ArchTag, 
        FT_REDUCE_TYPE::SUM_VMAD, CType, BRedYType, ZType>;

    using BlockFTGemvAIV = Gemv::Block::BlockFTGemvCENoSplitK<
        GemvDispatchPolicy,
        Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
        Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::THRE_FUSED,
        Gemv::helper::FT_ENC_TYPE::RCE,
        Gemv::helper::FT_COMP_TYPE::RSUB,
        Gemv::helper::FT_ABE_TYPE::CENTRAL_BLOCK_CHUNK_16,
        UBTileShapeCE, 
        UBTileShapeACast,
        UBBlockShapeCE, L1TileShape,
        AType, CType, 
        BRedYType, ZType, void, 
        TileFaultCopyCR, TileFaultSumVmad>;

    /*
    template <
        /// Tag indicating architecture
        class ArchTag,
        /// MatmulType for A matrix operand
        class AType,
        /// MatmulType for C matric operand
        class CType,
        /// MatmulType type for X vector operand
        class XType,
        /// MatmulType type for Y vector operand
        class YType,
        /// Output Result operand namely vector Z
        class ZType,
        /// MatmulTpe type for Bias operand
        class BiasType = void
    >
    struct TileCopyVerifyThreCompFusedAiv
    */
    using TileCopyVerify = Gemv::Tile::TileCopyVerifyThreCompFusedAiv<
        ArchTag, CType, CType, BRedYType, ZType, COMPZType, void>;

    using ThreCalcDispatchPolicy = Gemm::GemvAtlasA2;

    using TileMatrixAddforCRReduce = Gemv::Tile::TileMatmulAdd<
        typename SliceSumDispatchPolicy::ArchTag, MYType, MYType, void>;

    /*
    template <
    class ElementA
    >
    struct TileThreCalcChunk<Arch::AtlasA2,
        helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
        Gemm::GemmType<ElementA, layout::RowMajor>,
        Gemm::GemmType<float, layout::VectorLayout>,
        Gemm::GemmType<float, layout::VectorLayout>,
        void>
    */

    using TileThreCalcChunk = Gemv::Tile::TileThreCalcChunk<
        typename ThreCalcDispatchPolicy::ArchTag, 
        Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
        CType, ZType, ZType, void>; 

    /*
    struct TileStdEstRobust<Arch::AtlasA2,
                Gemm::GemmType<float, layout::VectorLayout>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
    */

    using TileStdEstVector = Gemv::Tile::TileStdEstRobust<
        typename ThreCalcDispatchPolicy::ArchTag,
        ZType,
        ZType>;
    /*
    struct BlockFTVerifyNoSplitKPreload <
    Gemm::GemvAtlasA2,
    Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
    Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::THRE_FUSED,
    Gemv::helper::FT_ENC_TYPE::RCE,
    Gemv::helper::FT_COMP_TYPE::RSUB,
    Gemv::helper::FT_ABE_TYPE::CENTRAL_BLOCK,
    UBTileShape_,
    UBBlockShape_,
    L1TileShape_,
    AType_,
    XType_,
    YType_,
    ZType_,
    BiasType_,
    TileCopy_,
    TileMatrixAdd_,
    TileThreCalc_,
    TileStdEst_>
    */

    using UBTileShapeVerify = GemvShape<L1TileShapeforFT::N / 2, L1TileShapeforFT::M / 2>;
    using UBBlockShapeVerify = GemvShape<L1TileShapeforFT::N / 2, L1TileShapeforFT::M>;

    using BlockChunkVerify = Gemv::Block::BlockFTVerifyNoSplitKPreload<
        GemvDispatchPolicy,
        Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
        Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::THRE_FUSED, 
        Gemv::helper::FT_ENC_TYPE::RCE, 
        Gemv::helper::FT_COMP_TYPE::RSUB,
        Gemv::helper::FT_ABE_TYPE::CENTRAL_BLOCK,
        UBTileShapeVerify, UBBlockShapeVerify, L1TileShapeforFT,
        CType, BRedYType, ZType, COMPZType, void, 
        TileCopyVerify, TileMatrixAddforCRReduce,
        TileThreCalcChunk, TileStdEstVector>;

    using UBTileShape = GemmShape<L1TileShapeAB::M, L1TileShapeAB::N, L1TileShapeAB::K>;
    using TileFaultCopy = Gemv::Tile::TileCopyGemvAiv<ArchTag, AType, XTypeAIV, ZType>;
    using TileFaultVmad = Gemv::Tile::TileVmad<ArchTag, AType, XTypeAIV, ZType>;
    // Kernel level
    /*
    template <
        class FTChunkShape_,
        class BlockMmadABr_,
        class BlockMmad_,
        class BlockSchedulerFirst_,
        class BlockScheduler_,
        class BlockFTGemvBRAIC_,
        class BlockFTGemvAEAIC_,
        class BlockFTSum_,
        class BlockFTGemvAIV_,
        class BlockSliceRed_,
        class BlockChunkVerify_
    >
    class MatmulAsVarABonAicChunkSpecRobustPreloadF16
    */
    // , BlockThresholdCalc
    // CompareBlockSUB
    // MatmulAsVarABonAicNoSplitSpecRobust
    using MatmulFTKernel = CubeSelf::Gemm::Kernel::MatmulAsVarABonAicChunkSpecRobustPreloadF16<
        FTChunkShape, 
        BlockMmadABr, BlockMmadPreload, 
        BlockSchedulerFirst, BlockScheduler,
        BlockFTGemvBRAIC, BlockFTGemvAEAIC,
        BlockFTSum, BlockFTGemvAIV, 
        BlockSliceRed, BlockChunkVerify>;
    // Prepare params

    /*
    MatmulAsVarABonAicNoSplitRelieveMixed
    */

    // TODO:  use adapter to activate the kernel
    using MatmulAdapter = CubeSelf::Gemm::Device::DeviceGemm<MatmulFTKernel>;
    ScalarType threshold{0.000f};

    FT_RCE_THRE_TYPE rce_thre_type = FT_RCE_THRE_TYPE::ROUND;

    if(options.thre_type >= 1){
        rce_thre_type = FT_RCE_THRE_TYPE::ROUND_WITH_ACC;
    }

    /*
    struct Arguments {
        Catlass::GemmCoord problemGemmShape;
        Catlass::GemvCoord problemShape;
        size_t elementSize;
        GM_ADDR ptrX; GM_ADDR ptrXV;
        GM_ADDR ptrA; GM_ADDR ptrB; GM_ADDR ptrC;
        GM_ADDR ptrZRow; GM_ADDR ptrZCol; GM_ADDR ptrZRow2; GM_ADDR ptrZCol2;
        GM_ADDR ptrCOMPZRow; GM_ADDR ptrCOMPZCol;
        GM_ADDR ptrBE; GM_ADDR ptrBEforAIV;
        GM_ADDR ptrBMaxSlice; GM_ADDR ptrBMinSlice;
        GM_ADDR ptrBMeanAbs; GM_ADDR ptrBMeanSquare; GM_ADDR ptrBVar; 
        GM_ADDR ptrVXforA; GM_ADDR ptrAMean; GM_ADDR ptrAMax; GM_ADDR ptrAMin; 
        GM_ADDR ptrThreZ; FT_ENC_TYPE enc_type;
        uint32_t UbNum; bool OutputWorkspace; ElementCOMPX threshold;
        float rounding_exponent; float size_beta;
        float e_max_raw; uint32_t reduce_cores; FT_RCE_THRE_TYPE rce_thre_type;
        bool outputThre; bool outputCE; uint32_t SplitKNum;
    };
    */

    float use_emax = options.e_max * 1.0f;
    if(k <= 1024){
        use_emax = use_emax * 1.0f;
    }else{
        use_emax = use_emax * std::sqrt(((k*1.0f / 1024*1.0f)*1.0f));
    }

    /*
    struct Arguments {
        Catlass::GemmCoord problemGemmShape; Catlass::GemvCoord problemShape;
        size_t elementSize;
        GM_ADDR ptrX; GM_ADDR ptrXTail;
        GM_ADDR ptrXVR2; GM_ADDR ptrXVR2Tail;
        GM_ADDR ptrA; GM_ADDR ptrACast; GM_ADDR ptrB; GM_ADDR ptrC;
        GM_ADDR ptrZRowR1; GM_ADDR ptrZRowR2;
        GM_ADDR ptrZRow2R1; GM_ADDR ptrZRow2R2;
        GM_ADDR ptrCR1Slice; GM_ADDR ptrCR2Slice;
        GM_ADDR ptrCOMPZRow; GM_ADDR ptrBR1; GM_ADDR ptrBR2;
        GM_ADDR ptrBRforAIV; GM_ADDR ptrBMaxSlice; GM_ADDR ptrBMinSlice;
        GM_ADDR ptrBMeanAbs; GM_ADDR ptrBMeanSquare; GM_ADDR ptrBVar; 
        GM_ADDR ptrXVR1forA; GM_ADDR ptrAMean; GM_ADDR ptrAMax; GM_ADDR ptrAMin;
        GM_ADDR ptrThreZ; FT_ENC_TYPE enc_type;
        ElementCOMPX threshold;
        float rounding_exponent; float size_beta; float e_max_raw;
        uint32_t reduce_cores; FT_RCE_THRE_TYPE rce_thre_type;
        bool outputThre; uint32_t SplitKNum;
    };  
    */

    typename MatmulFTKernel::Arguments arguments{
        options.problemGemmShape, options.problemShape, sizeof(GemvInTypeCforAB),
        deviceXV, deviceXVTail,
        deviceXR2, deviceXR2Tail,
        deviceA, deviceACast, deviceB, deviceC,
        deviceZRowR1, deviceZRowR2,
        deviceDRowR1, deviceDRowR2,
        deviceSliceCR1, deviceSliceCR2,
        deviceCOMPRow, deviceBR1, deviceBR2,
        deviceBRforAIV, deviceBMaxSlice, deviceBMinSlice,
        deviceBMeanAbs, deviceBMeanSquare, deviceBVar,
        deviceVXforAe, deviceAMean, deviceAMax, deviceAMin,
        deviceThre, FT_ENC_TYPE::RCE, threshold,
        options.round_exp, options.beta,
        use_emax, options.reduce_cores,
        rce_thre_type, true, options.split_ks,
        options.inject_row, options.inject_col, options.inject_bit, options.skip_gemm};

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));
    MatmulAdapter matmul_op;
    matmul_op.CanImplement(arguments);

    size_t sizeWorkspace = matmul_op.GetWorkspaceSize(arguments);
    uint8_t *deviceWorkspace = nullptr;
    if (sizeWorkspace > 0) {
        ACL_CHECK(
            aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST)
        );
    }

    // RunAdapter(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
    matmul_op.Initialize(arguments, deviceWorkspace);

    matmul_op(stream, aicCoreNum, fftsAddr);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    // ===== Fault injection: read C, flip a bit, write back, re-run with skip_gemm =====
    if (options.inject_row >= 0) {
        std::vector<GemvOutTypeC> hostC_inj(lenC);
        ACL_CHECK(aclrtMemcpy(hostC_inj.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

        int64_t inj_idx = (int64_t)options.inject_row * options.problemGemmShape.n() + options.inject_col;
        GemvOutTypeC val_before = hostC_inj[inj_idx];
        // BF16: reinterpret as uint16, flip the specified bit
        uint16_t ival = *(reinterpret_cast<uint16_t*>(&val_before));
        uint16_t ival_new = ival ^ (1u << options.inject_bit);
        GemvOutTypeC val_after = *(reinterpret_cast<GemvOutTypeC*>(&ival_new));
        hostC_inj[inj_idx] = val_after;
        printf("INJECT: C[%d][%d] bit=%d  before=%f after=%f (0x%04x->0x%04x)\n",
            options.inject_row, options.inject_col, options.inject_bit,
            (float)val_before, (float)val_after, (int)ival, (int)ival_new);

        ACL_CHECK(aclrtMemcpy(deviceC, sizeC, hostC_inj.data(), sizeC, ACL_MEMCPY_HOST_TO_DEVICE));

        // Re-run with skip_gemm=1
        arguments.skip_gemm = 1;
        matmul_op.Initialize(arguments, deviceWorkspace);
        matmul_op(stream, aicCoreNum, fftsAddr);
        ACL_CHECK(aclrtSynchronizeStream(stream));
    }

    ACL_CHECK(aclrtMemcpy(hostZRowR1.data(), sizeZRowR1, deviceZRowR1, sizeZRowR1, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostZRowR2.data(), sizeZRowR2, deviceZRowR2, sizeZRowR2, ACL_MEMCPY_DEVICE_TO_HOST));

    ACL_CHECK(aclrtMemcpy(hostDRowR1.data(), sizeZRowR1, deviceDRowR1, sizeZRowR1, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostDRowR2.data(), sizeZRowR2, deviceDRowR2, sizeZRowR2, ACL_MEMCPY_DEVICE_TO_HOST));
    
    ACL_CHECK(aclrtMemcpy(hostCOMPRow.data(), sizeCOMPRow, deviceCOMPRow, sizeCOMPRow, ACL_MEMCPY_DEVICE_TO_HOST));

    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));
    // DEBUG: print hostC[0..3] to verify injection
    printf("DEBUG hostC[0..3]: %f %f %f %f\n",
        (float)hostC[0], (float)hostC[1], (float)hostC[2], (float)hostC[3]);
    ACL_CHECK(aclrtMemcpy(hostACast.data(), sizeACast, deviceACast, sizeACast, ACL_MEMCPY_DEVICE_TO_HOST));
    
    ACL_CHECK(aclrtMemcpy(hostThre.data(), sizeThre, deviceThre, sizeThre, ACL_MEMCPY_DEVICE_TO_HOST));

    ACL_CHECK(aclrtMemcpy(hostBMeanAbs.data(), sizeBMeanAbs, deviceBMeanAbs, sizeBMeanAbs, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostBMeanSquare.data(), sizeBMeanSquare, deviceBMeanSquare, sizeBMeanSquare, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostBVar.data(), sizeBVar, deviceBVar, sizeBVar, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostBR1.data(), sizeBR1, deviceBR1, sizeBR1, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostBR2.data(), sizeBR2, deviceBR2, sizeBR2, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostBRforAIV.data(), sizeBRforAIV, deviceBRforAIV, sizeBRforAIV, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostBMaxSlice.data(), sizeBRforAIV, deviceBMaxSlice, sizeBRforAIV, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostBMinSlice.data(), sizeBRforAIV, deviceBMinSlice, sizeBRforAIV, ACL_MEMCPY_DEVICE_TO_HOST));

    ACL_CHECK(aclrtMemcpy(hostSliceCR1.data(), sizeSliceCR1, deviceSliceCR1, sizeSliceCR1, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostSliceCR2.data(), sizeSliceCR2, deviceSliceCR2, sizeSliceCR2, ACL_MEMCPY_DEVICE_TO_HOST));

    ACL_CHECK(aclrtMemcpy(hostAMax.data(), sizeAMax, deviceAMax, sizeAMax, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostAMean.data(), sizeAMean, deviceAMean, sizeAMean, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostAMin.data(), sizeAMin, deviceAMin, sizeAMin, ACL_MEMCPY_DEVICE_TO_HOST));

    for(uint32_t jj = 0; jj < 64; jj++){
        printf("A: %f; A Cast: %f\n", (float)hostA[192 + jj *k],hostACast[192 + jj * k]);
    }
   
    std::vector<GemvOutTypeC> hostGoldenCRowR1(lenZRowR1, (GemvOutTypeC)0.0f);
    std::vector<GemvOutTypeC> hostGoldenCRowR2(lenZRowR2, (GemvOutTypeC)0.0f);

    std::vector<GemvOutTypeC> hostGoldenRowR1(lenZRowR1, (GemvOutTypeC)0.0f);
    std::vector<GemvOutTypeC> hostGoldenRowR2(lenZRowR2, (GemvOutTypeC)0.0f);

    std::vector<GemvOutTypeC> hostBMeanAbsGolden(lenBMeanAbs,(GemvOutTypeC)0.0f);
    std::vector<GemvOutTypeC> hostBMeanSquareGolden(lenBMeanSquare, (GemvOutTypeC)0.0f);
    std::vector<GemvOutTypeC> hostBVarGolden(lenBVar, (GemvOutTypeC)0.0f);

    std::vector<GemvOutTypeC> hostYR1ForAB(static_cast<uint32_t>(k * splitNnumChunk), (GemvOutTypeC)0.0f);
    LayoutY layoutYR1ForAB{static_cast<uint32_t>(k * splitNnumChunk)};

    std::vector<GemvOutTypeC> hostYR2ForAB(static_cast<uint32_t>(k * splitNnumChunk), (GemvOutTypeC)0.0f);
    LayoutY layoutYR2ForAB{static_cast<uint32_t>(k * splitNnumChunk)};

    GemvCoord BRSliceShape{1, FTChunkShape::N};
    printf("BRSliceShape: {%d,%d}\n", BRSliceShape.m(), BRSliceShape.n());

    /*
    void ComputeGemvSliceMultiVector(
        const Catlass::GemvCoord &problemShape,
        const Catlass::GemvCoord &sliceShape,
        Element alpha, Element beta,
        const std::vector<ElementA> &dataA, const LayoutA &layoutA,
        const std::vector<ElementX> &dataX, const LayoutX &layoutX,
        const std::vector<ElementX> &dataXTail, const LayoutX &layoutXTail,
        const std::vector<ElementY> &dataYR1, const std::vector<ElementY> &dataYR2,
        const LayoutY &layoutY,
        std::vector<ElementGolden> &dataGoldenR1, std::vector<ElementGolden> &dataGoldenR2
    )
    */

    golden::ComputeGemvSliceMultiVector(options.problemShape, BRSliceShape, alpha, beta,
         hostC, layoutC, hostXVCR, layoutXRowChunk,
         hostXVTailCR, layoutXRowChunkTail,
         hostGoldenCRowR1, hostGoldenCRowR2,
         layoutZHost, hostGoldenCRowR1, hostGoldenCRowR2);

    std::vector<uint64_t> errorIndices = golden::CompareData(hostZRowR1, hostGoldenCRowR1, m);

    if (errorIndices.empty()) {
        std::cout << "CR1: RowSum Compare success." << std::endl;
    } else {
        std::cerr << "CR1: RowSum Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    errorIndices = golden::CompareData(hostZRowR2, hostGoldenCRowR2, m);

    if (errorIndices.empty()) {
        std::cout << "CR2: RowSum Compare success." << std::endl;
    } else {
        std::cerr << "CR2: RowSum Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    golden::ComputeGemvSliceMultiVector(problemShapeBR, BRSliceShape, alpha, beta,
         hostB, layoutB, hostXV, layoutXRowChunk,
         hostXVTail, layoutXRowChunkTail,
         hostYR1ForAB, hostYR2ForAB,
         layoutYR1ForAB, hostYR1ForAB, hostYR2ForAB);

    errorIndices = golden::CompareData(hostBR1, hostYR1ForAB, problemShapeBR.m());
    
    if (errorIndices.empty()) {
        std::cout << "BR1: Rowsum Compare success." << std::endl;
    } else {
        std::cerr << "BR1: Rowsum Compare failed." << std::endl;
    }

    errorIndices = golden::CompareData(hostBR2, hostYR2ForAB, problemShapeBR.m());
    
    if (errorIndices.empty()) {
        std::cout << "BR2: Rowsum Compare success." << std::endl;
    } else {
        std::cerr << "BR2: Rowsum Compare failed." << std::endl;
    }
        
    
    GemvCoord ABRSliceShape{static_cast<uint32_t>(splitNnumChunk), problemShapeABR.n()};
    golden::ComputeGemvSlice(problemShapeABR, ABRSliceShape, alpha, beta,
         hostA, layoutA, hostYR1ForAB, layoutYR1ForAB, 
         hostGoldenRowR1, layoutZHost, hostGoldenRowR1);

    errorIndices = golden::CompareData(hostDRowR1, hostGoldenRowR1, problemShapeABR.n());

    if (errorIndices.empty()) {
        std::cout << "ABR1: Rowsum Compare success." << std::endl;
    } else {
        std::cerr << "ABR1: Rowsum Compare failed." << std::endl;
    }

    golden::ComputeGemvSlice(problemShapeABR, ABRSliceShape, alpha, beta,
         hostA, layoutA, hostYR2ForAB, layoutYR2ForAB, 
         hostGoldenRowR2, layoutZHost, hostGoldenRowR2);

    errorIndices = golden::CompareData(hostDRowR2, hostGoldenRowR2, problemShapeABR.n());

    if (errorIndices.empty()) {
        std::cout << "ABR2: Rowsum Compare success." << std::endl;
    } else {
        std::cerr << "ABR2: Rowsum Compare failed." << std::endl;
    }

    // errorIndices = golden::CompareData(hostDRow, hostGoldenCRow, problemShapeABR.n());

    // if (errorIndices.empty()) {
    //     std::cout << "Matmul: Rowsum Compare success." << std::endl;
    // } else {
    //     std::cerr << "Matmul: Rowsum Compare failed." << std::endl;
    // }

    /*
    ComputeMeanAbsSquareVarSliceRobust(
    const Catlass::GemvCoord &problemShape,
    const Catlass::GemvCoord &sliceShape,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataMeanAbsGolden,
    std::vector<ElementGolden> &dataMeanSquareGolden,
    std::vector<ElementGolden> &dataVarGolden)
    */
    GemvCoord BReduceSliceShape{problemShapeBR.m(), FTChunkShape::N};
    golden::ComputeMeanAbsSquareVarSliceRobust(problemShapeBR, BReduceSliceShape,
        hostB, layoutB, hostBMeanAbsGolden, hostBMeanSquareGolden, hostBVarGolden);

    errorIndices = golden::CompareData(hostBMeanAbs, hostBMeanAbsGolden, splitNnumChunk);

    for(int j=0; j < splitNnumChunk; j++){
        printf("Expect B Mean ABS[%d]: %f\n",j,hostBMeanAbsGolden[j]);
        printf("Actual B Mean ABS[%d]: %f\n",j,hostBMeanAbs[j]);
    }

    if (errorIndices.empty()) {
        std::cout << "B Reduce Mean ABS Compare success." << std::endl;
    } else {
        std::cerr << "B Reduce Mean ABS Compare failed." << std::endl;
    }
    
    errorIndices = golden::CompareData(hostBMeanSquare, hostBMeanSquareGolden, splitNnumChunk);

    for(int j=0; j < splitNnumChunk; j++){
        printf("Expect B Mean square[%d]: %f\n",j,hostBMeanSquareGolden[j]);
        printf("Actual B Mean square[%d]: %f\n",j,hostBMeanSquare[j]);
        
    }

    if (errorIndices.empty()) {
        std::cout << "B Reduce Mean square Compare success." << std::endl;
    } else {
        std::cerr << "B Reduce Mean square Compare failed." << std::endl;
    }

    errorIndices = golden::CompareData(hostBVar, hostBVarGolden, splitNnumChunk);

    for(int j=0; j < splitNnumChunk; j++){
        printf("Expect B Var[%d]: %f\n",j,hostBVarGolden[j]);
        printf("Actual B Var[%d]: %f\n",j,hostBVar[j]);
        
    }

    if (errorIndices.empty()) {
        std::cout << "B Reduce Var Compare success." << std::endl;
    } else {
        std::cerr << "B Reduce Var Compare failed." << std::endl;
    }

    std::vector<GemvOutTypeC> hostThreGolden(lenThre);

    GemvCoord ThreSliceShape{L1TileShape::M, FTChunkShape::N};

    // std::vector<GemvOutTypeC> hostAMaxGolden2(lenARed);
    std::vector<GemvOutTypeC> hostAMaxGolden(lenThre);
    std::vector<GemvOutTypeC> hostAMeanGolden(lenThre);
    std::vector<GemvOutTypeC> hostAMinGolden(lenThre);
    std::vector<GemvOutTypeC> hostAStdGolden(lenThre);
    std::vector<GemvOutTypeC> hostAMeanAbsGolden(lenARed*2);

    /*
    void ComputeThresholdsASVARRobustTSlice(
    const Catlass::GemvCoord &problemShape,
    uint32_t splitNnum,
    const std::vector<ElementX> &dataBMeanabs,
    const std::vector<ElementX> &dataBMeanSquare,
    const std::Vector<ElementX> &dataBVar,
    uint32_t B_N_size, uint32_t B_N_tile,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataAMean,
    std::vector<ElementGolden> &dataAMax,
    std::vector<ElementGoldent> &dataAMin,
    std::vector<ElementGolden> &dataAStd,
    std::vector<ElementGolden> &dataAMeanAbs,
    std::vector<ElementGolden> &dataGolden, 
    float e_max,
    Catlass::Gemv::helper::FT_RCE_THRE_TYPE rce_thre_type 
)
    */
    golden::ComputeThresholdsASVARRobustTSlice(
        problemShapeABR, splitNnum,
        hostBMeanAbsGolden, hostBMeanSquareGolden, hostBVarGolden,
        options.problemShape.n(), FTChunkShape::N,
        hostA, layoutA, hostAMeanGolden, hostAMax, 
        hostAMinGolden, hostAStdGolden,hostAMeanAbsGolden,
        hostThreGolden, use_emax, rce_thre_type);

    // std::vector<GemvOutTypeC> hostAStdGolden(lenThre);

    // golden::ComputeThresholdsASVARTSlice(problemShapeABR, splitNnum,
    //      hostBMeanGolden, hostBMaxGolden, options.problemShape.n(),
    //      L1TileShape::N, hostA, layoutA, hostAMeanGolden, hostAMaxGolden,
    //      hostAStdGolden, hostThreGolden, options.e_max, rce_thre_type);

    
    errorIndices = golden::CompareData(hostThre, hostThreGolden, lenThre);
    if (errorIndices.empty()) {
        std::cout << "Threshold Compare success." << std::endl;
    } else {
        std::cerr << "Threshold Compare failed. Error count: " << errorIndices.size() << std::endl;
    }
    
    std::vector<uint8_t> hostGoldenCOMPRow(lenZRowR1,255);

    std::vector<uint64_t> totalErrorIdxRow;
    std::vector<uint64_t> totalErrorIdxRow_m;
    std::vector<uint64_t> totalErrorIdxRow_n;
    std::vector<float> totalErrorDataRow;
    std::vector<float> totalFailThresholds;

    errorIndices = golden::CompareData(hostAMean, hostAMeanGolden, options.problemShape.m());

    // for(int j=0; j < splitNnum; j++){
    //     printf("Expect A Mean[%d]: %f\n",j,hostAMeanGolden[j]);
    //     printf("Actual A Mean[%d]: %f\n",j,hostAMean[j]);
    // }

    if (errorIndices.empty()) {
        std::cout << "A Reduce Mean Compare success." << std::endl;
    } else {
        std::cerr << "A Reduce Mean Compare failed." << std::endl;
    }
    
    errorIndices = golden::CompareData(hostAMax, hostAMax, options.problemShape.m());

    if (errorIndices.empty()) {
        std::cout << "A Reduce Max Compare success." << std::endl;
    } else {
        std::cerr << "A Reduce Max Compare failed." << std::endl;
    }

    // for(int j=0; j < splitNnum; j++){
    //     printf("Expect A Max[%d]: %f\n",j,hostAMax[j]);
    //     printf("Actual A Max[%d]: %f\n",j,hostAMax[j]);
    // }

    errorIndices = golden::CompareData(hostAMin, hostAMinGolden, options.problemShape.m());

    if (errorIndices.empty()) {
        std::cout << "A Reduce Min Compare success." << std::endl;
    } else {
        std::cerr << "A Reduce Min Compare failed." << std::endl;
    }

    for(int j=0; j < splitNnum; j++){
        printf("Expect A Min[%d]: %f\n",j,hostAMinGolden[j]);
        printf("Actual A Min[%d]: %f\n",j,hostAMin[j]);
    }
    
    
    printf("%f\n", hostDRowR1[0]);
    printf("%f\n", hostGoldenCRowR1[0]);

    printf("%f\n", hostDRowR2[0]);
    printf("%f\n", hostGoldenCRowR2[0]);
    printf("Method: Verify with Computed Threshold\n");

    // errorIndices = golden::CompareDataAndIndexSliceWithThreshold(
    //     options.problemShape, hostGoldenCRow, hostDRow, hostThreGolden, 
    //     lenZRow, "CE", "ABE", totalErrorIdxRow, totalErrorIdxRow_m, totalErrorIdxRow_n,
    //     totalErrorDataRow, totalFailThresholds);

    

    /*
    std::vector<uint64_t> GetErrorDataAndIndexSliceWithThreshold(
    const Catlass::GemvCoord &problemShape,
    const std::vector<uint8_t>& result, 
    const std::vector<uint8_t>& expect,
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
    
    errorIndices = golden::GetErrorDataAndIndexSliceWithThreshold(
        options.problemShape,
        hostCOMPRow, hostGoldenCOMPRow, hostZRowR1, hostDRowR1, hostThre, 
        lenZRowR1, "CR1", "ABR1", totalErrorIdxRow, totalErrorIdxRow_m, totalErrorIdxRow_n,
        totalErrorDataRow, totalFailThresholds);
    
    if (errorIndices.empty()) {
        std::cout << "Row COMP OP compare success." << std::endl;
    } else {
        std::cerr << "Row COMP OP compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    printf("Total Error Idx len: %d\n", static_cast<int>(totalErrorIdxRow.size()));
    printf("Total Error Data len: %d\n", static_cast<int>(totalErrorDataRow.size()));


    for (int i = 0; i < 100; ++i) {
        ACL_CHECK(aclrtSynchronizeStream(stream));
        // RunAdapter(gemv_op, arguments, stream, aicCoreNum);
        // RunAdapter(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
        matmul_op(stream, aicCoreNum, fftsAddr);
        ACL_CHECK(aclrtSynchronizeStream(stream));
    }

    int num_repeat = 1;  // Injection test: only run once

    aclrtEvent start, stop;
    float temp_time = 0;
    float time = 0;
    ACL_CHECK(aclrtCreateEvent(&start));
    ACL_CHECK(aclrtCreateEvent(&stop));

    for (int i = 0; i < num_repeat; ++i) {
        ACL_CHECK(aclrtSynchronizeStream(stream));
        ACL_CHECK(aclrtRecordEvent(start, stream));

        // RunAdapter(gemv_op, arguments, stream, aicCoreNum);

        // RunAdapter(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
        matmul_op(stream, aicCoreNum, fftsAddr);
        // ACL_CHECK(aclrtSynchronizeStream(stream));
            
        ACL_CHECK(aclrtSynchronizeStream(stream));
        ACL_CHECK(aclrtRecordEvent(stop, stream));
        ACL_CHECK(aclrtSynchronizeEvent(stop));
        ACL_CHECK(aclrtEventElapsedTime(&temp_time, start, stop));
        time += temp_time;
    }
    // 4 * m * n + 4 * m*k + 4 * k*n
    // m*
    float total_op_nums = 1.0 * m*n*k*2;
    // / (1.0*1e12)
    std::cout << "m: " << m << ", n: " << n << ", k: "<< k <<","<< (float)(total_op_nums / (time / num_repeat * 1e-3))/ (1.0*1e12)  << " TFLOPS, " << (time / num_repeat) << " ms, repeat: " << num_repeat << std::endl;

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceACast));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));

    ACL_CHECK(aclrtFree(deviceXR2));
    ACL_CHECK(aclrtFree(deviceXR2Tail));
    ACL_CHECK(aclrtFree(deviceXV));
    ACL_CHECK(aclrtFree(deviceXVTail));

    ACL_CHECK(aclrtFree(deviceSliceCR1));
    ACL_CHECK(aclrtFree(deviceSliceCR2));

    ACL_CHECK(aclrtFree(deviceVXforAe));

    ACL_CHECK(aclrtFree(deviceZRowR1));
    ACL_CHECK(aclrtFree(deviceZRowR2));

    ACL_CHECK(aclrtFree(deviceDRowR1));
    ACL_CHECK(aclrtFree(deviceDRowR2));

    ACL_CHECK(aclrtFree(deviceCOMPRow));

    ACL_CHECK(aclrtFree(deviceThre));
    ACL_CHECK(aclrtFree(deviceBR1));
    ACL_CHECK(aclrtFree(deviceBR2));
    ACL_CHECK(aclrtFree(deviceBRforAIV));

    ACL_CHECK(aclrtFree(deviceBMaxSlice));
    ACL_CHECK(aclrtFree(deviceBMinSlice));

    ACL_CHECK(aclrtFree(deviceBMeanAbs));
    ACL_CHECK(aclrtFree(deviceBMeanSquare));
    ACL_CHECK(aclrtFree(deviceBVar));

    ACL_CHECK(aclrtFree(deviceAMax));
    ACL_CHECK(aclrtFree(deviceAMean));
    ACL_CHECK(aclrtFree(deviceAMin));

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