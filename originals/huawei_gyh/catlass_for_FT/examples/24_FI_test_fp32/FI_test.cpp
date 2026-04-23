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
#include "gemm/kernel/matmul_epilogue_asvar_thre_abft_no_splitk_aic_aiv_pipe_mixed_spec_robust_preload_FI.hpp"
#include "gemm/device/device_gemm.hpp" // catlass/

#include "fp16_t.h"
#include "bfloat16.h"

using namespace Catlass;

using DetectionMetrics = golden::DetectionMetrics;


using fp16_t = op::fp16_t;
using op_bfloat16 = op::bfloat16;

using GemmInTypeC = float;
// op_bfloat16;
using GemmInTypeN = float;
// bfloat16_t;

using GemmOutTypeC = float;
using GemmOutTypeN = float;

using GemvInTypeCforCE = float;
// op_bfloat16;
using GemvInTypeNforCE = float;
// bfloat16_t;

using GemvInTypeCforAB = float;
// op_bfloat16;
using GemvInTypeNforAB = float;
// bfloat16_t;

using GemvOutTypeC = float;
using GemvOutTypeN = float;

using ScalarTypeC = float;
using ScalarTypeN = float;

using ScalarType = float;


struct Options {
    const std::string HELPER = "18_matmul_ft m n k rt beta thre_type e_max red_cores split_ks inject_row_seed inject_col_seed inject_bit inject_k_unit inject_num_blk_limit [device_id]";

    GemmCoord problemGemmShape{128, 128, 128};
    GemvCoord problemShape{128, 128};

    int32_t deviceId{1};

    float round_exp{0.0f};
    float beta{1.0f};

    int thre_type{0};

    uint32_t reduce_cores{8};
    uint32_t split_ks{1};

    float e_max;

    // FI (Fault Injection) parameters
    uint32_t inject_row_seed{0};
    uint32_t inject_col_seed{0};
    uint32_t inject_bit{31};
    uint32_t inject_k_unit{1024};
    uint32_t inject_num_blk_limit{16};

    Options() = default;

    // inject_row_seed inject_col_seed inject_bit inject_k_unit inject_num_blk_limit
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
            INJECT_ROW_INDEX,
            INJECT_COL_INDEX,
            INJECT_BIT_INDEX,
            INJECT_K_UNIT_INDEX,
            INJECT_NUM_BLK_LIMIT_INDEX,
            DEVICE_ID_INDEX,
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

        // Parse FI parameters
        inject_row_seed = std::atoi(argv[INJECT_ROW_INDEX]);
        inject_col_seed = std::atoi(argv[INJECT_COL_INDEX]);
        inject_bit = std::atoi(argv[INJECT_BIT_INDEX]);
        inject_k_unit = std::atoi(argv[INJECT_K_UNIT_INDEX]);
        inject_num_blk_limit = std::atoi(argv[INJECT_NUM_BLK_LIMIT_INDEX]);

        if (argc == ARGS_MAX) {
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
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

    using L1TileShape = GemmShape<128, 256, 128>;
    using L0TileShape = GemmShape<128, 256, 32>;

    using UBTileShapeCE = GemvShape<64, L1TileShape::N>;
    using UBBlockShapeCE = GemvShape<L1TileShape::M, UBTileShapeCE::N>;
    // 64

    static constexpr uint32_t BYTE_FOR_EACH_BLK = 32;
    static constexpr uint32_t ELE_WORK_FOR_EACH_BLK = BYTE_FOR_EACH_BLK / sizeof(GemvOutTypeC);
    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemGemmShape.k();

    GemvCoord problemShapeCol{n, m};

    GemvCoord problemShapeBE{k,n};
    GemvCoord problemShapeABE{m,k};

    GemvCoord problemShapeETA{k,m};
    GemvCoord problemShapeETAB{n,k};

    uint32_t splitNnum = (n + L1TileShape::N - 1) / L1TileShape::N;
    uint32_t splitMnum = (m + L1TileShape::M - 1) / L1TileShape::M;


    uint32_t split_block_num = (splitNnum + ELE_WORK_FOR_EACH_BLK - 1) / ELE_WORK_FOR_EACH_BLK;

    uint32_t lenBMean = split_block_num * ELE_WORK_FOR_EACH_BLK + ELE_WORK_FOR_EACH_BLK;
    uint32_t lenBMax = lenBMean;
    uint32_t lenBMin = lenBMean;
    uint32_t lenBMeanAbs = lenBMean;
    uint32_t lenBMeanSquare = lenBMean;
    uint32_t lenBVar = lenBMean;

    /*
    calculate_expected_errors_per_blk(
        uint32_t KSize, uint32_t KInjectUnit,
        uint32_t inject_num_blk_limit, uint32_t BlockMSize)
    */

    uint32_t expected_errors_per_blk = golden::calculate_expected_errors_per_blk(
        k, options.inject_k_unit, options.inject_num_blk_limit, L1TileShape::M);
    
    /*
    uint32_t compute_align_expected_errors_per_blk(
        uint32_t expected_errors_per_blk, uint32_t element_size_in_byte)
    */

    uint32_t aligned_expected_errors_per_blk = golden::compute_align_expected_errors_per_blk(
        expected_errors_per_blk, sizeof(GemvOutTypeC));
    
    /*
    uint32_t calculate_ground_vector_size(
        const Catlass::GemvCoord &problemShape,
        uint32_t expected_errors_per_blk, 
        uint32_t BlockMSize, uint32_t BlockNSize, 
        uint32_t TileMSize, uint32_t element_size_in_byte)
    */
    uint32_t lenFIGround = golden::calculate_ground_vector_size(options.problemShape, 
        expected_errors_per_blk, 
        L1TileShape::M, L1TileShape::N, 
        UBTileShapeCE::M, sizeof(GemvOutTypeC));

    // uint32_t lenBMax = splitNnum * options.problemGemmShape.k();

    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenC = static_cast<size_t>(n) * m;

    // static_cast<size_t>(m) + static_cast<size_t>(n)

    size_t lenX =  static_cast<size_t>(n) * 1;
    size_t lenXV = static_cast<size_t>(n) * 1;
    size_t lenVXforAe = static_cast<size_t>(k) * 1;

    size_t lenDRow = (static_cast<size_t>(m)) * splitNnum;
    size_t lenDCol = (static_cast<size_t>(m)) * splitNnum;

    size_t lenCOMPCol = ((static_cast<size_t>(n)) + 8 - 1) / 8;
    size_t lenCOMPRow = (((static_cast<size_t>(m)) + 8 - 1) / 8) * splitNnum;

    size_t lenZRow = static_cast<size_t>(m) * splitNnum;
    size_t lenZCol = static_cast<size_t>(m) * splitNnum;
    size_t lenBE = static_cast<size_t>(k) * splitNnum;

    size_t lenThre = static_cast<size_t>(m) * splitNnum;

    size_t lenARed = static_cast<size_t>(m) * 1;

    size_t sizeA = lenA * sizeof(GemmInTypeC);
    size_t sizeB = lenB * sizeof(GemmInTypeC);
    size_t sizeC = lenC * sizeof(GemmOutTypeC);

    size_t sizeX = lenX * sizeof(GemvInTypeCforCE);
    size_t sizeXV = lenXV * sizeof(GemvInTypeCforAB);
    size_t sizeVXforAe = lenVXforAe * sizeof(GemvInTypeCforAB);
    size_t sizeBEforAIV = lenBE * sizeof(GemvInTypeCforCE);

    size_t sizeBE = lenBE * sizeof(GemvInTypeCforAB);

    size_t sizeZRow = lenZRow * sizeof(GemvOutTypeC);
    size_t sizeDRow = lenDRow * sizeof(GemvOutTypeC);

    size_t sizeCOMPRow = lenCOMPRow * sizeof(uint8_t);
    size_t sizeCOMPCol = lenCOMPCol * sizeof(uint8_t);

    size_t sizeZCol = lenZCol * sizeof(GemvOutTypeC);
    size_t sizeDCol = lenDCol * sizeof(GemvOutTypeC);

    size_t sizeThre = lenThre * sizeof(GemvOutTypeC);

    size_t sizeBMeanAbs = lenBMeanAbs * sizeof(GemvOutTypeC);
    size_t sizeBMeanSquare = lenBMeanSquare * sizeof(GemvOutTypeC);
    size_t sizeBVar = lenBVar * sizeof(GemvOutTypeC);

    size_t sizeAMean = lenARed * sizeof(GemvOutTypeC);
    size_t sizeAMax = lenARed * sizeof(GemvOutTypeC);
    size_t sizeAMin = lenARed * sizeof(GemvOutTypeC);

    size_t sizeFIGroundforID = lenFIGround * sizeof(int32_t);
    size_t sizeFIGroundforData = lenFIGround * sizeof(GemmOutTypeC);

    using LayoutX = layout::VectorLayout;
    using LayoutY = layout::VectorLayout;
    using LayoutCOMP = layout::VectorLayout;

    using LayoutA = layout::RowMajor;
    using LayoutACol = layout::ColumnMajor;

    using LayoutB = layout::RowMajor;
    using LayoutBCol = layout::ColumnMajor;

    using LayoutC = layout::RowMajor;
    using LayoutCCol = layout::ColumnMajor;

    using LayoutZ = layout::VectorLayout;
    using FT_COMP_TYPE = Catlass::Gemv::helper::FT_COMP_TYPE;

    LayoutX layoutXRow{n};
    LayoutX layoutXCol{m};

    LayoutC layoutC{m, n};
    LayoutCCol layoutCCol{n, m};

    LayoutA layoutA{m, k};
    LayoutACol layoutACol{k,m};

    LayoutB layoutB{k, n};
    LayoutBCol layoutBCol{n, k};

    LayoutZ layoutZ{m};
    LayoutZ layoutZCol{n};

    LayoutZ layoutZHost{m * splitNnum};

    LayoutZ layoutThre{m * splitNnum};

    ScalarType alpha{1.0};
    ScalarType beta{0.0};
    // FillRandomScalarData(alpha, -1.0f, 1.0f);
    // FillRandomScalarData(beta, -1.0f, 1.0f);

    float sum_base = 1.0;
    float mean_base = 1.0f / (1.0f * k);

    std::vector<uint8_t> hostCOMPRowRaw(lenCOMPRow,0);
    std::vector<uint8_t> hostCOMPColRaw(lenCOMPCol,0);

    std::vector<uint8_t> hostCOMPRow(lenCOMPRow,0);
    std::vector<uint8_t> hostCOMPCol(lenCOMPCol,0);
    
    std::vector<GemmOutTypeC> hostCRaw(lenC,0.0);

    std::vector<GemmOutTypeC> hostC(lenC,0.0);
    std::vector<GemmInTypeC> hostA(lenA);
    std::vector<GemmInTypeC> hostB(lenB);

    std::vector<GemvInTypeCforCE> hostX(lenX,(GemvInTypeCforCE)sum_base);
    std::vector<GemvInTypeCforAB> hostXV(lenXV,(GemvInTypeCforAB)sum_base);
    std::vector<GemvInTypeCforAB> hostVXforAe(lenVXforAe, (GemvInTypeCforAB)mean_base);

    printf("VX for Ae: %f\n",(float)hostVXforAe[0]);


    std::vector<GemvInTypeCforAB> hostBERaw(lenBE,(GemvInTypeCforAB)0.0f);
    std::vector<GemvInTypeCforCE> hostBEforAIVRaw(lenBE,(GemvInTypeCforCE)0.0f);

    std::vector<GemvInTypeCforAB> hostBE(lenBE,(GemvInTypeCforAB)0.0f);
    std::vector<GemvInTypeCforCE> hostBEforAIV(lenBE,(GemvInTypeCforCE)0.0f);

    std::vector<GemvInTypeCforCE> hostBMaxSliceRaw(lenBE,(GemvInTypeCforCE)0.0f);
    std::vector<GemvInTypeCforCE> hostBMinSliceRaw(lenBE,(GemvInTypeCforCE)0.0f);

    std::vector<GemvInTypeCforCE> hostBMaxSlice(lenBE,(GemvInTypeCforCE)0.0f);
    std::vector<GemvInTypeCforCE> hostBMinSlice(lenBE,(GemvInTypeCforCE)0.0f);

    std::vector<GemvOutTypeC> hostDRowRaw(lenZRow,0.0f);
    std::vector<GemvOutTypeC> hostDColRaw(lenZCol,0.0f);

    std::vector<GemvOutTypeC> hostZRowRaw(lenZRow,0.0f);
    std::vector<GemvOutTypeC> hostZColRaw(lenZCol,0.0f);

    std::vector<GemvOutTypeC> hostDRow(lenZRow,0.0f);
    std::vector<GemvOutTypeC> hostDCol(lenZCol,0.0f);

    std::vector<GemvOutTypeC> hostZRow(lenZRow,0.0f);
    std::vector<GemvOutTypeC> hostZCol(lenZCol,0.0f);

    std::vector<GemvOutTypeC> hostThreRaw(lenThre,0.0f);
    std::vector<GemvOutTypeC> hostAMeanRaw(lenARed, 0.0f);
    std::vector<GemvOutTypeC> hostAMaxRaw(lenARed, 0.0f);
    std::vector<GemvOutTypeC> hostAMinRaw(lenARed, 0.0f);

    std::vector<GemvOutTypeC> hostThre(lenThre,0.0f);
    std::vector<GemvOutTypeC> hostAMean(lenARed, 0.0f);
    std::vector<GemvOutTypeC> hostAMax(lenARed, 0.0f);
    std::vector<GemvOutTypeC> hostAMin(lenARed, 0.0f);


    std::vector<GemvOutTypeC> hostBMeanAbsRaw(lenBMeanAbs, 0.0f);
    std::vector<GemvOutTypeC> hostBMeanSquareRaw(lenBMeanSquare, 0.0f);
    std::vector<GemvOutTypeC> hostBVarRaw(lenBVar, 0.0f);

    std::vector<GemvOutTypeC> hostBMeanAbs(lenBMeanAbs, 0.0f);
    std::vector<GemvOutTypeC> hostBMeanSquare(lenBMeanSquare, 0.0f);
    std::vector<GemvOutTypeC> hostBVar(lenBVar, 0.0f);

    std::vector<uint32_t> hostSuccessRaw(1, 0);
    std::vector<int32_t> hostFIIndexRaw(lenFIGround, -1);
    std::vector<GemmOutTypeC> hostFIFlipDataRaw(lenFIGround, 0.0f);

    // FI (Fault Injection) success flag
    std::vector<uint32_t> hostSuccess(1, 0);
    std::vector<int32_t> hostFIIndex(lenFIGround, -1);
    std::vector<GemmOutTypeC> hostFIFlipData(lenFIGround, 0.0f);
    
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

    uint8_t* deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceX{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceX), sizeX, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceX, sizeX, hostX.data(), sizeX, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceXV{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceXV), sizeXV, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceXV, sizeXV, hostXV.data(), sizeXV, ACL_MEMCPY_HOST_TO_DEVICE));

    // hostVXforAe
    uint8_t* deviceVXforAe{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceVXforAe), sizeVXforAe, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceVXforAe, sizeVXforAe, hostVXforAe.data(), sizeVXforAe, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceBE{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceBE), sizeBE, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceBE, sizeBE, hostBERaw.data(), sizeBE, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceBEforAIV{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceBEforAIV), sizeBEforAIV, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceBEforAIV, sizeBEforAIV, hostBEforAIVRaw.data(), sizeBEforAIV, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceBMaxSlice{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceBMaxSlice), sizeBEforAIV, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceBMaxSlice, sizeBEforAIV, hostBMaxSliceRaw.data(), sizeBEforAIV, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceBMinSlice{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceBMinSlice), sizeBEforAIV, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceBMinSlice, sizeBEforAIV, hostBMinSliceRaw.data(), sizeBEforAIV, ACL_MEMCPY_HOST_TO_DEVICE));

    printf("size of A: %zu\n", sizeA);
    printf("size of B: %zu\n", sizeB);
    printf("size of C: %zu\n", sizeC);
    printf("size of X: %zu\n", sizeX);
    printf("size of Z: %zu\n", sizeZRow);
    printf("size of Z: %zu\n", sizeZCol);



    uint8_t* deviceZRow{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceZRow), sizeZRow, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceZRow, sizeZRow, hostZRowRaw.data(), sizeZRow, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceZCol{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceZCol), sizeZCol, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceZCol, sizeZCol, hostZColRaw.data(), sizeZCol, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceDRow{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceDRow), sizeZRow, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceDRow, sizeZRow, hostDRowRaw.data(), sizeZRow, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceDCol{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceDCol), sizeZCol, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceDCol, sizeZCol, hostDColRaw.data(), sizeZCol, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceCOMPRow{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceCOMPRow), sizeCOMPRow, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceCOMPRow, sizeCOMPRow, hostCOMPRowRaw.data(), sizeCOMPRow, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceCOMPCol{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceCOMPCol), sizeCOMPCol, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceCOMPCol, sizeCOMPCol, hostCOMPColRaw.data(), sizeCOMPCol, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceThre{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceThre), sizeThre, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceThre, sizeThre, hostThreRaw.data(), sizeThre, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceBMeanAbs{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceBMeanAbs), sizeBMeanAbs, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceBMeanAbs, sizeBMeanAbs, hostBMeanAbsRaw.data(), sizeBMeanAbs, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceBMeanSquare{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceBMeanSquare), sizeBMeanSquare, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceBMeanSquare, sizeBMeanSquare, hostBMeanSquareRaw.data(), sizeBMeanSquare, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceBVar{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceBVar), sizeBVar, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceBVar, sizeBVar, hostBVarRaw.data(), sizeBVar, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceAMean{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceAMean), sizeAMean, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceAMean, sizeAMean, hostAMeanRaw.data(), sizeAMean, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceAMax{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceAMax), sizeAMax, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceAMax, sizeAMax, hostAMaxRaw.data(), sizeAMax, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceAMin{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceAMin), sizeAMin, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceAMin, sizeAMin, hostAMinRaw.data(), sizeAMin, ACL_MEMCPY_HOST_TO_DEVICE));

    // ACL_CHECK(aclrtMemcpy(deviceThre, sizeThre, hostThre.data(), sizeThre, ACL_MEMCPY_HOST_TO_DEVICE));

    // FI (Fault Injection) success flag device memory
    uint8_t* deviceSuccess{nullptr};
    size_t sizeSuccess = hostSuccess.size() * sizeof(uint32_t);
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceSuccess), sizeSuccess, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceSuccess, sizeSuccess, hostSuccess.data(), sizeSuccess, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceFIIndex{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceFIIndex), sizeFIGroundforID, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceFIIndex, sizeFIGroundforID, hostFIIndex.data(), sizeFIGroundforID, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceFIFlipData{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceFIFlipData), sizeFIGroundforData, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceFIFlipData, sizeFIGroundforData, hostFIFlipData.data(), sizeFIGroundforData, ACL_MEMCPY_HOST_TO_DEVICE));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    using ArchTag = Arch::AtlasA2;
    using FT_REDUCE_TYPE = Gemv::helper::FT_REDUCE_TYPE;

    using LayoutMY = layout::RowMajor;
    using LayoutMX = layout::RowMajor;
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

    using L1TileShapeBE = GemvShape<128, 256>;
    using L0TileShapeBE = GemvShape<128, 64>;

    using UBBlockShapeBE = GemvShape<L1TileShapeBE::M*2, L1TileShapeBE::N*1>;
    // using UBBlockShapeBE = GemvShape<L1TileShapeBE::M*2, L1TileShapeBE::N*2>;

    using AType = Gemm::GemmType<GemmInTypeN, LayoutA>;
    using BType = Gemm::GemmType<GemmInTypeN, LayoutB>;
    
    using CType = Gemm::GemmType<GemmOutTypeN, LayoutC>;
    
    using XTypeAIC = Gemm::GemmType<GemvInTypeNforAB, LayoutX>;
    using YTypeBEAIC = Gemm::GemmType<GemvInTypeNforAB, LayoutMY>;
    using BiasType = void;

    using XTypeAIV = Gemm::GemmType<GemvInTypeNforCE, LayoutX>;

    using GemvDispatchPolicy = Gemm::GemvAtlasA2;
    using COMPDispatchPolicy = Gemm::GemvAtlasA2;
    using BeAICDispatchPolicy = Gemm::MmadAtlasA2Preload<enableUnitFlag, enableShuffleK>;
    using GEMVAICDispatchPolicy = Gemm::MmadAtlasA2Preload<enableUnitFlag, enableShuffleK>;
    using TileCopyGemvAic = Gemv::Tile::TileCopyGemvAic<typename BeAICDispatchPolicy::ArchTag, BType, XTypeAIC, YTypeBEAIC, BiasType>;
    using TileMmadGemvAic = Gemm::Tile::TileMmad<typename  BeAICDispatchPolicy::ArchTag, XTypeAIC, BType, BiasType>;

    /*
    struct BlockFTGemvBe<
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
    using BlockFTGemvAIC = Gemv::Block::BlockFTGemvBe<BeAICDispatchPolicy, 
        FT_AIC_BE_SCHEME::ROWCOMPLETE,
        UBBlockShapeBE, L1TileShapeBE, L0TileShapeBE, 
        BType, XTypeAIC, YTypeBEAIC, BiasType, TileCopyGemvAic, TileMmadGemvAic>;

    using ZType = Gemm::GemmType<GemvOutTypeN, LayoutZ>;

    static constexpr FT_AIC_BE_SCHEME BE_SCHEME = BlockFTGemvAIC::BE_SCHEME;

    constexpr uint32_t computeLength = 8192;

    using YType = Gemm::GemmType<GemvInTypeNforCE, LayoutY>;

    using BRedType = Gemm::GemmType<GemvInTypeNforCE, LayoutB>;
    using TileVmuls = Gemv::Tile::TileVmuls<ArchTag, XTypeAIV>;

    using MmadDispatchPolicy = CubeSelf::Gemm::MmadAtlasA2Pingpong<enableUnitFlag>;
    
    // using L0TileShapeforFT = GemvShape<L1TileShape::M,128>;
    // using L1TileShape = GemmShape<128, 240, 256>;
    // using L0TileShape = GemmShape<128, 240, 64>;

    using L1TileShapeFirst = GemmShape<256,256,128>;
    using L0TileShapeFirst = GemmShape<256,256,32>;

    using L1TileShapeforFT = GemmShape<L1TileShapeFirst::M, 16, L1TileShapeFirst::K>;
    using L0TileShapeforFT = GemmShape<L0TileShapeFirst::M, 16, L0TileShapeFirst::K>;

    // using L1TileShapeforFT = GemmShape<L1TileShapeFirst::M, 32, L1TileShapeFirst::K>;
    // using L0TileShapeforFT = GemmShape<L0TileShapeFirst::M, 32, L0TileShapeFirst::K>;

    /*
    using LayoutMY = layout::RowMajor;
    using LayoutMX = layout::ColumnMajor;
    */
    using MXType = Gemm::GemmType<GemvInTypeNforAB, LayoutMX>;
    using MYType = Gemm::GemmType<GemvOutTypeN, LayoutMY>;

    // using LayoutMY = layout::RowMajor;
    // using MYType = Gemm::GemmType<GemvOutTypeN, LayoutMY>;
    /*
    template<
        class DispatchPolicy,
        class L1TileShapeforFT,
        class L0TileShapeforFT,
        class AType,
        class BType,
        class CType,
        class XType,
        class YType,
        class BiasType,
        class TileCopyFTABonAic = CubeSelf::Gemm::Tile::TileCopyFTABonAic<typename DispatchPolicy::ArchTag, AType, BType, CType, XType, YType, BiasType>,
        class TileMmad = CubeSelf::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, XType, BiasType>
    >
    struct BlockMmadSpecABeNoSplitKRobust
    */
    using BlockMmadABe = CubeSelf::Gemm::Block::BlockMmadSpecABeNoSplitKRobust<
        MmadDispatchPolicy, 
        L1TileShapeforFT,
        L0TileShapeforFT, 
        AType, BType, CType, MXType, MYType, BiasType>;
    
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
        AType, BType, YType, ZType>;

    using UBTileShapeforB = GemvShape<64, L0TileShape::N>;
    using UBBlockShapeforB = GemvShape<UBTileShapeforB::M*2, UBTileShapeforB::N*2>;

    using UBTileShapeforA = GemvShape<64, 256>;

    using ARedType = Gemm::GemmType<GemvInTypeNforCE, LayoutA>;
    using TileFaultSum = Gemv::Tile::TileFaultSum<ArchTag, FT_REDUCE_TYPE::MAX_MIN, ARedType, ZType>;

    /*
    struct BlockFTSumNoSplitK <
    Gemm::GemvAtlasA2,
    Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
    Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::A_B_MIXED,
    UBTileShapeforB_,
    UBBlockShapeforB_,
    UBTileShapeforA_,
    L1TileShape_,
    AType_,
    BType_,
    XType_,
    YType_,
    BiasType_,
    TileCopy_,
    TileFaultSum_
    >  
    */
    
    using BlockFTSum = Gemv::Block::BlockFTSumNoSplitK<
        GemvDispatchPolicy,
        Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
        Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::A_B_MIXED,
        UBTileShapeforB, UBBlockShapeforB, UBTileShapeforA,
        L1TileShape, AType, BType, YType, ZType, void,
        TileFaultCopyRedAiv, TileFaultSum>;

    using SliceSumDispatchPolicy = Gemm::GemvAtlasA2;

    using SliceSumUBTileShape = GemvShape<8, 256>;
    using TileMatrixAddforABEReduce = Gemv::Tile::TileMatmulAdd<
        typename SliceSumDispatchPolicy::ArchTag, MYType, MYType, void>;
    using TileCopyMatrixAddforABEReduce = Gemv::Tile::TileCopyMatrixAddAiv<
        typename SliceSumDispatchPolicy::ArchTag, MYType, MYType, void>; 
    
    using BlockSliceSum = Gemv::Block::BlockSliceKMNSum<SliceSumDispatchPolicy,
        Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::ABE_FUSED_THRE,
        SliceSumUBTileShape, MYType, MYType, void, TileCopyMatrixAddforABEReduce, TileMatrixAddforABEReduce>;

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
    using TileFaultCopyBReduce = Gemv::Tile::TileCopyGemvAiv<ArchTag, BRedType, YType, ZType>;
    
    using BlockSliceRed = Gemv::Block::BlockSliceKMNSum<
        SliceSumDispatchPolicy,
        Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::A_B_ROBUST,
        SliceSumUBTileShape,
        UBTileShapeforBRed,
        MYType, BRedType, ZType,
        MYType, void,
        TileCopyMatrixAddforABEReduce, 
        TileFaultCopyBReduce,
        TileMatrixAddforABEReduce,
        TileFaultSumBReduce,
        MeanMaxTileVmuls>;

    

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
    /// Output Result operand namely vector Z
    class ZType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
    >
    struct TileCopyGemvThreCompFusedAivFI
    */
    // using TileFaultCopyABE = Gemv::Tile::TileCopyGemvAiv<ArchTag, AType, XType, YType>;
    using TileFaultCopyCE = Gemv::Tile::TileCopyGemvThreCompFusedAivFI<ArchTag, 
        CType, CType, YType, ZType, COMPZType, void>;

    using ThreCalcDispatchPolicy = Gemm::GemvAtlasA2;

    /*
    struct TileThreCalc<Arch::AtlasA2,
                helper::FT_THRESHOLD_ALGORITHM::ASVAR,
                Gemm::GemmType<ElementA, layout::RowMajor>,
                Gemm::GemmType<float, layout::VectorLayout>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
    */

    using TileThreCalc = Gemv::Tile::TileThreCalc<
        typename ThreCalcDispatchPolicy::ArchTag, 
        Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
        CType, ZType, ZType, BiasType>; 

    // using TileThreCalcRaw = Gemv::Tile::TileThreCalc<
    //     typename ThreCalcDispatchPolicy::ArchTag, 
    //     Gemv::helper::FT_THRESHOLD_ALGORITHM::AABFT,
    //     AType, YType, ZType, BiasType>; 

    /*
    struct TileStdEst<Arch::AtlasA2,
                Gemm::GemmType<float, layout::VectorLayout>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
    */

    using TileStdEst = Gemv::Tile::TileStdEstRobust<
        typename ThreCalcDispatchPolicy::ArchTag,
        ZType,
        ZType
    >;

    /*
    struct BlockFTGemvCENoSplitKPreload <
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
    TileFaultSum_,
    TileThreCalc_,
    TileStdEst_>
    */

    using TileFaultSumCSum = Gemv::Tile::TileFaultSum<ArchTag, FT_REDUCE_TYPE::SUM, CType, ZType>;

    using BlockFTGemvAIV = Gemv::Block::BlockFTGemvCENoSplitKPreloadFI<
        GemvDispatchPolicy,
        Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
        Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::THRE_FUSED, 
        Gemv::helper::FT_ENC_TYPE::RCE, 
        Gemv::helper::FT_COMP_TYPE::RSUB,
        Gemv::helper::FT_ABE_TYPE::CENTRAL_BLOCK,
        UBTileShapeCE, UBBlockShapeCE, L1TileShape,
        CType, YType, ZType, COMPZType, void, 
        TileFaultCopyCE, TileFaultSumCSum,
        TileThreCalc, TileStdEst>;

    using UBTileShape = GemmShape<L1TileShapeAB::M, L1TileShapeAB::N, L1TileShapeAB::K>;
    using TileFaultCopy = Gemv::Tile::TileCopyGemvAiv<ArchTag, AType, XTypeAIV, ZType>;
    using TileFaultVmad = Gemv::Tile::TileVmad<ArchTag, AType, XTypeAIV, ZType>;
   
    using BlockSumGemv = Gemv::Block::BlockSumGemv<GemvDispatchPolicy, UBTileShape, AType, XTypeAIV, ZType, void, TileFaultCopy, TileFaultVmad, TileVmuls>;
    // Kernel level
    /*
    template <
        class BlockMmadABe_,
        class BlockMmad_,
        class BlockSchedulerFirst_,
        class BlockScheduler_,
        class BlockFTGemvAIC_,
        class BlockFTSum_,
        class BlockFTGemvAIV_,
        class BlockSliceRed_,
        class BlockSliceSum_
    >
    class MatmulAsVarABonAicNoSplitMixedSpec
    */
    // , BlockThresholdCalc
    // CompareBlockSUB
    // MatmulAsVarABonAicNoSplitSpecRobust
    using MatmulFTKernel = CubeSelf::Gemm::Kernel::MatmulAsVarABonAicNoSplitSpecRobustPreloadFI<
        BlockMmadABe, BlockMmadPreload, BlockSchedulerFirst, BlockScheduler,
        BlockFTGemvAIC, BlockFTSum, BlockFTGemvAIV, 
        BlockSliceRed, BlockSliceSum>;
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
        Catlass::GemvCoord problemShape; size_t elementSize;
        GM_ADDR ptrX; GM_ADDR ptrXV;
        GM_ADDR ptrA; GM_ADDR ptrB; 
        GM_ADDR ptrC; GM_ADDR ptrZRow; GM_ADDR ptrZCol;
        GM_ADDR ptrZRow2;
        GM_ADDR ptrZCol2;
        GM_ADDR ptrCOMPZRow; GM_ADDR ptrCOMPZCol;
        GM_ADDR ptrBE; GM_ADDR ptrBEforAIV;
        GM_ADDR ptrBMaxSlice; GM_ADDR ptrBMinSlice;
        GM_ADDR ptrBMeanAbs; GM_ADDR ptrBMeanSquare;
        GM_ADDR ptrBVar; GM_ADDR ptrVXforA;
        GM_ADDR ptrAMean; GM_ADDR ptrAMax;
        GM_ADDR ptrAMin; GM_ADDR ptrThreZ;
        FT_ENC_TYPE enc_type;
        uint32_t UbNum;
        bool OutputWorkspace;
        ElementCOMPX threshold;
        float rounding_exponent;
        float size_beta;
        float e_max_raw;
        uint32_t reduce_cores;
        FT_RCE_THRE_TYPE rce_thre_type;
        bool outputThre;
        bool outputCE;
        uint32_t SplitKNum;
        uint32_t KInjectUnit;
        uint32_t inject_num_blk_limit;
        uint32_t inject_row_start;
        uint32_t inject_col;
        uint32_t inject_bit_raw;
        GM_ADDR success_ptr;
        GM_ADDR ptrFIIndex;
        GM_ADDR ptrFIFlipData;
    };
    */

    float use_emax = options.e_max * 1.0f;
    if(k <= 1024){
        use_emax = use_emax * 1.0f;
    }else{
        use_emax = use_emax * std::sqrt(((k*1.0f / 1024*1.0f)*1.0f));
    }

    

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));
    MatmulAdapter matmul_op;
    // matmul_op.CanImplement(arguments);

    // size_t sizeWorkspace = matmul_op.GetWorkspaceSize(arguments);
    // uint8_t *deviceWorkspace = nullptr;
    // if (sizeWorkspace > 0) {
    //     ACL_CHECK(
    //         aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST)
    //     );
    // }

    // RunAdapter(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
    // matmul_op.Initialize(arguments, deviceWorkspace);

    // matmul_op(stream, aicCoreNum, fftsAddr);

    typename MatmulFTKernel::Arguments arguments{
        options.problemGemmShape, options.problemShape, sizeof(GemvInTypeCforAB), 
        deviceX, deviceXV, deviceA, deviceB, deviceC, 
        deviceZRow, deviceZCol, deviceDRow, deviceDCol, 
        deviceCOMPRow, deviceCOMPCol, 
        deviceBE, deviceBEforAIV, deviceBMaxSlice, deviceBMinSlice,
        deviceBMeanAbs, deviceBMeanSquare, deviceBVar, 
        deviceVXforAe, deviceAMean, deviceAMax, deviceAMin,
        deviceThre, FT_ENC_TYPE::RCE, 1, false, threshold,
        options.round_exp, options.beta, 
        use_emax, options.reduce_cores, 
        rce_thre_type,true,true, options.split_ks,
        options.inject_k_unit, options.inject_num_blk_limit, options.inject_row_seed,
        options.inject_col_seed, options.inject_bit, 
        deviceSuccess, deviceFIIndex, deviceFIFlipData};

    RunAdapter(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    

    ACL_CHECK(aclrtMemcpy(hostZRow.data(), sizeZRow, deviceZRow, sizeZRow, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostZCol.data(), sizeZCol, deviceZCol, sizeZCol, ACL_MEMCPY_DEVICE_TO_HOST));

    ACL_CHECK(aclrtMemcpy(hostDRow.data(), sizeZRow, deviceDRow, sizeZRow, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostDCol.data(), sizeZCol, deviceDCol, sizeZCol, ACL_MEMCPY_DEVICE_TO_HOST));
    
    ACL_CHECK(aclrtMemcpy(hostCOMPRow.data(), sizeCOMPRow, deviceCOMPRow, sizeCOMPRow, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostCOMPCol.data(), sizeCOMPCol, deviceCOMPCol, sizeCOMPCol, ACL_MEMCPY_DEVICE_TO_HOST));

    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));
    
    ACL_CHECK(aclrtMemcpy(hostThre.data(), sizeThre, deviceThre, sizeThre, ACL_MEMCPY_DEVICE_TO_HOST));

    ACL_CHECK(aclrtMemcpy(hostBMeanAbs.data(), sizeBMeanAbs, deviceBMeanAbs, sizeBMeanAbs, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostBMeanSquare.data(), sizeBMeanSquare, deviceBMeanSquare, sizeBMeanSquare, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostBVar.data(), sizeBVar, deviceBVar, sizeBVar, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostBE.data(), sizeBE, deviceBE, sizeBE, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostBEforAIV.data(), sizeBEforAIV, deviceBEforAIV, sizeBEforAIV, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostBMaxSlice.data(), sizeBEforAIV, deviceBMaxSlice, sizeBEforAIV, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostBMinSlice.data(), sizeBEforAIV, deviceBMinSlice, sizeBEforAIV, ACL_MEMCPY_DEVICE_TO_HOST));

    ACL_CHECK(aclrtMemcpy(hostAMax.data(), sizeAMax, deviceAMax, sizeAMax, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostAMean.data(), sizeAMean, deviceAMean, sizeAMean, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostAMin.data(), sizeAMin, deviceAMin, sizeAMin, ACL_MEMCPY_DEVICE_TO_HOST));
   
    // Read back FI success flag
    ACL_CHECK(aclrtMemcpy(hostSuccess.data(), sizeSuccess, deviceSuccess, sizeSuccess, ACL_MEMCPY_DEVICE_TO_HOST));

    ACL_CHECK(aclrtMemcpy(hostFIIndex.data(), sizeFIGroundforID, deviceFIIndex, sizeFIGroundforID, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostFIFlipData.data(), sizeFIGroundforData, deviceFIFlipData, sizeFIGroundforData, ACL_MEMCPY_DEVICE_TO_HOST));
    
    // Print FI injection status
    printf("FI Injection Status:\n");
    printf("  Inject err seed: %u\n", options.inject_row_seed);
    printf("  Inject col seed: %u\n", options.inject_col_seed);
    printf("  Inject bit: %u\n", options.inject_bit);
    printf("  Success flag: %u\n", hostSuccess[0]);

    std::vector<GemvOutTypeC> hostGoldenCRow(lenZRow, (GemvOutTypeC)0.0f);
    std::vector<GemvOutTypeC> hostGoldenRow(lenZRow, (GemvOutTypeC)0.0f);
    std::vector<GemvOutTypeC> hostGoldenCol(lenZCol);
    std::vector<GemvOutTypeC> hostBMeanAbsGolden(lenBMeanAbs,(GemvOutTypeC)0.0f);
    std::vector<GemvOutTypeC> hostBMeanSquareGolden(lenBMeanSquare, (GemvOutTypeC)0.0f);
    std::vector<GemvOutTypeC> hostBVarGolden(lenBVar, (GemvOutTypeC)0.0f);

    std::vector<GemvInTypeCforCE> hostYForAB(static_cast<uint32_t>(k * splitNnum), (GemvInTypeCforCE)0.0f);
    LayoutY layoutYForAB{static_cast<uint32_t>(k * splitNnum)};

    GemvCoord BESliceShape{1, L1TileShape::N};
    printf("BESliceShape: {%d,%d}\n", BESliceShape.m(), BESliceShape.n());

    golden::ComputeGemvSlice(options.problemShape, BESliceShape, alpha, beta,
         hostC, layoutC, hostX, layoutXRow, hostGoldenCRow, layoutZHost, hostGoldenCRow);

    // hostGoldenCRow
    std::vector<uint64_t> errorIndices = golden::CompareData(hostZRow, hostZRow, m);

    if (errorIndices.empty()) {
        std::cout << "CE: RowSum Compare success." << std::endl;
    } else {
        std::cerr << "CE: RowSum Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    /*
    template<typename Element, class ElementA, class LayoutA, class ElementX, class LayoutX, class ElementY, class LayoutY, class ElementGolden>
    void ComputeGemvSlice(
        const Catlass::GemvCoord &problemShape,
        const Catlass::GemvCoord &sliceShape,
        Element alpha, Element beta,
        const std::vector<ElementA> &dataA, const LayoutA &layoutA,
        const std::vector<ElementX> &dataX, const LayoutX &layoutX,
        const std::vector<ElementY> &dataY, const LayoutY &layoutY,
        std::vector<ElementGolden> &dataGolden)
    */

    golden::ComputeGemvSlice(problemShapeBE, BESliceShape, 
        alpha, beta, hostB, layoutB, hostXV, layoutXRow, 
        hostYForAB, layoutYForAB, hostYForAB);
    
    // std::vector<uint64_t> errorIndices;
    if(BE_SCHEME == FT_AIC_BE_SCHEME::ROWCOMPLETE_BF){
        errorIndices = golden::CompareData(hostBEforAIV, hostYForAB, problemShapeBE.m());
    }else{
        errorIndices = golden::CompareData(hostBE, hostYForAB, problemShapeBE.m());
    }
    

    if (errorIndices.empty()) {
        std::cout << "Be: Rowsum Compare success." << std::endl;
    } else {
        std::cerr << "Be: Rowsum Compare failed." << std::endl;
    }
        
    
    GemvCoord ABESliceShape{static_cast<uint32_t>(splitNnum), problemShapeABE.n()};
    golden::ComputeGemvSlice(problemShapeABE, ABESliceShape, alpha, beta,
         hostA, layoutA, hostYForAB, layoutYForAB, 
         hostGoldenRow, layoutZHost, hostGoldenRow);

    errorIndices = golden::CompareData(hostDRow, hostDRow, problemShapeABE.n());

    if (errorIndices.empty()) {
        std::cout << "ABe: Rowsum Compare success." << std::endl;
    } else {
        std::cerr << "ABe: Rowsum Compare failed." << std::endl;
    }

    // errorIndices = golden::CompareData(hostDRow, hostGoldenCRow, problemShapeABE.n());

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
    GemvCoord BReduceSliceShape{problemShapeBE.m(), L1TileShape::N};
    golden::ComputeMeanAbsSquareVarSliceRobust(problemShapeBE, BReduceSliceShape,
        hostB, layoutB, hostBMeanAbsGolden, hostBMeanSquareGolden, hostBVarGolden);

    errorIndices = golden::CompareData(hostBMeanAbs, hostBMeanAbsGolden, splitNnum);

    for(int j=0; j < splitNnum; j++){
        printf("Expect B Mean ABS[%d]: %f\n",j,hostBMeanAbsGolden[j]);
        printf("Actual B Mean ABS[%d]: %f\n",j,hostBMeanAbs[j]);
        
    }

    if (errorIndices.empty()) {
        std::cout << "B Reduce Mean ABS Compare success." << std::endl;
    } else {
        std::cerr << "B Reduce Mean ABS Compare failed." << std::endl;
    }
    
    errorIndices = golden::CompareData(hostBMeanSquare, hostBMeanSquareGolden, splitNnum);

    for(int j=0; j < splitNnum; j++){
        printf("Expect B Mean square[%d]: %f\n",j,hostBMeanSquareGolden[j]);
        printf("Actual B Mean square[%d]: %f\n",j,hostBMeanSquare[j]);
        
    }

    if (errorIndices.empty()) {
        std::cout << "B Reduce Mean square Compare success." << std::endl;
    } else {
        std::cerr << "B Reduce Mean square Compare failed." << std::endl;
    }

    errorIndices = golden::CompareData(hostBVar, hostBVarGolden, splitNnum);

    for(int j=0; j < splitNnum; j++){
        printf("Expect B Var[%d]: %f\n",j,hostBVarGolden[j]);
        printf("Actual B Var[%d]: %f\n",j,hostBVar[j]);
        
    }

    if (errorIndices.empty()) {
        std::cout << "B Reduce Var Compare success." << std::endl;
    } else {
        std::cerr << "B Reduce Var Compare failed." << std::endl;
    }

    std::vector<GemvOutTypeC> hostThreGolden(lenThre);

    GemvCoord ThreSliceShape{L1TileShape::M, L1TileShape::N};

    std::vector<GemvOutTypeC> hostAMaxGolden(lenThre);
    std::vector<GemvOutTypeC> hostAMeanGolden(lenThre);
    std::vector<GemvOutTypeC> hostAMinGolden(lenThre);
    std::vector<GemvOutTypeC> hostAStdGolden(lenThre);
    std::vector<GemvOutTypeC> hostAMeanAbsGolden(lenThre);

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
        problemShapeABE, splitNnum,
        hostBMeanAbsGolden, hostBMeanSquareGolden, hostBVarGolden,
        options.problemShape.n(), L1TileShape::N,
        hostA, layoutA, hostAMeanGolden, hostAMaxGolden, 
        hostAMinGolden, hostAStdGolden,hostAMeanAbsGolden,
        hostThreGolden, use_emax, rce_thre_type);

    // std::vector<GemvOutTypeC> hostAStdGolden(lenThre);

    // golden::ComputeThresholdsASVARTSlice(problemShapeABE, splitNnum,
    //      hostBMeanGolden, hostBMaxGolden, options.problemShape.n(),
    //      L1TileShape::N, hostA, layoutA, hostAMeanGolden, hostAMaxGolden,
    //      hostAStdGolden, hostThreGolden, options.e_max, rce_thre_type);

    
    errorIndices = golden::CompareData(hostThre, hostThreGolden, lenThre);
    if (errorIndices.empty()) {
        std::cout << "Threshold Compare success." << std::endl;
    } else {
        std::cerr << "Threshold Compare failed. Error count: " << errorIndices.size() << std::endl;
    }
    
    std::vector<uint8_t> hostGoldenCOMPRow(lenZRow,255);

    std::vector<uint64_t> totalErrorIdxRow;
    std::vector<uint64_t> totalErrorIdxRow_m;
    std::vector<uint64_t> totalErrorIdxRow_n;
    std::vector<int32_t> totalDetectedErrorIdxinFI;
    std::vector<int32_t> totalDetectedErrorBlkIdxinFI;
    std::vector<float> totalErrorDataRow;
    std::vector<float> totalOriginalDataElem;
    std::vector<float> totalFailThresholds;

    std::vector<int32_t> true_detected_error_idies_set;
    std::vector<int32_t> false_detected_error_idies_set;
    std::vector<int32_t> false_detected_error_blks_set;


    errorIndices = golden::CompareData(hostAMean, hostAMeanGolden, options.problemShape.m());

    for(int j=0; j < splitNnum; j++){
        printf("Expect A Mean[%d]: %f\n",j,hostAMeanGolden[j]);
        printf("Actual A Mean[%d]: %f\n",j,hostAMean[j]);
    }

    if (errorIndices.empty()) {
        std::cout << "A Reduce Mean Compare success." << std::endl;
    } else {
        std::cerr << "A Reduce Mean Compare failed." << std::endl;
    }
    
    errorIndices = golden::CompareData(hostAMax, hostAMaxGolden, options.problemShape.m());

    if (errorIndices.empty()) {
        std::cout << "A Reduce Max Compare success." << std::endl;
    } else {
        std::cerr << "A Reduce Max Compare failed." << std::endl;
    }

    // for(int j=0; j < splitNnum; j++){
    //     printf("Expect A Max[%d]: %f\n",j,hostAMaxGolden[j]);
    //     printf("Actual A Max[%d]: %f\n",j,hostAMax[j]);
    // }

    errorIndices = golden::CompareData(hostAMin, hostAMinGolden, options.problemShape.m());

    if (errorIndices.empty()) {
        std::cout << "A Reduce Min Compare success." << std::endl;
    } else {
        std::cerr << "A Reduce Min Compare failed." << std::endl;
    }

    // for(int j=0; j < splitNnum; j++){
    //     printf("Expect A Min[%d]: %f\n",j,hostAMaxGolden[j]);
    //     printf("Actual A Min[%d]: %f\n",j,hostAMax[j]);
    // }
    
    
    printf("%f\n", hostDRow[0]);
    printf("%f\n", hostGoldenCRow[0]);
    printf("Method: Verify with Computed Threshold\n");

    // errorIndices = golden::CompareDataAndIndexSliceWithThreshold(
    //     options.problemShape, hostGoldenCRow, hostDRow, hostThreGolden, 
    //     lenZRow, "CE", "ABE", totalErrorIdxRow, totalErrorIdxRow_m, totalErrorIdxRow_n,
    //     totalErrorDataRow, totalFailThresholds);

    

    /*
    template<class ElementData>
    std::vector<uint64_t> GetErrorDataAndIndexSliceWithThresholdFI(
        const Catlass::GemvCoord &problemShape,
        const std::vector<uint8_t>& result, 
        const std::vector<uint8_t>& expect,
        const std::vector<ElementData> &actualdata, 
        const std::vector<ElementData> &expectdata,
        const std::vector<ElementData> &thresholddata, 
        const std::vector<ElementData> &rawOutputdata,
        uint32_t computeNum, const char* IdNameAct, const char* IdNameExp,
        std::vector<uint64_t>& total_error_idies,
        std::vector<int32_t>& total_error_idies_for_FI,
        std::vector<int32_t>& total_error_blk_idies_for_FI,
        std::vector<uint64_t>& total_error_idies_m,
        std::vector<uint64_t>& total_error_idies_n, 
        std::vector<ElementData>& total_error_data,
        std::vector<ElementData>& total_fail_threshold_data,
        std::vector<ElementData>& total_original_data,
        uint32_t BlockMSize, uint32_t BlockNSize,
        uint32_t FI_col_pos,
        bool ifShowError)
    */
    
    errorIndices = golden::GetErrorDataAndIndexSliceWithThresholdFI(
        options.problemShape,
        hostCOMPRow, hostGoldenCOMPRow, hostZRow, 
        hostDRow, 
        hostThre, hostC, 
        lenZRow, "CE", "ABE", 
        totalErrorIdxRow, totalDetectedErrorIdxinFI,
        totalDetectedErrorBlkIdxinFI,
        totalErrorIdxRow_m, totalErrorIdxRow_n,
        totalErrorDataRow, totalFailThresholds, totalOriginalDataElem,
        L1TileShape::M, L1TileShape::N, options.inject_col_seed, true
    );
    
    // if (errorIndices.empty()) {
    //     std::cout << "Row COMP OP compare success." << std::endl;
    // } else {
    //     std::cerr << "Row COMP OP compare failed. Error count: " << errorIndices.size() << std::endl;
    // }

    

    /*
    DetectionMetrics CalculateDetectionMetrics(
    const std::vector<int32_t>& detected_error_idies_for_FI, 
    const std::vector<int32_t>& ground_error_idies_for_FI,
    const Catlass::GemvCoord &problemShape, 
    std::vector<int32_t>& true_detected_error_idies_set,
    std::vector<int32_t>& false_detected_error_idies_set,
    std::vector<int32_t>& false_detected_error_blks_set,
    uint32_t BlockMSize, uint32_t BlockNSize)
    */



    DetectionMetrics metrics = golden::CalculateDetectionMetrics(
        totalDetectedErrorIdxinFI, hostFIIndex, options.problemShape,
        true_detected_error_idies_set, false_detected_error_idies_set, false_detected_error_blks_set,
        L1TileShape::M, L1TileShape::N);

    uint64_t total_detected_error_counts = static_cast<uint64_t>(totalErrorIdxRow.size());
    
    printf("Detection Results Iter 0:\n");
    printf("Total Detected Error count: %d\n", static_cast<int>(totalErrorIdxRow.size()));
    printf("True Detected Error Count: %d\n", metrics.detected_errs());
    printf("Total Inject Error Count: %d\n", metrics.inject_errs());
    printf("Detection Rate: %.2f%%\n", metrics.dr() * 100);

    printf("False Detected Error Count: %d\n", metrics.fa_rows());
    printf("False Detected Error Block Num: %d\n", metrics.fa_blocks());
    printf("Total Involved Rows in Detection: %d\n", metrics.total_rows_count());
    printf("Total Involved Blocks in Detection: %d\n", metrics.total_blocks_count());
    printf("Row-level False Alarm Rate: %.2f%%\n", metrics.rfp() * 100);
    printf("Block-level False Alarm Rate: %.2f%%\n", metrics.bfp() * 100);

    uint32_t inject_row_seed_used = options.inject_row_seed;
    uint32_t inject_col_seed_used = options.inject_col_seed;

    

    for(int i = 1; i < 10000; i++){
        if (i % L1TileShape::N == 0){
            golden::FillRandomData(hostA, -1.0f, 1.0f);
            golden::FillRandomData(hostB, -1.0f, 1.0f);
    
            ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
            ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

            inject_row_seed_used = (inject_row_seed_used + 1) % L1TileShape::M;
        }

        ACL_CHECK(aclrtMemcpy(deviceC, sizeC, hostCRaw.data(), sizeC, ACL_MEMCPY_HOST_TO_DEVICE));

        ACL_CHECK(aclrtMemcpy(deviceBE, sizeBE, hostBERaw.data(), sizeBE, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(deviceBEforAIV, sizeBEforAIV, hostBEforAIVRaw.data(), sizeBEforAIV, ACL_MEMCPY_HOST_TO_DEVICE));

        ACL_CHECK(aclrtMemcpy(deviceBMaxSlice, sizeBEforAIV, hostBMaxSliceRaw.data(), sizeBEforAIV, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(deviceBMinSlice, sizeBEforAIV, hostBMinSliceRaw.data(), sizeBEforAIV, ACL_MEMCPY_HOST_TO_DEVICE));

        
        ACL_CHECK(aclrtMemcpy(deviceZRow, sizeZRow, hostZRowRaw.data(), sizeZRow, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(deviceDRow, sizeZRow, hostDRowRaw.data(), sizeZRow, ACL_MEMCPY_HOST_TO_DEVICE));

        ACL_CHECK(aclrtMemcpy(deviceCOMPRow, sizeCOMPRow, hostCOMPRowRaw.data(), sizeCOMPRow, ACL_MEMCPY_HOST_TO_DEVICE));

        ACL_CHECK(aclrtMemcpy(deviceThre, sizeThre, hostThreRaw.data(), sizeThre, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(deviceBMeanAbs, sizeBMeanAbs, hostBMeanAbsRaw.data(), sizeBMeanAbs, ACL_MEMCPY_HOST_TO_DEVICE));

        ACL_CHECK(aclrtMemcpy(deviceBMeanSquare, sizeBMeanSquare, hostBMeanSquareRaw.data(), sizeBMeanSquare, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(deviceBVar, sizeBVar, hostBVarRaw.data(), sizeBVar, ACL_MEMCPY_HOST_TO_DEVICE));

        ACL_CHECK(aclrtMemcpy(deviceAMean, sizeAMean, hostAMeanRaw.data(), sizeAMean, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(deviceAMax, sizeAMax, hostAMaxRaw.data(), sizeAMax, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(deviceAMin, sizeAMin, hostAMinRaw.data(), sizeAMin, ACL_MEMCPY_HOST_TO_DEVICE));

        // FI (Fault Injection) success flag device memory
    
        ACL_CHECK(aclrtMemcpy(deviceSuccess, sizeSuccess, hostSuccess.data(), sizeSuccess, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(deviceFIIndex, sizeFIGroundforID, hostFIIndex.data(), sizeFIGroundforID, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(deviceFIFlipData, sizeFIGroundforData, hostFIFlipData.data(), sizeFIGroundforData, ACL_MEMCPY_HOST_TO_DEVICE));

        ACL_CHECK(aclrtSynchronizeStream(stream));

        inject_col_seed_used = (inject_col_seed_used + 1) % L1TileShape::N;

        typename MatmulFTKernel::Arguments arguments_iter{
            options.problemGemmShape, options.problemShape, sizeof(GemvInTypeCforAB), 
            deviceX, deviceXV, deviceA, deviceB, deviceC, 
            deviceZRow, deviceZCol, deviceDRow, deviceDCol, 
            deviceCOMPRow, deviceCOMPCol, 
            deviceBE, deviceBEforAIV, deviceBMaxSlice, deviceBMinSlice,
            deviceBMeanAbs, deviceBMeanSquare, deviceBVar, 
            deviceVXforAe, deviceAMean, deviceAMax, deviceAMin,
            deviceThre, FT_ENC_TYPE::RCE, 1, false, threshold,
            options.round_exp, options.beta, 
            use_emax, options.reduce_cores, 
            rce_thre_type,true,true, options.split_ks,
            options.inject_k_unit, options.inject_num_blk_limit, inject_col_seed_used,
            inject_col_seed_used, options.inject_bit, 
            deviceSuccess, deviceFIIndex, deviceFIFlipData};

        RunAdapter(matmul_op, arguments_iter, stream, aicCoreNum, fftsAddr);

        ACL_CHECK(aclrtSynchronizeStream(stream));
        
        ACL_CHECK(aclrtMemcpy(hostZRow.data(), sizeZRow, deviceZRow, sizeZRow, ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_CHECK(aclrtMemcpy(hostDRow.data(), sizeZRow, deviceDRow, sizeZRow, ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_CHECK(aclrtMemcpy(hostCOMPRow.data(), sizeCOMPRow, deviceCOMPRow, sizeCOMPRow, ACL_MEMCPY_DEVICE_TO_HOST));
        if(i % L1TileShape::N == 0){
            ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));
            ACL_CHECK(aclrtMemcpy(hostThre.data(), sizeThre, deviceThre, sizeThre, ACL_MEMCPY_DEVICE_TO_HOST));

            ACL_CHECK(aclrtMemcpy(hostBMeanAbs.data(), sizeBMeanAbs, deviceBMeanAbs, sizeBMeanAbs, ACL_MEMCPY_DEVICE_TO_HOST));
            ACL_CHECK(aclrtMemcpy(hostBMeanSquare.data(), sizeBMeanSquare, deviceBMeanSquare, sizeBMeanSquare, ACL_MEMCPY_DEVICE_TO_HOST));
            ACL_CHECK(aclrtMemcpy(hostBVar.data(), sizeBVar, deviceBVar, sizeBVar, ACL_MEMCPY_DEVICE_TO_HOST));

            ACL_CHECK(aclrtMemcpy(hostBMaxSlice.data(), sizeBEforAIV, deviceBMaxSlice, sizeBEforAIV, ACL_MEMCPY_DEVICE_TO_HOST));
            ACL_CHECK(aclrtMemcpy(hostBMinSlice.data(), sizeBEforAIV, deviceBMinSlice, sizeBEforAIV, ACL_MEMCPY_DEVICE_TO_HOST));

            ACL_CHECK(aclrtMemcpy(hostAMax.data(), sizeAMax, deviceAMax, sizeAMax, ACL_MEMCPY_DEVICE_TO_HOST));
            ACL_CHECK(aclrtMemcpy(hostAMean.data(), sizeAMean, deviceAMean, sizeAMean, ACL_MEMCPY_DEVICE_TO_HOST));
            ACL_CHECK(aclrtMemcpy(hostAMin.data(), sizeAMin, deviceAMin, sizeAMin, ACL_MEMCPY_DEVICE_TO_HOST));
        }
        
        ACL_CHECK(aclrtMemcpy(hostBE.data(), sizeBE, deviceBE, sizeBE, ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_CHECK(aclrtMemcpy(hostBEforAIV.data(), sizeBEforAIV, deviceBEforAIV, sizeBEforAIV, ACL_MEMCPY_DEVICE_TO_HOST));
    
        // Read back FI success flag
        ACL_CHECK(aclrtMemcpy(hostSuccess.data(), sizeSuccess, deviceSuccess, sizeSuccess, ACL_MEMCPY_DEVICE_TO_HOST));

        ACL_CHECK(aclrtMemcpy(hostFIIndex.data(), sizeFIGroundforID, deviceFIIndex, sizeFIGroundforID, ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_CHECK(aclrtMemcpy(hostFIFlipData.data(), sizeFIGroundforData, deviceFIFlipData, sizeFIGroundforData, ACL_MEMCPY_DEVICE_TO_HOST));

        ACL_CHECK(aclrtSynchronizeStream(stream));

        totalErrorIdxRow.clear();
        totalErrorIdxRow_m.clear();
        totalErrorIdxRow_n.clear();
        totalDetectedErrorIdxinFI.clear();
        totalDetectedErrorBlkIdxinFI.clear();
        totalErrorDataRow.clear();
        totalOriginalDataElem.clear();
        totalFailThresholds.clear();

        true_detected_error_idies_set.clear();
        false_detected_error_idies_set.clear();
        false_detected_error_blks_set.clear();

        if(i % 2000 == 0){
            errorIndices = golden::GetErrorDataAndIndexSliceWithThresholdFI(
                options.problemShape,
                hostCOMPRow, hostGoldenCOMPRow, hostZRow, 
                hostDRow, 
                hostThre, hostC, 
                lenZRow, "CE", "ABE", 
                totalErrorIdxRow, totalDetectedErrorIdxinFI,
                totalDetectedErrorBlkIdxinFI,
                totalErrorIdxRow_m, totalErrorIdxRow_n,
                totalErrorDataRow, totalFailThresholds, totalOriginalDataElem,
                L1TileShape::M, L1TileShape::N, options.inject_col_seed, true);
        }else{
            errorIndices = golden::GetErrorDataAndIndexSliceWithThresholdFI(
                options.problemShape,
                hostCOMPRow, hostGoldenCOMPRow, hostZRow, 
                hostDRow, 
                hostThre, hostC, 
                lenZRow, "CE", "ABE", 
                totalErrorIdxRow, totalDetectedErrorIdxinFI,
                totalDetectedErrorBlkIdxinFI,
                totalErrorIdxRow_m, totalErrorIdxRow_n,
                totalErrorDataRow, totalFailThresholds, totalOriginalDataElem,
                L1TileShape::M, L1TileShape::N, options.inject_col_seed, false);
        }

        DetectionMetrics metrics_iter = golden::CalculateDetectionMetrics(
        totalDetectedErrorIdxinFI, hostFIIndex, options.problemShape,
        true_detected_error_idies_set, false_detected_error_idies_set, false_detected_error_blks_set,
        L1TileShape::M, L1TileShape::N);

        uint64_t iter_detected_error_counts = static_cast<uint64_t>(totalErrorIdxRow.size());
        total_detected_error_counts += iter_detected_error_counts;

        if(i % 1000 == 0)
        {
            printf("Detection Results Iter %d:\n", i);
            printf("Total Detected Error count: %d\n", static_cast<int>(totalErrorIdxRow.size()));
            printf("True Detected Error Count: %d\n", metrics_iter.detected_errs());
            printf("Total Inject Error Count: %d\n", metrics_iter.inject_errs());
            printf("Detection Rate: %.2f%%\n", metrics_iter.dr() * 100);
            printf("False Detected Error Count: %d\n", metrics_iter.fa_rows());
            printf("False Detected Error Block Num: %d\n", metrics_iter.fa_blocks());
            printf("Total Involved Rows in Detection: %d\n", metrics_iter.total_rows_count());
            printf("Total Involved Blocks in Detection: %d\n", metrics_iter.total_blocks_count());
            printf("Row-level False Alarm Rate: %.2f%%\n", metrics_iter.rfp() * 100);
            printf("Block-level False Alarm Rate: %.2f%%\n", metrics_iter.bfp() * 100);
        }

        metrics = golden::UpdateDetectionMetrics(metrics, metrics_iter);
    }

    printf("Global Detection Results:\n");
    printf("Total Detected Error count: %d\n", static_cast<int>(total_detected_error_counts));
    printf("True Detected Error Count: %d\n", metrics.detected_errs());
    printf("Total Inject Error Count: %d\n", metrics.inject_errs());
    printf("Detection Rate: %.2f%%\n", metrics.dr() * 100);

    printf("False Detected Error Count: %d\n", metrics.fa_rows());
    printf("False Detected Error Block Num: %d\n", metrics.fa_blocks());
    printf("Total Involved Rows in Detection: %d\n", metrics.total_rows_count());
    printf("Total Involved Blocks in Detection: %d\n", metrics.total_blocks_count());
    printf("Row-level False Alarm Rate: %.2f%%\n", metrics.rfp() * 100);
    printf("Block-level False Alarm Rate: %.2f%%\n", metrics.bfp() * 100);


    // printf("Total Error Data len: %d\n", static_cast<int>(totalErrorDataRow.size()));


    // for (int i = 0; i < 100; ++i) {
    //     ACL_CHECK(aclrtSynchronizeStream(stream));
    //     // RunAdapter(gemv_op, arguments, stream, aicCoreNum);
    //     // RunAdapter(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
    //     matmul_op(stream, aicCoreNum, fftsAddr);
    //     ACL_CHECK(aclrtSynchronizeStream(stream));
    // }

    // int num_repeat = 10000;

    // aclrtEvent start, stop;
    // float temp_time = 0;
    // float time = 0;
    // ACL_CHECK(aclrtCreateEvent(&start));
    // ACL_CHECK(aclrtCreateEvent(&stop));

    // for (int i = 0; i < num_repeat; ++i) {
    //     ACL_CHECK(aclrtSynchronizeStream(stream));
    //     ACL_CHECK(aclrtRecordEvent(start, stream));

    //     // RunAdapter(gemv_op, arguments, stream, aicCoreNum);

    //     // RunAdapter(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
    //     matmul_op(stream, aicCoreNum, fftsAddr);
    //     // ACL_CHECK(aclrtSynchronizeStream(stream));
            
    //     ACL_CHECK(aclrtSynchronizeStream(stream));
    //     ACL_CHECK(aclrtRecordEvent(stop, stream));
    //     ACL_CHECK(aclrtSynchronizeEvent(stop));
    //     ACL_CHECK(aclrtEventElapsedTime(&temp_time, start, stop));
    //     time += temp_time;
    // }
    // // 4 * m * n + 4 * m*k + 4 * k*n
    // // m*
    // float total_op_nums = 1.0 * m*n*k*2;
    // // / (1.0*1e12)
    // std::cout << "m: " << m << ", n: " << n << ", k: "<< k <<","<< (float)(total_op_nums / (time / num_repeat * 1e-3))/ (1.0*1e12)  << " TFLOPS, " << (time / num_repeat) << " ms, repeat: " << num_repeat << std::endl;

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));

    ACL_CHECK(aclrtFree(deviceX));
    ACL_CHECK(aclrtFree(deviceXV));
    ACL_CHECK(aclrtFree(deviceVXforAe));

    ACL_CHECK(aclrtFree(deviceZRow));
    ACL_CHECK(aclrtFree(deviceZCol));

    ACL_CHECK(aclrtFree(deviceDRow));
    ACL_CHECK(aclrtFree(deviceDCol));

    ACL_CHECK(aclrtFree(deviceCOMPRow));
    ACL_CHECK(aclrtFree(deviceCOMPCol));

    ACL_CHECK(aclrtFree(deviceThre));
    ACL_CHECK(aclrtFree(deviceBE));
    ACL_CHECK(aclrtFree(deviceBEforAIV));

    ACL_CHECK(aclrtFree(deviceBMaxSlice));
    ACL_CHECK(aclrtFree(deviceBMinSlice));

    ACL_CHECK(aclrtFree(deviceBMeanAbs));
    ACL_CHECK(aclrtFree(deviceBMeanSquare));
    ACL_CHECK(aclrtFree(deviceBVar));

    ACL_CHECK(aclrtFree(deviceAMax));
    ACL_CHECK(aclrtFree(deviceAMean));
    ACL_CHECK(aclrtFree(deviceAMin));

    ACL_CHECK(aclrtFree(deviceSuccess));
    ACL_CHECK(aclrtFree(deviceFIIndex));
    ACL_CHECK(aclrtFree(deviceFIFlipData));

    // if (sizeWorkspace > 0) {
    //     ACL_CHECK(aclrtFree(deviceWorkspace));
    // }

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