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

// #include "catlass/gemv/kernel/kernel_gemv_aic_FT.hpp"
// #include "catlass/gemv/kernel/kernel_gemv_FT_double.hpp"
// #include "catlass/gemv/kernel/kernel_gemv_aic_FT.hpp"
#include "catlass/gemv/kernel/kernel_gemv_FT_double_thre.hpp"
#include "catlass/gemv/tile/tile_copy.hpp"
#include "catlass/gemv/helper.hpp"

#include "catlass/gemv/tile/tile_threshold_compute.hpp"

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

using namespace Catlass;

using ScalarType = float;

using GemmInTypeC = float;
// op_bfloat16;
// op_bfloat16;
using GemmInTypeN = float;
// bfloat16_t;
// using GemmOutTypeC = float;
// using GemmOutTypeN = float;
using GemmOutTypeC = float;
using GemmOutTypeN = float;
    
using ScalarTypeC = float;
using ScalarTypeN = float;


struct Options {
    const std::string HELPER = "18_gemv_aic m n k gol_path in_path out_path lim exp1 exp2 exp3 rt beta thre_type [device_id]";

    GemmCoord problemGemmShape{128, 128, 128};
    GemvCoord problemShape{128, 128};

    int32_t deviceId{1};

    // int32_t deviceId{0};
    int32_t make_golden{0};
    int32_t limit{1};

    int32_t gol_exp{1};
    int32_t in_exp{1};
    int32_t out_exp{1};

    std::string in_path = "./";
    std::string gol_path = "./";
    std::string out_path = "./";

    // GemmOutTypeN threshold{0.02f};
    // int32_t thre_expon{0};

    float round_exp{0.0f};
    float beta{1.0f};

    int thre_type{0};

    Options() = default;

    int Parse(int argc, const char** argv) {
        // std::stof(argv[1]);
        enum ArgsIndex {
            M_INDEX = 1,
            N_INDEX,
            K_INDEX,
            GOL_PATH_INDEX,
            IN_PATH_INDEX,
            OUT_PATH_INDEX,
            LIM_INDEX,
            GOL_EXP_INDEX,
            IN_EXP_INDEX,
            OUT_EXP_INDEX,
            RT_INDEX,
            BETA_INDEX,
            THRE_TYPE_INDEX,
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

        limit = std::atoi(argv[LIM_INDEX]);

        gol_path = argv[GOL_PATH_INDEX];
        in_path = argv[IN_PATH_INDEX];
        out_path = argv[OUT_PATH_INDEX];

        in_exp = std::atoi(argv[IN_EXP_INDEX]);
        out_exp = std::atoi(argv[OUT_EXP_INDEX]);
        gol_exp = std::atoi(argv[GOL_EXP_INDEX]);

        round_exp = static_cast<float>(std::stof(argv[RT_INDEX]));
        beta = static_cast<float>(std::stof(argv[BETA_INDEX]));

        thre_type = std::atoi(argv[THRE_TYPE_INDEX]);

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
    std::cout<<"Device ID: "<<options.deviceId<<std::endl;
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemGemmShape.k();

    int32_t gol_exp = options.gol_exp;
    int32_t in_exp = options.in_exp;
    int32_t out_exp = options.out_exp;

    int32_t limit = options.limit;

    std::string gol_path = options.gol_path;
    std::string in_path = options.in_path;
    std::string out_path = options.out_path;

    GemmOutTypeN threshold = 0.000f;


    GemvCoord problemShapeCol{n, m};

    GemvCoord problemShapeBE{k,n};
    GemvCoord problemShapeABE{m,k};

    GemvCoord problemShapeETA{k,m};
    GemvCoord problemShapeETAB{n,k};


    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenC = static_cast<size_t>(n) * m;

    size_t lenX = (static_cast<size_t>(m) + static_cast<size_t>(n)) * 1;
    size_t lenXV = (static_cast<size_t>(m) + static_cast<size_t>(n)) * 1;

    size_t lenDRow = (static_cast<size_t>(m)) * 1;
    size_t lenDCol = (static_cast<size_t>(n)) * 1;

    size_t lenCOMPCol = ((static_cast<size_t>(n)) + 8 - 1) / 8;
    size_t lenCOMPRow = ((static_cast<size_t>(m)) + 8 - 1) / 8;

    size_t lenZRow = static_cast<size_t>(m) * 1;
    size_t lenZCol = static_cast<size_t>(n) * 1;

    size_t lenThre = static_cast<size_t>(m) * 1;

    size_t sizeA = lenA * sizeof(GemmInTypeC);
    size_t sizeB = lenB * sizeof(GemmInTypeC);
    size_t sizeC = lenC * sizeof(GemmOutTypeC);

    size_t sizeX = lenX * sizeof(GemmInTypeC);
    size_t sizeXV = lenXV * sizeof(GemmInTypeC);

    size_t sizeZRow = lenZRow * sizeof(GemmOutTypeC);
    size_t sizeDRow = lenDRow * sizeof(GemmOutTypeC);

    size_t sizeCOMPRow = lenCOMPRow * sizeof(uint8_t);
    size_t sizeCOMPCol = lenCOMPCol * sizeof(uint8_t);

    size_t sizeZCol = lenZCol * sizeof(GemmOutTypeC);
    size_t sizeDCol = lenDCol * sizeof(GemmOutTypeC);

    size_t sizeThre = lenThre * sizeof(GemmOutTypeC);

    size_t sizeWorkspace;

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
    using FT_RCE_THRE_TYPE = Gemv::helper::FT_RCE_THRE_TYPE;

    LayoutX layoutXRow{n};
    LayoutX layoutXCol{m};

    LayoutC layoutC{m, n};
    LayoutCCol layoutCCol{n,m};

    LayoutA layoutA{m, k};
    LayoutACol layoutACol{k,m};

    LayoutB layoutB{k, n};
    LayoutBCol layoutBCol{n, k};

    LayoutZ layoutZ{m};
    LayoutZ layoutZCol{n};

    LayoutZ layoutThre{m};

    ScalarType alpha{1.0};
    ScalarType beta{0.0};
    // FillRandomScalarData(alpha, -1.0f, 1.0f);
    // FillRandomScalarData(beta, -1.0f, 1.0f);

    GemmInTypeC sum_base = 1.0;

    std::vector<uint8_t> hostCOMPRow(lenCOMPRow,0);
    std::vector<uint8_t> hostCOMPCol(lenCOMPCol,0);
    
    std::vector<GemmOutTypeC> hostC(lenC);
    std::vector<GemmInTypeC> hostA(lenA);
    std::vector<GemmInTypeC> hostB(lenB);

    std::vector<GemmOutTypeC> hostGolden(lenC);

    std::vector<GemmInTypeC> hostX(lenX,sum_base);
    std::vector<GemmInTypeC> hostXV(lenXV,sum_base);

    std::vector<GemmOutTypeC> hostDRow(lenZRow,0.0f);
    std::vector<GemmOutTypeC> hostDCol(lenZCol,0.0f);

    std::vector<GemmOutTypeC> hostZRow(lenZRow,0.0f);
    std::vector<GemmOutTypeC> hostZCol(lenZCol,0.0f);

    std::vector<GemmOutTypeC> hostThre(lenThre,0.0f);

    golden::FillRandomData(hostC, -1.0f, 1.0f);

    FT_RCE_THRE_TYPE rce_thre_type = FT_RCE_THRE_TYPE::ROUND;
    std::string out_threshold_type = "round";

    if(options.thre_type >= 1){
        rce_thre_type = FT_RCE_THRE_TYPE::ROUND_WITH_ACC;
        out_threshold_type = "round_accum";
    }
    // golden::FillRandomData(hostA, -1.0f, 1.0f);
    // golden::FillRandomData(hostB, -1.0f, 1.0f);

    char c_golden_name_buf[100];
    snprintf(c_golden_name_buf, sizeof(c_golden_name_buf), 
        "exp_%d_%u_%u_%u_lim_%d/C.bin", gol_exp, m, n, k, limit);
    std::string C_golden_file_name = c_golden_name_buf;
    std::string C_golden_file_path = gol_path+ "/" + C_golden_file_name;

    char a_input_name_buf[100];
    snprintf(a_input_name_buf, sizeof(a_input_name_buf), 
        "exp_%d_%u_%u_%u_lim_%d/A.bin", gol_exp, m, n, k, limit);
    std::string A_input_file_name = a_input_name_buf;
    std::string A_input_file_path = gol_path + "/" + A_input_file_name;

    char b_input_name_buf[100];
    snprintf(b_input_name_buf, sizeof(b_input_name_buf), 
        "exp_%d_%u_%u_%u_lim_%d/B.bin", gol_exp, m, n, k, limit);
    std::string B_input_file_name = b_input_name_buf;
    std::string B_input_file_path = gol_path + "/" + B_input_file_name;

    char c_input_name_buf[100];
    snprintf(c_input_name_buf, sizeof(c_input_name_buf), "exp_%d_in_%d_%u_%u_%u",
        in_exp, gol_exp, m, n, k);
    
    char c_output_name_buf[100];
    snprintf(c_output_name_buf, sizeof(c_output_name_buf), "exp_%d_in_%d_%u_%u_%u_thre_est_%s",
        in_exp, gol_exp, m, n, k, out_threshold_type.c_str());

    std::string C_input_fpath = c_input_name_buf;
    std::string C_input_base_path = in_path + "/" + C_input_fpath;

    std::string C_output_fpath = c_output_name_buf;
    std::string C_output_base_path = out_path + "/" + C_output_fpath;

    golden::loadVectorFromBinaryFile(C_golden_file_path, hostC);


    golden::loadVectorFromBinaryFile(A_input_file_path, hostA);

    golden::loadVectorFromBinaryFile(B_input_file_path, hostB);

    golden::loadVectorFromBinaryFile(C_golden_file_path, hostGolden);

    


    uint8_t* deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));
    // ACL_CHECK(aclrtMemcpy(deviceC, sizeC, hostC.data(), sizeC, ACL_MEMCPY_HOST_TO_DEVICE));

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


    uint8_t* deviceZRow{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceZRow), sizeZRow, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceZRow, sizeZRow, hostZRow.data(), sizeZRow, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceZCol{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceZCol), sizeZCol, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceZCol, sizeZCol, hostZCol.data(), sizeZCol, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceDRow{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceDRow), sizeZRow, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceDRow, sizeZRow, hostDRow.data(), sizeZRow, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceDCol{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceDCol), sizeZCol, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceDCol, sizeZCol, hostZCol.data(), sizeZCol, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceCOMPRow{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceCOMPRow), sizeCOMPRow, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceCOMPRow, sizeCOMPRow, hostCOMPRow.data(), sizeCOMPRow, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceCOMPCol{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceCOMPCol), sizeCOMPCol, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceCOMPCol, sizeCOMPCol, hostCOMPCol.data(), sizeCOMPCol, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceThre{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceThre), sizeThre, ACL_MEM_MALLOC_HUGE_FIRST));
    // ACL_CHECK(aclrtMemcpy(deviceThre, sizeThre, hostThre.data(), sizeThre, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceWorkspace{nullptr};

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    using ArchTag = Arch::AtlasA2;
    using LayoutMY = layout::RowMajor;

    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;

    using DispatchPolicy = Gemm::MmadAtlasA2Preload<enableUnitFlag, enableShuffleK>;
    using FT_ENC_TYPE = Gemv::helper::FT_ENC_TYPE;
    

    using L1TileShapeC = GemvShape<32, 512>;
    using L0TileShapeC = GemvShape<32, 256>;

    using L1TileShapeAB = GemmShape<128, 128, 128>;
    using L0TileShapeAB = GemmShape<128, 128, 128>;

    using AType = Gemm::GemmType<GemmInTypeN, LayoutA>;
    using BType = Gemm::GemmType<GemmInTypeN, LayoutB>;
    using CType = Gemm::GemmType<GemmOutTypeN, LayoutC>;
    using XType = Gemm::GemmType<GemmOutTypeN, LayoutX>;
    using MYType = Gemm::GemmType<GemmOutTypeN, LayoutMY>;
    using BiasType = void;
    using TileCopy = Gemv::Tile::TileCopyGemvAic<typename DispatchPolicy::ArchTag, CType, XType, MYType, BiasType>;
    using TileMmad = Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, XType, CType, BiasType>;

    using BlockAICGemv = Gemv::Block::BlockFTGemvDouble<DispatchPolicy, FT_ENC_TYPE::BOTHC, L1TileShapeC, L0TileShapeC, CType, XType, MYType, BiasType, TileCopy, TileMmad>;

    // Block level, define BlockEpilogue
    using EpilogueBlockDispatchPolicy = Epilogue::EpilogueAtlasA2Gemv;
    using EYType = Gemm::GemmType<GemmOutTypeN, LayoutZ>;
    using ZType = Gemm::GemmType<GemmOutTypeN, LayoutZ>;
    using AXType = Gemm::GemmType<GemmOutTypeN, LayoutZ>;

    using ComputeType = AXType;
    constexpr uint32_t computeLength = 8192;

    using TileElemWiseAddGemv = Epilogue::Tile::TileElemWiseAdd<ArchTag, ComputeType, computeLength>;
    using TileElemWiseMulsGemv = Epilogue::Tile::TileElemWiseMuls<ArchTag, ComputeType, computeLength>;

    using EpilogueTileCopy = Epilogue::Tile::TileCopy<ArchTag, EYType, AXType, ZType>;

    using BlockEpilogue = Epilogue::Block::BlockEpilogue<EpilogueBlockDispatchPolicy, AXType, EYType, ZType, TileElemWiseAddGemv, TileElemWiseMulsGemv, EpilogueTileCopy>;

    using GemvDispatchPolicy = Gemm::GemvAtlasA2;
    using COMPDispatchPolicy = Gemm::GemvAtlasA2;

    using YType = Gemm::GemmType<GemmOutTypeN, LayoutY>;
    using TileVmuls = Gemv::Tile::TileVmuls<ArchTag, XType>;

    using UBTileShape = GemmShape<L1TileShapeAB::M, L1TileShapeAB::N, L1TileShapeAB::K>;
    using TileFaultCopy = Gemv::Tile::TileCopyGemvAiv<ArchTag, AType, XType, YType>;
    using TileFaultVmad = Gemv::Tile::TileVmad<ArchTag, AType, XType, YType>;
    using TileFaultSum = Gemv::Tile::TileFaultSum<ArchTag, AType, YType>;
    using BlockSumGemv = Gemv::Block::BlockSumGemv<GemvDispatchPolicy, UBTileShape, AType, XType, YType, void, TileFaultCopy, TileFaultVmad, TileVmuls>;

    using COMPZType = Gemm::GemmType<uint8_t, LayoutZ>;

    using UBTileShapeCOMP = GemvShape<1,1024>;
    using TileCompareCopy = Gemv::Tile::TileCopyCompareAiv<FT_COMP_TYPE::RSUB, typename COMPDispatchPolicy::ArchTag, COMPZType, ZType, ZType, void>;
    using CompareBlock = Gemv::Block::BlockCompare<COMPDispatchPolicy,FT_COMP_TYPE::RSUB, UBTileShapeCOMP, COMPZType, ZType, ZType, TileCompareCopy>;

    using TileCompareCopySUB = Gemv::Tile::TileCopyCompareAiv<FT_COMP_TYPE::SUB, typename COMPDispatchPolicy::ArchTag, COMPZType, ZType, ZType, void>;
    using CompareBlockSUB = Gemv::Block::BlockCompare<COMPDispatchPolicy,FT_COMP_TYPE::SUB, UBTileShapeCOMP, COMPZType, ZType, ZType, TileCompareCopySUB>;

    using ThreCalcDispatchPolicy = Gemm::GemvAtlasA2;
    using ThreCalcUBTileShape = GemvShape<32,512>;

    using ThreCalcTileCopy = Gemv::Tile::TileCopyMatrixReduceAiv<typename ThreCalcDispatchPolicy::ArchTag, CType, XType, ZType, BiasType>;
    // using TileVmad = Gemv::Tile::TileVmad<typename DispatchPolicy::ArchTag, AType, XType, YType, BiasType>;
    using ThreCalcTileVmuls = Gemv::Tile::TileVmuls<typename ThreCalcDispatchPolicy::ArchTag, ZType>;

    using TileThreCalc = Gemv::Tile::TileThreCalc<typename ThreCalcDispatchPolicy::ArchTag, CType, XType, ZType, BiasType>; 

    using BlockThresholdCalc = Gemv::Block::BlockThresholdCalc<ThreCalcDispatchPolicy, Gemv::helper::FT_ENC_TYPE::RCE, 
        ThreCalcUBTileShape, CType, XType, ZType, BiasType, 
        ThreCalcTileCopy, TileThreCalc, ThreCalcTileVmuls>;

    // kernle levels
    // using GemvKernel = Gemv::Kernel::KernelGemvFTDouble<BlockAICGemv,BlockSumGemv,CompareBlock,BlockEpilogue>;

    using GemvKernel = Gemv::Kernel::KernelGemvFTDoubleThre<
        BlockAICGemv,BlockSumGemv,
        CompareBlock,CompareBlockSUB,
        BlockThresholdCalc,void>;

    // TODO:  use adapter to activate the kernel
    using GemvAdapter = Gemv::Device::DeviceGemv<GemvKernel>;

    

    // GemvKernel::Arguments arguments{options.problemGemmShape, 
    // options.problemShape, sizeof(GemmOutTypeC), deviceX, deviceXV, 
    //     deviceA, deviceB, deviceC, deviceZRow, deviceZCol, deviceDRow, deviceDCol, deviceCOMPRow, deviceCOMPCol,
    //      FT_ENC_TYPE::BOTHC,1,false,threshold};

    GemvKernel::Arguments arguments{options.problemGemmShape, options.problemShape, 
        sizeof(GemmOutTypeC), deviceX, deviceXV, 
        deviceA, deviceB, deviceC, deviceZRow, deviceZCol, deviceDRow, deviceDCol,
        deviceCOMPRow, deviceCOMPCol, deviceThre, 
        FT_ENC_TYPE::RCE, 1, false, threshold, 
        options.round_exp, options.beta, rce_thre_type};
    GemvAdapter gemv_op;

    // RunAdapter(gemv_op, arguments, stream, aicCoreNum, fftsAddr);

    for (int i = 0; i < 100; ++i) {
        ACL_CHECK(aclrtSynchronizeStream(stream));
        RunAdapter(gemv_op, arguments, stream, aicCoreNum, fftsAddr);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    int32_t num_repeat = 100;
    for(int jj = 0; jj < num_repeat; ++jj){
        char c_iter_name_buf[100];
        snprintf(c_iter_name_buf, sizeof(c_iter_name_buf), "iter_%d/C.bin", jj);
        
        std::string C_iter_fpath = c_iter_name_buf;
        std::string C_iter_file_path = C_input_base_path + "/" + C_iter_fpath;

        golden::loadVectorFromBinaryFile(C_iter_file_path, hostC);
        ACL_CHECK(aclrtMemcpy(deviceC, sizeC, hostC.data(), sizeC, ACL_MEMCPY_HOST_TO_DEVICE));

        ACL_CHECK(aclrtSynchronizeStream(stream));

        RunAdapter(gemv_op, arguments, stream, aicCoreNum, fftsAddr);

        ACL_CHECK(aclrtSynchronizeStream(stream));
        
        ACL_CHECK(aclrtMemcpy(hostZRow.data(), sizeZRow, deviceZRow, sizeZRow, ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_CHECK(aclrtMemcpy(hostZCol.data(), sizeZCol, deviceZCol, sizeZCol, ACL_MEMCPY_DEVICE_TO_HOST));
        
        ACL_CHECK(aclrtMemcpy(hostDRow.data(), sizeZRow, deviceDRow, sizeZRow, ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_CHECK(aclrtMemcpy(hostDCol.data(), sizeZCol, deviceDCol, sizeZCol, ACL_MEMCPY_DEVICE_TO_HOST));
        
        ACL_CHECK(aclrtMemcpy(hostCOMPRow.data(), sizeCOMPRow, deviceCOMPRow, sizeCOMPRow, ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_CHECK(aclrtMemcpy(hostCOMPCol.data(), sizeCOMPCol, deviceCOMPCol, sizeCOMPCol, ACL_MEMCPY_DEVICE_TO_HOST));

        ACL_CHECK(aclrtMemcpy(hostThre.data(), sizeThre, deviceThre, sizeThre, ACL_MEMCPY_DEVICE_TO_HOST));
        
        if(jj < 5 || jj > (num_repeat - 5)){
            std::vector<float> hostGoldenRow(lenZRow);
            std::vector<float> hostGoldenCol(lenZCol);
            std::vector<float> hostYForAB(k, 0.0f);
            LayoutY layoutYForAB{k};
            // Host
            // golden::ComputeGemv(problemShapeCol, alpha, beta, hostC, layoutCCol, hostX, layoutXCol, hostX, layoutZCol, hostGoldenCol, layoutZCol);
            golden::ComputeGemv(options.problemShape, alpha, beta, hostC, layoutC, hostX, layoutXRow, hostX, layoutZ, hostGoldenRow, layoutZ);

            // std::vector<uint64_t> errorIndices = golden::CompareData(hostZCol, hostGoldenCol, n);

            // if (errorIndices.empty()) {
            //     std::cout << "Iter: "<<jj <<" ColSum Compare success." << std::endl;
            // } else {
            //     std::cerr  << "Iter: "<<jj <<" ColSum Compare failed. Error count: " << errorIndices.size() << std::endl;
            // }

            std::vector<uint64_t> errorIndices = golden::CompareData(hostZRow, hostGoldenRow, m);
            if (errorIndices.empty()) {
                std::cout << "Iter: "<<jj <<" RowSum Compare success." << std::endl;
            } else {
                std::cout << "Iter: "<<jj <<" RowSum Compare failed. Error count: " << errorIndices.size() << std::endl;
            }

            golden::ComputeGemv(problemShapeBE, alpha, beta, hostB, layoutB, hostX, layoutXRow, hostYForAB, layoutYForAB, hostYForAB, layoutYForAB);
            golden::ComputeGemv(problemShapeABE, alpha, beta, hostA, layoutA, hostYForAB, layoutYForAB, hostGoldenRow, layoutZ, hostGoldenRow, layoutZ);

    

            golden::FillRandomData<float>(hostYForAB, 0.0f, 0.0f);
            // golden::ComputeGemv(problemShapeETA, alpha, beta, hostA, layoutACol, hostX, layoutXCol, hostYForAB, layoutYForAB, hostYForAB, layoutYForAB);
            // golden::ComputeGemv(problemShapeETAB, alpha, beta, hostB, layoutBCol, hostYForAB, layoutYForAB, hostGoldenCol, layoutZCol, hostGoldenCol, layoutZCol);
            // errorIndices = golden::CompareData(hostDCol, hostGoldenCol, n);

            // if (errorIndices.empty()) {
            //     std::cout << "ColSum Compare success." << std::endl;
            // } else {
            //     std::cerr << "ColSum Compare failed." << std::endl;
            // }

            errorIndices = golden::CompareData(hostDRow, hostGoldenRow, n);

            if (errorIndices.empty()) {
                std::cout << "Rowsum Compare success." << std::endl;
            } else {
                std::cerr << "Rowsum Compare failed." << std::endl;
            }

            if(jj == 0){
                std::vector<float> hostMaxGolden(lenThre);
                golden::ComputeThresholds(options.problemShape, options.round_exp, options.beta, hostC, layoutC, hostMaxGolden, layoutThre);
                // golden::ComputeGemv(options.problemShape, alpha, beta, hostA, layoutA, hostX, layoutX, hostY, layoutY, hostGolden, layoutY);
                errorIndices = golden::CompareData(hostThre, hostMaxGolden, lenThre);
                if (errorIndices.empty()) {
                    std::cout << "Threshold Compare success." << std::endl;
                } else {
                    std::cerr << "Threshold Compare failed. Error count: " << errorIndices.size() << std::endl;
                }
            }
        }
    
        std::vector<uint8_t> hostGoldenCOMPRow(lenZRow,255);

        std::vector<uint64_t> totalErrorIdxRow;
        std::vector<GemmOutTypeC> totalErrorDataRow;
        std::vector<GemmOutTypeC> totalFailThresholds;

        totalErrorIdxRow.clear();
        totalErrorDataRow.clear();
        totalFailThresholds.clear();
        // golden::ComputeGemv(options.problemShape, alpha, beta, hostA, layoutA, hostX, layoutX, hostY, layoutY, hostGolden, layoutY);
        // std::vector<uint64_t> errorIndices = golden::GetErrorDataWithIndexTotal(
        //          hostCOMPRow, hostGoldenCOMPRow, hostZRow, 
        //         hostDRow, lenZRow, "CE", "ABE", totalErrorIdxRow, 
        //         totalErrorDataRow);

        std::vector<uint64_t> errorIndices = golden::GetErrorDataAndIndexTotalWithThreshold(
            hostCOMPRow, hostGoldenCOMPRow,hostZRow, hostDRow, hostThre, 
            lenZRow, "CE", "ABE", totalErrorIdxRow, totalErrorDataRow,
            totalFailThresholds);
        
        printf("Iter %d: Verify: Row Threshold Estimation Method: %s\n",jj, out_threshold_type.c_str());
        printf("%f\n", hostZRow[0]);
        printf("%f\n", hostDRow[0]);
        if (errorIndices.empty()) {
            std::cout << "Row COMP OP compare success." << std::endl;
        } else {
            std::cerr << "Row COMP OP compare failed. Error count: " << errorIndices.size() << std::endl;
        }

        printf("Total Error Idx len: %d\n", static_cast<int>(totalErrorIdxRow.size()));
        printf("Total Error Data len: %d\n", static_cast<int>(totalErrorDataRow.size()));

        int32_t show_len = (totalErrorIdxRow.size() > 20) ? 20 : static_cast<int32_t>(totalErrorIdxRow.size());

        for(int32_t kk=0; kk < show_len; kk++){
            printf("Data Id: %lu, Real Data: %f, Error Data: %f\n",totalErrorIdxRow[kk],hostDRow[totalErrorIdxRow[kk]],totalErrorDataRow[kk]);
        }

        char ce_file_name_buf[100];
        snprintf(ce_file_name_buf, sizeof(ce_file_name_buf), "iter_%d/CE.bin",jj);
        std::string CE_vector_path = ce_file_name_buf;

        golden::saveVectorToBinaryFile(C_output_base_path+"/"+CE_vector_path, hostZRow);

        char abe_file_name_buf[100];
        snprintf(abe_file_name_buf, sizeof(abe_file_name_buf), "iter_%d/ABE.bin",jj);
        std::string ABE_vector_path = abe_file_name_buf;

        golden::saveVectorToBinaryFile(C_output_base_path+"/"+ABE_vector_path, hostDRow);

        char thre_file_name_buf[100];
        snprintf(thre_file_name_buf, sizeof(thre_file_name_buf), "iter_%d/Threshold.bin",jj);
        std::string Thre_vector_path = thre_file_name_buf;

        golden::saveVectorToBinaryFile(C_output_base_path+"/"+Thre_vector_path, hostThre);

        char row_comp_name_buf[100];
        snprintf(row_comp_name_buf, sizeof(row_comp_name_buf), "iter_%d/Verify_Row.bin",jj);
        std::string Row_verify_path = row_comp_name_buf;
        golden::saveVectorToBinaryFile(C_output_base_path+"/"+Row_verify_path, hostCOMPRow);

        char row_error_idx_buf[100];
        snprintf(row_error_idx_buf, sizeof(row_error_idx_buf), "iter_%d/Error_Row_Idx.bin",jj);
        std::string Row_error_idx_path = row_error_idx_buf;
        golden::saveVectorToBinaryFile(C_output_base_path+"/"+Row_error_idx_path, totalErrorIdxRow);

        char row_error_data_buf[100];
        snprintf(row_error_data_buf, sizeof(row_error_data_buf), "iter_%d/Error_Row.bin",jj);
        std::string Row_error_data_path = row_error_data_buf;
        golden::saveVectorToBinaryFile(C_output_base_path+"/"+Row_error_data_path, totalErrorDataRow);

        char row_error_threshold_buf[100];
        snprintf(row_error_threshold_buf, sizeof(row_error_threshold_buf), "iter_%d/Failed_Row_Threshold.bin",jj);
        std::string Row_error_threshold_path = row_error_threshold_buf;
        golden::saveVectorToBinaryFile(C_output_base_path+"/"+Row_error_threshold_path, totalFailThresholds);
    }

    

    

    // int num_repeat = 10000;

    // aclrtEvent start, stop;
    // float temp_time = 0;
    // float time = 0;
    // ACL_CHECK(aclrtCreateEvent(&start));
    // ACL_CHECK(aclrtCreateEvent(&stop));

    // for (int i = 0; i < num_repeat; ++i) {
    //     ACL_CHECK(aclrtSynchronizeStream(stream));
    //     ACL_CHECK(aclrtRecordEvent(start, stream));

    //     RunAdapter(gemv_op, arguments, stream, aicCoreNum, fftsAddr);
            
    //     ACL_CHECK(aclrtSynchronizeStream(stream));
    //     ACL_CHECK(aclrtRecordEvent(stop, stream));
    //     ACL_CHECK(aclrtSynchronizeEvent(stop));
    //     ACL_CHECK(aclrtEventElapsedTime(&temp_time, start, stop));
    //     time += temp_time;
    // }

    // uint32_t total_op_nums = 4 * m * n + 4 * m*k + 4 * k*n;
    // std::cout << "m: " << m << ", n: " << n << ", k: "<< k <<","<< (float)total_op_nums / (time / num_repeat * 1e-3) / 1e9 << " GFLOPS, " << (time / num_repeat) << " ms, repeat: " << num_repeat << std::endl;

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));

    ACL_CHECK(aclrtFree(deviceX));
    ACL_CHECK(aclrtFree(deviceXV));

    ACL_CHECK(aclrtFree(deviceZRow));
    ACL_CHECK(aclrtFree(deviceZCol));

    ACL_CHECK(aclrtFree(deviceDRow));
    ACL_CHECK(aclrtFree(deviceDCol));

    ACL_CHECK(aclrtFree(deviceCOMPRow));
    ACL_CHECK(aclrtFree(deviceCOMPCol));

    ACL_CHECK(aclrtFree(deviceThre));

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