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
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_add.hpp"
#include "gemm/block/block_mmad.hpp" // catlass/
#include "gemm/block/block_swizzle.hpp" // catlass/
#include "gemm/dispatch_policy.hpp" // catlass/
#include "gemm/kernel/matmul_epilogue.hpp" // catlass/
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

#include "catlass/status.hpp"
#include "gemm/device/device_gemm.hpp" // catlass/
#include <string>

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

using namespace Catlass;
using fp16_t = op::fp16_t;

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
    const std::string HELPER = "03_matmul_add m n k in_path data_path out_path lim exp1 exp2 enc_type [device_id, make_golden]";

    GemmCoord problemShape{4096, 4096, 4096};
    int32_t deviceId{0};
    int32_t make_golden{0};
    int32_t limit{1};
    int32_t in_exp{0};
    int32_t out_exp{1};
    int32_t enc_type{0};
    std::string in_path = "./";
    std::string out_path = "./";
    std::string data_path = "./";


    Options() = default;

    int Parse(int argc, const char **argv)
    {
        enum ArgsIndex {
            M_INDEX = 1,
            N_INDEX,
            K_INDEX,
            IN_PATH_INDEX,
            DATA_PATH_INDEX,
            OUT_PATH_INDEX,
            LIM_INDEX,
            IN_EXP_INDEX,
            OUT_EXP_INDEX,
            ENC_INDEX,
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

        limit = std::atoi(argv[LIM_INDEX]);
        in_path = argv[IN_PATH_INDEX];
        data_path = argv[DATA_PATH_INDEX];
        out_path = argv[OUT_PATH_INDEX];
        in_exp = std::atoi(argv[IN_EXP_INDEX]);
        out_exp = std::atoi(argv[OUT_EXP_INDEX]);
        enc_type = std::atoi(argv[ENC_INDEX]);


        if (argc >= GOLDEN) {
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        }
        if (argc == ARGS_MAX){
            make_golden = std::atoi(argv[GOLDEN]);
        }
        return 0;
    }
};

template <class Adapter>
void RunAdapterComp(Adapter compare_op, typename Adapter::Arguments args, aclrtStream stream, uint32_t aicCoreNum)
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

template <class Adapter>
void RunAdapterMul(Adapter matmul_op, typename Adapter::Arguments args, aclrtStream stream,
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

void Run(Options const &options)
{
    std::cout<<"Device ID: "<<options.deviceId<<std::endl;
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();
    int32_t make_golden = options.make_golden;
    int32_t in_exp = options.in_exp;
    int32_t out_exp = options.out_exp;
    int32_t limit = options.limit;
    std::string in_path = options.in_path;
    std::string out_path = options.out_path;
    std::string data_path = options.data_path;
    int32_t enc_input_id = options.enc_type;

    // Compute the length of each matrix and the size of each buffer
    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenD = static_cast<size_t>(m) * n;
    // size_t lenX = lenD;

    size_t sizeA = lenA * sizeof(GemmInTypeC);
    size_t sizeB = lenB * sizeof(GemmInTypeC);
    size_t sizeD = lenD * sizeof(GemmOutTypeC);

    size_t lenX = static_cast<size_t>(m) * n;
    size_t lenY = static_cast<size_t>(m) * n;
    
    size_t lenZ = (lenX + 8 -1)/8;

    size_t sizeX = lenX * sizeof(GemmOutTypeC);
    size_t sizeY = lenY * sizeof(GemmOutTypeC);
    size_t sizeZ = lenZ * sizeof(uint8_t);

    // Define the layout of each matrix
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutD{m, n};

    using LayoutX = layout::VectorLayout;
    using LayoutY = layout::VectorLayout;
    using LayoutZ = layout::VectorLayout;
    
    LayoutX layoutX{m*n};
    LayoutY layoutY{m*n};
    LayoutZ layoutZ{(m*n + 8 - 1)/8};

    // Prepare input data A, B, and X
    std::vector<GemmInTypeC> hostA(lenA);
    std::vector<GemmInTypeC> hostB(lenB);
    std::vector<GemmOutTypeC> hostX(lenX,0.0f);
    
    std::vector<GemmOutTypeC> hostGolden(lenD);
    std::vector<uint8_t> hostZ(lenZ);

    GemmInTypeC up_limit_data = static_cast<GemmInTypeC>(options.limit * 1.0f);
    GemmInTypeC lower_limit_data = 0 - up_limit_data;

    // std::string in_path = options.in_path;
    // std::string out_path = options.out_path;

    char c_golden_name_buf[100];
    snprintf(c_golden_name_buf, sizeof(c_golden_name_buf), 
        "exp_%d_%u_%u_%u_lim_%d/C.bin", in_exp, m, n, k, limit);
    std::string C_golden_file_name = c_golden_name_buf;
    std::string C_golden_file_path = in_path+ "/" + C_golden_file_name;

    char a_input_name_buf[100];
    snprintf(a_input_name_buf, sizeof(a_input_name_buf), 
        "exp_%d_%u_%u_%u_lim_%d/A.bin", in_exp, m, n, k, limit);
    std::string A_input_file_name = a_input_name_buf;
    std::string A_input_file_path = in_path + "/" + A_input_file_name;

    char b_input_name_buf[100];
    snprintf(b_input_name_buf, sizeof(b_input_name_buf), 
        "exp_%d_%u_%u_%u_lim_%d/B.bin", in_exp, m, n, k, limit);
    std::string B_input_file_name = b_input_name_buf;
    std::string B_input_file_path = in_path + "/" + B_input_file_name;

    char c_output_name_buf[100];
    snprintf(c_output_name_buf, sizeof(c_output_name_buf), "exp_%d_in_%d_%u_%u_%u",
        out_exp, in_exp, m, n, k);
    std::string C_output_fpath = c_output_name_buf;
    std::string C_output_base_path = out_path + "/" + C_output_fpath;
    std::string C_data_base_path = data_path + "/" + C_output_fpath;


    golden::loadVectorFromBinaryFile(A_input_file_path, hostA);

    golden::loadVectorFromBinaryFile(B_input_file_path, hostB);

    golden::loadVectorFromBinaryFile(C_golden_file_path, hostGolden);

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

    uint8_t *deviceX{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceX), sizeX, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceX, sizeX, hostGolden.data(), sizeX, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceZ{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceZ), sizeZ, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceZ, sizeZ, hostZ.data(), sizeZ, ACL_MEMCPY_HOST_TO_DEVICE));

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};

    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    // auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    // Define ArchTag
    using ArchTag = Arch::AtlasA2;

    // Block level, define BlockMmad
    constexpr bool enableUnitFlag = true;
    using MmadDispatchPolicy = CubeSelf::Gemm::MmadAtlasA2Pingpong<enableUnitFlag>;
    using L1TileShape = GemmShape<128, 128, 128>;
    using L0TileShape = GemmShape<128, 128, 64>;
    using AType = Gemm::GemmType<GemmInTypeN, LayoutA>;
    using BType = Gemm::GemmType<GemmInTypeN, LayoutB>;
    using CType = Gemm::GemmType<GemmOutTypeN, LayoutC>;
    using BlockMmad = CubeSelf::Gemm::Block::BlockMmad<MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

    using UBTileShapeCOMP = GemvShape<1,8192>;


    GemmOutTypeN threshold{1.0f/256.0f};

    // Block level, define BlockEpilogue
    using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2ElemWiseOneSource;
    using XType = CType;
    using DType = CType;
    using ComputeType = CType;
    constexpr uint32_t computeLength = 16384;
    using TileElemWiseEpilogue = Epilogue::Tile::TileElemWiseAdd<ArchTag, ComputeType, computeLength>;
    using EpilogueTileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, XType, DType>;
    using BlockEpilogue = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy, CType, XType, DType,
        TileElemWiseEpilogue, EpilogueTileCopy>;

    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    using COMPDispatchPolicy = Catlass::Gemm::GemvAtlasA2;

    using XCType = Gemm::GemmType<GemmOutTypeN, LayoutX>;
    using YCType = Gemm::GemmType<GemmOutTypeN, LayoutY>;
    using ZType = Gemm::GemmType<uint8_t, LayoutZ>;
    using BiasType = void;

    using FT_COMP_TYPE = Catlass::Gemv::helper::FT_COMP_TYPE;
    
    FT_COMP_TYPE use_type = FT_COMP_TYPE::XOR;
    std::string use_type_str = "XOR";
    if(enc_input_id < 1){
        use_type = FT_COMP_TYPE::XOR;
    }else{
        use_type = FT_COMP_TYPE::SUB;
        use_type_str = "SUB";
    }

    
    

    
    std::vector<GemmOutTypeC> hostD(lenD);
    Catlass::GemvCoord problemShapeComp{m,n};
    int32_t num_repeat = 100;

    if (m > n) {
        // Define BlockScheduler
        // Swizzle offset is 3 and direction is 0.
        using BlockScheduler = typename CubeSelf::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
        // Kernel level
        using MatmulKernel = CubeSelf::Gemm::Kernel::MatmulEpilogue<BlockMmad, BlockEpilogue, BlockScheduler>;
        // Prepare params
        typename MatmulKernel::Arguments arguments{
            options.problemShape, sizeof(GemmOutTypeC), deviceA, deviceB, deviceD};
        using MatmulAdapter = CubeSelf::Gemm::Device::DeviceGemm<MatmulKernel>;
        MatmulAdapter matmul_op;
        
        using BlockEpilogueComp = void;

        if(enc_input_id < 1){
            using TileCompareCopy = Gemv::Tile::TileCopyCompareAiv<FT_COMP_TYPE::XOR, typename COMPDispatchPolicy::ArchTag, ZType, XCType, YCType, BiasType>;
            using CompareBlock = Gemv::Block::BlockCompare<COMPDispatchPolicy,FT_COMP_TYPE::XOR, UBTileShapeCOMP, ZType, XCType, YCType, TileCompareCopy>;

            using ElementWork = typename std::conditional<
                (CompareBlock::COMP_TYPE == FT_COMP_TYPE::XOR),
                uint16_t,
                typename std::conditional<(CompareBlock::COMP_TYPE == FT_COMP_TYPE::COMPARE), int32_t, GemmOutTypeN>::type>::type;
            
            using CompareKernel = Gemv::Kernel::KernelCompareAiv<CompareBlock, BlockEpilogue>;
        
            uint32_t UbNum = 2;
            bool OutputWorkspace = false;

            typename CompareKernel::Arguments arguments_comp{problemShapeComp, deviceX, deviceD, deviceZ, UbNum, false, threshold};
            using CompareAdapter = Gemv::Device::DeviceGemv<CompareKernel>;
            CompareAdapter compare_op;
            compare_op.CanImplement(arguments_comp);
            std::vector<uint8_t> hostCompGolden(lenZ,255);

            for (int i = 0; i < 100; ++i) {
                ACL_CHECK(aclrtSynchronizeStream(stream));
                // matmul_op(stream, aicCoreNum, fftsAddr);
                // RunAdapterMul(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
                RunAdapterComp(compare_op, arguments_comp, stream, aivCoreNum);
            }
            ACL_CHECK(aclrtSynchronizeStream(stream));

            for(int jj = 0; jj < num_repeat; ++jj){
                // ACL_CHECK(aclrtSynchronizeStream(stream));
                // RunAdapterMul(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
                // matmul_op(stream, aicCoreNum, fftsAddr);
                char c_iter_name_buf[100];
                snprintf(c_iter_name_buf, sizeof(c_iter_name_buf), "iter_%d/C.bin", jj);
        
                std::string C_iter_fpath = c_iter_name_buf;
                std::string C_iter_file_path = C_data_base_path + "/" + C_iter_fpath;

                golden::loadVectorFromBinaryFile(C_iter_file_path, hostD);
                ACL_CHECK(aclrtMemcpy(deviceD, sizeD, hostD.data(), sizeD, ACL_MEMCPY_HOST_TO_DEVICE));

                ACL_CHECK(aclrtSynchronizeStream(stream));
                // ACL_CHECK(aclrtMemcpy(hostD.data(), sizeD, deviceD, sizeD, ACL_MEMCPY_DEVICE_TO_HOST));

                std::vector<uint64_t> errorIndices = golden::CompareData(hostD, hostGolden, lenX);
                if (errorIndices.empty()) {
                    std::cout << "Compare success." << std::endl;
                } else {
                    std::cout << "Compare failed. Error count: " << errorIndices.size() << std::endl;
                }

                std::vector<uint8_t> hostCompRes(lenZ);
        
                // kernel level
                RunAdapterComp(compare_op, arguments_comp, stream, aivCoreNum);
                ACL_CHECK(aclrtSynchronizeStream(stream));
                ACL_CHECK(aclrtMemcpy(hostCompRes.data(), sizeZ, deviceZ, sizeZ, ACL_MEMCPY_DEVICE_TO_HOST));
                printf("%f\n", hostGolden[0]);
                printf("%f\n", hostD[0]);
                printf("%hhu\n",hostCompRes[0]);
                printf("%hhu\n",hostCompRes[lenZ - 1]);
        
                std::vector<uint64_t> totalErrorIdx;
                std::vector<GemmOutTypeC> totalErrorData;

                totalErrorIdx.clear();
                totalErrorData.clear();
                // errorIndices = golden::CompareData(hostCompRes, hostGolden, lenZ);
                errorIndices = golden::GetErrorDataWithIndexTotal(hostCompRes, hostCompGolden, hostD, 
                    hostGolden, lenZ, "NPU", "Real", totalErrorIdx, 
                    totalErrorData);

                printf("Iter %d: Method: %s Threshold: %f\n",jj,use_type_str.c_str(),threshold);
                if (errorIndices.empty()) {
                    std::cout << "Bit compare success." << std::endl;
                } else {
                    std::cerr << "Bit compare failed. Error count: " << errorIndices.size() << std::endl;
                }

                printf("Total Error Idx len: %d\n", static_cast<int>(totalErrorIdx.size()));
                printf("Total Error Data len: %d\n", static_cast<int>(totalErrorData.size()));

                int32_t show_len = (totalErrorIdx.size() > 20) ? 20 : static_cast<int32_t>(totalErrorIdx.size());

                for(int32_t kk=0; kk < show_len; kk++){
                    printf("Data Id: %lu, Real Data: %f, Error Data: %f\n",totalErrorIdx[kk],hostGolden[totalErrorIdx[kk]],totalErrorData[kk]);
                }

                // char c_file_name_buf[100];
                // snprintf(c_file_name_buf, sizeof(c_file_name_buf), "iter_%d/C.bin",jj);
                // std::string C_matrix_path = c_file_name_buf;

                // golden::saveVectorToBinaryFile(C_output_base_path+"/"+C_matrix_path, hostD);

                char c_comp_name_buf[100];
                snprintf(c_comp_name_buf, sizeof(c_comp_name_buf), "iter_%d/Verify_Z_%s.bin",jj, use_type_str.c_str());
                std::string C_verify_path = c_comp_name_buf;
                golden::saveVectorToBinaryFile(C_output_base_path+"/"+C_verify_path, hostCompRes);

                char c_error_idx_buf[100];
                snprintf(c_error_idx_buf, sizeof(c_error_idx_buf), "iter_%d/Error_Idx_%s.bin",jj, use_type_str.c_str());
                std::string C_error_idx_path = c_error_idx_buf;
                golden::saveVectorToBinaryFile(C_output_base_path+"/"+C_error_idx_path, totalErrorIdx);

                char c_error_data_buf[100];
                snprintf(c_error_data_buf, sizeof(c_error_data_buf), "iter_%d/Error_C_%s.bin",jj, use_type_str.c_str());
                std::string C_error_data_path = c_error_data_buf;
                golden::saveVectorToBinaryFile(C_output_base_path+"/"+C_error_data_path, totalErrorData);
            }
        }else{
            using TileCompareCopy = Gemv::Tile::TileCopyCompareAiv<FT_COMP_TYPE::SUB, typename COMPDispatchPolicy::ArchTag, ZType, XCType, YCType, BiasType>;
            using CompareBlock = Gemv::Block::BlockCompare<COMPDispatchPolicy,FT_COMP_TYPE::SUB, UBTileShapeCOMP, ZType, XCType, YCType, TileCompareCopy>;

            using ElementWork = typename std::conditional<
                (CompareBlock::COMP_TYPE == FT_COMP_TYPE::XOR),
                uint16_t,
                typename std::conditional<(CompareBlock::COMP_TYPE == FT_COMP_TYPE::COMPARE), 
                    int32_t, GemmOutTypeN>::type>::type;

            using CompareKernel = Gemv::Kernel::KernelCompareAiv<CompareBlock, BlockEpilogue>;
        
            uint32_t UbNum = 2;
            bool OutputWorkspace = false;

            typename CompareKernel::Arguments arguments_comp{problemShapeComp, deviceX, deviceD, deviceZ, UbNum, false, threshold};
            using CompareAdapter = Gemv::Device::DeviceGemv<CompareKernel>;
            CompareAdapter compare_op;
            compare_op.CanImplement(arguments_comp);
            std::vector<uint8_t> hostCompGolden(lenZ,255);

            for (int i = 0; i < 100; ++i) {
                ACL_CHECK(aclrtSynchronizeStream(stream));
                // matmul_op(stream, aicCoreNum, fftsAddr);
                // RunAdapterMul(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
                RunAdapterComp(compare_op, arguments_comp, stream, aivCoreNum);
            }
            ACL_CHECK(aclrtSynchronizeStream(stream));

            for(int jj = 0; jj < num_repeat; ++jj){
                // ACL_CHECK(aclrtSynchronizeStream(stream));
                // RunAdapterMul(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
                // matmul_op(stream, aicCoreNum, fftsAddr);
                char c_iter_name_buf[100];
                snprintf(c_iter_name_buf, sizeof(c_iter_name_buf), "iter_%d/C.bin", jj);
        
                std::string C_iter_fpath = c_iter_name_buf;
                std::string C_iter_file_path = C_data_base_path + "/" + C_iter_fpath;

                golden::loadVectorFromBinaryFile(C_iter_file_path, hostD);
                ACL_CHECK(aclrtMemcpy(deviceD, sizeD, hostD.data(), sizeD, ACL_MEMCPY_HOST_TO_DEVICE));

                ACL_CHECK(aclrtSynchronizeStream(stream));
                // ACL_CHECK(aclrtMemcpy(hostD.data(), sizeD, deviceD, sizeD, ACL_MEMCPY_DEVICE_TO_HOST));

                std::vector<uint64_t> errorIndices = golden::CompareData(hostD, hostGolden, lenX);
                if (errorIndices.empty()) {
                    std::cout << "Compare success." << std::endl;
                } else {
                    std::cout << "Compare failed. Error count: " << errorIndices.size() << std::endl;
                }

                std::vector<uint8_t> hostCompRes(lenZ);
        
                // kernel level
                RunAdapterComp(compare_op, arguments_comp, stream, aivCoreNum);
                ACL_CHECK(aclrtSynchronizeStream(stream));
                ACL_CHECK(aclrtMemcpy(hostCompRes.data(), sizeZ, deviceZ, sizeZ, ACL_MEMCPY_DEVICE_TO_HOST));
                printf("%f\n", hostGolden[0]);
                printf("%f\n", hostD[0]);
                printf("%hhu\n",hostCompRes[0]);
                printf("%hhu\n",hostCompRes[lenZ - 1]);
        
                std::vector<uint64_t> totalErrorIdx;
                std::vector<GemmOutTypeC> totalErrorData;

                totalErrorIdx.clear();
                totalErrorData.clear();
                // errorIndices = golden::CompareData(hostCompRes, hostGolden, lenZ);
                errorIndices = golden::GetErrorDataWithIndexTotal(hostCompRes, hostCompGolden, hostD, 
                    hostGolden, lenZ, "NPU", "Real", totalErrorIdx, 
                    totalErrorData);

                printf("Iter %d: Method: %s Threshold:%f \n",jj,use_type_str.c_str(),threshold);
                if (errorIndices.empty()) {
                    std::cout << "Bit compare success." << std::endl;
                } else {
                    std::cerr << "Bit compare failed. Error count: " << errorIndices.size() << std::endl;
                }

                printf("Total Error Idx len: %d\n", static_cast<int>(totalErrorIdx.size()));
                printf("Total Error Data len: %d\n", static_cast<int>(totalErrorData.size()));

                int32_t show_len = (totalErrorIdx.size() > 20) ? 20 : static_cast<int32_t>(totalErrorIdx.size());

                for(int32_t kk=0; kk < show_len; kk++){
                    printf("Data Id: %lu, Real Data: %f, Error Data: %f\n",totalErrorIdx[kk],hostGolden[totalErrorIdx[kk]],totalErrorData[kk]);
                }

                // char c_file_name_buf[100];
                // snprintf(c_file_name_buf, sizeof(c_file_name_buf), "iter_%d/C.bin",jj);
                // std::string C_matrix_path = c_file_name_buf;

                // golden::saveVectorToBinaryFile(C_output_base_path+"/"+C_matrix_path, hostD);

                char c_comp_name_buf[100];
                snprintf(c_comp_name_buf, sizeof(c_comp_name_buf), "iter_%d/Verify_Z_%s.bin",jj, use_type_str.c_str());
                std::string C_verify_path = c_comp_name_buf;
                golden::saveVectorToBinaryFile(C_output_base_path+"/"+C_verify_path, hostCompRes);

                char c_error_idx_buf[100];
                snprintf(c_error_idx_buf, sizeof(c_error_idx_buf), "iter_%d/Error_Idx_%s.bin",jj, use_type_str.c_str());
                std::string C_error_idx_path = c_error_idx_buf;
                golden::saveVectorToBinaryFile(C_output_base_path+"/"+C_error_idx_path, totalErrorIdx);

                char c_error_data_buf[100];
                snprintf(c_error_data_buf, sizeof(c_error_data_buf), "iter_%d/Error_C_%s.bin",jj, use_type_str.c_str());
                std::string C_error_data_path = c_error_data_buf;
                golden::saveVectorToBinaryFile(C_output_base_path+"/"+C_error_data_path, totalErrorData);
            }
        }

    } else {
        // Define BlockScheduler
        // Swizzle offset is 3 and direction is 1.
        using BlockScheduler = typename CubeSelf::Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
        // Kernel level
        using MatmulKernel = CubeSelf::Gemm::Kernel::MatmulEpilogue<BlockMmad, BlockEpilogue, BlockScheduler>;
        // Prepare params
        typename MatmulKernel::Arguments arguments{
            options.problemShape, sizeof(GemmOutTypeN), deviceA, deviceB, deviceD};
        using MatmulAdapter = CubeSelf::Gemm::Device::DeviceGemm<MatmulKernel>;
        MatmulAdapter matmul_op;
        // using CompareBlock = Gemv::Block::BlockCompare<COMPDispatchPolicy,use_type, UBTileShapeCOMP, ZType, XCType, YCType, TileCompareCopy>;
        using BlockEpilogueComp = void;

        if(enc_input_id < 1){
            using TileCompareCopy = Gemv::Tile::TileCopyCompareAiv<FT_COMP_TYPE::XOR, typename COMPDispatchPolicy::ArchTag, ZType, XCType, YCType, BiasType>;
            using CompareBlock = Gemv::Block::BlockCompare<COMPDispatchPolicy,FT_COMP_TYPE::XOR, UBTileShapeCOMP, ZType, XCType, YCType, TileCompareCopy>;

            using ElementWork = typename std::conditional<
                (CompareBlock::COMP_TYPE == FT_COMP_TYPE::XOR),
                uint16_t,
                typename std::conditional<(CompareBlock::COMP_TYPE == FT_COMP_TYPE::COMPARE), 
                int32_t, GemmOutTypeN>::type>::type;

            using CompareKernel = Gemv::Kernel::KernelCompareAiv<CompareBlock, BlockEpilogue>;
        
            uint32_t UbNum = 2;
            bool OutputWorkspace = false;

            typename CompareKernel::Arguments arguments_comp{problemShapeComp, deviceX, deviceD, deviceZ, UbNum, false, threshold};
            using CompareAdapter = Gemv::Device::DeviceGemv<CompareKernel>;
            CompareAdapter compare_op;
            compare_op.CanImplement(arguments_comp);
            std::vector<uint8_t> hostCompGolden(lenZ,255);

            for (int i = 0; i < 100; ++i) {
                ACL_CHECK(aclrtSynchronizeStream(stream));
                // matmul_op(stream, aicCoreNum, fftsAddr);
                // RunAdapterMul(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
                RunAdapterComp(compare_op, arguments_comp, stream, aivCoreNum);
            }
            ACL_CHECK(aclrtSynchronizeStream(stream));

            for(int jj = 0; jj < num_repeat; ++jj){
                // ACL_CHECK(aclrtSynchronizeStream(stream));
                // RunAdapterMul(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
                // matmul_op(stream, aicCoreNum, fftsAddr);
                char c_iter_name_buf[100];
                snprintf(c_iter_name_buf, sizeof(c_iter_name_buf), "iter_%d/C.bin", jj);
        
                std::string C_iter_fpath = c_iter_name_buf;
                std::string C_iter_file_path = C_data_base_path + "/" + C_iter_fpath;

                golden::loadVectorFromBinaryFile(C_iter_file_path, hostD);
                ACL_CHECK(aclrtMemcpy(deviceD, sizeD, hostD.data(), sizeD, ACL_MEMCPY_HOST_TO_DEVICE));

                ACL_CHECK(aclrtSynchronizeStream(stream));
                // ACL_CHECK(aclrtMemcpy(hostD.data(), sizeD, deviceD, sizeD, ACL_MEMCPY_DEVICE_TO_HOST));

                std::vector<uint64_t> errorIndices = golden::CompareData(hostD, hostGolden, lenX);
                if (errorIndices.empty()) {
                    std::cout << "Compare success." << std::endl;
                } else {
                    std::cout << "Compare failed. Error count: " << errorIndices.size() << std::endl;
                }

                std::vector<uint8_t> hostCompRes(lenZ);
        
                // kernel level
                RunAdapterComp(compare_op, arguments_comp, stream, aivCoreNum);
                ACL_CHECK(aclrtSynchronizeStream(stream));
                ACL_CHECK(aclrtMemcpy(hostCompRes.data(), sizeZ, deviceZ, sizeZ, ACL_MEMCPY_DEVICE_TO_HOST));
                printf("%f\n", hostGolden[0]);
                printf("%f\n", hostD[0]);
                printf("%hhu\n",hostCompRes[0]);
                printf("%hhu\n",hostCompRes[lenZ - 1]);
        
                std::vector<uint64_t> totalErrorIdx;
                std::vector<GemmOutTypeC> totalErrorData;

                totalErrorIdx.clear();
                totalErrorData.clear();
                // errorIndices = golden::CompareData(hostCompRes, hostGolden, lenZ);
                errorIndices = golden::GetErrorDataWithIndexTotal(hostCompRes, hostCompGolden, hostD, 
                    hostGolden, lenZ, "NPU", "Real", totalErrorIdx, 
                    totalErrorData);

                printf("Iter %d: Method: %s Threshold: %f\n",jj,use_type_str.c_str(),threshold);
                if (errorIndices.empty()) {
                    std::cout << "Bit compare success." << std::endl;
                } else {
                    std::cerr << "Bit compare failed. Error count: " << errorIndices.size() << std::endl;
                }

                printf("Total Error Idx len: %d\n", static_cast<int>(totalErrorIdx.size()));
                printf("Total Error Data len: %d\n", static_cast<int>(totalErrorData.size()));

                int32_t show_len = (totalErrorIdx.size() > 20) ? 20 : static_cast<int32_t>(totalErrorIdx.size());

                for(int32_t kk=0; kk < show_len; kk++){
                    printf("Data Id: %lu, Real Data: %f, Error Data: %f\n",totalErrorIdx[kk],hostGolden[totalErrorIdx[kk]],totalErrorData[kk]);
                }

                // char c_file_name_buf[100];
                // snprintf(c_file_name_buf, sizeof(c_file_name_buf), "iter_%d/C.bin",jj);
                // std::string C_matrix_path = c_file_name_buf;

                // golden::saveVectorToBinaryFile(C_output_base_path+"/"+C_matrix_path, hostD);

                char c_comp_name_buf[100];
                snprintf(c_comp_name_buf, sizeof(c_comp_name_buf), "iter_%d/Verify_Z_%s.bin",jj, use_type_str.c_str());
                std::string C_verify_path = c_comp_name_buf;
                golden::saveVectorToBinaryFile(C_output_base_path+"/"+C_verify_path, hostCompRes);

                char c_error_idx_buf[100];
                snprintf(c_error_idx_buf, sizeof(c_error_idx_buf), "iter_%d/Error_Idx_%s.bin",jj, use_type_str.c_str());
                std::string C_error_idx_path = c_error_idx_buf;
                golden::saveVectorToBinaryFile(C_output_base_path+"/"+C_error_idx_path, totalErrorIdx);

                char c_error_data_buf[100];
                snprintf(c_error_data_buf, sizeof(c_error_data_buf), "iter_%d/Error_C_%s.bin",jj, use_type_str.c_str());
                std::string C_error_data_path = c_error_data_buf;
                golden::saveVectorToBinaryFile(C_output_base_path+"/"+C_error_data_path, totalErrorData);
            }
        }else{
            using TileCompareCopy = Gemv::Tile::TileCopyCompareAiv<FT_COMP_TYPE::SUB, typename COMPDispatchPolicy::ArchTag, ZType, XCType, YCType, BiasType>;
            using CompareBlock = Gemv::Block::BlockCompare<COMPDispatchPolicy,FT_COMP_TYPE::SUB, UBTileShapeCOMP, ZType, XCType, YCType, TileCompareCopy>;

            using ElementWork = typename std::conditional<
                (CompareBlock::COMP_TYPE == FT_COMP_TYPE::XOR),
                uint16_t,
                typename std::conditional<(CompareBlock::COMP_TYPE == FT_COMP_TYPE::COMPARE), 
                    int32_t, GemmOutTypeN>::type>::type;

            using CompareKernel = Gemv::Kernel::KernelCompareAiv<CompareBlock, BlockEpilogue>;
        
            uint32_t UbNum = 2;
            bool OutputWorkspace = false;

            typename CompareKernel::Arguments arguments_comp{problemShapeComp, deviceX, deviceD, deviceZ, UbNum, false, threshold};
            using CompareAdapter = Gemv::Device::DeviceGemv<CompareKernel>;
            CompareAdapter compare_op;
            compare_op.CanImplement(arguments_comp);
            std::vector<uint8_t> hostCompGolden(lenZ,255);

            for (int i = 0; i < 100; ++i) {
                ACL_CHECK(aclrtSynchronizeStream(stream));
                // matmul_op(stream, aicCoreNum, fftsAddr);
                // RunAdapterMul(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
                RunAdapterComp(compare_op, arguments_comp, stream, aivCoreNum);
            }

            ACL_CHECK(aclrtSynchronizeStream(stream));

            for(int jj = 0; jj < num_repeat; ++jj){
                // ACL_CHECK(aclrtSynchronizeStream(stream));
                // RunAdapterMul(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
                // matmul_op(stream, aicCoreNum, fftsAddr);
                char c_iter_name_buf[100];
                snprintf(c_iter_name_buf, sizeof(c_iter_name_buf), "iter_%d/C.bin", jj);
        
                std::string C_iter_fpath = c_iter_name_buf;
                std::string C_iter_file_path = C_data_base_path + "/" + C_iter_fpath;

                golden::loadVectorFromBinaryFile(C_iter_file_path, hostD);
                ACL_CHECK(aclrtMemcpy(deviceD, sizeD, hostD.data(), sizeD, ACL_MEMCPY_HOST_TO_DEVICE));

                ACL_CHECK(aclrtSynchronizeStream(stream));
                // ACL_CHECK(aclrtMemcpy(hostD.data(), sizeD, deviceD, sizeD, ACL_MEMCPY_DEVICE_TO_HOST));

                std::vector<uint64_t> errorIndices = golden::CompareData(hostD, hostGolden, lenX);
                if (errorIndices.empty()) {
                    std::cout << "Compare success." << std::endl;
                } else {
                    std::cout << "Compare failed. Error count: " << errorIndices.size() << std::endl;
                }

                std::vector<uint8_t> hostCompRes(lenZ);
        
                // kernel level
                RunAdapterComp(compare_op, arguments_comp, stream, aivCoreNum);
                ACL_CHECK(aclrtSynchronizeStream(stream));
                ACL_CHECK(aclrtMemcpy(hostCompRes.data(), sizeZ, deviceZ, sizeZ, ACL_MEMCPY_DEVICE_TO_HOST));
                printf("%f\n", hostGolden[0]);
                printf("%f\n", hostD[0]);
                printf("%hhu\n",hostCompRes[0]);
                printf("%hhu\n",hostCompRes[lenZ - 1]);
        
                std::vector<uint64_t> totalErrorIdx;
                std::vector<GemmOutTypeC> totalErrorData;

                totalErrorIdx.clear();
                totalErrorData.clear();
                // errorIndices = golden::CompareData(hostCompRes, hostGolden, lenZ);
                errorIndices = golden::GetErrorDataWithIndexTotal(hostCompRes, hostCompGolden, hostD, 
                    hostGolden, lenZ, "NPU", "Real", totalErrorIdx, 
                    totalErrorData);

                printf("Iter %d: Method: %s Threshold: %f\n",jj,use_type_str.c_str(),threshold);
                if (errorIndices.empty()) {
                    std::cout << "Bit compare success." << std::endl;
                } else {
                    std::cerr << "Bit compare failed. Error count: " << errorIndices.size() << std::endl;
                }

                printf("Total Error Idx len: %d\n", static_cast<int>(totalErrorIdx.size()));
                printf("Total Error Data len: %d\n", static_cast<int>(totalErrorData.size()));

                int32_t show_len = (totalErrorIdx.size() > 20) ? 20 : static_cast<int32_t>(totalErrorIdx.size());

                for(int32_t kk=0; kk < show_len; kk++){
                    printf("Data Id: %lu, Real Data: %f, Error Data: %f\n",totalErrorIdx[kk],hostGolden[totalErrorIdx[kk]],totalErrorData[kk]);
                }

                // char c_file_name_buf[100];
                // snprintf(c_file_name_buf, sizeof(c_file_name_buf), "iter_%d/C.bin",jj);
                // std::string C_matrix_path = c_file_name_buf;

                // golden::saveVectorToBinaryFile(C_output_base_path+"/"+C_matrix_path, hostD);

                char c_comp_name_buf[100];
                snprintf(c_comp_name_buf, sizeof(c_comp_name_buf), "iter_%d/Verify_Z_%s.bin",jj, use_type_str.c_str());
                std::string C_verify_path = c_comp_name_buf;
                golden::saveVectorToBinaryFile(C_output_base_path+"/"+C_verify_path, hostCompRes);

                char c_error_idx_buf[100];
                snprintf(c_error_idx_buf, sizeof(c_error_idx_buf), "iter_%d/Error_Idx_%s.bin",jj, use_type_str.c_str());
                std::string C_error_idx_path = c_error_idx_buf;
                golden::saveVectorToBinaryFile(C_output_base_path+"/"+C_error_idx_path, totalErrorIdx);

                char c_error_data_buf[100];
                snprintf(c_error_data_buf, sizeof(c_error_data_buf), "iter_%d/Error_C_%s.bin",jj, use_type_str.c_str());
                std::string C_error_data_path = c_error_data_buf;
                golden::saveVectorToBinaryFile(C_output_base_path+"/"+C_error_data_path, totalErrorData);
            }
        }

        
        
        // using CompareKernel = Gemv::Kernel::KernelCompareAiv<CompareBlock, BlockEpilogue>;
        
        // uint32_t UbNum = 2;
        // bool OutputWorkspace = false;

        // typename CompareKernel::Arguments arguments_comp{problemShapeComp, deviceX, deviceD, deviceZ, UbNum, false, threshold};
        // using CompareAdapter = Gemv::Device::DeviceGemv<CompareKernel>;
        // CompareAdapter compare_op;
        // compare_op.CanImplement(arguments_comp);
        // std::vector<uint8_t> hostCompGolden(lenZ,255);

        // for (int i = 0; i < 100; ++i) {
        //     ACL_CHECK(aclrtSynchronizeStream(stream));
        //     // matmul_op(stream, aicCoreNum, fftsAddr);
        //     // RunAdapterMul(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
        //     RunAdapterComp(compare_op, arguments_comp, stream, aivCoreNum);
        // }

        // ACL_CHECK(aclrtSynchronizeStream(stream));

        // for(int jj = 0; jj < num_repeat; ++jj){
        //     // ACL_CHECK(aclrtSynchronizeStream(stream));
        //     // RunAdapterMul(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
        //     // // matmul_op(stream, aicCoreNum, fftsAddr);
        //     // ACL_CHECK(aclrtSynchronizeStream(stream));
        //     // ACL_CHECK(aclrtMemcpy(hostD.data(), sizeD, deviceD, sizeD, ACL_MEMCPY_DEVICE_TO_HOST));

        //     char c_iter_name_buf[100];
        //     snprintf(c_iter_name_buf, sizeof(c_iter_name_buf), "iter_%d/C.bin", jj);
        
        //     std::string C_iter_fpath = c_iter_name_buf;
        //     std::string C_iter_file_path = C_output_base_path + "/" + C_iter_fpath;

        //     golden::loadVectorFromBinaryFile(C_iter_file_path, hostD);
        //     ACL_CHECK(aclrtMemcpy(deviceD, sizeD, hostD.data(), sizeD, ACL_MEMCPY_HOST_TO_DEVICE));

        //     ACL_CHECK(aclrtSynchronizeStream(stream));

        //     std::vector<uint64_t> errorIndices = golden::CompareData(hostD, hostGolden, lenX);
        //     if (errorIndices.empty()) {
        //         std::cout << "Compare success." << std::endl;
        //     } else {
        //         std::cout << "Compare failed. Error count: " << errorIndices.size() << std::endl;
        //     }

        //     std::vector<uint8_t> hostCompRes(lenZ);
        
        //     // kernel level
        //     RunAdapterComp(compare_op, arguments_comp, stream, aivCoreNum);
        //     ACL_CHECK(aclrtSynchronizeStream(stream));
        //     ACL_CHECK(aclrtMemcpy(hostCompRes.data(), sizeZ, deviceZ, sizeZ, ACL_MEMCPY_DEVICE_TO_HOST));
        //     printf("%f\n", hostGolden[0]);
        //     printf("%f\n", hostD[0]);
        //     printf("%hhu\n",hostCompRes[0]);
        //     printf("%hhu\n",hostCompRes[lenZ - 1]);
        
        //     std::vector<uint64_t> totalErrorIdx;
        //     std::vector<GemmOutTypeC> totalErrorData;

        //     totalErrorIdx.clear();
        //     totalErrorData.clear();
        //     // errorIndices = golden::CompareData(hostCompRes, hostGolden, lenZ);
        //     errorIndices = golden::GetErrorDataWithIndexTotal(hostCompRes, hostCompGolden, hostD, 
        //         hostGolden, lenZ, "NPU", "Real", totalErrorIdx, 
        //         totalErrorData);

        //     printf("Iter %d: Method: %s\n",jj,use_type_str.c_str());
        //     if (errorIndices.empty()) {
        //         std::cout << "Bit compare success." << std::endl;
        //     } else {
        //         std::cerr << "Bit compare failed. Error count: " << errorIndices.size() << std::endl;
        //     }

        //     printf("Total Error Idx len: %d\n", static_cast<int>(totalErrorIdx.size()));
        //     printf("Total Error Data len: %d\n", static_cast<int>(totalErrorData.size()));

        //     int32_t show_len = (totalErrorIdx.size() > 20) ? 20 : static_cast<int32_t>(totalErrorIdx.size());

        //     for(int32_t jj=0; jj < show_len; jj++){
        //         printf("Data Id: %lu, Real Data: %f, Error Data: %f\n",totalErrorIdx[jj],hostGolden[totalErrorIdx[jj]],totalErrorData[jj]);
        //     }

        //     // char c_file_name_buf[100];
        //     // snprintf(c_file_name_buf, sizeof(c_file_name_buf), "iter_%d/C.bin",jj);
        //     // std::string C_matrix_path = c_file_name_buf;

        //     // golden::saveVectorToBinaryFile(C_output_base_path+"/"+C_matrix_path, hostD);

        //     char c_comp_name_buf[100];
        //     snprintf(c_comp_name_buf, sizeof(c_comp_name_buf), "iter_%d/Verify_Z_%s.bin",jj, use_type_str.c_str());
        //     std::string C_verify_path = c_comp_name_buf;
        //     golden::saveVectorToBinaryFile(C_output_base_path+"/"+C_verify_path, hostCompRes);

        //     char c_error_idx_buf[100];
        //     snprintf(c_error_idx_buf, sizeof(c_error_idx_buf), "iter_%d/Error_Idx_%s.bin",jj, use_type_str.c_str());
        //     std::string C_error_idx_path = c_error_idx_buf;
        //     golden::saveVectorToBinaryFile(C_output_base_path+"/"+C_error_idx_path, totalErrorIdx);

        //     char c_error_data_buf[100];
        //     snprintf(c_error_data_buf, sizeof(c_error_data_buf), "iter_%d/Error_C_%s.bin",jj, use_type_str.c_str());
        //     std::string C_error_data_path = c_error_data_buf;
        //     golden::saveVectorToBinaryFile(C_output_base_path+"/"+C_error_data_path, totalErrorData);
        // }
    }


    

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceD));
    ACL_CHECK(aclrtFree(deviceX));
    ACL_CHECK(aclrtFree(deviceZ));


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
