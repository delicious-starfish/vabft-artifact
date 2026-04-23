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
#include <bitset>
#include "acl/acl.h"
// #include "kernel_operator.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclnnop/aclnn_matmul.h"
#include "acl/acl_base.h"

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
#include "catlass/gemm_coord.hpp"

using namespace Catlass;

using ScalarType = float;

#define CHECK_ACL(x)                                                                        \
    do                                                                                      \
    {                                                                                       \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE)                                                        \
        {                                                                                   \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        }                                                                                   \
    } while (0);

#define CHECK_RET(cond, return_expr) \
    do                               \
    {                                \
        if (!(cond))                 \
        {                            \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do                                  \
    {                                   \
        printf(message, ##__VA_ARGS__); \
    } while (0)

struct Options{
    const std::string HELPER = "17_compare_aiv m n k r [device_id]";

    uint32_t M = 32;
    uint32_t N = 32;
    uint32_t K = 32;
    uint32_t repeat = 2;
    uint32_t deviceId{0};

    Options() = default;
    
    GemmCoord problemShape{M, N, K};

    int Parse(int argc, const char **argv)
    {
        enum ArgsIndex {
            M_INDEX = 1,
            N_INDEX,
            K_INDEX,
            REPEAT_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };
        if (argc > ARGS_MAX || argc <= K_INDEX) 
        {
            std::cerr << HELPER << std::endl;
            return -1;
        }
        problemShape.m() = std::atoi(argv[M_INDEX]);
        problemShape.n() = std::atoi(argv[N_INDEX]);
        problemShape.k() = std::atoi(argv[K_INDEX]);

        repeat = std::atoi(argv[REPEAT_INDEX]);
        printf("Repeat Time for DMR: %d\n",repeat);

        if (argc == ARGS_MAX)
        {
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        }
        return 0;
    }
};

template <class Adapter>
void RunAdapter(Adapter compare_op, typename Adapter::Arguments args, aclrtStream stream, uint32_t aicCoreNum)
{
    size_t sizeWorkspace = compare_op.GetWorkspaceSize(args);
    // printf("Workspace for compare: %zu\n",sizeWorkspace);
    uint8_t *deviceWorkspace = nullptr;
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    // printf("start to initialize the compare op\n");
    compare_op.Initialize(args, deviceWorkspace);
    // printf("finished initialize compare op\n");
    compare_op(stream, aicCoreNum);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    // printf("finished compare\n");

    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }
}

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape)
    {
        shapeSize *= i;
    }
    return shapeSize;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape,
                    void **deviceAddr, aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--)
    {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template<class ElementRandom>
void FillRandomScalarData(ElementRandom &scalarData, ElementRandom low, ElementRandom high)
{
    scalarData = static_cast<ElementRandom>(low + (static_cast<ElementRandom>(rand()) / static_cast<ElementRandom>(RAND_MAX)) * (high - low));
}

void Init(Options options, aclrtStream *stream)
{
    // 固定写法，AscendCL初始化
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(stream));
}

int WarmupFunc(aclrtStream & stream, uint32_t warmup, aclTensor * self_warm, aclTensor * mat2_warm, aclTensor * out_warm){
    int8_t cubeMathType = 0;
    /*
    workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
    executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。
    */
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;

    // 根据第一段接口计算出的workspaceSize申请device内存
    uint8_t *workspaceAddr = nullptr;

    for (uint32_t i = 0; i < warmup; i++)
    {
        workspaceSize = 0;
        auto ret = aclnnMatmulGetWorkspaceSize(self_warm, mat2_warm, out_warm, cubeMathType, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclnnMatmulGetWorkspaceSize failed. ERROR: %d\n", ret);
                  return ret);

        if (workspaceSize > 0)
        {
            ret = aclrtMalloc(reinterpret_cast<void **>(&workspaceAddr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                      return ret);
        }
        ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmul failed. ERROR: %d\n", ret);
                  return ret);

        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
                  return ret);
        if (workspaceSize > 0)
        {
            ret = aclrtFree(workspaceAddr);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("free workspace failed. ERROR: %d\n", ret);
                      return ret);
        }
    }

    return 0;
}

void InitialRepeatData(uint8_t* outDeviceAddr, uint8_t* outDeviceVerify,
    std::vector<std::vector<float>> & outputHostArr,
    aclTensor * outTensor, aclTensor * outTensorVerify,
    std::vector<std::vector<uint8_t>> & compHostZArr,
    uint8_t * compDeviceZAddr,
    std::vector<int64_t> outShape, uint32_t repeat_time,bool IfFreeGM)
{
    uint32_t lenX = static_cast<uint32_t>(outShape[0]) * static_cast<uint32_t>(outShape[1]);
    uint32_t lenZ = (static_cast<uint32_t>(outShape[0]) * static_cast<uint32_t>(outShape[1]) + 8 - 1)/8;
    // printf("Matrix Element: %d; Verify Element: %d\n",lenX, lenZ);

    if(IfFreeGM){
        // std::vector<void *> outDeviceAddrArr,
        // std::vector<uint8_t *> compDeviceZAddrArr
        aclDestroyTensor(outTensor);
        aclDestroyTensor(outTensorVerify);

        ACL_CHECK(aclrtFree(outDeviceAddr));
        ACL_CHECK(aclrtFree(outDeviceVerify));
        ACL_CHECK(aclrtFree(compDeviceZAddr));
        
    }

    // resize 到 repeat_time，避免越界
    outputHostArr.resize(repeat_time);
    compHostZArr.resize(repeat_time);


    for(uint id=0; id < repeat_time; id++){
        std::vector<float> outHostData(lenX, 0);
        std::vector<uint8_t> hostZ(lenZ, 0);

        outputHostArr[id] = outHostData;
        compHostZArr[id] = hostZ;
    }

    outDeviceAddr= nullptr;
    outTensor = nullptr;

    outDeviceVerify = nullptr;
    outTensorVerify = nullptr;

    compDeviceZAddr = nullptr;

    size_t sizeZ = lenZ * sizeof(uint8_t);
    auto ret = CreateAclTensor(outputHostArr[0], outShape, reinterpret_cast<void **>(&outDeviceAddr), aclDataType::ACL_FLOAT, &outTensor);
    ret = CreateAclTensor(outputHostArr[0], outShape, reinterpret_cast<void **>(&outDeviceVerify), aclDataType::ACL_FLOAT, &outTensorVerify);

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&compDeviceZAddr), sizeZ, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(compDeviceZAddr, sizeZ, compHostZArr[0].data(), sizeZ, ACL_MEMCPY_HOST_TO_DEVICE));    
}

// std::vector<std::vector<uint8_t>> & compHostZArr,
template<class CompareKernel, class CompareAdapter, class Element>
float RunDMREXP(CompareAdapter & compare_op, aclTensor * self, aclTensor * mat2,
    uint8_t * outDeviceAddr, uint8_t *outDeviceVerify,
    aclTensor * outTensor, aclTensor * outTensorVerify, std::vector<std::vector<Element>> & outputHostArr,
    uint8_t * compDeviceZAddr,std::vector<std::vector<uint8_t>> & compHostZArr,
    uint32_t repeat_time,
    GemvCoord testproblemShape, Element threshold, 
    uint32_t aicCoreNum,
    aclrtEvent &start, aclrtEvent &stop, aclrtStream & stream, bool CopyToHost, bool IfFreeGM, bool IsTest){

    int8_t cubeMathType = 0;
    // uint64_t workspaceSize = 0;
    // outputHostArr
    uint32_t UbNum = 2;
    size_t lenX = static_cast<size_t>(testproblemShape.m()) * testproblemShape.n();
    size_t lenZ = (lenX + 8 - 1) / 8;
    
    size_t sizeZ = lenZ * sizeof(uint8_t);
    
    float temp_time = 0;
    float time = 0;

    uint8_t * workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor * executor;
    // void *workspaceAddr = nullptr;
    for(uint32_t id = 0; id < repeat_time; id++){
        // printf("Round time: %u\n",id);
        if(id == 0 || (!IsTest)){
            workspaceSize = 0;
            auto ret = aclnnMatmulGetWorkspaceSize(self, mat2, outTensor, cubeMathType, &workspaceSize, &executor);

            if (workspaceSize > 0)
            {
                ret = aclrtMalloc(reinterpret_cast<void **>(&workspaceAddr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                //           return ret);
            }
            // printf("aclnn workspace size: %lu\n",workspaceSize);
            CHECK_ACL(aclrtSynchronizeStream(stream)); 
            CHECK_ACL(aclrtRecordEvent(start, stream));
        
            ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
            ret = aclrtSynchronizeStream(stream);
            CHECK_ACL(aclrtRecordEvent(stop, stream));
            CHECK_ACL(aclrtSynchronizeEvent(stop));
            CHECK_ACL(aclrtEventElapsedTime(&temp_time, start, stop));
            time += temp_time;
            ret = aclrtFree(workspaceAddr);
            // printf("Round %u matmual down!!!\n",id);
            if(CopyToHost){
                auto ret = aclrtMemcpy(outputHostArr[0].data(), lenX * sizeof(Element), outDeviceAddr,
                      lenX * sizeof(Element), ACL_MEMCPY_DEVICE_TO_HOST);
            }
            // printf("Copy Compute Data Down!!!\n");
        } else{

            workspaceSize = 0;
            auto ret = aclnnMatmulGetWorkspaceSize(self, mat2, outTensorVerify, cubeMathType, &workspaceSize, &executor);

            if (workspaceSize > 0)
            {
                ret = aclrtMalloc(reinterpret_cast<void **>(&workspaceAddr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            }

            CHECK_ACL(aclrtSynchronizeStream(stream)); 
            CHECK_ACL(aclrtRecordEvent(start, stream));
            ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
            ret = aclrtSynchronizeStream(stream);
            typename CompareKernel::Arguments arguments{testproblemShape, outDeviceAddr, 
                    outDeviceVerify, compDeviceZAddr, UbNum, false, threshold};
            RunAdapter(compare_op, arguments, stream, aicCoreNum);
            CHECK_ACL(aclrtRecordEvent(stop, stream));
            CHECK_ACL(aclrtSynchronizeEvent(stop));
            CHECK_ACL(aclrtEventElapsedTime(&temp_time, start, stop));
            time += temp_time;
            ret = aclrtFree(workspaceAddr); 
            // printf("compare test down!!!\n");
            ret = aclrtSynchronizeStream(stream);
            if(CopyToHost){
                std::vector<uint8_t> hostGolden(lenZ,255);
                std::vector<uint8_t> hostResult(lenZ,0);
                auto ret = aclrtMemcpy(outputHostArr[id].data(), lenX * sizeof(Element), outDeviceVerify,
                      lenX * sizeof(Element), ACL_MEMCPY_DEVICE_TO_HOST);
                
                
                // printf("start to get test result: %d\n",ret);
                // printf("%zu\n",hostResult.size());
                // printf("%zu\n",sizeZ);

                (aclrtMemcpy(compHostZArr[id-1].data(), sizeZ, compDeviceZAddr, 
                    sizeZ, ACL_MEMCPY_DEVICE_TO_HOST));
                // printf("copy Z down!!!\n");
            
                printf("first time: \n");
                for(int tmpi = 7; tmpi >=0; --tmpi){
                    // outputHostArr[0][tmpi]
                    printf("%f ", outputHostArr[0][tmpi]);
                }
                printf("\n");
                printf("Round {%u}: \n",(id+1));
                for(int tmpi = 7; tmpi >=0; tmpi--){
                    printf("%f ", outputHostArr[id][tmpi]);
                }
                printf("\n");
                std::cout<<"Verify Result"<<"("<<0<<"): " << std::bitset<8>(compHostZArr[id-1][0]) << std::endl;

                printf("first time: \n");
                for(int tmpi = lenX-1; tmpi >=lenX-8; tmpi--){
                    printf("%f ", outputHostArr[0][tmpi]);
                }
                printf("\n");

                printf("Round {%u}: \n",(id+1));
                for(int tmpi = lenX-1; tmpi >=lenX-8; tmpi--){
                    printf("%f ", outputHostArr[id][tmpi]);
                }
                printf("\n");
                std::cout<<"Verify Result"<<"("<<(lenZ-1)<<"): " << std::bitset<8>(compHostZArr[id-1][lenZ-1]) << std::endl;

                std::vector<uint64_t> errorIndices = golden::CompareData(compHostZArr[id-1], hostGolden, lenZ);

                printf("Method: XOR Time: %d \n",id);
                if (errorIndices.empty()) {
                    std::cout << "compare success." << std::endl;
                } else {
                    std::cerr << "compare failed. Error count: " << errorIndices.size() << std::endl;
                }
            }  
        }
    }
    
    if(IfFreeGM){
        // std::vector<void *> outDeviceAddrArr,
        // std::vector<uint8_t *> compDeviceZAddrArr
        aclDestroyTensor(outTensor);
        aclDestroyTensor(outTensorVerify);

        ACL_CHECK(aclrtFree(outDeviceAddr));
        ACL_CHECK(aclrtFree(outDeviceVerify));
        
        ACL_CHECK(aclrtFree(compDeviceZAddr));
    }

    return time; 
}

void Run(Options options){

    aclrtStream stream{nullptr};

    Init(options,&stream);

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();

    GemvCoord testproblemShape{m, n};

    uint32_t repeat_time = options.repeat;
    std::vector<int64_t> selfShape = {m, k};
    std::vector<int64_t> mat2Shape = {k, n};
    std::vector<int64_t> outShape = {m, n};

    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    
    size_t lenX = static_cast<size_t>(m) * n;
    size_t lenY = static_cast<size_t>(m) * n;
    size_t lenZ = (lenX + 8 -1)/8;

    size_t sizeZ = lenZ * sizeof(uint8_t);

    ScalarType threshold{0.0002f};

    aclrtEvent start, stop;

    ACL_CHECK(aclrtCreateEvent(&start));
    ACL_CHECK(aclrtCreateEvent(&stop));

    void *selfDeviceAddr = nullptr;
    void *mat2DeviceAddr = nullptr;

    uint8_t * outDeviceAddr = nullptr;
    aclTensor * outTensor = nullptr;

    uint8_t * outDeviceVerify = nullptr;
    aclTensor * outTensorVerify = nullptr;

    uint8_t * compDeviceZAddr = nullptr;

    aclTensor *self = nullptr;
    aclTensor *mat2 = nullptr;

    std::vector<float> selfHostData(lenA);
    std::vector<float> mat2HostData(lenB);
    std::vector<std::vector<float>> outputHostArr;
    std::vector<std::vector<uint8_t>> compHostZArr;

    for(uint id=0; id < repeat_time; id++){
        std::vector<float> outHostData(lenX, 0);
        std::vector<uint8_t> hostZ(lenZ, 0);

        outputHostArr.push_back(outHostData);
        compHostZArr.push_back(hostZ);
        // hostTestRes.push_back();
    }
    
    golden::FillRandomData(selfHostData,  -1.0f, 1.0f);
    golden::FillRandomData(mat2HostData,  -1.0f, 1.0f);

    std::cout << "this is selfhostdata " << (selfHostData[1]) << std::endl;
    std::cout << "this is mat2hostdata " << (mat2HostData[1]) << std::endl;

    // 创建self aclTensor
    auto ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    // CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建mat2 aclTensor
    ret = CreateAclTensor(mat2HostData, mat2Shape, &mat2DeviceAddr, aclDataType::ACL_FLOAT, &mat2);

    ret = CreateAclTensor(outputHostArr[0], outShape, reinterpret_cast<void **>(&outDeviceAddr), aclDataType::ACL_FLOAT, &outTensor);
    ret = CreateAclTensor(outputHostArr[0], outShape, reinterpret_cast<void **>(&outDeviceVerify), aclDataType::ACL_FLOAT, &outTensorVerify);
    // printf("%d\n",ret);

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&compDeviceZAddr), sizeZ, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(compDeviceZAddr, sizeZ, compHostZArr[0].data(), sizeZ, ACL_MEMCPY_HOST_TO_DEVICE));
    printf("size of CompHostZArr:%zu\n",compHostZArr[0].size());
    printf("size of CompHostZArr:%zu\n",compHostZArr[1].size());
    printf("output space: %zu\n", sizeZ);
    // CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 初始化out tensor
    // InitialRepeatData(outDeviceAddr, outDeviceVerify, outputHostArr,
    //     outTensor, outTensorVerify, compHostZArr,
    //     compDeviceZAddr, outShape, repeat_time, false);

    using UBTileShape = GemvShape<1,8192>;

    size_t sizeX = lenX * sizeof(float);
    size_t sizeY = lenY * sizeof(float);
    // size_t sizeZ = lenZ * sizeof(uint8_t);

    using LayoutX = layout::VectorLayout;
    using LayoutY = layout::VectorLayout;
    using LayoutZ = layout::VectorLayout;
    
    LayoutX layoutX{m*n};
    LayoutY layoutY{m*n};
    LayoutZ layoutZ{(m*n + 8 - 1)/8};
    
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::GemvAtlasA2;

    using XType = Gemm::GemmType<float, LayoutX>;
    using YType = Gemm::GemmType<float, LayoutY>;
    using ZType = Gemm::GemmType<uint8_t, LayoutZ>;
    using BiasType = void;
    
    using FT_COMP_TYPE = Catlass::Gemv::helper::FT_COMP_TYPE;

    using TileCopy = Gemv::Tile::TileCopyCompareAiv<FT_COMP_TYPE::XOR, typename DispatchPolicy::ArchTag, ZType, XType, YType, BiasType>;

    using CompareBlock = Gemv::Block::BlockCompare<DispatchPolicy,FT_COMP_TYPE::XOR, UBTileShape, ZType, XType, YType, TileCopy>;
    using BlockEpilogue = void;

    // kernel level
    using ElementWork = typename std::conditional<
        (CompareBlock::COMP_TYPE == FT_COMP_TYPE::XOR),
        uint16_t,
        typename std::conditional<(CompareBlock::COMP_TYPE == FT_COMP_TYPE::COMPARE), int32_t, float>::type>::type;

    size_t sizeW = RoundUp(static_cast<uint32_t>(lenX*sizeof(float)),static_cast<uint32_t>(sizeof(ElementWork)));
    size_t lenW = sizeW / sizeof(ElementWork);

    printf("workspace len: %zu\n", lenW);

    using CompareKernel = Gemv::Kernel::KernelCompareAiv<CompareBlock, BlockEpilogue>; 
    using CompareAdapter = Gemv::Device::DeviceGemv<CompareKernel>;
    CompareAdapter compare_op;
    // compare_op.CanImplement(arguments);

    int num_repeat = 10000;

    float temp_time = 0;
    float time = 0;
    float first_time = 0;

    /*
    1. 首先，测试功能
    */
    // first_time = RunDMREXP<CompareKernel,CompareAdapter,float>(compare_op, self, mat2,
    //     outDeviceAddrArr, outTensorArr, outputHostArr, compDeviceZAddrArr, compHostZArr, repeat_time,
    //     testproblemShape, threshold, aicCoreNum, start, stop, stream, true, true, true);
    // compHostZArr, 
    printf("Stage 1: start functional test!!!\n");
    first_time = RunDMREXP<CompareKernel,CompareAdapter,float>(compare_op, self, mat2,
        outDeviceAddr, outDeviceVerify, outTensor, outTensorVerify, outputHostArr,
        compDeviceZAddr, compHostZArr, repeat_time, testproblemShape, threshold, aicCoreNum,
        start, stop, stream, true, false, true);
    printf("Stage 2: finished functional test!!!\n");
    
    
    /*
    2. 进行warmup
    */

    // InitialRepeatData(outDeviceAddrArr, outputHostArr, outTensorArr, compHostZArr,
    //     compDeviceZAddrArr, outShape, repeat_time, false);
    printf("Stage 2: start warmup!!!\n");
    WarmupFunc(stream, 100, self, mat2, outTensor);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    printf("Stage 2: finished warmup!!!\n");

    

    /*
    3. 重复实验
    */

    // InitialRepeatData(outDeviceAddrArr, outputHostArr, outTensorArr, compHostZArr,
    //     compDeviceZAddrArr, outShape, repeat_time, true);
    // ACL_CHECK(aclrtSynchronizeStream(stream));

    for (int i = 0; i < num_repeat; ++i) {

        ACL_CHECK(aclrtSynchronizeStream(stream));
        temp_time = RunDMREXP<CompareKernel,CompareAdapter,float>(compare_op, self, mat2,
        outDeviceAddr, outDeviceVerify, outTensor, outTensorVerify, outputHostArr,
        compDeviceZAddr, compHostZArr, repeat_time, testproblemShape, threshold, aicCoreNum,
        start, stop, stream, false, false, true);
        time += temp_time;
        
        // if(i == num_repeat - 1){
        //     InitialRepeatData(outDeviceAddrArr, outputHostArr, outTensorArr, compHostZArr,
        //         compDeviceZAddrArr, outShape, 1, false);
        // }else{
        //     InitialRepeatData(outDeviceAddrArr, outputHostArr, outTensorArr, compHostZArr,
        //         compDeviceZAddrArr, outShape, repeat_time, false);
        // }
        
        ACL_CHECK(aclrtSynchronizeStream(stream));
    }

    std::cout << "m: " << m << ", n: " << n << ", k: " << k << ", " << (float)2 * m * n * k / (time / num_repeat * 1e-3) / 1e12 << " TFLOPS, " << (time / num_repeat) << " ms, repeat: " << num_repeat << std::endl;


    /*
    4. 对照，即不进行校验时的性能
    */

    aclDestroyTensor(outTensor);
    aclDestroyTensor(outTensorVerify);
    aclrtFree(outDeviceAddr);
    aclrtFree(outDeviceVerify);
    aclrtFree(compDeviceZAddr);

    ret = CreateAclTensor(outputHostArr[0], outShape, reinterpret_cast<void **>(&outDeviceAddr), aclDataType::ACL_FLOAT, &outTensor);
    ret = CreateAclTensor(outputHostArr[0], outShape, reinterpret_cast<void **>(&outDeviceVerify), aclDataType::ACL_FLOAT, &outTensorVerify);
    printf("%d\n",ret);

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&compDeviceZAddr), sizeZ, ACL_MEM_MALLOC_HUGE_FIRST));

    temp_time = 0;
    time = 0;

    for (int i = 0; i < num_repeat; ++i) {

        ACL_CHECK(aclrtSynchronizeStream(stream));
        temp_time = RunDMREXP<CompareKernel,CompareAdapter,float>(compare_op, self, mat2,
        outDeviceAddr, outDeviceVerify, outTensor, outTensorVerify, outputHostArr,
        compDeviceZAddr, compHostZArr, 1, testproblemShape, threshold, aicCoreNum,
        start, stop, stream, false, false, false);
        time += temp_time;
        
        // if(i < num_repeat - 1){
        //     InitialRepeatData(outDeviceAddrArr, outputHostArr, outTensorArr, compHostZArr,
        //     compDeviceZAddrArr, outShape, 1, false);
        // }
        ACL_CHECK(aclrtSynchronizeStream(stream));
    }

    std::cout << "m: " << m << ", n: " << n << ", k: " << k << ", " << (float)2 * m * n * k / (time / num_repeat * 1e-3) / 1e12 << " TFLOPS, " << (time / num_repeat) << " ms, repeat: " << num_repeat << std::endl;

    /*
    最终，销毁全部的tensor与内存占用，stream等
    */
    aclDestroyTensor(self);
    aclDestroyTensor(mat2);
    aclDestroyTensor(outTensor);
    aclDestroyTensor(outTensorVerify);
    // aclDestroyTensor(out);


    aclrtFree(selfDeviceAddr);
    aclrtFree(mat2DeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(outDeviceVerify);
    aclrtFree(compDeviceZAddr);
    // aclrtFree(outDeviceAddr);

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