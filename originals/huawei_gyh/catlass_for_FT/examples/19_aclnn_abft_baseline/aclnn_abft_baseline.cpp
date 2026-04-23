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
#include "catlass/gemv/block/block_gemv.hpp"
#include "catlass/gemv/block/block_large_local_compare.hpp"
#include "catlass/gemv/kernel/kernel_gemv_FT_double_total_aiv.hpp"

// #include "catlass/gemv/kernel/kernel_gemv_aic_FT.hpp"
// #include "catlass/gemv/kernel/kernel_gemv_FT_double.hpp"
#include "catlass/gemv/tile/tile_copy.hpp"
#include "catlass/gemv/helper.hpp"

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

#include "catlass/status.hpp"
#include "gemm/device/device_gemm.hpp" // catlass/

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
    uint32_t repeat = 1;
    uint32_t deviceId{0};

    Options() = default;
    
    GemmCoord problemGemmShape{M, N, K};
    GemvCoord problemShape{M, N};

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
        problemGemmShape.m() = std::atoi(argv[M_INDEX]);
        problemGemmShape.n() = std::atoi(argv[N_INDEX]);
        problemGemmShape.k() = std::atoi(argv[K_INDEX]);

        problemShape.m() = std::atoi(argv[M_INDEX]);
        problemShape.n() = std::atoi(argv[N_INDEX]);

        repeat = std::atoi(argv[REPEAT_INDEX]);
        printf("Repeat Time for ABFT: %d\n",repeat);

        if (argc == ARGS_MAX)
        {
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        }
        return 0;
    }
};

// template <class Adapter>
// void RunAdapter(Adapter gemv_op, typename Adapter::Arguments args, aclrtStream stream, uint32_t aicCoreNum)
// {
//     size_t sizeWorkspace = gemv_op.GetWorkspaceSize(args);
//     uint8_t *deviceWorkspace = nullptr;
//     if (sizeWorkspace > 0) {
//         ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
//     }
//     gemv_op.Initialize(args, deviceWorkspace);
//     printf("Initialized ABFT OP!!! aicCore: %d workspace: %zu\n", aicCoreNum,sizeWorkspace);
//     gemv_op(stream, aicCoreNum);
//     ACL_CHECK(aclrtSynchronizeStream(stream));
//     printf("ABFT OP Done!!! ");

//     // if(sizeWorkspace > 0 && args.OutputWorkspace){
//     //     ACL_CHECK(aclrtMemcpy((*hostWorkspace).data(), sizeWorkspace, deviceWorkspace, sizeWorkspace, ACL_MEMCPY_DEVICE_TO_HOST));
//     // }

//     if (sizeWorkspace > 0) {
//         ACL_CHECK(aclrtFree(deviceWorkspace));
//     }
// }

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
    printf("Initialized ABFT OP!!! aicCore: %d workspace: %zu\n", aicCoreNum,sizeWorkspace);
    // printf("Initialized!!!!\n");
    gemv_op(stream, aicCoreNum, fftsAddr);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    printf("ABFT OP Done!!! ");
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

// void InitialRepeatData(uint8_t* outDeviceAddr, uint8_t* outDeviceVerify,
//     std::vector<std::vector<float>> & outputHostArr,
//     aclTensor * outTensor, aclTensor * outTensorVerify,
//     std::vector<std::vector<uint8_t>> & compHostZArr,
//     uint8_t * compDeviceZRowAddr,
//     std::vector<int64_t> outShape, uint32_t repeat_time,bool IfFreeGM)
// {
//     uint32_t lenX = static_cast<uint32_t>(outShape[0]) * static_cast<uint32_t>(outShape[1]);
//     uint32_t lenCOMPRow = (static_cast<uint32_t>(outShape[0]) * static_cast<uint32_t>(outShape[1]) + 8 - 1)/8;
//     // printf("Matrix Element: %d; Verify Element: %d\n",lenX, lenZ);

//     if(IfFreeGM){
//         // std::vector<void *> outDeviceAddrArr,
//         // std::vector<uint8_t *> compDeviceZRowAddrArr
//         aclDestroyTensor(outTensor);
//         aclDestroyTensor(outTensorVerify);

//         ACL_CHECK(aclrtFree(outDeviceAddr));
//         ACL_CHECK(aclrtFree(outDeviceVerify));
//         ACL_CHECK(aclrtFree(compDeviceZRowAddr));
        
//     }

//     // resize 到 repeat_time，避免越界
//     outputHostArr.resize(repeat_time);
//     compHostZArr.resize(repeat_time);


//     for(uint id=0; id < repeat_time; id++){
//         std::vector<float> outHostData(lenX, 0);
//         std::vector<uint8_t> hostZ(lenZ, 0);

//         outputHostArr[id] = outHostData;
//         compHostZArr[id] = hostZ;
//     }

//     outDeviceAddr= nullptr;
//     outTensor = nullptr;

//     outDeviceVerify = nullptr;
//     outTensorVerify = nullptr;

//     compDeviceZRowAddr = nullptr;

//     size_t sizeCOMPRow = lenCOMPRow * sizeof(uint8_t);
//     auto ret = CreateAclTensor(outputHostArr[0], outShape, reinterpret_cast<void **>(&outDeviceAddr), aclDataType::ACL_FLOAT, &outTensor);
//     ret = CreateAclTensor(outputHostArr[0], outShape, reinterpret_cast<void **>(&outDeviceVerify), aclDataType::ACL_FLOAT, &outTensorVerify);

//     ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&compDeviceZRowAddr), sizeCOMPRow, ACL_MEM_MALLOC_HUGE_FIRST));
//     ACL_CHECK(aclrtMemcpy(compDeviceZRowAddr, sizeCOMPRow, compHostZArr[0].data(), sizeCOMPRow, ACL_MEMCPY_HOST_TO_DEVICE));    
// }

// std::vector<std::vector<uint8_t>> & compHostZArr,
// GemvAdapter GemvKernel


using FT_COMP_TYPE = Catlass::Gemv::helper::FT_COMP_TYPE;
using FT_ENC_TYPE = Catlass::Gemv::helper::FT_ENC_TYPE;

template<class GemvKernel, class GemvAdapter, class Element, class ElementSum, 
    class AType, class BType, class CType>
float RunABFTEXP(Options const &options, GemvAdapter & gemv_op, aclTensor * self, aclTensor * mat2,
    uint8_t * selfDeviceAddr, uint8_t * mat2DeviceAddr,
    uint8_t * outDeviceAddr, aclTensor * outTensor,
    uint8_t * deviceX,
    uint8_t * deviceZRow, uint8_t * deviceZCol,
    uint8_t * deviceDRow, uint8_t * deviceDCol,
    std::vector<std::vector<Element>> & outputHostArr,
    std::vector<std::vector<ElementSum>> & outZRowHostArr,
    std::vector<std::vector<ElementSum>> & outZColHostArr,
    std::vector<std::vector<ElementSum>> & outDRowHostArr,
    std::vector<std::vector<ElementSum>> & outDColHostArr,
    uint8_t * compDeviceZRowAddr, uint8_t * compDeviceZColAddr, 
    std::vector<std::vector<uint8_t>> & compHostZRowArr, 
    std::vector<std::vector<uint8_t>> & compHostZColArr,
    uint32_t repeat_time,
    GemmCoord testproblemGemmShape,
    GemvCoord testproblemShape,
    ElementSum threshold, 
    uint32_t aicCoreNum,
    aclrtEvent &start, aclrtEvent &stop, 
    aclrtStream & stream, 
    bool CopyToHost, bool IfFreeGM, bool IsTest, 
    FT_ENC_TYPE enc_type, uint64_t & fftsAddr){

    // 

    int8_t cubeMathType = 0;
    // uint64_t workspaceSize = 0;
    // outputHostArr
    uint32_t UbNum = 1;
    size_t lenC = static_cast<size_t>(testproblemGemmShape.m()) * testproblemGemmShape.n();
    size_t lenRow = static_cast<size_t>(testproblemGemmShape.m());
    size_t lenCol = static_cast<size_t>(testproblemGemmShape.n());

    size_t lenCOMPRow = (static_cast<size_t>(testproblemGemmShape.m()) + 8 - 1) / 8;
    size_t lenCOMPCol = (static_cast<size_t>(testproblemGemmShape.n()) + 8 - 1) / 8;
    
    size_t sizeCOMPRow = lenCOMPRow * sizeof(uint8_t);
    size_t sizeCOMPCol = lenCOMPCol * sizeof(uint8_t);
    
    float temp_time = 0;
    float time = 0;

    uint8_t * workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;

    
    printf("aclnn compute start!!!\n");
    for(uint32_t id = 0; id < repeat_time; id++){
        workspaceSize = 0;
        aclOpExecutor * executor;
        
        auto ret = aclnnMatmulGetWorkspaceSize(self, mat2, outTensor, cubeMathType, &workspaceSize, &executor);

        if (workspaceSize > 0)
        {
            ret = aclrtMalloc(reinterpret_cast<void **>(&workspaceAddr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            //           return ret);
        }
        CHECK_ACL(aclrtSynchronizeStream(stream)); 
        CHECK_ACL(aclrtRecordEvent(start, stream));
        ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
        ret = aclrtSynchronizeStream(stream);
        if (workspaceSize > 0)
        {
            ret = aclrtFree(workspaceAddr);
            // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("free workspace failed. ERROR: %d\n", ret);
            //           return ret);
        }
        // aclnnMatmulDestroy(executor);
        printf("aclnn compute down");
        if(IsTest){
            typename GemvKernel::Arguments arguments{testproblemGemmShape, testproblemShape, sizeof(Element), 
                deviceX, selfDeviceAddr, mat2DeviceAddr, outDeviceAddr, 
                deviceZRow, deviceZCol, deviceDRow, deviceDCol, 
                compDeviceZRowAddr, compDeviceZColAddr,
                FT_ENC_TYPE::BOTHC,UbNum,false,threshold};

            RunAdapter(gemv_op, arguments, stream, aicCoreNum, fftsAddr);
            // RunAdapter(gemv_op, arguments, stream, aicCoreNum);
        }
        CHECK_ACL(aclrtRecordEvent(stop, stream));
        CHECK_ACL(aclrtSynchronizeEvent(stop));
        CHECK_ACL(aclrtEventElapsedTime(&temp_time, start, stop));
        time += temp_time;

        

        ret = aclrtSynchronizeStream(stream);

        if(CopyToHost){
            auto ret = aclrtMemcpy(outputHostArr[id].data(), lenC * sizeof(Element), outDeviceAddr,
                      lenC * sizeof(Element), ACL_MEMCPY_DEVICE_TO_HOST);

            ret = aclrtMemcpy(outZRowHostArr[id].data(), lenRow * sizeof(ElementSum), deviceZRow,
                      lenRow * sizeof(ElementSum), ACL_MEMCPY_DEVICE_TO_HOST);
            ret = aclrtMemcpy(outDRowHostArr[id].data(), lenRow * sizeof(ElementSum), deviceDRow,
                      lenRow * sizeof(ElementSum), ACL_MEMCPY_DEVICE_TO_HOST);

            ret = aclrtMemcpy(outZColHostArr[id].data(), lenCol * sizeof(ElementSum), deviceZCol,
                      lenCol * sizeof(ElementSum), ACL_MEMCPY_DEVICE_TO_HOST);
            ret = aclrtMemcpy(outDColHostArr[id].data(), lenCol * sizeof(ElementSum), deviceDCol,
                      lenCol * sizeof(ElementSum), ACL_MEMCPY_DEVICE_TO_HOST);
            
            ret = aclrtMemcpy(compHostZRowArr[id].data(), lenCOMPRow * sizeof(uint8_t), compDeviceZRowAddr,
                      lenCOMPRow * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
            ret = aclrtMemcpy(compHostZColArr[id].data(), lenCOMPCol * sizeof(uint8_t), compDeviceZColAddr,
                      lenCOMPCol * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);

            std::vector<uint8_t> hostGoldenCOMPRow(lenCOMPRow,255);
            std::vector<uint8_t> hostGoldenCOMPCol(lenCOMPCol,255);

            // printf("start to get test result: %d\n",ret);
            if(enc_type == FT_ENC_TYPE::BOTHC || enc_type == FT_ENC_TYPE::CE){
                printf("\t\t***Row Sum Check***:\t\t\n");
                printf("CE: \t");
                for(int tmpi = 15; tmpi >=0; --tmpi){
                    // outputHostArr[0][tmpi]
                    printf("%f ", outZRowHostArr[id][tmpi]);
                }
                printf("\n");
                printf("ABE:\t");
                for(int tmpi = 15; tmpi >=0; tmpi--){
                    printf("%f ", outDRowHostArr[id][tmpi]);
                }
                printf("\n");
                std::cout<<"Verify Result: " <<std::bitset<8>(compHostZRowArr[id][1])<<std::bitset<8>(compHostZRowArr[id][0]) << std::endl;
            }
            if(enc_type == FT_ENC_TYPE::BOTHC || enc_type == FT_ENC_TYPE::ETC){
                printf("\t\t***Column Sum Check***:\t\t\n");
                printf("ETC: \t");
                for(int tmpi = 15; tmpi >= 0; tmpi--){
                    printf("%f ", outZColHostArr[id][tmpi]);
                }
                printf("\n");

                printf("ETAB:\t");
                for(int tmpi = 15; tmpi >= 0; tmpi--){
                    printf("%f ", outZColHostArr[id][tmpi]);
                }
                printf("\n");
                std::cout<<"Verify Result: " << std::bitset<8>(compHostZColArr[id][1]) << std::bitset<8>(compHostZColArr[id][0]) << std::endl;
            }
            
            if(enc_type == FT_ENC_TYPE::BOTHC || enc_type == FT_ENC_TYPE::CE){
                std::vector<uint64_t> errorIndices = golden::CompareData(compHostZRowArr[id], hostGoldenCOMPRow, lenCOMPRow);
                printf("Method: SUB Row Sum \n");
                if (errorIndices.empty()) {
                    std::cout << "compare success." << std::endl;
                } else {
                    std::cerr << "compare failed. Error count: " << errorIndices.size() << std::endl;
                }
            }

            if(enc_type == FT_ENC_TYPE::BOTHC || enc_type == FT_ENC_TYPE::ETC){
                std::vector<uint64_t> errorIndices = golden::CompareData(compHostZColArr[id], hostGoldenCOMPCol, lenCOMPCol);
                printf("Method: SUB Column Sum \n");
                if (errorIndices.empty()) {
                    std::cout << "compare success." << std::endl;
                } else {
                    std::cerr << "compare failed. Error count: " << errorIndices.size() << std::endl;
                }
            }
        }
    }
    
    if(IfFreeGM){
        aclDestroyTensor(outTensor);
        ACL_CHECK(aclrtFree(outDeviceAddr));

        ACL_CHECK(aclrtFree(deviceDCol));
        ACL_CHECK(aclrtFree(deviceDRow));
        ACL_CHECK(aclrtFree(deviceZCol));
        ACL_CHECK(aclrtFree(deviceZRow));
        
        ACL_CHECK(aclrtFree(compDeviceZRowAddr));
        ACL_CHECK(aclrtFree(compDeviceZColAddr));
    }

    return time; 
}

void Run(Options options){

    aclrtStream stream{nullptr};

    Init(options,&stream);

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemGemmShape.k();

    GemmCoord testproblemGemmShape{m,n,k};
    GemvCoord testproblemShape{m,n};

    GemvCoord problemShapeCol{n, m};

    GemvCoord problemShapeBE{k,n};
    GemvCoord problemShapeABE{m,k};

    GemvCoord problemShapeETA{k,m};
    GemvCoord problemShapeETAB{n,k};

    uint32_t repeat_time = options.repeat;
    std::vector<int64_t> selfShape = {m, k};
    std::vector<int64_t> mat2Shape = {k, n};
    std::vector<int64_t> outShape = {m, n};

    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenC = static_cast<size_t>(n) * m;

    size_t lenX = (static_cast<size_t>(m) + static_cast<size_t>(n)) * 1;

    size_t lenDRow = (static_cast<size_t>(m)) * 1;
    size_t lenDCol = (static_cast<size_t>(n)) * 1;

    size_t lenCOMPCol = ((static_cast<size_t>(n)) + 8 - 1) / 8;
    size_t lenCOMPRow = ((static_cast<size_t>(m)) + 8 - 1) / 8;

    size_t lenZRow = static_cast<size_t>(m) * 1;
    size_t lenZCol = static_cast<size_t>(n) * 1;

    size_t sizeA = lenA * sizeof(float);
    size_t sizeB = lenB * sizeof(float);
    size_t sizeC = lenC * sizeof(float);

    size_t sizeX = lenX * sizeof(float);

    size_t sizeZRow = lenZRow * sizeof(float);
    size_t sizeDRow = lenDRow * sizeof(float);

    size_t sizeCOMPRow = lenCOMPRow * sizeof(uint8_t);
    size_t sizeCOMPCol = lenCOMPCol * sizeof(uint8_t);

    size_t sizeZCol = lenZCol * sizeof(float);
    size_t sizeDCol = lenDCol * sizeof(float);

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

    ScalarType alpha{1.0};
    ScalarType beta{0.0};

    float sum_base = 1.0;

    std::vector<float> hostX(lenX,sum_base);


    ScalarType threshold{0.0002f};

    aclrtEvent start, stop;

    ACL_CHECK(aclrtCreateEvent(&start));
    ACL_CHECK(aclrtCreateEvent(&stop));

    uint8_t *selfDeviceAddr = nullptr;
    uint8_t *mat2DeviceAddr = nullptr;

    uint8_t * outDeviceAddr = nullptr;
    aclTensor * outTensor = nullptr;

    uint8_t * compDeviceZRowAddr = nullptr;
    uint8_t * compDeviceZColAddr = nullptr;

    aclTensor *self = nullptr;
    aclTensor *mat2 = nullptr;

    std::vector<float> selfHostData(lenA);
    std::vector<float> mat2HostData(lenB);
    std::vector<std::vector<float>> outputHostArr;

    std::vector<std::vector<uint8_t>> compHostZRowArr;
    std::vector<std::vector<uint8_t>> compHostZColArr;

    std::vector<std::vector<float>> outZRowHostArr;
    std::vector<std::vector<float>> outZColHostArr;

    std::vector<std::vector<float>> outDRowHostArr;
    std::vector<std::vector<float>> outDColHostArr;

    for(uint id=0; id < repeat_time; id++){
        
        std::vector<float> hostDRow(lenZRow,0.0f);
        std::vector<float> hostDCol(lenZCol,0.0f);

        std::vector<float> hostZRow(lenZRow,0.0f);
        std::vector<float> hostZCol(lenZCol,0.0f);
    
        std::vector<float> outHostData(lenC, 0);

        std::vector<uint8_t> hostCOMPRow(lenCOMPRow,0);
        std::vector<uint8_t> hostCOMPCol(lenCOMPCol,0);

        outputHostArr.push_back(outHostData);
        compHostZRowArr.push_back(hostCOMPRow);
        compHostZColArr.push_back(hostCOMPCol);

        outZRowHostArr.push_back(hostZRow);
        outZColHostArr.push_back(hostZCol);

        outDRowHostArr.push_back(hostDRow);
        outDColHostArr.push_back(hostDCol);
    }

    golden::FillRandomData(selfHostData,  -1.0f, 1.0f);
    golden::FillRandomData(mat2HostData,  -1.0f, 1.0f);

    std::cout << "this is selfhostdata " << (selfHostData[1]) << std::endl;
    std::cout << "this is mat2hostdata " << (mat2HostData[1]) << std::endl;

    // 创建self aclTensor
    auto ret = CreateAclTensor(selfHostData, selfShape, reinterpret_cast<void **>(&selfDeviceAddr), aclDataType::ACL_FLOAT, &self);
    // CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建mat2 aclTensor
    ret = CreateAclTensor(mat2HostData, mat2Shape, reinterpret_cast<void **>(&mat2DeviceAddr), aclDataType::ACL_FLOAT, &mat2);

    ret = CreateAclTensor(outputHostArr[0], outShape, reinterpret_cast<void **>(&outDeviceAddr), aclDataType::ACL_FLOAT, &outTensor);

    uint8_t* deviceX{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceX), sizeX, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceX, sizeX, hostX.data(), sizeX, ACL_MEMCPY_HOST_TO_DEVICE));


    uint8_t* deviceZRow{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceZRow), sizeZRow, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceZRow, sizeZRow, outZRowHostArr[0].data(), sizeZRow, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceZCol{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceZCol), sizeZCol, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceZCol, sizeZCol, outZColHostArr[0].data(), sizeZCol, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceDRow{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceDRow), sizeZRow, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceDRow, sizeZRow, outDRowHostArr[0].data(), sizeZRow, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceDCol{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceDCol), sizeZCol, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceDCol, sizeZCol, outDColHostArr[0].data(), sizeZCol, ACL_MEMCPY_HOST_TO_DEVICE));
    // printf("%d\n",ret);

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&compDeviceZRowAddr), sizeCOMPRow, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(compDeviceZRowAddr, sizeCOMPRow, compHostZRowArr[0].data(), sizeCOMPRow, ACL_MEMCPY_HOST_TO_DEVICE));

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&compDeviceZColAddr), sizeCOMPCol, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(compDeviceZColAddr, sizeCOMPCol, compHostZColArr[0].data(), sizeCOMPCol, ACL_MEMCPY_HOST_TO_DEVICE));
    printf("size of CompHostZRowArr:%zu\n",compHostZRowArr[0].size());
    printf("size of CompHostZColArr:%zu\n",compHostZColArr[0].size());
    printf("output space: %zu\n", (sizeCOMPRow+sizeCOMPCol));

    

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();

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

    using AType = Gemm::GemmType<float, LayoutA>;
    using BType = Gemm::GemmType<float, LayoutB>;
    using CType = Gemm::GemmType<float, LayoutC>;
    using XType = Gemm::GemmType<float, LayoutX>;
    using MYType = Gemm::GemmType<float, LayoutMY>;
    using BiasType = void;
   
    
    using ZType = Gemm::GemmType<float, LayoutZ>;

    constexpr uint32_t computeLength = 8192;

    using GemvDispatchPolicy = Gemm::GemvAtlasA2;
    using COMPDispatchPolicy = Gemm::GemvAtlasA2;

    using YType = Gemm::GemmType<float, LayoutY>;
    using TileVmuls = Gemv::Tile::TileVmuls<ArchTag, XType>;

    using UBTileShape = GemmShape<L1TileShapeAB::M, L1TileShapeAB::N, L1TileShapeAB::K>;
    using TileFaultCopy = Gemv::Tile::TileCopyGemvAiv<ArchTag, AType, XType, YType>;
    using TileFaultVmad = Gemv::Tile::TileVmad<ArchTag, AType, XType, YType>;
    using TileFaultSum = Gemv::Tile::TileFaultSum<ArchTag, AType, YType>;
    using BlockSumGemv = Gemv::Block::BlockSumGemv<GemvDispatchPolicy, UBTileShape, AType, XType, YType, void, TileFaultCopy, TileFaultVmad, TileVmuls>;

    using COMPZType = Gemm::GemmType<uint8_t, LayoutZ>;
    
    using UBTileShapeCOMP = GemvShape<1,1024>;
    using TileCompareCopy = Gemv::Tile::TileCopyCompareAiv<FT_COMP_TYPE::SUB, typename COMPDispatchPolicy::ArchTag, COMPZType, ZType, ZType, void>;
    using CompareBlock = Gemv::Block::BlockCompare<COMPDispatchPolicy,FT_COMP_TYPE::SUB, UBTileShapeCOMP, COMPZType, ZType, ZType, TileCompareCopy>;

    // kernle levels
    /*
    template<
    class BlockGemv_,
    class BlockSumGemv_,
    class BlockCompare_,
    class BlockEpilogue_
    >
    */
    // BlockAICGemv,
    using GemvKernel = Gemv::Kernel::KernelGemvFTDoubleTotalAiv<BlockSumGemv,CompareBlock>;

    // TODO:  use adapter to activate the kernel
    using GemvAdapter = Gemv::Device::DeviceGemv<GemvKernel>;
    // ScalarType threshold{0.0002f};

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));
    printf("%lu\n",fftsAddr);
    printf("%u\n",fftsLen);

    GemvAdapter gemv_op;

    
    // compare_op.CanImplement(arguments);

    int num_repeat = 10000;

    float temp_time = 0;
    float time = 0;
    float first_time = 0;

    /*
    1. 首先，测试功能
    */
    // first_time = RunABFTEXP<CompareKernel,GemvAdapter,float>(compare_op, self, mat2,
    //     outDeviceAddrArr, outTensorArr, outputHostArr, compDeviceZRowAddrArr, compHostZArr, repeat_time,
    //     testproblemGemmShape, threshold, aicCoreNum, start, stop, stream, true, true, true);
    // compHostZArr, 
    printf("Stage 1: start functional test!!!\n");
    // template<class GemvKernel, class GemvAdapter, class Element, class ElementSum, 
    // class AType, class BType, class CType>
    first_time = RunABFTEXP<GemvKernel, GemvAdapter,float,float, AType, BType, CType>(
        options, gemv_op, self, mat2, selfDeviceAddr, mat2DeviceAddr,
        outDeviceAddr, outTensor, deviceX, deviceZRow, deviceZCol,
        deviceDRow, deviceDCol, outputHostArr, outZRowHostArr, outZColHostArr, 
        outDRowHostArr, outDColHostArr, compDeviceZRowAddr, compDeviceZColAddr, 
        compHostZRowArr, compHostZColArr, repeat_time, testproblemGemmShape,
        testproblemShape, threshold, aicCoreNum, start, stop, stream, true, false, true, 
        FT_ENC_TYPE::BOTHC, fftsAddr);
        // , fftsAddr
    printf("Stage 1: finished functional test!!!\n");
    
    
    /*
    2. 进行warmup
    */

    // InitialRepeatData(outDeviceAddrArr, outputHostArr, outTensorArr, compHostZArr,
    //     compDeviceZRowAddrArr, outShape, repeat_time, false);
    // printf("Stage 2: start warmup!!!\n");
    // WarmupFunc(stream, 100, self, mat2, outTensor);
    // ACL_CHECK(aclrtSynchronizeStream(stream));
    // printf("Stage 2: finished warmup!!!\n");

    

    /*
    3. 重复实验
    */

    // InitialRepeatData(outDeviceAddrArr, outputHostArr, outTensorArr, compHostZArr,
    //     compDeviceZRowAddrArr, outShape, repeat_time, true);
    // ACL_CHECK(aclrtSynchronizeStream(stream));

    // for (int i = 0; i < num_repeat; ++i) {

    //     ACL_CHECK(aclrtSynchronizeStream(stream));
    //     temp_time = RunABFTEXP<GemvKernel, GemvAdapter,float,float>(gemv_op, self, mat2,
    //         selfDeviceAddr, mat2DeviceAddr, outDeviceAddr, outTensor, deviceX, deviceXV, 
    //         deviceZRow, deviceZCol, deviceDRow, deviceDCol, outputHostArr,
    //         outZRowHostArr, outZColHostArr, outDRowHostArr, outDColHostArr,
    //         compDeviceZRowAddr, compDeviceZColAddr, compHostZRowArr, compHostZColArr, repeat_time,
    //         testproblemGemmShape, testproblemShape, threshold, aicCoreNum, start, stop, stream, 
    //         false, false, false, FT_ENC_TYPE::BOTHC);
    //     time += temp_time;
        
    //     // if(i == num_repeat - 1){
    //     //     InitialRepeatData(outDeviceAddrArr, outputHostArr, outTensorArr, compHostZArr,
    //     //         compDeviceZRowAddrArr, outShape, 1, false);
    //     // }else{
    //     //     InitialRepeatData(outDeviceAddrArr, outputHostArr, outTensorArr, compHostZArr,
    //     //         compDeviceZRowAddrArr, outShape, repeat_time, false);
    //     // }
        
    //     ACL_CHECK(aclrtSynchronizeStream(stream));
    // }

    // std::cout << "m: " << m << ", n: " << n << ", k: " << k << ", " << (float)2 * m * n * k / (time / num_repeat * 1e-3) / 1e12 << " TFLOPS, " << (time / num_repeat) << " ms, repeat: " << num_repeat << std::endl;


    /*
    4. 对照，即不进行校验时的性能
    */

    aclDestroyTensor(outTensor);
    aclrtFree(outDeviceAddr);

    aclrtFree(compDeviceZRowAddr);
    aclrtFree(compDeviceZColAddr);

    aclrtFree(deviceZCol);
    aclrtFree(deviceDCol);
    aclrtFree(deviceZRow);
    aclrtFree(deviceDRow);
    
    std::vector<float> outputHostInit(lenC, 0.0);

    ret = CreateAclTensor(outputHostInit, outShape, reinterpret_cast<void **>(&outDeviceAddr), aclDataType::ACL_FLOAT, &outTensor);

    printf("%d\n",ret);

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&compDeviceZRowAddr), sizeCOMPRow, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&compDeviceZColAddr), sizeCOMPRow, ACL_MEM_MALLOC_HUGE_FIRST));

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceZRow), sizeZRow, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceZCol), sizeZCol, ACL_MEM_MALLOC_HUGE_FIRST));

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceDRow), sizeZRow, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceDCol), sizeZCol, ACL_MEM_MALLOC_HUGE_FIRST));

    temp_time = 0;
    time = 0;

    // for (int i = 0; i < num_repeat; ++i) {

    //     ACL_CHECK(aclrtSynchronizeStream(stream));
    //     temp_time = RunABFTEXP<GemvKernel, GemvAdapter,float,float>(gemv_op, self, mat2,
    //         selfDeviceAddr, mat2DeviceAddr, outDeviceAddr, outTensor, deviceX, deviceXV, 
    //         deviceZRow, deviceZCol, deviceDRow, deviceDCol, outputHostArr,
    //         outZRowHostArr, outZColHostArr, outDRowHostArr, outDColHostArr,
    //         compDeviceZRowAddr, compDeviceZColAddr, compHostZRowArr, compHostZColArr, repeat_time,
    //         testproblemGemmShape, testproblemShape, threshold, aicCoreNum, start, stop, stream, 
    //         true, false, false, FT_ENC_TYPE::BOTHC);
    //     time += temp_time;
        
    //     // if(i < num_repeat - 1){
    //     //     InitialRepeatData(outDeviceAddrArr, outputHostArr, outTensorArr, compHostZArr,
    //     //     compDeviceZRowAddrArr, outShape, 1, false);
    //     // }
    //     ACL_CHECK(aclrtSynchronizeStream(stream));
    // }

    // std::cout << "m: " << m << ", n: " << n << ", k: " << k << ", " << (float)2 * m * n * k / (time / num_repeat * 1e-3) / 1e12 << " TFLOPS, " << (time / num_repeat) << " ms, repeat: " << num_repeat << std::endl;

    /*
    最终，销毁全部的tensor与内存占用，stream等
    */
    aclDestroyTensor(self);
    aclDestroyTensor(mat2);
    aclDestroyTensor(outTensor);
    // aclDestroyTensor(out);


    aclrtFree(selfDeviceAddr);
    aclrtFree(mat2DeviceAddr);
    aclrtFree(outDeviceAddr);

    aclrtFree(compDeviceZRowAddr);
    aclrtFree(compDeviceZColAddr);

    aclrtFree(deviceZCol);
    aclrtFree(deviceDCol);
    aclrtFree(deviceZRow);
    aclrtFree(deviceDRow);

    aclrtFree(deviceX);
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