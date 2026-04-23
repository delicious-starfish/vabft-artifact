/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_ARCH_CROSS_CORE_SYNC_HPP
#define CATLASS_ARCH_CROSS_CORE_SYNC_HPP

#include "catlass/catlass.hpp"

namespace Catlass::Arch {

/*
因为同一flagId的计数器最多设置16次，从0开始到15，加上翻转，所以这里设置为14
*/
constexpr uint32_t MAX_REVERSE_DEPTH = 14;

using FlagID = uint16_t;
constexpr FlagID AIV_INTER_BLOCK_BARRIER = 8;
constexpr FlagID AIC_INTER_BLOCK_BARRIER = 9;
constexpr FlagID AIV_INTER_SUBBLOCK_BARRIER = 10;
constexpr FlagID FFTS_MAX_FLAG = 7;

struct CrossCoreFlag {
    CATLASS_DEVICE
    CrossCoreFlag() : id(0) {}

    CATLASS_DEVICE
    CrossCoreFlag(FlagID id) : id(id) {}

    FlagID id;
};

template <uint32_t REVERSE_DEPTH_ = MAX_REVERSE_DEPTH>
struct CrossCoreFlagWithReverse {
    CATLASS_DEVICE
    CrossCoreFlagWithReverse() : id(0), reverseId(0) {}

    CATLASS_DEVICE
    CrossCoreFlagWithReverse(FlagID id, FlagID reverseId) : id(id), reverseId(reverseId) {}

    FlagID id;
    FlagID reverseId;
    uint32_t count{ 0 };
};

template <uint8_t MODE, int32_t CORE_TYPE>
struct BarrierFlag {
    static_assert(MODE != MODE, "Unsupported cross core barrier flag, can not find the specialization.");
};

template <>
struct BarrierFlag<0x0, AscendC::AIV> {
    static constexpr FlagID ID = AIV_INTER_BLOCK_BARRIER;
};

template <>
struct BarrierFlag<0x0, AscendC::AIC> {
    static constexpr FlagID ID = AIC_INTER_BLOCK_BARRIER;
};

template <>
struct BarrierFlag<0x1, AscendC::AIV> {
    static constexpr FlagID ID = AIV_INTER_SUBBLOCK_BARRIER;
};

template <uint8_t MODE, pipe_t PIPE>
CATLASS_DEVICE
void CrossCoreBarrier()
{
    constexpr FlagID flagId = BarrierFlag<MODE, g_coreType>::ID;
    AscendC::CrossCoreSetFlag<MODE, PIPE>(flagId);
    AscendC::CrossCoreWaitFlag(flagId);
}

template <uint8_t MODE, pipe_t PIPE>
CATLASS_DEVICE
void CrossCoreSetFlag(CrossCoreFlag &flag)
{
    AscendC::CrossCoreSetFlag<MODE, PIPE>(flag.id);
}

CATLASS_DEVICE
void CrossCoreWaitFlag(CrossCoreFlag &flag)
{
    AscendC::CrossCoreWaitFlag(flag.id);
}

template <uint8_t MODE, pipe_t PIPE, uint32_t REVERSE_DEPTH>
CATLASS_DEVICE
void CrossCoreSetFlagWithReverse(CrossCoreFlagWithReverse<REVERSE_DEPTH> &flag)
{
    AscendC::CrossCoreSetFlag<MODE, PIPE>(flag.id);
    if (++flag.count >= REVERSE_DEPTH) {
        AscendC::CrossCoreWaitFlag(flag.reverseId);
        flag.count = 0;
    }
}

template <uint8_t MODE, pipe_t PIPE, uint32_t REVERSE_DEPTH>
CATLASS_DEVICE
void CrossCoreWaitFlagWithReverse(CrossCoreFlagWithReverse<REVERSE_DEPTH> &flag)
{
    AscendC::CrossCoreWaitFlag(flag.id);
    if (++flag.count >= REVERSE_DEPTH) {
        AscendC::CrossCoreSetFlag<MODE, PIPE>(flag.reverseId);
        flag.count = 0;
    }
}

}  // namespace Catlass::Arch

#endif // CATLASS_ARCH_CROSS_CORE_SYNC_HPP

/*
AscendC::CrossCoreSetFlag
I. 功能说明
面向分离架构的核间同步控制接口。
该接口和CrossCoreWaitFlag接口配合使用。
使用时需传入核间同步的标记ID(flagId)，每个ID对应一个初始值为0的计数器。
执行CrossCoreSetFlag后ID对应的计数器增加1；
执行CrossCoreWaitFlag时如果对应的计数器数值为0则阻塞不执行；
如果对应的计数器大于0，则计数器减一，同时后续指令开始执行。

同步控制分为以下几种模式：

1. 模式0：AI Core核间的同步控制。
对于AIC场景，同步所有的AIC核，直到所有的AIC核都执行到CrossCoreSetFlag时，
CrossCoreWaitFlag后续的指令才会执行；
对于AIV场景，同步所有的AIV核，直到所有的AIV核都执行到CrossCoreSetFlag时，
CrossCoreWaitFlag后续的指令才会执行。

2. 模式1：AI Core内部，AIV核之间的同步控制。(这里在AIV 上调用AscendC::CrossCoreSetFlag/AscendC::CrossCoreWaitFlag时，内部有默认的同步，
即一个AI Core 上的两个AIV 会在调用这些指令时默认同步调用)
如果两个AIV核都运行了CrossCoreSetFlag，
CrossCoreWaitFlag后续的指令才会执行。

3. 模式2：AI Core内部，AIC与AIV之间的同步控制。(这里在AIV 上调用AscendC::CrossCoreSetFlag/AscendC::CrossCoreWaitFlag时，内部有默认的同步，
即一个AI Core 上的两个AIV 会在调用这些指令时默认同步调用)
在AIC核执行CrossCoreSetFlag之后， 两个AIV上CrossCoreWaitFlag后续的指令才会继续执行；

两个AIV都执行CrossCoreSetFlag后，(这里在AIV 上调用AscendC::CrossCoreSetFlag/AscendC::CrossCoreWaitFlag时，内部有默认的同步，
即一个AI Core 上的两个AIV 会在调用这些指令时默认同步调用)
AIC上CrossCoreWaitFlag后续的指令才能执行。

II. 函数原型
    template <uint8_t modeId, pipe_t pipe>
    __aicore__ inline void CrossCoreSetFlag(uint16_t flagId)

III. 参数说明
                            表1 模板参数说明
    参数名                                  描述
    modeId                          核间同步的模式，取值如下：
                                    模式0：AI Core核间的同步控制。
                                    
                                    模式1：AI Core内部，Vector核（AIV）之间的同步控制。
                                    
                                    模式2：AI Core内部，
                                          Cube核（AIC）与Vector核（AIV）之间的同步控制。
    
    pipe                            设置这条指令所在的流水类型，
                                    流水类型可参考硬件流水类型。
                                    即阻塞或指示可用的指令作用于的流水线的类型 

                            表2 参数说明
    参数名          输入/输出              描述
    flagId            输入              核间同步的标记。
                                        Atlas A2 训练系列产品/Atlas 800I A2 推理产品/
                                        A200I A2 Box 异构组件，取值范围是0-10。
                                        
                                        Atlas A3 训练系列产品/Atlas A3 推理系列产品，
                                        取值范围是0-10。

IV. 返回值: 无

V. 约束说明
1. 因为Matmul高阶API内部实现中使用了本接口进行核间同步控制，
   所以不建议开发者同时使用该接口和Matmul高阶API，否则会有flagID冲突的风险。

2. 同一flagId的计数器最多设置16次。

VI. 调用示例
// 使用模式0的方式同步所有的AIV核
if (g_coreType == AscendC::AIV) {
    AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(0x8);
    AscendC::CrossCoreWaitFlag(0x8);
}

// 使用模式1的方式同步当前AICore内的所有AIV子核
if (g_coreType == AscendC::AIV) {
    AscendC::CrossCoreSetFlag<0x1, PIPE_MTE3>(0x8);
    AscendC::CrossCoreWaitFlag(0x8);
}

// 注意：如果调用高阶API,无需开发者处理AIC和AIV的同步
// AIC侧做完Matmul计算后通知AIV进行后处理
if (g_coreType == AscendC::AIC) {
    // Matmul处理
    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x8);
}

// AIV侧等待AIC Set消息, 进行Vector后处理
if (g_coreType == AscendC::AIV) {
    AscendC::CrossCoreWaitFlag(0x8);
    // Vector后处理
}
*/

/*
AscendC::CrossCoreWaitFlag

I. 功能说明
面向分离架构的核间同步控制接口。该接口和CrossCoreSetFlag接口配合使用。
具体使用方法请参考CrossCoreSetFlag。

II. 函数原型
    template <uint8_t modeId, pipe_t pipe>
    __aicore__ inline void CrossCoreWaitFlag(uint16_t flagId)

III. 参数说明
                    表1 模板参数说明
    参数名                          描述
    modeId              核间同步的模式，取值如下:
                        模式0：AI Core核间的同步控制。
                        模式1：AI Core内部，Vector核（AIV）之间的同步控制。
                        模式2：AI Core内部，Cube核（AIC）与Vector核（AIV）之间的同步控制。
    
    pipe                设置这条指令所在的流水类型，流水类型可参考硬件流水类型。

                    表2 参数说明
    参数名              输入/输出               描述
    flagId                输入              核间同步的标记。
                                            Atlas A2 训练系列产品/
                                            Atlas 800I A2 推理产品/
                                            A200I A2 Box 异构组件，取值范围是0-10。

                                            Atlas A3 训练系列产品/
                                            Atlas A3 推理系列产品，取值范围是0-10。

IV. 返回值: 无

V. 约束说明
    1. CrossCoreWaitFlag必须与CrossCoreSetFlag接口配合使用，避免计算核一直处于阻塞阶段。
    
    2.如果执行CrossCoreWaitFlag时该flagId的计数器的值为0，
    则CrossCoreWaitFlag之后的所有指令都将被阻塞，直到该flagId的计数器的值不为0。
    同一个flagId的计数器最多设置16次。

*/