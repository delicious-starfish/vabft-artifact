/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_COPY_L0C_TO_GM_HPP
#define CATLASS_GEMM_TILE_COPY_L0C_TO_GM_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Tile {

enum class ScaleGranularity {
    UNDEFINED = -1,
    NO_QUANT = 0,
    PER_TENSOR,
    PER_CHANNEL,
    PER_GROUP
};

template <
    class ArchTag,
    class ElementSrc,
    class ElementDst,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT
>
struct CopyL0CToGmQuantMode {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm, can not find the specialization.");
};

// CopyL0CToGm cast fp32 to fp16
template <>
struct CopyL0CToGmQuantMode<
    Catlass::Arch::AtlasA2,
    float, half,
    ScaleGranularity::NO_QUANT
> {
    static constexpr auto VALUE = QuantMode_t::F322F16;
};

// CopyL0CToGm cast fp32 to bf16
template <>
struct CopyL0CToGmQuantMode<
    Catlass::Arch::AtlasA2,
    float, bfloat16_t,
    ScaleGranularity::NO_QUANT
> {
    static constexpr auto VALUE = QuantMode_t::F322BF16;
};

// CopyL0CToGm output fp32
template <>
struct CopyL0CToGmQuantMode<
    Catlass::Arch::AtlasA2,
    float, float,
    ScaleGranularity::NO_QUANT
> {
    static constexpr auto VALUE = QuantMode_t::NoQuant;
};

// CopyL0CToGm output int32
template <>
struct CopyL0CToGmQuantMode<
    Catlass::Arch::AtlasA2,
    int32_t, int32_t,
    ScaleGranularity::NO_QUANT
> {
    static constexpr auto VALUE = QuantMode_t::NoQuant;
};

// CopyL0CToGm cast int32_t to fp16
template <>
struct CopyL0CToGmQuantMode<
    Catlass::Arch::AtlasA2,
    int32_t, half,
    ScaleGranularity::PER_TENSOR
> {
    static constexpr auto VALUE = QuantMode_t::DEQF16;
};

template <>
struct CopyL0CToGmQuantMode<
    Catlass::Arch::AtlasA2,
    int32_t, half,
    ScaleGranularity::PER_CHANNEL
> {
    static constexpr auto VALUE = QuantMode_t::VDEQF16;
};

template <
    class ArchTag,
    class ElementAccumulator,
    class GmType,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    bool ReluEnable = false
>
struct CopyL0CToGm {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm, can not find the specialization.");
};

template <
    class ArchTag,
    class ElementAccumulator,
    class GmType,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    bool ReluEnable = false
>
struct CopyL0CToL1 {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to L1, can not find the specialization.");
};

template <
    class ElementAccumulator_,
    class ElementDst_,
    bool ReluEnable_
>
struct CopyL0CToGm<Catlass::Arch::AtlasA2,
                   ElementAccumulator_,
                   Gemm::GemmType<ElementDst_, layout::RowMajor>,
                   ScaleGranularity::NO_QUANT,
                   ReluEnable_>
{
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = Catlass::layout::zN;
    using LayoutDst = Catlass::layout::RowMajor;
    static constexpr auto quantPre = CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst,
        ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementDst> const &dst, AscendC::LocalTensor<ElementSrc> const &src,
        LayoutDst const &dstLayout, LayoutSrc const &srcLayout, uint8_t unitFlag = 0)
    {
         /*
                表3 Fixpipe搬运参数结构体说明
            参数名称            数据类型                        含义
            nSize               输入                源NZ矩阵在N方向上的大小。
                                                    不使能NZ2ND功能
                                                        若使能channelSplit功能，nSize必须为8的倍数，
                                                        取值范围：nSize∈[1, 4095]。

                                                        若不使能channelSplit功能，nSize必须为16的倍数，
                                                        取值范围：nSize∈[1, 4095]。

                                                    使能NZ2ND功能: nSize取值范围 ∈[1, 4095]。

            mSize               输入                源NZ矩阵在M方向上的大小。
                                                    不使能NZ2ND功能
                                                        取值范围：mSize∈[1, 65535]。
                                                    使能NZ2ND功能
                                                        取值范围：mSize∈[1, 8192]。
    
            srcStride           输入                源NZ矩阵中相邻Z排布的起始地址偏移，
                                                    取值范围：srcStride∈[0, 65535]， 
                                                    单位：C0_Size(16*sizeof(T)，T为srcLocal的数据类型)。
    
            dstStride           输入                不使能NZ2ND功能
                                                    目的NZ矩阵中相邻Z排布的起始地址偏移，取值不为0， 
                                                    单位：datablock(32Bytes)。

                                                    使能NZ2ND功能
                                                    目的ND矩阵每一行中的元素个数，取值不为0 ，
                                                    单位：element。

            quantPre            输入                QuantMode_t是一个枚举类型，用于控制量化模式，
                                                    默认值为QuantMode_t::NoQuant，即不使能量化功能。
                                                    QuantMode_t取值如下：
                                                    NoQuant，不使能量化功能
                                                    F322F16，float量化成half
                                                    F322BF16，float量化成bfloat16_t
                                                    DEQF16，int32_t量化成half, scalar量化
                                                    VDEQF16， int32_t量化成half，tensor量化
                                                    QF322B8_PRE，float量化成uint8_t/int8_t，scalar量化
                                                    VQF322B8_PRE，float量化成uint8_t/int8_t，tensor量化
                                                    REQ8，int32_t量化成uint8_t/int8_t，scalar量化
                                                    VREQ8，int32_t量化成uint8_t/int8_t，tensor量化

            deqScalar           输入                scalar量化参数，表示单个scale值，
                                                    quantPre量化模式为scalar量化时需要设置该参数。
                                                    支持的数据类型为uint64_t。

            ndNum               输入                源NZ矩阵的数目，也就是传输ND矩阵的数目，
                                                    取值范围：ndNum∈[1, 65535]

            srcNdStride         输入                不同NZ矩阵起始地址之间的间隔，
                                                    取值范围：srcNdStride∈[1, 512]，单位：1024B。
                                                    当ndNum配置为1时，srcNdStride配置为0即可，不生效。

            dstNdStride         输入                目的相邻ND矩阵起始地址之间的偏移，
                                                    取值范围：dstNdstride∈[1, 65535]，单位：element。
                                                    当ndNum配置为1时，dstNdStride配置为0即可，不生效。

            reluEn              输入                是否使能relu的开关，false：不使能relu功能；
                                                    true：使能relu功能。
    
            unitFlag            输入                预留参数，用户无需关心，使用默认值0即可。
        
            isChannelSplit      输入                是否使能通道拆分的功能。默认为false，不使能该功能。
                                                    仅在src和dst都为float时才能使能通道拆分，
                                                    且不能同时使能ChannelSplit和NZ2ND功能。
        */
        AscendC::FixpipeParamsV220 intriParams;

        // Fixpipe layout information
        intriParams.nSize = dstLayout.shape(1);
        intriParams.mSize = dstLayout.shape(0);
        intriParams.srcStride = srcLayout.stride(3) / srcLayout.stride(0);
        intriParams.dstStride = dstLayout.stride(0);

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>(dst, src, intriParams);
    }
};

template <
    class ElementAccumulator_,
    class ElementDst_,
    bool ReluEnable_
>
struct CopyL0CToL1<Catlass::Arch::AtlasA2,
                   ElementAccumulator_,
                   Gemm::GemmType<ElementDst_, layout::zN>,
                   ScaleGranularity::NO_QUANT,
                   ReluEnable_>
{
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = Catlass::layout::zN;
    using LayoutDst = Catlass::layout::zN;
    static constexpr auto quantPre = CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst,
        ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<ElementDst> const &dst, 
        AscendC::LocalTensor<ElementSrc> const &src,
        LayoutDst const &dstLayout, LayoutSrc const &srcLayout, 
        uint8_t unitFlag = 0, bool isChannelSplit=false)
    {

        AscendC::FixpipeParamsM300 intriParams;

        /*
                表3 Fixpipe搬运参数结构体说明
            参数名称            数据类型                        含义
            nSize               输入                源NZ矩阵在N方向上的大小。
                                                    不使能NZ2ND功能
                                                        若使能channelSplit功能，nSize必须为8的倍数，
                                                        取值范围：nSize∈[1, 4095]。

                                                        若不使能channelSplit功能，nSize必须为16的倍数，
                                                        取值范围：nSize∈[1, 4095]。

                                                    使能NZ2ND功能: nSize取值范围 ∈[1, 4095]。

            mSize               输入                源NZ矩阵在M方向上的大小。
                                                    不使能NZ2ND功能
                                                        取值范围：mSize∈[1, 65535]。
                                                        使能NZ2ND功能
                                                        取值范围：mSize∈[1, 8192]。
    
            srcStride           输入                源NZ矩阵中相邻Z排布的起始地址偏移，
                                                    取值范围：srcStride∈[0, 65535]， 
                                                    单位：C0_Size(16*sizeof(T)，T为srcLocal的数据类型)。
    
            dstStride           输入                不使能NZ2ND功能
                                                    目的NZ矩阵中相邻Z排布的起始地址偏移，取值不为0， 
                                                    单位：datablock(32Bytes)。

                                                    使能NZ2ND功能
                                                    目的ND矩阵每一行中的元素个数，取值不为0 ，
                                                    单位：element。

            quantPre            输入                QuantMode_t是一个枚举类型，用于控制量化模式，
                                                    默认值为QuantMode_t::NoQuant，即不使能量化功能。
                                                    QuantMode_t取值如下：
                                                    NoQuant，不使能量化功能
                                                    F322F16，float量化成half
                                                    F322BF16，float量化成bfloat16_t
                                                    DEQF16，int32_t量化成half, scalar量化
                                                    VDEQF16， int32_t量化成half，tensor量化
                                                    QF322B8_PRE，float量化成uint8_t/int8_t，scalar量化
                                                    VQF322B8_PRE，float量化成uint8_t/int8_t，tensor量化
                                                    REQ8，int32_t量化成uint8_t/int8_t，scalar量化
                                                    VREQ8，int32_t量化成uint8_t/int8_t，tensor量化

            deqScalar           输入                scalar量化参数，表示单个scale值，
                                                    quantPre量化模式为scalar量化时需要设置该参数。
                                                    支持的数据类型为uint64_t。

            ndNum               输入                源NZ矩阵的数目，也就是传输ND矩阵的数目，
                                                    取值范围：ndNum∈[1, 65535]

            srcNdStride         输入                不同NZ矩阵起始地址之间的间隔，
                                                    取值范围：srcNdStride∈[1, 512]，单位：1024B。
                                                    当ndNum配置为1时，srcNdStride配置为0即可，不生效。

            dstNdStride         输入                目的相邻ND矩阵起始地址之间的偏移，
                                                    取值范围：dstNdstride∈[1, 65535]，单位：element。
                                                    当ndNum配置为1时，dstNdStride配置为0即可，不生效。

            reluEn              输入                是否使能relu的开关，false：不使能relu功能；
                                                    true：使能relu功能。
    
            unitFlag            输入                预留参数，用户无需关心，使用默认值0即可。
        
            isChannelSplit      输入                是否使能通道拆分的功能。默认为false，不使能该功能。
                                                    仅在src和dst都为float时才能使能通道拆分，
                                                    且不能同时使能ChannelSplit和NZ2ND功能。
        */

        // Fixpipe layout information
        intriParams.nSize = dstLayout.shape(2) * dstLayout.shape(3);
        intriParams.mSize = dstLayout.shape(0) * dstLayout.shape(1);
        intriParams.srcStride = srcLayout.stride(3) / srcLayout.shape(2);
        intriParams.dstStride = dstLayout.stride(3) / (BYTE_PER_C0 / sizeof(ElementDst));

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;
        intriParams.isChannelSplit = isChannelSplit;

        /*
        template <typename DstT, typename SrcT,
         const FixpipeConfig& config = CFG_ROW_MAJOR>
        void Fixpipe(const LocalTensor<DstT>& dstLocal, 
            const LocalTensor<SrcT>& srcLocal, 
            const FixpipeParamsM300& intriParams)
        */

        // Call AscendC Fixpipe

        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_NZ>(dst, src, intriParams);
    }
};

template <
    class ElementAccumulator_,
    class ElementDst_,
    bool ReluEnable_
>
struct CopyL0CToGm<Catlass::Arch::AtlasA2,
                   ElementAccumulator_,
                   Gemm::GemmType<ElementDst_, layout::zN>,
                   ScaleGranularity::NO_QUANT,
                   ReluEnable_>
{
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = Catlass::layout::zN;
    using LayoutDst = Catlass::layout::zN;
    static constexpr auto quantPre = CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst,
        ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementDst> const &dst, AscendC::LocalTensor<ElementSrc> const &src,
        LayoutDst const &dstLayout, LayoutSrc const &srcLayout, uint8_t unitFlag = 0)
    {
        AscendC::FixpipeParamsV220 intriParams;

        // Fixpipe layout information
        intriParams.nSize = dstLayout.shape(2) * dstLayout.shape(3);
        intriParams.mSize = dstLayout.shape(0) * dstLayout.shape(1);
        intriParams.srcStride = srcLayout.stride(3) / srcLayout.shape(2);
        intriParams.dstStride = dstLayout.stride(3) / (BYTE_PER_C0 / sizeof(ElementDst));

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_NZ>(dst, src, intriParams);
    }
};

///////////////////////////////////////////CopyL0CToGmTla/////////////////////////////////////////////////
template <
    class ArchTag,
    class TensorSrc,
    class TensorDst,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    bool ReluEnable = false,
    class Enable = void
>
struct CopyL0CToGmTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm, can not find the specialization.");
};

template <
    class TensorSrc_,
    class ElementDst_,
    class LayoutDst_,
    bool ReluEnable_
>
struct CopyL0CToGmTla<Catlass::Arch::AtlasA2,
                   TensorSrc_,
                   tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, AscendC::TPosition::GM>,
                   ScaleGranularity::NO_QUANT,
                   ReluEnable_,
                   std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>>
{
    using ArchTag = Catlass::Arch::AtlasA2;
    using TensorDst = tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, AscendC::TPosition::GM>;
    using ElementDst = ElementDst_;
    using TensorSrc = TensorSrc_;
    using ElementSrc = typename TensorSrc::Element;
    static constexpr auto quantPre = CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst,
        ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    CATLASS_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint8_t unitFlag = 0)
    {
        AscendC::FixpipeParamsV220 intriParams;

        // Fixpipe layout information
        intriParams.nSize = tla::get<1>(dstTensor.shape());
        intriParams.mSize = tla::get<0>(dstTensor.shape());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>(
            dstTensor.data(), srcTensor.data(), intriParams);
    }
};


/////////////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_COPY_L0C_TO_GM_HPP
