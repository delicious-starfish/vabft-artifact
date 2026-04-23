/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMV_HELPER_HPP
#define CATLASS_GEMV_HELPER_HPP

#include "catlass/catlass.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/gemm_type.hpp"

namespace Catlass::Gemv::helper {

template<class Element>
struct UBAlignHelper
{
    static constexpr uint32_t ALIGN = BYTE_PER_BLK / sizeof(Element);
    static constexpr uint32_t ALIGN_REMAIN = 8;
};

template<class Element>
struct UBTransposeAlignHelper
{
    static constexpr uint32_t H_ALIGN = 16;
    static constexpr uint32_t W_ALIGN = 16;
    static constexpr uint32_t BLK_ALIGN = BYTE_PER_BLK / sizeof(Element);
    static constexpr uint32_t C_ALIGN = BYTE_PER_BLK / sizeof(Element);
    static constexpr uint32_t C_UNIT_UPPER_LIMIT = 24;
};

template<class GmAType>
struct AtomicAddSelector
{
    static_assert(DEPENDENT_FALSE<GmAType>,
        "Unsupported layout selector, can not find the specialization.");
};

template<class Element>
struct AtomicAddSelector<Gemm::GemmType<Element, layout::RowMajor>>
{
    static constexpr bool value = false;
};

template<class Element>
struct AtomicAddSelector<Gemm::GemmType<Element, layout::ColumnMajor>>
{
    static constexpr bool value = true;
};

template <class Element, class Layout>
struct L1AlignHelper
{
    static_assert(DEPENDENT_FALSE<Element>, "Unsupported align helper, can not find the specialization.");
};

template <class Element>
struct L1AlignHelper<Element, layout::RowMajor>
{
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t M_ALIGNED = C0_NUM_PER_FRACTAL;
    static constexpr uint32_t N_ALIGNED = ELE_NUM_PER_C0;
};

template <class Element>
struct L1AlignHelper<Element, layout::ColumnMajor>
{
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    static constexpr uint32_t getNAligned()
    {
        if constexpr (std::is_same<Element, int8_t>::value) {
            return ELE_NUM_PER_C0 / sizeof(Element);
        } else {
            return C0_NUM_PER_FRACTAL;
        }
    }

    static constexpr uint32_t getMAligned()
    {
        if constexpr (std::is_same<Element, int8_t>::value) {
            return ELE_NUM_PER_C0 / sizeof(Element);
        } else {
            return C0_NUM_PER_FRACTAL;
        }
    }

    static constexpr uint32_t N_ALIGNED = getNAligned();
    static constexpr uint32_t M_ALIGNED = getMAligned();
};

template <class Element>
struct L1AlignHelper<Element, layout::VectorLayout>
{
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t M_ALIGNED = C0_NUM_PER_FRACTAL;
    static constexpr uint32_t N_ALIGNED = ELE_NUM_PER_C0;
};

////////////////////////////////
// new add  gemvaic selector
template<class GmAType, class GmBType>
struct L1AndL0TypeSelectorGemv{
    static_assert(DEPENDENT_FALSE<GmAType>,
        "Unsupported layout selector, can not find the specialization.");
    static_assert(DEPENDENT_FALSE<GmBType>,
        "Unsupported layout selector, can not find the specialization.");
};

enum class FT_ENC_TYPE {
    CE = 0,
    ETC,
    BOTHC,
    RCE,
    NO, 
    BE,
    ETA,
    ABE,
    ETAB, 
    BOTHAB
};

enum class FT_COMP_TYPE {
    SUB = 0,
    RSUB,
    XOR,
    COMPARE
};

enum class FT_RCE_THRE_TYPE {
    ROUND = 0,
    ROUND_WITH_ACC
};

enum class FT_PIPELINE_TYPE {
    SPLIT_K = 0,
    NO_SPLIT_K,
    NO_PIPELINE
};

enum class FT_L02L1_TYPE{
    FIX_PIPE = 0,
    CO12DST,
    ENHANCED
};

enum class FT_AIV_PIPE_FUSE_TYPE{
    NO_FUSED = 0,
    ABE_FUSED_THRE,
    A_B_MIXED,
    A_B_MIXED_BF,
    A_B_MIXED_CHUNK,
    A_B_MIXED_CHUNK_BF,
    A_B_MIXED_CHUNK_F16,
    A_B_ROBUST,
    A_B_ROBUST_BF,
    A_B_SIMPLIFIED,
    A_B_SIMPLIFIED_BF,
    THRE_FUSED
};

enum class FT_THRESHOLD_ALGORITHM{
    AABFT = 0,
    ASVAR,
    ASVAR_ROBUST,
    ASVAR_SIMPLIFIED
};

enum class FT_REDUCE_TYPE{
    SUM = 0,
    MAX,
    SUM_MAX,
    MAX_MIN,
    MAX_MIN_WITH_CAST,
    SUM_MAX_MIXED,
    SUM_MAX_ABE,
    MEAN_SQUARE,
    VAR,
    VAR_SIMPLIFIED,
    SUM_VMAD
};

enum class FT_AIC_BE_SCHEME{
    ROWCOMPLETE = 0,
    COLCOMPLETE,
    ROWCOMPLETE_BF
};

enum class FT_ABE_TYPE{
    TILING_BLOCK = 0,
    CENTRAL_BLOCK,
    CENTRAL_BLOCK_CHUNK_32,
    CENTRAL_BLOCK_CHUNK_16,
    WAIT_CENTRAL_BLOCK
};

enum class MATRIX_SIMPLING_TYPE{
    CONTINUOUS_SIMPLING,
    STRIDED_SIMPLING
};

enum class VEC_PADDING_TYPE{
    ALIGNED,
    PADDING
};

enum class VEC_ADD_TYPE{
    COUNT,
    MASK
};

template<class Element>
struct L1AndL0TypeSelectorGemv<Gemm::GemmType<Element, layout::VectorLayout>, Gemm::GemmType<Element, layout::RowMajor>>{
    using L1AType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
    using L1BType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
    using L1BColType = Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::A1>;

    using L0AType = Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::A2>;
    using L0BType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::B2>;
    using L0BColType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>;
};

template<class Element>
struct L1AndL0TypeSelectorGemv<Gemm::GemmType<Element, layout::VectorLayout>, Gemm::GemmType<Element, layout::zN>>{
    using L1AType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
    using L1BType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
    using L1BColType = Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::A1>;

    using L0AType = Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::A2>;
    using L0BType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::B2>;
    using L0BColType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>;
};

template<class Element>
struct L1AndL0TypeSelectorGemv<Gemm::GemmType<Element, layout::VectorLayout>, Gemm::GemmType<Element, layout::ColumnMajor>>{
    using L1AType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
    using L1BType = Gemm::GemmType<Element, layout::nN, AscendC::TPosition::A1>;
    using L1BColType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>;

    using L0AType = Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::A2>;
    using L0BType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::B2>;
    using L0BColType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>;
};

template<>
struct L1AndL0TypeSelectorGemv<Gemm::GemmType<int8_t, layout::VectorLayout>, Gemm::GemmType<int8_t, layout::ColumnMajor>>{
    using L1AType = Gemm::GemmType<int8_t, layout::zN, AscendC::TPosition::A1>;
    using L1BType = Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::B1>;
    using L1BColType = Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::A1>;

    using L0AType = Gemm::GemmType<int8_t, layout::zZ, AscendC::TPosition::A2>;
    using L0BType = Gemm::GemmType<int8_t, layout::zN, AscendC::TPosition::B2>;
    using L0BColType = Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::B2>;
};

} // namespace Catlass::Gemv::helper

#endif // CATLASS_GEMV_HELPER_HPP