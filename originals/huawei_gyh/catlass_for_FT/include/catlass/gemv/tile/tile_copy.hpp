/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMV_TILE_TILE_COPY_HPP
#define CATLASS_GEMV_TILE_TILE_COPY_HPP

#include "catlass/catlass.hpp"
#include "catlass/detail/tag_to_layout.hpp"

#include "catlass/gemv/tile/vec_copy_gm_to_ub.hpp"
#include "catlass/gemv/tile/vec_copy_ub_to_gm.hpp"
#include "catlass/gemv/tile/matrix_copy_gm_to_ub.hpp"
#include "catlass/gemv/tile/matrix_copy_ub_to_gm.hpp"
// include/catlass/gemv/tile/matrix_copy_gm_to_ub_simpling.hpp
#include "catlass/gemv/tile/matrix_copy_gm_to_ub_simpling.hpp"
// include/catlass/gemv/tile/vec_copy_gm_to_ub_pad.hpp
#include "catlass/gemv/tile/vec_copy_gm_to_ub_pad.hpp"
// include/catlass/gemv/tile/vec_copy_ub_to_gm_pad.hpp
#include "catlass/gemv/tile/vec_copy_ub_to_gm_pad.hpp"

#include "catlass/gemm/tile/copy_gm_to_l1.hpp"
#include "catlass/gemm/tile/copy_l0c_to_gm.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0a.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0b.hpp"

#include "catlass/gemm/helper.hpp"
#include "catlass/gemv/helper.hpp"
#include "catlass/gemm/gemm_type.hpp"

namespace Catlass::Gemv::Tile {

template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    /// MatmulType type for X vector operand
    class XType,
    /// MatmulType type for Y vector operand
    class YType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileCopyGemvAiv {
    using MATRIX_SIMPLING_TYPE = Gemv::helper::MATRIX_SIMPLING_TYPE;

    using ElementA = typename AType::Element;
    using ElementX = typename XType::Element;
    using ElementY = typename YType::Element;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    // the function of aiv
    using VecCopyGmToUb = Gemv::Tile::VecCopyGmToUB<ArchTag, XType>;
    static constexpr bool is_atoadd = Gemv::helper::AtomicAddSelector<AType>::value;
    using VecCopyUbToGm = Gemv::Tile::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
    using MatrixCopyGmToUb = Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;

    using MatrixCopyGmToUbSimplingContinue = Gemv::Tile::MatrixCopyGmToUBSimpling<
        ArchTag, AType, MATRIX_SIMPLING_TYPE::CONTINUOUS_SIMPLING>;

    using MatrixCopyGmToUbSimplingStrided = Gemv::Tile::MatrixCopyGmToUBSimpling<
        ArchTag, AType, MATRIX_SIMPLING_TYPE::STRIDED_SIMPLING>;
};


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
struct TileCopyFTRedAiv {
    using MATRIX_SIMPLING_TYPE = Gemv::helper::MATRIX_SIMPLING_TYPE;
    using ElementA = typename AType::Element;
    using LayoutA = typename AType::Layout;

    using ElementB = typename BType::Element;
    using ElementX = typename XType::Element;
    using ElementY = typename YType::Element;

    using AOutType = Gemm::GemmType<ElementY, LayoutA>;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    // the function of aiv
    // using ElementCInL1 = typename XType::Element;
    using VecCopyGmToUb = Gemv::Tile::VecCopyGmToUB<ArchTag, XType>;

    static constexpr bool is_atoadd = Gemv::helper::AtomicAddSelector<AType>::value;
    
    using LayoutVX = Catlass::layout::VectorLayout;
    using VXType = Catlass::Gemm::GemmType<ElementX, LayoutVX>;
    
    
    using VecCopyUbToGmforBMax = Gemv::Tile::VecCopyUBToGm<ArchTag, XType, is_atoadd>;
    using VecCopyUbToGmforAMax = Gemv::Tile::VecCopyUBToGm<ArchTag, VXType, is_atoadd>;
    using VecCopyUbToGmforAMean = Gemv::Tile::VecCopyUBToGm<ArchTag, YType, is_atoadd>;

    using MatrixCopyGmToUb = Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;

    /*
    template <class Element>
    struct MatrixCopyGmToUBSimpling<
        Arch::AtlasA2, 
        Gemm::GemmType<Element, layout::RowMajor>, 
        Gemv::helper::MATRIX_SIMPLING_TYPE::CONTINUOUS_SIMPLING>
    */
    using MatrixCopyGmToUbSimplingContinue = Gemv::Tile::MatrixCopyGmToUBSimpling<
        ArchTag, AType, MATRIX_SIMPLING_TYPE::CONTINUOUS_SIMPLING>;

    using MatrixCopyGmToUbSimplingStrided = Gemv::Tile::MatrixCopyGmToUBSimpling<
        ArchTag, AType, MATRIX_SIMPLING_TYPE::STRIDED_SIMPLING>;

    using MatrixCopyUbToGmforAOut = Gemv::Tile::MatrixCopyUBToGm<ArchTag, AOutType>;
};




template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    /// MatmulType type for X vector operand
    class XType,
    /// MatmulType type for Y vector operand
    class YType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileCopyGemvAic {
    using ElementA = typename AType::Element;
    using ElementX = typename XType::Element;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;
    
    using ElementYforBFAIV = float;
    using LayoutY = typename YType::Layout;
    using YTypeforBFAIV = Gemm::GemmType<ElementYforBFAIV, LayoutY>;

    // the function of aic
    using L1XType = typename Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L1AType;
    using L1AType = typename Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L1BType;
    using L1AColType = typename Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L1BColType;

    using L0AType = typename Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L0AType;
    using L0BType = typename Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L0BType;
    using L0BColType = typename Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L0BColType;


    using CopyGmToL1A = Gemm::Tile::CopyGmToL1<ArchTag, XType, L1XType>;
    using CopyGmToL1B = Gemm::Tile::CopyGmToL1<ArchTag, AType, L1AType>;
    using CopyGmToL1BCol = Gemm::Tile::CopyGmToL1<ArchTag, AType, L1AColType>;


    using CopyL1ToL0A = Gemm::Tile::CopyL1ToL0A<ArchTag, L1XType, L0AType>;
    using CopyL1ToL0B = Gemm::Tile::CopyL1ToL0B<ArchTag, L1AType, L0BType>;
    using CopyL1ToL0BCol = Gemm::Tile::CopyL1ToL0B<ArchTag, L1AColType, L0BColType>;
    using CopyL1ToL0BBoth = Gemm::Tile::CopyL1ToL0B<ArchTag, L1AType, L0BColType>;


    using CopyL0CToGm = Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, YType>;
    using CopyL0CToGmforBFAIV = Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, YTypeforBFAIV>;

    
};


template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    /// MatmulType type for X vector operand
    class XType,
    /// MatmulType type for Y vector operand
    class YType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileCopyGemvAicMultiVector {
    using ElementA = typename AType::Element;
    using ElementX = typename XType::Element;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;
    
    using ElementYforBFAIV = float;
    using LayoutY = typename YType::Layout;
    using MXType = Gemm::GemmType<ElementX, LayoutY>;

    using LayoutVX = Catlass::layout::VectorLayout;
    using VXType = Catlass::Gemm::GemmType<ElementX, LayoutVX>;

    using YTypeforBFAIV = Gemm::GemmType<ElementYforBFAIV, LayoutY>;
    using YTypeforAccAIV = Gemm::GemmType<ElementYforBFAIV, LayoutY>;

    // the function of aic
    // using L1XType = typename Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L1AType;
    using L1XType = typename Gemv::helper::L1AndL0TypeSelectorGemv<VXType, AType>::L1AType;

    using L1AType = typename Gemv::helper::L1AndL0TypeSelectorGemv<VXType, AType>::L1BType;
    using L1AColType = typename Gemv::helper::L1AndL0TypeSelectorGemv<VXType, AType>::L1BColType;

    using L0AType = typename Gemv::helper::L1AndL0TypeSelectorGemv<VXType, AType>::L0AType;
    using L0XType = typename Gemv::helper::L1AndL0TypeSelectorGemv<VXType, AType>::L0AType;

    using L0BType = typename Gemv::helper::L1AndL0TypeSelectorGemv<VXType, AType>::L0BType;
    using L0BColType = typename Gemv::helper::L1AndL0TypeSelectorGemv<VXType, AType>::L0BColType;

    using CopyGmToL1A = Gemm::Tile::CopyGmToL1<ArchTag, MXType, L1XType>;
    using CopyGmToL1X = Gemm::Tile::CopyGmToL1<ArchTag, MXType, L1XType>;
    using CopyGmToL1VX = Gemm::Tile::CopyGmToL1<ArchTag, VXType, L1XType>;

    using CopyGmToL1B = Gemm::Tile::CopyGmToL1<ArchTag, AType, L1AType>;
    using CopyGmToL1BCol = Gemm::Tile::CopyGmToL1<ArchTag, AType, L1AColType>;


    using CopyL1ToL0A = Gemm::Tile::CopyL1ToL0A<ArchTag, L1XType, L0AType>;
    using CopyL1ToL0X = Gemm::Tile::CopyL1ToL0A<ArchTag, L1XType, L0XType>;

    using CopyL1ToL0B = Gemm::Tile::CopyL1ToL0B<ArchTag, L1AType, L0BType>;
    using CopyL1ToL0BCol = Gemm::Tile::CopyL1ToL0B<ArchTag, L1AColType, L0BColType>;
    using CopyL1ToL0BBoth = Gemm::Tile::CopyL1ToL0B<ArchTag, L1AType, L0BColType>;


    using CopyL0CToGm = Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, YType>;
    using CopyL0CToGmforBFAIV = Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, YTypeforBFAIV>;
    using CopyL0CToGmforAccAIV = Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, YTypeforAccAIV>;
};

template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    /// MatmulType type for X vector operand
    class XType,
    /// MatmulType type for Y vector operand
    class YType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileCopyMatrixReduceAiv {
    using ElementA = typename AType::Element;
    using ElementX = typename XType::Element;
    using ElementY = typename YType::Element;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    // the function of aiv
    using VecCopyGmToUb = Gemv::Tile::VecCopyGmToUB<ArchTag, XType>;
    // Gemv::helper::AtomicAddSelector<AType>::value
    static constexpr bool is_atoadd = false;
    using VecCopyUbToGm = Gemv::Tile::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
    using MatrixCopyGmToUb = Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;
};

template <
    Catlass::Gemv::helper::FT_COMP_TYPE COMP_TYPE_,
    /// Tag indicating architecture
    class ArchTag,
    /// Output Result operand namely vector Z
    class ZType,
    /// MatmulType type for X vector operand
    class XType,
    /// MatmulType type for Y vector operand
    class YType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileCopyCompareAiv {
    using ElementZ = typename ZType::Element;
    using ElementX = typename XType::Element;
    using ElementY = typename YType::Element;
    using FT_COMP_TYPE = Catlass::Gemv::helper::FT_COMP_TYPE;

    using ElementWork = typename std::conditional<
        (COMP_TYPE_ == FT_COMP_TYPE::XOR),
        uint16_t,
        typename std::conditional<(COMP_TYPE_ == FT_COMP_TYPE::COMPARE), int32_t, ElementX>::type>::type;
    
    
    // using ElementAccumulator =
    //     typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    // the function of aiv
    
    using VecCopyGmToUbX = Gemv::Tile::VecCopyGmToUB<ArchTag, XType>;
    using VecCopyGmToUbY = Gemv::Tile::VecCopyGmToUB<ArchTag, YType>;
    using VecCopyGmToUbW = Gemv::Tile::VecCopyGmToUB<ArchTag, XType>;
    
    using VecCopyUbToGmZ = Gemv::Tile::VecCopyUBToGm<ArchTag, ZType>;

    using WType = Gemm::GemmType<ElementWork, Catlass::layout::VectorLayout>;
    using VecCopyUbToGmW = Gemv::Tile::VecCopyUBToGm<ArchTag, WType>;



    // static constexpr bool is_atoadd = Gemv::helper::AtomicAddSelector<AType>::value;
    // using VecCopyUbToGm = Gemv::Tile::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
    // using MatrixCopyGmToUb = Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;
};

template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    /// MatmulType type for X vector operand
    class XType,
    /// MatmulType type for Y vector operand
    class YType,
    /// Output Result operand namely vector Z
    class ZType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileCopyMatrixReduceAivFused {
    using ElementA = typename AType::Element;
    using ElementX = typename XType::Element;
    using ElementY = typename YType::Element;
    using ElementZ = typename ZType::Element;

    using FT_COMP_TYPE = Catlass::Gemv::helper::FT_COMP_TYPE;

    using ElementWork = ElementY;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    // the function of aiv
    using VecCopyGmToUb = Gemv::Tile::VecCopyGmToUB<ArchTag, XType>;
    // Gemv::helper::AtomicAddSelector<AType>::value
    static constexpr bool is_atoadd = false;
    using VecCopyUbToGm = Gemv::Tile::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
    using MatrixCopyGmToUb = Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;

    using VecCopyGmToUbInX = Gemv::Tile::VecCopyGmToUB<ArchTag, YType>;
    using VecCopyGmToUbInY = Gemv::Tile::VecCopyGmToUB<ArchTag, YType>;
    using VecCopyGmToUbW = Gemv::Tile::VecCopyGmToUB<ArchTag, YType>;
    
    using VecCopyUbToGmZ = Gemv::Tile::VecCopyUBToGm<ArchTag, ZType>;

    using WType = Gemm::GemmType<ElementWork, Catlass::layout::VectorLayout>;
    using VecCopyUbToGmW = Gemv::Tile::VecCopyUBToGm<ArchTag, WType>;
};

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
struct TileCopyGemvThreCompFusedAiv {
    using ElementA = typename AType::Element;
    using ElementC = typename CType::Element;

    using ElementX = typename XType::Element;
    using ElementY = typename YType::Element;

    using FT_COMP_TYPE = Catlass::Gemv::helper::FT_COMP_TYPE;

    using ElementWork = ElementY;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    // the function of aiv
    using VecCopyGmToUb = Gemv::Tile::VecCopyGmToUB<ArchTag, XType>;
    static constexpr bool is_atoadd = Gemv::helper::AtomicAddSelector<AType>::value;
    using VecCopyUbToGm = Gemv::Tile::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
    using MatrixCopyGmToUb = Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;

    using MatrixCopyGmToUbforThre = Gemv::Tile::MatrixCopyGmToUB<ArchTag, CType>;

    using VecCopyGmToUbInY = Gemv::Tile::VecCopyGmToUB<ArchTag, YType>;
    using VecCopyGmToUbW = Gemv::Tile::VecCopyGmToUB<ArchTag, YType>;
    
    using VecCopyUbToGmZ = Gemv::Tile::VecCopyUBToGm<ArchTag, ZType>;
    using VecCopyUbToGmforThre = Gemv::Tile::VecCopyUBToGm<ArchTag, YType>;

    using WType = Gemm::GemmType<ElementWork, Catlass::layout::VectorLayout>;
    using VecCopyUbToGmW = Gemv::Tile::VecCopyUBToGm<ArchTag, WType>;

};

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
struct TileCopyVerifyThreCompFusedAiv {
    using ElementA = typename AType::Element;
    using ElementC = typename CType::Element;

    using ElementX = typename XType::Element;
    using ElementY = typename YType::Element;

    using ElementCompZ = typename ZType::Element;

    using LayoutMX = layout::RowMajor;
    using LayoutMY = layout::RowMajor;
    using LayoutMZ = layout::RowMajor;

    using MXType = Gemm::GemmType<ElementX, LayoutMX>;
    using MYType = Gemm::GemmType<ElementY, LayoutMY>;
    using MZType = Gemm::GemmType<ElementCompZ, LayoutMZ>;


    using FT_COMP_TYPE = Catlass::Gemv::helper::FT_COMP_TYPE;

    using ElementWork = ElementY;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    // the function of aiv
    using VecCopyGmToUb = Gemv::Tile::VecCopyGmToUB<ArchTag, XType>;
    static constexpr bool is_atoadd = Gemv::helper::AtomicAddSelector<AType>::value;
    using VecCopyUbToGm = Gemv::Tile::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
    using MatrixCopyGmToUb = Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;

    using MatrixCopyGmToUbforThre = Gemv::Tile::MatrixCopyGmToUB<ArchTag, CType>;

    using VecCopyGmToUbInY = Gemv::Tile::VecCopyGmToUB<ArchTag, YType>;
    using VecCopyGmToUbW = Gemv::Tile::VecCopyGmToUB<ArchTag, YType>;
    
    using VecCopyUbToGmZ = Gemv::Tile::VecCopyUBToGm<ArchTag, ZType>;
    using VecCopyUbToGmforThre = Gemv::Tile::VecCopyUBToGm<ArchTag, YType>;

    using WType = Gemm::GemmType<ElementWork, Catlass::layout::VectorLayout>;
    using VecCopyUbToGmW = Gemv::Tile::VecCopyUBToGm<ArchTag, WType>;

    using MatrixCopyGmToUbforThreAIn = Gemv::Tile::MatrixCopyGmToUB<ArchTag, MXType>;
    using MatrixCopyGmToUbforABRIn = Gemv::Tile::MatrixCopyGmToUB<ArchTag, MYType>;
    using MatrixCopyGmToUbforCRIn = Gemv::Tile::MatrixCopyGmToUB<ArchTag, CType>;

    using MatrixCopyUbToGmforZ = Gemv::Tile::MatrixCopyUBToGm<ArchTag, MZType>;
    using MatrixCopyUbToGmforCROut = Gemv::Tile::MatrixCopyUBToGm<ArchTag, MYType>;

    using MatrixCopyUbToGmforThreOut = Gemv::Tile::MatrixCopyUBToGm<ArchTag, MYType>;

};

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
struct TileCopyGemvThreCompFusedAivFI {
    using ElementA = typename AType::Element;
    using ElementC = typename CType::Element;

    using ElementX = typename XType::Element;
    using ElementY = typename YType::Element;

    using FT_COMP_TYPE = Catlass::Gemv::helper::FT_COMP_TYPE;

    using ElementWork = ElementY;

    using ElementFIID = int32_t;
    using ElementFIData = ElementA;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    // the function of aiv
    using VecCopyGmToUb = Gemv::Tile::VecCopyGmToUB<ArchTag, XType>;
    static constexpr bool is_atoadd = Gemv::helper::AtomicAddSelector<AType>::value;
    using VecCopyUbToGm = Gemv::Tile::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
    using MatrixCopyGmToUb = Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;

    using MatrixCopyGmToUbforThre = Gemv::Tile::MatrixCopyGmToUB<ArchTag, CType>;

    using VecCopyGmToUbInY = Gemv::Tile::VecCopyGmToUB<ArchTag, YType>;
    using VecCopyGmToUbW = Gemv::Tile::VecCopyGmToUB<ArchTag, YType>;
    
    using VecCopyUbToGmZ = Gemv::Tile::VecCopyUBToGm<ArchTag, ZType>;
    using VecCopyUbToGmforThre = Gemv::Tile::VecCopyUBToGm<ArchTag, YType>;

    using WType = Gemm::GemmType<ElementWork, Catlass::layout::VectorLayout>;
    using VecCopyUbToGmW = Gemv::Tile::VecCopyUBToGm<ArchTag, WType>;

    using FIIndexType = Gemm::GemmType<ElementFIID, Catlass::layout::VectorLayout>;
    using VecCopyUbToGmforFIIndex = Gemv::Tile::VecCopyUBToGm<ArchTag, FIIndexType, false>;

    using FIDataType = Gemm::GemmType<ElementFIData, Catlass::layout::VectorLayout>;
    using VecCopyUbToGmforFIData = Gemv::Tile::VecCopyUBToGm<ArchTag, FIDataType, false>;


};

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
struct TileCopyGemvCastFusedAiv {
    using ElementA = typename AType::Element;
    using ElementC = typename CType::Element;

    using ElementX = typename XType::Element;
    using ElementY = typename YType::Element;


    using ElementWork = ElementY;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    // the function of aiv
    using VecCopyGmToUb = Gemv::Tile::VecCopyGmToUB<ArchTag, XType>;
    static constexpr bool is_atoadd = Gemv::helper::AtomicAddSelector<AType>::value;
    using VecCopyUbToGm = Gemv::Tile::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
    
    using MatrixCopyGmToUbforA = Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;
    using MatrixCopyGmToUbforC = Gemv::Tile::MatrixCopyGmToUB<ArchTag, CType>;

    using VecCopyGmToUbInY = Gemv::Tile::VecCopyGmToUB<ArchTag, YType>;
    using VecCopyGmToUbW = Gemv::Tile::VecCopyGmToUB<ArchTag, YType>;
    
    using WType = Gemm::GemmType<ElementWork, Catlass::layout::VectorLayout>;
    using VecCopyUbToGmW = Gemv::Tile::VecCopyUBToGm<ArchTag, WType>;

    using MatrixCopyUbToGmforA = Gemv::Tile::MatrixCopyUBToGm<ArchTag, CType>;

};


template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    /// MatmulType type for Y vector operand
    class YType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileCopyMatrixSliceSumAiv {
    using ElementA = typename AType::Element;
    using ElementY = typename YType::Element;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementY>::ElementAccumulator;

    // the function of aiv
    using VecCopyGmToUb = Gemv::Tile::VecCopyGmToUB<ArchTag, YType>;
    // Gemv::helper::AtomicAddSelector<AType>::value
    static constexpr bool is_atoadd = false;
    using VecCopyUbToGm = Gemv::Tile::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
    using MatrixCopyGmToUb = Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;
};

template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    /// MatmulType for Y matrix operand
    class YType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileCopyMatrixAddAiv {
    using ElementA = typename AType::Element;
    using ElementY = typename YType::Element;

    using LayoutVX = Catlass::layout::VectorLayout;
    using VXType = Catlass::Gemm::GemmType<ElementA, LayoutVX>;

    using LayoutVY = Catlass::layout::VectorLayout;
    using VYType = Catlass::Gemm::GemmType<ElementY, LayoutVY>;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementY>::ElementAccumulator;

    using VecCopyGmToUb = Gemv::Tile::VecCopyGmToUB<ArchTag, VXType>;
    static constexpr bool is_atoadd = false;
    using VecCopyUbToGm = Gemv::Tile::VecCopyUBToGm<ArchTag, VYType, is_atoadd>;

    // the function of aiv
    using MatrixCopyGmToUb = Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;
    // MatrixCopyUbtoGm
    using MatrixCopyUbToGm = Gemv::Tile::MatrixCopyUBToGm<ArchTag, YType>;
};


template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    /// MatmulType for Y matrix operand
    class YType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileCopyMatrixAddVectorizedAiv {
    using ElementA = typename AType::Element;
    using ElementY = typename YType::Element;

    using LayoutVX = Catlass::layout::VectorLayout;
    using VXType = Catlass::Gemm::GemmType<ElementA, LayoutVX>;

    using LayoutVY = Catlass::layout::VectorLayout;
    using VYType = Catlass::Gemm::GemmType<ElementY, LayoutVY>;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementY>::ElementAccumulator;

    /*
    template <
        class ArchTag_,
        class VType_,
        Gemv::helper::VEC_PADDING_TYPE::ALIGNED
    >
    struct VecCopyGmToUBPadding
    */
    using VecCopyGmToUbCommon = Gemv::Tile::VecCopyGmToUBPadding<ArchTag, VXType, Gemv::helper::VEC_PADDING_TYPE::ALIGNED>;
    /*
    template <
        class ArchTag_,
        class VType_,
        Gemv::helper::VEC_PADDING_TYPE::PADDING
    >
    struct VecCopyGmToUBPadding
    */
    using VecCopyGmToUbTail = Gemv::Tile::VecCopyGmToUBPadding<ArchTag, VXType, Gemv::helper::VEC_PADDING_TYPE::PADDING>;
    static constexpr bool is_atoadd = false;
    /*
    template <class Element>
    struct VecCopyUBToGmPadding<Arch::AtlasA2, 
        Gemm::GemmType<Element, layout::VectorLayout>, 
        Gemv::helper::VEC_PADDING_TYPE::ALIGNED,
        false>
    */
    using VecCopyUbToGmCommon = Gemv::Tile::VecCopyUBToGmPadding<ArchTag, VYType, Gemv::helper::VEC_PADDING_TYPE::ALIGNED, is_atoadd>;
    
    /*
    template <class Element>
    struct VecCopyUBToGmPadding<Arch::AtlasA2, 
        Gemm::GemmType<Element, layout::VectorLayout>, 
        Gemv::helper::VEC_PADDING_TYPE::PADDING,
        false>
    */
    using VecCopyUbToGmTail = Gemv::Tile::VecCopyUBToGmPadding<ArchTag, VYType, Gemv::helper::VEC_PADDING_TYPE::PADDING, is_atoadd>;
    
    // the function of aiv
    using MatrixCopyGmToUb = Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;
    using MatrixCopyUbToGm = Gemv::Tile::MatrixCopyUBToGm<ArchTag, YType>;
};

} // namespace Catlass::Gemv::Tile

#endif // CATLASS_GEMV_TILE_TILE_COPY_HPP
