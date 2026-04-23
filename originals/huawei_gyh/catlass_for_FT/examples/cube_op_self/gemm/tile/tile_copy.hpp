#ifndef CATLASS_GEMM_TILE_TILE_COPY_HPP_SELF
#define CATLASS_GEMM_TILE_TILE_COPY_HPP_SELF
#include <type_traits>
#include "catlass/catlass.hpp"
#include "catlass/detail/tag_to_layout.hpp"
#include "tla/tensor.hpp"


// catlass/
// catlass/
// catlass/
// catlass/
// catlass/
// catlass/
// catlass/

#include "gemm/tile/copy_gm_to_l1.hpp"
#include "gemm/tile/copy_l0c_to_gm.hpp"
#include "gemm/tile/copy_l1_to_l0a.hpp"
#include "gemm/tile/copy_l1_to_l0b.hpp"
#include "gemm/tile/copy_l1_to_bt.hpp"
#include "gemm/tile/copy_gm_to_ub.hpp"
#include "gemm/tile/copy_ub_to_gm.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemv/helper.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0a.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0b.hpp"
#include "catlass/gemm/tile/copy_gm_to_l1.hpp"
#include "catlass/gemm/tile/copy_l0c_to_gm.hpp"

namespace CubeSelf::Gemm::Tile{
template <
    // Tag indicating architecture
    class ArchTag,
    // GemmType for A matrix operand;
    class AType,
    // GemmType for B matrix operand;
    class BType,
    // GemmType for C matrix operand;
    class CType,
    // GemmType for Bias operand;
    class BiasType = void
>
struct TileCopy {
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;

    using ElementAccumulator = 
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using CopyGmToL1A = CubeSelf::Gemm::Tile::CopyGmToL1<ArchTag, AType>;
    using CopyGmToL1B = CubeSelf::Gemm::Tile::CopyGmToL1<ArchTag, BType>;

    using CopyL1ToL0A = CubeSelf::Gemm::Tile::CopyL1ToL0A<
        ArchTag, typename Catlass::Gemm::helper::L1ATypeSelector<AType>::L1AType>;
    
    using CopyL1ToL0B = CubeSelf::Gemm::Tile::CopyL1ToL0B<
        ArchTag, typename Catlass::Gemm::helper::L1BTypeSelector<BType>::L1BType>;
    
    using CopyL0CToGm = CubeSelf::Gemm::Tile::CopyL0CToGm<ArchTag,ElementAccumulator,CType>;
    using BiasTypeSelector = Catlass::Gemm::helper::L1BiasTypeSelector<BiasType, ElementAccumulator>;

    using CopyGmToL1Bias = std::conditional_t<std::is_same_v<BiasType,void>,void,
                            CubeSelf::Gemm::Tile::CopyGmToL1<
                                ArchTag,
                                typename BiasTypeSelector::GMBiasType,
                                typename BiasTypeSelector::L1BiasType>>;
    
    using CopyL1ToBT = std::conditional_t<std::is_same_v<BiasType,void>,void,
                            CubeSelf::Gemm::Tile::CopyL1ToBT<ArchTag,
                                typename BiasTypeSelector::L1BiasType,
                                typename BiasTypeSelector::L0BiasType>>;
    
    
};


template <
    // Tag indicating architecture
    class ArchTag,
    // GemmType for A matrix operand;
    class AType,
    // GemmType for B matrix operand;
    class BType,
    // GemmType for C matrix operand;
    class CType,
    // GemvType for X vector operand;
    class XType,
    // GemvType for Y vector operand;
    class YType,
    // GemmType for Bias operand;
    class BiasType = void,
    Catlass::Gemv::helper::FT_L02L1_TYPE COPY_TYPE_ = Catlass::Gemv::helper::FT_L02L1_TYPE::FIX_PIPE
>
struct TileCopyFT {
    using FT_L02L1_TYPE = Catlass::Gemv::helper::FT_L02L1_TYPE;
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;

    using ElementCInL1 = typename XType::Element;

    using CTypeForL1 = Catlass::Gemm::GemmType<ElementCInL1,Catlass::layout::zN>;
    using L1XType = typename Catlass::Gemv::helper::L1AndL0TypeSelectorGemv<XType, CTypeForL1>::L1AType;
    using L1CTypeforFT = typename Catlass::Gemv::helper::L1AndL0TypeSelectorGemv<XType, CTypeForL1>::L1BType;

    using L0XType = typename Catlass::Gemv::helper::L1AndL0TypeSelectorGemv<XType, CTypeForL1>::L0AType;
    using L0CTypeforFT = typename Catlass::Gemv::helper::L1AndL0TypeSelectorGemv<XType, CTypeForL1>::L0BType;
    
    using ElementAccumulator = 
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using CopyGmToL1A = CubeSelf::Gemm::Tile::CopyGmToL1<ArchTag, AType>;
    using CopyGmToL1B = CubeSelf::Gemm::Tile::CopyGmToL1<ArchTag, BType>;

    using CopyGmToL1X = Catlass::Gemm::Tile::CopyGmToL1<ArchTag, XType, L1XType>;
    
    using CopyL1ToL0A = CubeSelf::Gemm::Tile::CopyL1ToL0A<
        ArchTag, typename Catlass::Gemm::helper::L1ATypeSelector<AType>::L1AType>;
    
    using CopyL1ToL0B = CubeSelf::Gemm::Tile::CopyL1ToL0B<
        ArchTag, typename Catlass::Gemm::helper::L1BTypeSelector<BType>::L1BType>;

    using CopyL1ToL0X = Catlass::Gemm::Tile::CopyL1ToL0A<ArchTag, L1XType, L0XType>;

    using CopyL1ToL0CforFT = Catlass::Gemm::Tile::CopyL1ToL0B<ArchTag, L1CTypeforFT, L0CTypeforFT>;

    using CopyL0CToGm = CubeSelf::Gemm::Tile::CopyL0CToGm<ArchTag,ElementAccumulator,CType>;

    using CopyL0CToGmforFT = Catlass::Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, YType>;

    /*
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
    
    template <
    class ArchTag,
    class ElementAccumulator,
    class L1Type,
    Catlass::Gemv::helper::FT_L02L1_TYPE CopyType,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    bool ReluEnable = false
    >
    struct CopyL0CToL1
    */

    using CopyL0CToL1forFT =  CubeSelf::Gemm::Tile::CopyL0CToL1<ArchTag,
        ElementAccumulator, CTypeForL1, COPY_TYPE_>;

    using BiasTypeSelector = Catlass::Gemm::helper::L1BiasTypeSelector<BiasType, ElementAccumulator>;

    using CopyGmToL1Bias = std::conditional_t<std::is_same_v<BiasType,void>,void,
                            CubeSelf::Gemm::Tile::CopyGmToL1<
                                ArchTag,
                                typename BiasTypeSelector::GMBiasType,
                                typename BiasTypeSelector::L1BiasType>>;
    
    using CopyL1ToBT = std::conditional_t<std::is_same_v<BiasType,void>,void,
                            CubeSelf::Gemm::Tile::CopyL1ToBT<ArchTag,
                                typename BiasTypeSelector::L1BiasType,
                                typename BiasTypeSelector::L0BiasType>>;
    
    
};

template <
    // Tag indicating architecture
    class ArchTag,
    // GemmType for A matrix operand;
    class AType,
    // GemmType for B matrix operand;
    class BType,
    // GemmType for C matrix operand;
    class CType,
    // GemvType for X vector operand;
    class XType,
    // GemvType for Y vector operand;
    class YType,
    // GemmType for Bias operand;
    class BiasType = void
>
struct TileCopyFTABonAic {
    using FT_L02L1_TYPE = Catlass::Gemv::helper::FT_L02L1_TYPE;
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;

    using ElementX = typename XType::Element;
    using LayoutX = typename XType::Layout;

    using CopyGmToL1B = CubeSelf::Gemm::Tile::CopyGmToL1<ArchTag, BType>;


    // using LayoutXInL1 = typename std::conditional<
    //     std::is_same<LayoutX, Catlass::layout::ColumnMajor>::value,
    //     Catlass::layout::nZ,
    //     Catlass::layout::zN>::type;

    // using ElementCInL1 = typename XType::Element;
    using LayoutVX = Catlass::layout::VectorLayout;
    using VXType = Catlass::Gemm::GemmType<ElementX, LayoutVX>;
    using L1XType = typename Catlass::Gemv::helper::L1AndL0TypeSelectorGemv<VXType, AType>::L1AType;

    // using LayoutXInL1 = typename CopyGmToL1B::LayoutDst;

    // using L1AType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
    // using L1XType = Catlass::Gemm::GemmType<ElementX,LayoutXInL1, AscendC::TPosition::A1>;
    // using L1XType = typename Catlass::Gemv::helper::L1AndL0TypeSelectorGemv<XType, CTypeForL1>::L1AType;

    // using L0XType = typename Catlass::Gemv::helper::L1AndL0TypeSelectorGemv<XType, CTypeForL1>::L0AType;
    // using LayoutXInL0 = Catlass::layout::nZ;
    // using L0XType = Catlass::Gemm::GemmType<ElementX, LayoutXInL0, AscendC::TPosition::B2>;
    using L0XType = typename Catlass::Gemv::helper::L1AndL0TypeSelectorGemv<VXType, AType>::L0AType;

    using ElementAccumulator = 
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using CopyGmToL1A = CubeSelf::Gemm::Tile::CopyGmToL1<ArchTag, AType>;

    using LayoutAInL1 = typename CopyGmToL1A::LayoutDst;
    
    /*
    template <class ArchTag, class Element>
        struct CopyGmToL1<
        ArchTag, Gemm::GemmType<Element, layout::ColumnMajor>, 
        Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>> {
    */
    using CopyGmToL1VX = Catlass::Gemm::Tile::CopyGmToL1<ArchTag, VXType, L1XType>;
    using CopyGmToL1X = Catlass::Gemm::Tile::CopyGmToL1<ArchTag, XType, L1XType>;
    
    using CopyL1ToL0A = CubeSelf::Gemm::Tile::CopyL1ToL0A<
        ArchTag, typename Catlass::Gemm::helper::L1ATypeSelector<AType>::L1AType>;
    
    using CopyL1ToL0B = CubeSelf::Gemm::Tile::CopyL1ToL0B<
        ArchTag, typename Catlass::Gemm::helper::L1BTypeSelector<BType>::L1BType>;

    using CopyL1ToL0X = Catlass::Gemm::Tile::CopyL1ToL0A<ArchTag, L1XType, L0XType>;

    using L1ATypeforFT = typename Catlass::Gemm::helper::L1ATypeSelector<AType>::L1AType;
    using L0ATypeforFT = typename Catlass::Gemv::helper::L1AndL0TypeSelectorGemv<VXType, AType>::L0BType;

    using CopyL1ToL0AforFT = Catlass::Gemm::Tile::CopyL1ToL0B<ArchTag, L1ATypeforFT, L0ATypeforFT>;

    using CopyL0CToGm = CubeSelf::Gemm::Tile::CopyL0CToGm<ArchTag,ElementAccumulator, CType>;

    using CopyL0CToGmforABE = Catlass::Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, YType>;

    using BiasTypeSelector = Catlass::Gemm::helper::L1BiasTypeSelector<BiasType, ElementAccumulator>;

    using CopyGmToL1Bias = std::conditional_t<std::is_same_v<BiasType,void>,void,
                            CubeSelf::Gemm::Tile::CopyGmToL1<
                                ArchTag,
                                typename BiasTypeSelector::GMBiasType,
                                typename BiasTypeSelector::L1BiasType>>;
    
    using CopyL1ToBT = std::conditional_t<std::is_same_v<BiasType,void>,void,
                            CubeSelf::Gemm::Tile::CopyL1ToBT<ArchTag,
                                typename BiasTypeSelector::L1BiasType,
                                typename BiasTypeSelector::L0BiasType>>;
    
    
};

template <
    // Tag indicating architecture
    class ArchTag,
    // GemmType for A matrix operand;
    class AType,
    // GemmType for B matrix operand;
    class BType,
    // GemmType for C matrix operand;
    class CType,
    // GemvType for X vector operand;
    class XType,
    // GemvType for Y vector operand;
    class YType,
    // GemmType for Bias operand;
    class BiasType = void
>
struct TileCopyFTABonAicMultiVector {
    using FT_L02L1_TYPE = Catlass::Gemv::helper::FT_L02L1_TYPE;
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;

    using ElementAforFT = float;
    using LayoutA = typename AType::Layout;

    using ATypeforFT = Catlass::Gemm::GemmType<ElementAforFT, LayoutA>;

    using ElementX = typename XType::Element;

    using ElementXforFT = float;
    using LayoutX = typename XType::Layout;

    using XTypeforFT = Catlass::Gemm::GemmType<ElementXforFT, LayoutX>;

    using CopyGmToL1B = CubeSelf::Gemm::Tile::CopyGmToL1<ArchTag, BType>;


    // using LayoutXInL1 = typename std::conditional<
    //     std::is_same<LayoutX, Catlass::layout::ColumnMajor>::value,
    //     Catlass::layout::nZ,
    //     Catlass::layout::zN>::type;

    // using ElementCInL1 = typename XType::Element;
    using LayoutVX = Catlass::layout::VectorLayout;
    using VXType = Catlass::Gemm::GemmType<ElementXforFT, LayoutVX>;
    using L1XType = typename Catlass::Gemv::helper::L1AndL0TypeSelectorGemv<VXType, ATypeforFT>::L1AType;

    // using LayoutXInL1 = typename CopyGmToL1B::LayoutDst;

    // using L1AType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
    // using L1XType = Catlass::Gemm::GemmType<ElementX,LayoutXInL1, AscendC::TPosition::A1>;
    // using L1XType = typename Catlass::Gemv::helper::L1AndL0TypeSelectorGemv<XType, CTypeForL1>::L1AType;

    // using L0XType = typename Catlass::Gemv::helper::L1AndL0TypeSelectorGemv<XType, CTypeForL1>::L0AType;
    // using LayoutXInL0 = Catlass::layout::nZ;
    // using L0XType = Catlass::Gemm::GemmType<ElementX, LayoutXInL0, AscendC::TPosition::B2>;
    using L0XType = typename Catlass::Gemv::helper::L1AndL0TypeSelectorGemv<VXType, ATypeforFT>::L0AType;

    using ElementAccumulator = 
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementAforFT, ElementXforFT>::ElementAccumulator;

    using CopyGmToL1A = CubeSelf::Gemm::Tile::CopyGmToL1<ArchTag, ATypeforFT>;

    using LayoutAInL1 = typename CopyGmToL1A::LayoutDst;
    
    /*
    template <class ArchTag, class Element>
        struct CopyGmToL1<
        ArchTag, Gemm::GemmType<Element, layout::ColumnMajor>, 
        Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>> {
    */
    using CopyGmToL1VX = Catlass::Gemm::Tile::CopyGmToL1<ArchTag, VXType, L1XType>;
    using CopyGmToL1X = Catlass::Gemm::Tile::CopyGmToL1<ArchTag, XTypeforFT, L1XType>;
    
    using CopyL1ToL0A = CubeSelf::Gemm::Tile::CopyL1ToL0A<
        ArchTag, typename Catlass::Gemm::helper::L1ATypeSelector<ATypeforFT>::L1AType>;
    
    using CopyL1ToL0B = CubeSelf::Gemm::Tile::CopyL1ToL0B<
        ArchTag, typename Catlass::Gemm::helper::L1BTypeSelector<BType>::L1BType>;

    using CopyL1ToL0X = Catlass::Gemm::Tile::CopyL1ToL0A<ArchTag, L1XType, L0XType>;

    using L1ATypeforFT = typename Catlass::Gemm::helper::L1ATypeSelector<ATypeforFT>::L1AType;
    using L0ATypeforFT = typename Catlass::Gemv::helper::L1AndL0TypeSelectorGemv<VXType, ATypeforFT>::L0BType;

    using CopyL1ToL0AforFT = Catlass::Gemm::Tile::CopyL1ToL0B<ArchTag, L1ATypeforFT, L0ATypeforFT>;

    using CopyL0CToGm = CubeSelf::Gemm::Tile::CopyL0CToGm<ArchTag,ElementAccumulator, CType>;

    using CopyL0CToGmforABR = Catlass::Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, YType>;

    using BiasTypeSelector = Catlass::Gemm::helper::L1BiasTypeSelector<BiasType, ElementAccumulator>;

    using CopyGmToL1Bias = std::conditional_t<std::is_same_v<BiasType,void>,void,
                            CubeSelf::Gemm::Tile::CopyGmToL1<
                                ArchTag,
                                typename BiasTypeSelector::GMBiasType,
                                typename BiasTypeSelector::L1BiasType>>;
    
    using CopyL1ToBT = std::conditional_t<std::is_same_v<BiasType,void>,void,
                            CubeSelf::Gemm::Tile::CopyL1ToBT<ArchTag,
                                typename BiasTypeSelector::L1BiasType,
                                typename BiasTypeSelector::L0BiasType>>;
    
    
};

template <
    // Tag indicating architecture
    class ArchTag,
    // GemmType for A matrix operand;
    class AType,
    // GemmType for B matrix operand;
    class BType,
    // GemmType for C matrix operand;
    class CType,
    // GemvType for X vector operand;
    class XType,
    class XTypeCol,
    // GemvType for Y vector operand;
    class YType,
    // GemmType for Bias operand;
    class BiasType = void
>
struct TileCopyFTABonAicAuged {
    using FT_L02L1_TYPE = Catlass::Gemv::helper::FT_L02L1_TYPE;
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;

    using ElementX = typename XType::Element;

    using LayoutX = typename XType::Layout;
    using LayoutXCol = typename XTypeCol::Layout;

    using CopyGmToL1B = CubeSelf::Gemm::Tile::CopyGmToL1<ArchTag, BType>;


    // using LayoutXInL1 = typename std::conditional<
    //     std::is_same<LayoutX, Catlass::layout::ColumnMajor>::value,
    //     Catlass::layout::nZ,
    //     Catlass::layout::zN>::type;

    // using ElementCInL1 = typename XType::Element;
    using LayoutVX = Catlass::layout::VectorLayout;
    using LayoutVXCol = Catlass::layout::ColumnMajor;

    using VXType = Catlass::Gemm::GemmType<ElementX, LayoutVX>;
    using VXTypeCol = Catlass::Gemm::GemmType<ElementX, LayoutVXCol>;

    using L1XType = typename Catlass::Gemv::helper::L1AndL0TypeSelectorGemv<VXType, AType>::L1AType;
    /*
    template<class Element>
    struct L1BTypeSelector<Gemm::GemmType<Element, layout::ColumnMajor>> {
        using L1BType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>;
    };
    */
    using L1XTypeCol = typename Catlass::Gemm::helper::L1BTypeSelector<VXTypeCol>::L1BType;

    // using LayoutXInL1 = typename CopyGmToL1B::LayoutDst;

    // using L1AType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
    // using L1XType = Catlass::Gemm::GemmType<ElementX,LayoutXInL1, AscendC::TPosition::A1>;
    // using L1XType = typename Catlass::Gemv::helper::L1AndL0TypeSelectorGemv<XType, CTypeForL1>::L1AType;

    // using L0XType = typename Catlass::Gemv::helper::L1AndL0TypeSelectorGemv<XType, CTypeForL1>::L0AType;
    
    using LayoutXInL0Col = Catlass::layout::nZ;
    
    using L0XType = typename Catlass::Gemv::helper::L1AndL0TypeSelectorGemv<VXType, AType>::L0AType;
    using L0XTypeCol = Catlass::Gemm::GemmType<ElementX, LayoutXInL0Col, AscendC::TPosition::B2>;

    using ElementAccumulator = 
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using CopyGmToL1A = CubeSelf::Gemm::Tile::CopyGmToL1<ArchTag, AType>;

    using LayoutAInL1 = typename CopyGmToL1A::LayoutDst;
    
    /*
    template <class ArchTag, class Element>
        struct CopyGmToL1<
        ArchTag, Gemm::GemmType<Element, layout::ColumnMajor>, 
        Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>> {
    */
    using CopyGmToL1VX = Catlass::Gemm::Tile::CopyGmToL1<ArchTag, VXType, L1XType>;
    using CopyGmToL1VXCol = Catlass::Gemm::Tile::CopyGmToL1<ArchTag, VXTypeCol, L1XTypeCol>;

    using CopyGmToL1X = Catlass::Gemm::Tile::CopyGmToL1<ArchTag, XType, L1XType>;
    using CopyGmToL1XCol = Catlass::Gemm::Tile::CopyGmToL1<ArchTag, XTypeCol, L1XTypeCol>;
    
    using CopyL1ToL0A = CubeSelf::Gemm::Tile::CopyL1ToL0A<
        ArchTag, typename Catlass::Gemm::helper::L1ATypeSelector<AType>::L1AType>;
    
    using CopyL1ToL0B = CubeSelf::Gemm::Tile::CopyL1ToL0B<
        ArchTag, typename Catlass::Gemm::helper::L1BTypeSelector<BType>::L1BType>;

    using CopyL1ToL0X = Catlass::Gemm::Tile::CopyL1ToL0A<ArchTag, L1XType, L0XType>;
    using CopyL1ToL0XCol = CubeSelf::Gemm::Tile::CopyL1ToL0B<ArchTag, L1XTypeCol, L0XTypeCol>;

    using L1ATypeforFT = typename Catlass::Gemm::helper::L1ATypeSelector<AType>::L1AType;
    using L0ATypeforFT = typename Catlass::Gemv::helper::L1AndL0TypeSelectorGemv<VXType, AType>::L0BType;

    using CopyL1ToL0AforFT = Catlass::Gemm::Tile::CopyL1ToL0B<ArchTag, L1ATypeforFT, L0ATypeforFT>;

    using CopyL0CToGm = CubeSelf::Gemm::Tile::CopyL0CToGm<ArchTag,ElementAccumulator, CType>;

    using CopyL0CToGmforABE = Catlass::Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, YType>;

    using BiasTypeSelector = Catlass::Gemm::helper::L1BiasTypeSelector<BiasType, ElementAccumulator>;

    using CopyGmToL1Bias = std::conditional_t<std::is_same_v<BiasType,void>,void,
                            CubeSelf::Gemm::Tile::CopyGmToL1<
                                ArchTag,
                                typename BiasTypeSelector::GMBiasType,
                                typename BiasTypeSelector::L1BiasType>>;
    
    using CopyL1ToBT = std::conditional_t<std::is_same_v<BiasType,void>,void,
                            CubeSelf::Gemm::Tile::CopyL1ToBT<ArchTag,
                                typename BiasTypeSelector::L1BiasType,
                                typename BiasTypeSelector::L0BiasType>>;

    //(cSize * 2 + 1) * 16 * 8 * 4B
    /*
    auto h0 = 16; // 当数据类型的位宽为8时，h0 = 32；其他情况下，h0 = 16
    auto w0 = 32 / sizeof(type);  // type代表数据类型
    auto tmpBufferSize = (cSize  * 2 + 1)  * h0 * w0 * sizeof(type);
    */                            
    // (cSize  * 2 + 1)  * h0 * w0 * sizeof(type);
    // TRANSPOSE_NHWC2NCHW 
    
    
};

template <
    /// Tag indicating architecture
    class ArchTag,
    class TensorA,
    class LayoutTagA,
    class TensorB,
    class LayoutTagB,
    class TensorC,
    class LayoutTagC,
    class TensorBias = void,
    class LayoutTagBias = void
>
struct PackedTileCopyTla {
    using ElementA = typename TensorA::Element;
    using ElementB = typename TensorB::Element;
    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using LayoutL1A = Catlass::detail::TagToLayout_t<ElementA,
        typename Catlass::Gemm::helper::L1ATypeSelector<Catlass::Gemm::GemmType<ElementA, LayoutTagA>>::L1AType::Layout>;
    using LayoutL1B = Catlass::detail::TagToLayout_t<ElementB,
        typename Catlass::Gemm::helper::L1BTypeSelector<Catlass::Gemm::GemmType<ElementB, LayoutTagB>>::L1BType::Layout>;

    using LayoutL0A = Catlass::detail::TagToLayout_t<ElementA,Catlass::layout::zZ>;
    using LayoutL0B = Catlass::detail::TagToLayout_t<ElementB,Catlass::layout::nZ>;

    using LayoutL0C = typename Catlass::detail::LayoutL0C;

    using TensorL1A = tla::Tensor<AscendC::LocalTensor<ElementA>,LayoutL1A,AscendC::TPosition::A1>;
    using TensorL1B = tla::Tensor<AscendC::LocalTensor<ElementB>,LayoutL1B,AscendC::TPosition::B1>;

    using TensorL0A = tla::Tensor<AscendC::LocalTensor<ElementA>,LayoutL0A,AscendC::TPosition::A2>;
    using TensorL0B = tla::Tensor<AscendC::LocalTensor<ElementB>,LayoutL0B,AscendC::TPosition::B2>;
    using TensorL0C = tla::Tensor<AscendC::LocalTensor<ElementAccumulator>,LayoutL0C,AscendC::TPosition::CO1>;

    using L1AAlignHelper = Catlass::Gemm::helper::L1AlignHelper<ElementA, LayoutTagA>;
    using L1BAlignHelper = Catlass::Gemm::helper::L1AlignHelper<ElementB, LayoutTagB>;

    using CopyGmToL1A = CubeSelf::Gemm::Tile::TileCopyTla<ArchTag, TensorA,TensorL1A>;
    using CopyGmToL1B = CubeSelf::Gemm::Tile::TileCopyTla<ArchTag, TensorB,TensorL1B>;

    using CopyL1ToL0A = CubeSelf::Gemm::Tile::TileCopyTla<ArchTag, TensorL1A, TensorL0A>;
    using CopyL1ToL0B = CubeSelf::Gemm::Tile::TileCopyTla<ArchTag, TensorL1B, TensorL0B>;

    using CopyL0CToGm = CubeSelf::Gemm::Tile::CopyL0CToGmTla<ArchTag, TensorL0C, TensorC>;
};

template <
    /// Tag indicating architecture
    class ArchTag,
    class TensorA,
    class LayoutTagA,
    class TensorB,
    class LayoutTagB,
    class TensorC,
    class LayoutTagC,
    class TensorBias = void,
    class LayoutTagBias = void,
    bool IS_PADDING_A = false,
    bool IS_PADDING_B = false
>
struct PaddingPackedTileCopyTla {
    static_assert(std::is_same_v<LayoutTagA, Catlass::layout::RowMajor> || std::is_same_v<LayoutTagA, Catlass::layout::ColumnMajor>,
        "Unsupported layout, only can be RowMajor and ColumnMajor");
    static_assert(std::is_same_v<LayoutTagB, Catlass::layout::RowMajor> || std::is_same_v<LayoutTagB, Catlass::layout::ColumnMajor>,
        "Unsupported layout, only can be RowMajor and ColumnMajor");
    using ElementA = typename TensorA::Element;
    using ElementB = typename TensorB::Element;
    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using LayoutTagL1A = typename Catlass::Gemm::helper::L1ATypeSelector<Catlass::Gemm::GemmType<ElementA, LayoutTagA>>::L1AType::Layout;
    using LayoutTagL1B = typename Catlass::Gemm::helper::L1BTypeSelector<Catlass::Gemm::GemmType<ElementB, LayoutTagB>>::L1BType::Layout;
    using LayoutL1A = Catlass::detail::TagToLayout_t<ElementA, LayoutTagL1A>;
    using LayoutL1B = Catlass::detail::TagToLayout_t<ElementB, LayoutTagL1B>;
    using LayoutL0A = Catlass::detail::TagToLayout_t<ElementA, Catlass::layout::zZ>;
    using LayoutL0B = Catlass::detail::TagToLayout_t<ElementB, Catlass::layout::nZ>;
    using LayoutL0C = typename Catlass::detail::LayoutL0C;

    using TensorL1A = tla::Tensor<AscendC::LocalTensor<ElementA>, LayoutL1A, AscendC::TPosition::A1>;
    using TensorL1B = tla::Tensor<AscendC::LocalTensor<ElementB>, LayoutL1B, AscendC::TPosition::A1>;
    using TensorL0A = tla::Tensor<AscendC::LocalTensor<ElementA>, LayoutL0A, AscendC::TPosition::A2>;
    using TensorL0B = tla::Tensor<AscendC::LocalTensor<ElementB>, LayoutL0B, AscendC::TPosition::B2>;
    using TensorL0C = tla::Tensor<AscendC::LocalTensor<ElementAccumulator>, LayoutL0C, AscendC::TPosition::CO1>;

    using L1AAlignHelper = Catlass::Gemm::helper::L1AlignHelper<ElementA, LayoutTagA>;
    using L1BAlignHelper = Catlass::Gemm::helper::L1AlignHelper<ElementB, LayoutTagB>;

    using LayoutPaddingTagA = std::conditional_t<std::is_same_v<LayoutTagA, Catlass::layout::RowMajor>,
        Catlass::layout::PaddingRowMajor, Catlass::layout::PaddingColumnMajor>;
    using LayoutPaddingTagB = std::conditional_t<std::is_same_v<LayoutTagB, Catlass::layout::RowMajor>,
        Catlass::layout::PaddingRowMajor, Catlass::layout::PaddingColumnMajor>;

    using CopyGmToL1A = std::conditional_t<
        IS_PADDING_A,
        Gemm::Tile::TileCopyTlaExt<ArchTag, TensorA, TensorL1A, LayoutPaddingTagA, LayoutTagL1A>,
        Gemm::Tile::TileCopyTlaExt<ArchTag, TensorA, TensorL1A, LayoutTagA, LayoutTagL1A>
    >;
    using CopyGmToL1B = std::conditional_t<
        IS_PADDING_B,
        Gemm::Tile::TileCopyTlaExt<ArchTag, TensorB, TensorL1B, LayoutPaddingTagB, LayoutTagL1B>,
        Gemm::Tile::TileCopyTlaExt<ArchTag, TensorB, TensorL1B, LayoutTagB, LayoutTagL1B>
    >;

    using CopyL1ToL0A = Gemm::Tile::TileCopyTla<ArchTag, TensorL1A, TensorL0A>;
    using CopyL1ToL0B = Gemm::Tile::TileCopyTla<ArchTag, TensorL1B, TensorL0B>;
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGmTla<ArchTag, TensorL0C, TensorC>;
};

///////////////////////////////////
/// new add
template<
    /// Tag indicating architecture
    class ArchTag,
    /// GemmType for A matrix operand
    class AType,
    /// GemmType type for B matrix operand
    class BType,
    /// GemmType type for C matrix operand
    class CType,
    /// GemmType type for Bias operand
    class BiasType = void
>
struct TileCopyGemm {

    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;

    using ElementAccumulator = 
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA,ElementB>::ElementAccumulator;
    
    
    // change structual
    using L1AType = typename Catlass::Gemm::helper::L1AndL0TypeSelectorGemm<AType,BType>::L1AType;
    using L1BType = typename Catlass::Gemm::helper::L1AndL0TypeSelectorGemm<AType,BType>::L1BType;

    using L0AType = typename Catlass::Gemm::helper::L1AndL0TypeSelectorGemm<AType,BType>::L0AType;
    using L0BType = typename Catlass::Gemm::helper::L1AndL0TypeSelectorGemm<AType,BType>::L0BType;

    using CopyGmToL1A = CubeSelf::Gemm::Tile::CopyGmToL1<ArchTag,AType,L1AType>;
    using CopyGmToL1B = CubeSelf::Gemm::Tile::CopyGmToL1<ArchTag,BType,L1BType>;
    using CopyL1ToL0A = CubeSelf::Gemm::Tile::CopyL1ToL0A<ArchTag,L1AType,L0AType>;
    using CopyL1ToL0B = CubeSelf::Gemm::Tile::CopyL1ToL0B<ArchTag,L1BType,L0BType>;

    using CopyL0CToGm = CubeSelf::Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, CType>;

};
//////////////////////////////

} // namespace CubeSelf::Gemm::Tile

#endif // CATLASS_GEMM_TILE_TILE_COPY_HPP_SELF

/*
CubeSelf::KernelAdapter<
    CubeSelf::Gemm::Kernel::MatmulEpilogue<
        CubeSelf::Gemm::Block::BlockMmad<CubeSelf::Gemm::MmadAtlasA2Pingpong<true>, 
        Catlass::GemmShape<128, 256, 256>, 
        Catlass::GemmShape<128, 256, 64>, 
        Catlass::Gemm::GemmType<half, Catlass::layout::RowMajor, AscendC::TPosition::GM>, 
        Catlass::Gemm::GemmType<half, Catlass::layout::RowMajor, AscendC::TPosition::GM>,
        Catlass::Gemm::GemmType<half, Catlass::layout::RowMajor, AscendC::TPosition::GM>, 
        void, 
        CubeSelf::Gemm::Tile::TileCopy<Catlass::Arch::AtlasA2, 
            Catlass::Gemm::GemmType<half, Catlass::layout::RowMajor, AscendC::TPosition::GM>, 
            Catlass::Gemm::GemmType<half, Catlass::layout::RowMajor, AscendC::TPosition::GM>, 
            Catlass::Gemm::GemmType<half, Catlass::layout::RowMajor, AscendC::TPosition::GM>>,
        CubeSelf::Gemm::Tile::TileMmad<Catlass::Arch::AtlasA2,
            Catlass::Gemm::GemmType<half, Catlass::layout::RowMajor, AscendC::TPosition::GM>, 
            Catlass::Gemm::GemmType<half, Catlass::layout::RowMajor, AscendC::TPosition::GM>, 
            void>>, 
        Catlass::Epilogue::Block::BlockEpilogue<Catlass::Epilogue::EpilogueAtlasA2ElemWiseOneSource, Catlass::Gemm::GemmType<half, Catlass::layout::RowMajor, AscendC::TPosition::GM>, Catlass::Gemm::GemmType<half, Catlass::layout::RowMajor, AscendC::TPosition::GM>, Catlass::Gemm::GemmType<half, Catlass::layout::RowMajor, AscendC::TPosition::GM>, Catlass::Epilogue::Tile::TileElemWiseAdd<Catlass::Arch::AtlasA2, Catlass::Gemm::GemmType<half, Catlass::layout::RowMajor, AscendC::TPosition::GM>, 16384>, Catlass::Epilogue::Tile::TileCopy<Catlass::Arch::AtlasA2, Catlass::Gemm::GemmType<half, Catlass::layout::RowMajor, AscendC::TPosition::GM>, Catlass::Gemm::GemmType<half, Catlass::layout::RowMajor, AscendC::TPosition::GM>, Catlass::Gemm::GemmType<half, Catlass::layout::RowMajor, AscendC::TPosition::GM>>>, CubeSelf::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>>>
*/