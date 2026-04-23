#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_HPP_SELF
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_HPP_SELF
// catlass/ catlass/


#include "catlass/catlass.hpp"
#include "gemm/tile/tile_copy.hpp"
#include "gemm/tile/tile_mmad.hpp"
#include "catlass/gemv/helper.hpp"

namespace CubeSelf::Gemm::Block{

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
struct BlockMmad {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};

/*
template<
    bool ENABLE_UNIT_FLAG_,
    class L1TileShape_,
    class L0TileShape_,
    class AType_,
    class BType_,
    class CType_,
    class BiasType_,
    class TileCopy_,
    class TileMmad_
>
struct BlockMmadPreload<
    CubeSelf::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L0TileShape_,
    AType_,
    BType_,
    CType_,
    BiasType_,
    TileCopy_,
    TileMmad_
>
*/

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
struct BlockMmadPreload {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadPreload is not implemented for this DispatchPolicy");
};


/*
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
*/
// using TileMmadAIC = Gemm::Tile::TileMmad<typename GEMVAICDispatchPolicy::ArchTag, XType, CType, BiasType>;
// class TileMmadforFT = CubeSelf::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag,>
//  = Catlass::Gemv::helper::FT_L02L1_TYPE::FIX_PIPE

/*
 template <
    class ArchTag,
    class AType,
    class BType,
    class CType,
    class XType,
    class YType,
    class BiasType = void,
    Catlass::Gemv::helper::FT_L02L1_TYPE COPY_TYPE_ = Catlass::Gemv::helper::FT_L02L1_TYPE::FIX_PIPE
>
struct TileCopyFT
*/

template<
    class DispatchPolicy,
    Catlass::Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    Catlass::Gemv::helper::FT_L02L1_TYPE COPY_TYPE_,
    class L1TileShape,
    class L0TileShape,
    class L0TileShapeforFT,
    class AType,
    class BType,
    class CType,
    class XType,
    class YType,
    class BiasType = void,
    class TileCopyFT = CubeSelf::Gemm::Tile::TileCopyFT<typename DispatchPolicy::ArchTag, AType, BType, CType, XType, YType, BiasType, COPY_TYPE_>,
    class TileMmad = CubeSelf::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>
>
struct BlockMmadFTNOSPLIT {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};

/*
struct BlockMmadFTABeNoSplitK<
    CubeSelf::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L1TileShapeforFT_,
    L0TileShape_,
    L0TileShapeforFT_,
    AType_,
    BType_,
    CType_,
    XType_,
    YType_,
    BiasType_,
    TileCopyFTABonAic_,
    TileMmad_
>
*/
template<
    class DispatchPolicy,
    class L1TileShape,
    class L1TileShapeforFT,
    class L0TileShape,
    class L0TileShapeforFT,
    class AType,
    class BType,
    class CType,
    class XType,
    class YType,
    class BiasType = void,
    class TileCopyFTABonAic = CubeSelf::Gemm::Tile::TileCopyFTABonAic<typename DispatchPolicy::ArchTag, AType, BType, CType, XType, YType, BiasType>,
    class TileMmad = CubeSelf::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>
>
struct BlockMmadFTABeNoSplitK {
    /*
    L1TileShape_,
    L1TileShapeforFT_,
    L0TileShape_,
    L0TileShapeforFT_,
    */
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};

template<
    class DispatchPolicy,
    class L1TileShape,
    class L1TileShapeforFT,
    class L0TileShape,
    class L0TileShapeforFT,
    class AType,
    class BType,
    class CType,
    class XType,
    class XColType,
    class YType,
    class BiasType = void,
    class TileCopyFTABonAicAuged = CubeSelf::Gemm::Tile::TileCopyFTABonAicAuged<typename DispatchPolicy::ArchTag, AType, BType, CType, XType, XColType, YType, BiasType>,
    class TileMmad = CubeSelf::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>
>
struct BlockMmadFTABeAugedNoSplitK{
    /*
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
    */
    /*
    CubeSelf::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L1TileShapeforFT_,
    L0TileShape_,
    L0TileShapeforFT_,
    AType_,
    BType_,
    CType_,
    XType_,
    XColType_,
    YType_,
    BiasType_,
    TileCopyFTABonAicAuged_,
    TileMmad_
    */
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadFTABeAugedNoSplitK is not implemented for this DispatchPolicy");
};


template<
    class DispatchPolicy,
    class L1TileShape,
    class L1TileShapeforFT,
    class L0TileShape,
    class L0TileShapeforFT,
    class AType,
    class BType,
    class CType,
    class XType,
    class XColType,
    class YType,
    class BiasType = void,
    class TileCopyFTABonAicAuged = CubeSelf::Gemm::Tile::TileCopyFTABonAicAuged<typename DispatchPolicy::ArchTag, AType, BType, CType, XType, XColType, YType, BiasType>,
    class TileMmad = CubeSelf::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>
>
struct BlockMmadFTABeAugedNoSplitKGemv{
    /*
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
    */
    /*
    CubeSelf::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L1TileShapeforFT_,
    L0TileShape_,
    L0TileShapeforFT_,
    AType_,
    BType_,
    CType_,
    XType_,
    XColType_,
    YType_,
    BiasType_,
    TileCopyFTABonAicAuged_,
    TileMmad_
    */
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadFTABeAugedNoSplitKGemv is not implemented for this DispatchPolicy");
};

template<
    class DispatchPolicy,
    class L1TileShape,
    class L1TileShapeforFT,
    class L0TileShape,
    class L0TileShapeforFT,
    class AType,
    class BType,
    class CType,
    class XType,
    class XColType,
    class YType,
    class BiasType = void,
    class TileCopyFTABonAicAuged = CubeSelf::Gemm::Tile::TileCopyFTABonAicAuged<typename DispatchPolicy::ArchTag, AType, BType, CType, XType, XColType, YType, BiasType>,
    class TileMmad = CubeSelf::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>
>
struct BlockMmadFTABeAugedNoSplitKRobust{
    /*
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
    */
    /*
    CubeSelf::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L1TileShapeforFT_,
    L0TileShape_,
    L0TileShapeforFT_,
    AType_,
    BType_,
    CType_,
    XType_,
    XColType_,
    YType_,
    BiasType_,
    TileCopyFTABonAicAuged_,
    TileMmad_
    */
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadFTABeAugedNoSplitKRobust is not implemented for this DispatchPolicy");
};

template<
    class DispatchPolicy,
    Catlass::Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    Catlass::Gemv::helper::FT_L02L1_TYPE COPY_TYPE_,
    class L1TileShape,
    class L0TileShape,
    class L0TileShapeforFT,
    class AType,
    class BType,
    class CType,
    class XType,
    class YType,
    class BiasType = void,
    class TileCopyFT = CubeSelf::Gemm::Tile::TileCopyFT<typename DispatchPolicy::ArchTag, AType, BType, CType, XType, YType, BiasType, COPY_TYPE_>,
    class TileMmad = CubeSelf::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>
>
struct BlockMmadFTSpiltK {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadFTSpiltK is not implemented for this DispatchPolicy");
};

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
struct BlockMmadSpecABeNoSplitK {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadSpecABeNoSplitK is not implemented for this DispatchPolicy");
};

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
struct BlockMmadSpecABeNoSplitKRobust {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadSpecABeNoSplitK is not implemented for this DispatchPolicy");
};

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
    class TileCopyFTABonAic = CubeSelf::Gemm::Tile::TileCopyFTABonAicMultiVector<typename DispatchPolicy::ArchTag, AType, BType, CType, XType, YType, BiasType>,
    class TileMmad = CubeSelf::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, XType, BiasType>
>
struct BlockMmadSpecABrNoSplitKRobust {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadSpecABrNoSplitK is not implemented for this DispatchPolicy");
};

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
struct BlockMmadSpecAeNoSplitKRobust {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadSpecAeNoSplitK is not implemented for this DispatchPolicy");
};

/*
struct BlockMmadFTABeNoSplitKRobust<
    CubeSelf::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L1TileShapeforFT_,
    L0TileShape_,
    L0TileShapeforFT_,
    AType_,
    BType_,
    CType_,
    XType_,
    YType_,
    BiasType_,
    TileCopyFTABonAic_,
    TileMmad_
>
*/

template<
    class DispatchPolicy,
    class L1TileShape,
    class L1TileShapeforFT,
    class L0TileShape,
    class L0TileShapeforFT,
    class AType,
    class BType,
    class CType,
    class XType,
    class YType,
    class BiasType = void,
    class TileCopyFTABonAic = CubeSelf::Gemm::Tile::TileCopyFTABonAic<typename DispatchPolicy::ArchTag, AType, BType, CType, XType, YType, BiasType>,
    class TileMmad = CubeSelf::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>
>
struct BlockMmadFTABeNoSplitKRobust {
    /*
    L1TileShape_,
    L1TileShapeforFT_,
    L0TileShape_,
    L0TileShapeforFT_,
    */
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};

template <
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
struct BlockMmadFault {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadFault is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    class L1TileShape,
    class L0TileShape,
    class TensorA,
    class TensorB,
    class TensorC,
    class TensorBias = void,
    class TileCopy = CubeSelf::Gemm::Tile::PackedTileCopyTla<
        typename DispatchPolicy::ArchTag, 
        TensorA, Catlass::layout::RowMajor,
        TensorB, Catlass::layout::RowMajor, 
        TensorC, Catlass::layout::RowMajor, 
        TensorBias, Catlass::layout::RowMajor>,
    class TileMmad = CubeSelf::Gemm::Tile::TileMmadTla<
        typename DispatchPolicy::ArchTag, typename TileCopy::TensorL0A,
        typename TileCopy::TensorL0B, typename TileCopy::TensorL0C>
>
struct BlockMmadTla {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadTla is not implemented for this DispatchPolicy");
};

/// new add for the reason that i am using the dispatchpolicy which is same as the policy of the optimized_matmul
// so i add a new one class to avoid the conflict
template<
    class DispatchPolicy,
    class L1TileShape,
    class L0TileShape,
    class AType,
    class BType,
    class CType,
    class BiasType = void,
    class TileCopy = CubeSelf::Gemm::Tile::TileCopyGemm<
        typename DispatchPolicy::ArchTag, AType, BType, CType, BiasType>,
    class TileMmad = CubeSelf::Gemm::Tile::TileMmad<
        typename DispatchPolicy::ArchTag, AType, BType, BiasType>
>
struct BlockGemm {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};

} // namespace CubeSelf::Gemm::Block

// catlass/ catlass/
#include "gemm/block/block_mmad_pingpong.hpp"
#include "gemm/block/block_mmad_pingpong_bias.hpp"
#include "gemm/block/block_mmad_pingpong_fault_no_splitk.hpp"
#include "gemm/block/block_mmad_pingpong_fault_splitk.hpp"
// examples/cube_op_self/gemm/block/block_mmad_pingpong_fault_abe_no_splitk.hpp
#include "gemm/block/block_mmad_pingpong_fault_abe_no_splitk.hpp"
// examples/cube_op_self/gemm/block/block_mmad_pingpong_fault_no_splitk.hpp
// examples/cube_op_self/gemm/block/block_mmad_pingpong_fault_abe_spec_no_splitk.hpp
#include "gemm/block/block_mmad_pingpong_fault_abe_spec_no_splitk.hpp"
#include "gemm/block/block_mmad_pingpong_fault_abe_spec_no_splitk_robust.hpp"
#include "gemm/block/block_mmad_pingpong_fault_abe_no_splitk_robust.hpp"
#include "gemm/block/block_mmad_pingpong_fault_abe_auged_no_splitk_robust.hpp"
#include "gemm/block/block_mmad_pingpong_fault_abe_auged_no_splitk.hpp"
#include "gemm/block/block_mmad_pingpong_preload.hpp"
#include "gemm/block/block_mmad_pingpong_fault_abe_auged_no_splitk_gemv.hpp"
#include "gemm/block/block_mmad_pingpong_fault_abr_spec_no_splitk_robust.hpp"
#include "gemm/block/block_mmad_pingpong_fault_ae_spec_no_splitk_robust.hpp"
#endif
