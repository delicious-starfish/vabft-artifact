#ifndef CATLASS_GEMV_BLOCK_BLOCK_MATRIX_ADD_HPP
#define CATLASS_GEMV_BLOCK_BLOCK_MATRIX_ADD_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemv_coord.hpp"
#include "catlass/gemv/helper.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemv/tile/tile_matmul_elem_add.hpp"
// #include "catlass/gemv/tile/tile_vmad.hpp"
// #include "catlass/gemv/tile/tile_fault_sum.hpp"

namespace Catlass::Gemv::Block {

template <
    class UBTileShape_,
    class UBBlockShape_,
    class L1TileShape_,
    class AType_,
    class YType_,
    class BiasType_,
    class TileCopy_,
    class TileMatrixAdd_
>
struct BlockMatrixAdd <
    Gemm::GemvAtlasA2,
    UBTileShape_,
    UBBlockShape_,
    L1TileShape_,
    AType_,
    YType_,
    BiasType_,
    TileCopy_,
    TileMatrixAdd_
> {
public:
    // Type Aliases
    using DispatchPolicy = Gemm::GemvAtlasA2;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using UBTileShape = UBTileShape_;
    using UBBlockShape = UBBlockShape_;
    using L1TileShape = L1TileShape_;

    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;

    using ElementY = typename YType_::Element;
    using LayoutY = typename YType_::Layout;

    using TileMatrixAdd = TileMatrixAdd_;

    // // the function of aiv
    // using MatrixCopyGmToUb = Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;
    // using MatrixCopyUbtoGm = Gemv::Tile::MatrixCopyUBToGm<ArchTag, YType>;

    using MatrixCopyGmToUb = typename TileCopy_::MatrixCopyGmToUb;
    using MatrixCopyUbtoGm = typename TileCopy_::MatrixCopyUbtoGm;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    using UBAlignHelper = Gemv::helper::UBAlignHelper<ElementA>;

    using TensorCoord = layout::VectorLayout::TensorCoord;

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t Abuf_SOURCE_SIZE1_ = 64 * 1024;
    static constexpr uint32_t Abuf_SOURCE_SIZE2_ = 64 * 1024;
    static constexpr uint32_t Ybuf_SIZE_ = 64 * 1024;

    static_assert(L1TileShape::M == UBBlockShape::M,
        "The situation where the basic Tile of UB and L1 for MMA differ on the m axes is not supported yet");

    static_assert(UBTileShape::N == L1TileShape::N,
        "The situation where the basic Tile of UB and L1 for MMA differ on the n axes is not supported yet");

    static_assert(UBBlockShape::N == UBTileShape::N,
        "The situation where the basic Tile of UB and L1 for MMA differ on the n axes is not supported yet");


    CATLASS_DEVICE
    BlockMatrixAdd() {}

    /// Construct
    CATLASS_DEVICE
    BlockMatrixAdd(Arch::Resource<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbA1Offset = UBufAddrStart;
        uint32_t UbA2Offset = UBufAddrStart + Abuf_SOURCE_SIZE1_;
        uint32_t UbYOffset = UBufAddrStart + Abuf_SOURCE_SIZE1_ + Abuf_SOURCE_SIZE2_;

        UbA1Tensor = resource.ubBuf.template GetBufferByByte<ElementA>(UbA1Offset);
        UbA2Tensor = resource.ubBuf.template GetBufferByByte<ElementA>(UbA2Offset);
        UbYTensor = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset);

        UbInA1Event = 0;
        UbInA2Event = 1;
        UbOutEvent = 2;

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInA1Event);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInA2Event);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutEvent);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutEvent);
    }

    /// Construct
    CATLASS_DEVICE
    BlockMatrixAdd(Arch::ResourceAIV<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbA1Offset = UBufAddrStart;
        uint32_t UbA2Offset = UBufAddrStart + Abuf_SOURCE_SIZE1_;
        uint32_t UbYOffset = UBufAddrStart + Abuf_SOURCE_SIZE1_ + Abuf_SOURCE_SIZE2_;

        UbA1Tensor = resource.ubBuf.template GetBufferByByte<ElementA>(UbA1Offset);
        UbA2Tensor = resource.ubBuf.template GetBufferByByte<ElementA>(UbA2Offset);
        UbYTensor = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset);

        UbInA1Event = 0;
        UbInA2Event = 1;
        UbOutEvent = 2;

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInA1Event);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInA2Event);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutEvent);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutEvent);
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockMatrixAdd()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInA1Event);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInA2Event);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(UbOutEvent);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutEvent);
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmA1, 
        AscendC::GlobalTensor<ElementA> const &gmA2,
        LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementY> const &gmY, LayoutY const &layoutY,
        GemvCoord const &actualShape, uint32_t aiv_part_num)
    {
        
        TileMRound = RoundUp(UBTileShape::M, UBAlignHelper::ALIGN);
        TileNRound = RoundUp(UBTileShape::N, UBAlignHelper::ALIGN);
        
        BlockMRound = RoundUp(UBBlockShape::M, UBAlignHelper::ALIGN);
        BlockNRound = RoundUp(UBBlockShape::N, UBAlignHelper::ALIGN);

        m_actual_total = (actualShape.m() < BlockMRound) ? actualShape.m() : BlockMRound;
        n_actual_total = (actualShape.n() < BlockNRound) ? actualShape.n() : BlockNRound;


        m_actual_part = m_actual_total / aiv_part_num;

        uint32_t M_start_offset = AscendC::GetSubBlockIdx() * m_actual_part;

        if(AscendC::GetSubBlockIdx() == (aiv_part_num -1)) {
            m_actual_part = m_actual_total - M_start_offset;
        }
        

        y_actual_total = m_actual_total;
        x_actual_total = n_actual_total;

        uint32_t A_row_offset = M_start_offset;
        uint32_t A_col_offset = 0;
        uint32_t A_block_offset = A_row_offset * layoutA.stride(0) + A_col_offset * layoutA.stride(1);

        uint32_t Y_row_offset = M_start_offset;
        uint32_t Y_col_offset = 0;
        uint32_t Y_block_offset = Y_row_offset * layoutY.stride(0) + Y_col_offset * layoutY.stride(1);

        m_actual = m_actual_part;
        n_actual = n_actual_total;

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutEvent);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInA1Event));
        auto layoutA1InUb = layoutA.GetTileLayout(MakeCoord(m_actual_part, TileNRound));
        auto layoutTileA1 = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
        matrixCopyGmToUb(UbA1Tensor, gmA1[A_block_offset], layoutA1InUb, layoutTileA1);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInA1Event));

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInA2Event));
        auto layoutA2InUb = layoutA.GetTileLayout(MakeCoord(m_actual_part, TileNRound));
        auto layoutTileA2 = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
        matrixCopyGmToUb(UbA2Tensor, gmA2[A_block_offset], layoutA2InUb, layoutTileA2);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInA2Event));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(UbOutEvent);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(UbOutEvent);
        auto layoutComputeInUb = layoutA.GetTileLayout(MakeCoord(m_actual_part, TileNRound));
        auto layoutTileCompute = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));

        /*
        CATLASS_DEVICE
        void operator()(
            AscendC::LocalTensor<ElementY> dstTensor,
            AscendC::LocalTensor<ElementA> srcTensor_m1,
            AscendC::LocalTensor<ElementA> srcTensor_m2,
            LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
        */
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInA1Event));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInA2Event));
        tileMatrixAdd(UbYTensor,UbA1Tensor,UbA2Tensor,layoutComputeInUb,layoutTileCompute);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInA1Event));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInA2Event));

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEvent));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEvent));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(UbOutEvent);

        // auto layoutXInL1 = LayoutXInL1::template MakeLayout<ElementX>(L1XAlignHelper::M_ALIGNED, L1TileShape::N);
        auto layoutYInUb = LayoutY::template MakeLayoutInUb<ElementY>(MakeCoord(m_actual_part, TileNRound));;
        auto layoutDstY = layoutY.GetTileLayout(MakeCoord(m_actual, n_actual));

        // vecCopyUbToGm(gmZ[TileY_Row_offset], UbYTensorList[UbOutListId], layoutDstY, layoutComputeInUb);
        /*
        CATLASS_DEVICE
        void operator()(
            AscendC::GlobalTensor<Element> const &dstTensor,
            AscendC::LocalTensor<Element> const &srcTensor,
            LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
        */
        matrixCopyUbToGm(gmY[Y_block_offset], UbYTensor, layoutDstY, layoutYInUb);

        AscendC::PipeBarrier<PIPE_MTE3>();

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutEvent);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutEvent);
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> UbA1Tensor;
    AscendC::LocalTensor<ElementA> UbA2Tensor;
    AscendC::LocalTensor<ElementY> UbYTensor;

    // Multi-stage event id list
    int32_t UbInA1Event;
    int32_t UbInA2Event;
    int32_t UbOutEvent;

    uint32_t m_actual, n_actual, x_actual, y_actual;
    uint32_t m_actual_total, n_actual_total, x_actual_total, y_actual_total;
    uint32_t m_actual_part;
    uint32_t TileMRound, TileNRound;
    uint32_t BlockMRound, BlockNRound;
    uint32_t TaskSplit;
    uint32_t MatrixOffset;
    uint32_t strideACol, strideARow;
    uint32_t strideOut;
    uint32_t splitNnum;
    uint32_t tileMnum;

    TileMatrixAdd tileMatrixAdd;

    MatrixCopyGmToUb matrixCopyGmToUb;
    MatrixCopyUbtoGm matrixCopyUbToGm;
};

} // namespace Catlass::Gemv::Block

#endif // CATLASS_GEMV_BLOCK_BLOCK_GEMV_AIV_HPP
