#ifndef CATLASS_GEMV_BLOCK_BLOCK_SUM_AIV_BEFT_NOSPLITK_HPP
#define CATLASS_GEMV_BLOCK_BLOCK_SUM_AIV_BEFT_NOSPLITK_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemv_coord.hpp"
#include "catlass/gemv/helper.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemv/tile/tile_vmad.hpp"
#include "catlass/gemv/tile/tile_fault_sum.hpp"

namespace Catlass::Gemv::Block {

template <
    class UBTileShape_,
    class UBBlockShape_,
    class L1TileShape_,
    class AType_,
    class XType_,
    class YType_,
    class BiasType_,
    class TileCopy_,
    class TileFaultSum_
>
struct BlockFTSumNoSplitK <
    Gemm::GemvAtlasA2,
    Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR,
    Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::ABE_FUSED_THRE,
    UBTileShape_,
    UBBlockShape_,
    L1TileShape_,
    AType_,
    XType_,
    YType_,
    BiasType_,
    TileCopy_,
    TileFaultSum_
> {
public:
    // Type Aliases
    using DispatchPolicy = Gemm::GemvAtlasA2;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using UBTileShape = UBTileShape_;
    using UBBlockShape = UBBlockShape_;
    using L1TileShape = L1TileShape_;

    using FT_AIV_PIPE_FUSE_TYPE = Gemv::helper::FT_AIV_PIPE_FUSE_TYPE;
    using FT_THRESHOLD_ALGORITHM = Gemv::helper::FT_THRESHOLD_ALGORITHM;

    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;

    using LayoutACol = typename std::conditional<
        std::is_same<LayoutA, Catlass::layout::RowMajor>::value,
        Catlass::layout::ColumnMajor,
        Catlass::layout::RowMajor>::type;
    using AColType = Gemm::GemmType<ElementA, LayoutACol>;

    using ElementX = typename XType_::Element;
    using LayoutX = typename XType_::Layout;
    using ElementY = typename YType_::Element;
    using LayoutY = typename YType_::Layout;

    using TileFaultSum = TileFaultSum_;

    using VecCopyGmToUb = typename TileCopy_::VecCopyGmToUb;
    using VecCopyUbToGm = typename TileCopy_::VecCopyUbToGm;
    using MatrixCopyGmToUb = typename TileCopy_::MatrixCopyGmToUb;

    using TileCopyCol_ = Gemv::Tile::TileCopyGemvAiv<typename DispatchPolicy::ArchTag, AColType, XType_, YType_, BiasType_>;


    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    using UBAlignHelper = Gemv::helper::UBAlignHelper<ElementA>;

    using TensorCoord = layout::VectorLayout::TensorCoord;

    using FT_REDUCE_TYPE = Gemv::helper::FT_REDUCE_TYPE;

    // ,void

    using TileFaultSumBE = Gemv::Tile::TileFaultSum<ArchTag, FT_REDUCE_TYPE::SUM_MAX, AType_, YType_>;

    static constexpr FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE = Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::ABE_FUSED_THRE;
    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR;


    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t Abuf_SIZE_ = 128 * 1024;
    static constexpr uint32_t YSumbuf_SIZE_ = 16 * 1024;
    static constexpr uint32_t YMaxbuf_SIZE_ = 16 * 1024;
    static constexpr uint32_t workspace_SIZE_ = 32 * 1024;

    static_assert(UBTileShape::N == L1TileShape::N,
        "The situation where the basic Tile of UB and L1 for MMA differ on the n axes is not supported yet");

    static_assert((UBBlockShape::N % UBTileShape::N) == 0,
        "The situation where the basic Tile of UB and L1 for MMA differ on the n axes is not supported yet");


    CATLASS_DEVICE
    BlockFTSumNoSplitK() {}

    /// Construct
    CATLASS_DEVICE
    BlockFTSumNoSplitK(Arch::Resource<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbAOffset = UBufAddrStart;
        uint32_t UbYSumOffset = UBufAddrStart + Abuf_SIZE_;
        uint32_t UbYMaxOffset = UBufAddrStart + Abuf_SIZE_ + YSumbuf_SIZE_;
        uint32_t UbWOffset = UBufAddrStart + Abuf_SIZE_ + YSumbuf_SIZE_ + YMaxbuf_SIZE_;
        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SIZE_ / 2));
            UbYSumTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYSumOffset + i * (YSumbuf_SIZE_ / 2));
            UbYMaxTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYMaxOffset + i * (YMaxbuf_SIZE_ / 2));
            UbWTensorList[i] =
                resource.ubBuf.template GetBufferByByte<ElementA>(UbWOffset + i * (workspace_SIZE_ / 2));

            // Assign event ID for each stages
            UbInAEventList[i] = i;
            UbOutSumEventList[i] = i;
            UbOutMaxEventList[i] = i + STAGES;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutSumEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutMaxEventList[i]);
        }
    }

    /// Construct
    CATLASS_DEVICE
    BlockFTSumNoSplitK(Arch::ResourceAIV<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbAOffset = UBufAddrStart;
        uint32_t UbYSumOffset = UBufAddrStart + Abuf_SIZE_;
        uint32_t UbYMaxOffset = UBufAddrStart + Abuf_SIZE_ + YSumbuf_SIZE_;
        uint32_t UbWOffset = UBufAddrStart + Abuf_SIZE_ + YSumbuf_SIZE_ + YMaxbuf_SIZE_;
        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SIZE_ / 2));
            UbYSumTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYSumOffset + i * (YSumbuf_SIZE_ / 2));
            UbYMaxTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYMaxOffset + i * (YMaxbuf_SIZE_ / 2));
            UbWTensorList[i] =
                resource.ubBuf.template GetBufferByByte<ElementA>(UbWOffset + i * (workspace_SIZE_ / 2));

             // Assign event ID for each stages
            UbInAEventList[i] = i;
            UbOutSumEventList[i] = i;
            UbOutMaxEventList[i] = i + STAGES;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutSumEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutMaxEventList[i]);
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockFTSumNoSplitK()
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(UbOutSumEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(UbOutMaxEventList[i]);
        }
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementY> const &gmZSum, 
        AscendC::GlobalTensor<ElementY> const &gmZMax,
        LayoutY const &layoutY,
        GemvCoord const &actualShape)
    {
        

        TileMRound = RoundUp(UBTileShape::M, UBAlignHelper::ALIGN);
        TileNRound = RoundUp(UBTileShape::N, UBAlignHelper::ALIGN);
        BlockMRound = RoundUp(UBBlockShape::M, UBAlignHelper::ALIGN);
        BlockNRound = RoundUp(UBBlockShape::N, UBAlignHelper::ALIGN);

        strideACol = layoutA.stride(1) * TileNRound;
        strideARow = layoutA.stride(0) * TileMRound;

        strideOut = layoutA.stride(0);

        uint32_t NloopBlock = 1;
        uint32_t MBLoopIdx = 0;

        uint32_t MloopBlock = 1;
        uint32_t NBLoopIdx = 0;

        m_actual_total = (actualShape.m() < BlockMRound) ? actualShape.m() : BlockMRound;
        n_actual_total = (actualShape.n() < BlockNRound) ? actualShape.n() : BlockNRound;

        splitNnum = (n_actual_total + TileNRound - 1) / TileNRound;
        tileMnum = (m_actual_total + TileMRound - 1) / TileMRound;

        uint32_t Nloop = splitNnum;
        uint32_t Mloop = tileMnum;

        y_actual_total = m_actual_total;
        x_actual_total = n_actual_total;

        uint32_t A_row_offset = MBLoopIdx * BlockMRound;
        uint32_t A_col_offset = NBLoopIdx * BlockNRound;
        uint32_t A_block_offset = A_row_offset * layoutA.stride(0) + A_col_offset * layoutA.stride(1);

        m_actual = (m_actual_total < TileMRound) ? m_actual_total : TileMRound;
        n_actual = (n_actual_total < TileNRound) ? n_actual_total : TileNRound;

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
        auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
        auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
        matrixCopyGmToUb(UbATensorList[UbInListId], gmA[A_block_offset], layoutAInUb, layoutTileA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));

        // main loop
        for (uint32_t NLoopIdx = 0; NLoopIdx < Nloop; NLoopIdx++) {
            n_actual = (NLoopIdx == Nloop - 1) ? (n_actual_total - NLoopIdx * TileNRound) : TileNRound;
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutSumEventList[UbOutListId]));
            auto UbYSumTensor = UbYSumTensorList[UbOutListId];
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutMaxEventList[UbOutListId]));
            auto UbYMaxTensor = UbYMaxTensorList[UbOutListId];
            // AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbOutEventList[UbOutListId]));
            // AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbOutEventList[UbOutListId]));
             

            AscendC::Duplicate<ElementY>(UbYSumTensor, (ElementY)0.0, m_actual_total);
            AscendC::Duplicate<ElementY>(UbYMaxTensor, (ElementY)0.0, m_actual_total);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutSumEventList[UbOutListId]));
            // AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutMaxEventList[UbOutListId]));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutSumEventList[UbOutListId]));
            // AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutMaxEventList[UbOutListId]));

            auto UbYSumTile = UbYSumTensor[A_row_offset]; 
            auto UbYMaxTile = UbYMaxTensor[A_row_offset];

            uint32_t TileA_Col_offset = NLoopIdx * strideACol;
            uint32_t TileY_Row_offset = NLoopIdx * strideOut;

            for(uint32_t MLoopIdx = 0; MLoopIdx < Mloop; MLoopIdx++) {
                m_actual = (MLoopIdx == Mloop - 1) ? (m_actual_total - MLoopIdx * TileMRound) : TileMRound;
                y_actual = m_actual;
                x_actual = n_actual;
                uint32_t TileA_Row_offset = MLoopIdx * strideARow;
                uint32_t TileY_col_offset = MLoopIdx * TileMRound;

                uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;

                if (MLoopIdx < (Mloop - 1)) {
                    uint32_t MLoopIdxNext = MLoopIdx + 1;
                    uint32_t m_actual_next = (MLoopIdxNext == Mloop - 1) ? (m_actual_total - MLoopIdxNext * TileMRound) : TileMRound;
                    uint32_t n_actual_next = n_actual;
                            
                    uint32_t y_actual_next = m_actual_next;
                    uint32_t x_actual_next = n_actual_next;
                    uint32_t TileA_Row_offset_next = MLoopIdxNext * strideARow;

                    // Get L1 tensor for next stage
                    auto matrixTensor = UbATensorList[UbInListIdNext];
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                    auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                    matrixCopyGmToUb(matrixTensor, gmA[TileA_Row_offset_next + TileA_Col_offset], layoutAInUb, layoutTileA);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
                }else if((MLoopIdx == (Mloop - 1)) && (NLoopIdx < (Nloop - 1))) {
                    uint32_t NLoopIdxNext = NLoopIdx + 1;
                    uint32_t MLoopIdxNext = 0;
                    uint32_t m_actual_next = (MLoopIdxNext == Mloop - 1) ? (m_actual_total - MLoopIdxNext * TileMRound) : TileMRound;
                    uint32_t n_actual_next = (NLoopIdxNext == Nloop - 1) ? (n_actual_total - NLoopIdxNext * TileNRound) : TileNRound;
                            
                    uint32_t y_actual_next = m_actual_next;
                    uint32_t x_actual_next = n_actual_next;
                    uint32_t TileA_Row_offset_next = MLoopIdxNext * strideARow;
                    uint32_t TileA_Col_offset_next = NLoopIdxNext * strideACol;

                    // Get L1 tensor for next stage
                    auto matrixTensor = UbATensorList[UbInListIdNext];
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                    auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                    matrixCopyGmToUb(matrixTensor, gmA[TileA_Row_offset_next + TileA_Col_offset_next], layoutAInUb, layoutTileA);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
                auto layoutComputeInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileCompute = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));

                /*
                AscendC::LocalTensor<ElementY> dstTensorSum,
                AscendC::LocalTensor<ElementY> dstTensorMax,
                AscendC::LocalTensor<ElementA> srcTensor_m,
                AscendC::LocalTensor<ElementAccumulator> sum_workspace,
                LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
                */
                tileFaultSumBE(UbYSumTile[TileY_col_offset],
                        UbYMaxTile[TileY_col_offset],
                        UbATensorList[UbInListId],
                        UbWTensorList[UbInListId],
                        layoutComputeInUb,
                        layoutTileCompute);
                
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
                UbInListId = UbInListIdNext;
            }

            // uint32_t ubTileOutOffset = TileY_Row_offset;
            auto layoutDstY = layoutY.GetTileLayout(TensorCoord(y_actual_total));
            auto layoutComputeInUb = layoutY.GetTileLayout(TensorCoord(y_actual_total));

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutSumEventList[UbOutListId]));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutSumEventList[UbOutListId]));

            vecCopyUbToGm(gmZSum[TileY_Row_offset], UbYSumTensorList[UbOutListId], layoutDstY, layoutComputeInUb);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutSumEventList[UbOutListId]));


            vecCopyUbToGm(gmZMax[TileY_Row_offset], UbYMaxTensorList[UbOutListId], layoutDstY, layoutComputeInUb);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutMaxEventList[UbOutListId]));
            UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
        }
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> UbATensorList[STAGES];
    AscendC::LocalTensor<ElementY> UbYSumTensorList[STAGES];
    AscendC::LocalTensor<ElementY> UbYMaxTensorList[STAGES];
    AscendC::LocalTensor<ElementA> UbWTensorList[STAGES];

    // Multi-stage event id list
    int32_t UbInAEventList[STAGES];
    int32_t UbOutSumEventList[STAGES];
    int32_t UbOutMaxEventList[STAGES];

    // The id of current stage
    uint32_t UbOutListId{0};
    uint32_t UbInListId{0};

    uint32_t m_actual, n_actual, x_actual, y_actual;
    uint32_t m_actual_total, n_actual_total, x_actual_total, y_actual_total;
    uint32_t TileMRound, TileNRound;
    uint32_t BlockMRound, BlockNRound;
    uint32_t strideACol, strideARow;
    uint32_t strideOut;
    uint32_t splitNnum;
    uint32_t tileMnum;

    TileFaultSumBE tileFaultSumBE;

    MatrixCopyGmToUb matrixCopyGmToUb;
    VecCopyGmToUb vecCopyGmToUb;
    VecCopyUbToGm vecCopyUbToGm;
};

} // namespace Catlass::Gemv::Block

#endif // CATLASS_GEMV_BLOCK_BLOCK_GEMV_AIV_HPP
