#ifndef CATLASS_GEMV_BLOCK_BLOCK_SUM_AIV_AE_BMAX_MIXED_NOSPLITK_HPP
#define CATLASS_GEMV_BLOCK_BLOCK_SUM_AIV_AE_BMAX_MIXED_NOSPLITK_HPP

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
#include "catlass/gemv/tile/tile_vmuls.hpp"

namespace Catlass::Gemv::Block {

template <
    class UBTileShapeforB_,
    class UBBlockShapeforB_,
    class UBTileShapeforA_,
    class L1TileShape_,
    class AType_,
    class BType_,
    class XType_,
    class YType_,
    class BiasType_,
    class TileCopy_,
    class TileFaultSum_
>
struct BlockFTSumNoSplitK <
    Gemm::GemvAtlasA2,
    Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR,
    Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::A_B_MIXED,
    UBTileShapeforB_,
    UBBlockShapeforB_,
    UBTileShapeforA_,
    L1TileShape_,
    AType_,
    BType_,
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
    using UBTileShapeforB = UBTileShapeforB_;
    using UBBlockShapeforB = UBBlockShapeforB_;
    using UBTileShapeforA = UBTileShapeforA_;
    using L1TileShape = L1TileShape_;

    using FT_AIV_PIPE_FUSE_TYPE = Gemv::helper::FT_AIV_PIPE_FUSE_TYPE;
    using FT_THRESHOLD_ALGORITHM = Gemv::helper::FT_THRESHOLD_ALGORITHM;

    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;

    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;

    using ElementX = typename XType_::Element;
    using LayoutX = typename XType_::Layout;

    using ElementY = typename YType_::Element;
    using LayoutY = typename YType_::Layout;

    using TileFaultSum = TileFaultSum_;

    using VecCopyGmToUb = typename TileCopy_::VecCopyGmToUb;

    using VecCopyUbToGmforAMax = typename TileCopy_::VecCopyUbToGmforAMax;
    using VecCopyUbToGmforAMean = typename TileCopy_::VecCopyUbToGmforAMean;
    using VecCopyUbToGmforBMax = typename TileCopy_::VecCopyUbToGmforBMax;

    using MatrixCopyGmToUb = typename TileCopy_::MatrixCopyGmToUb;


    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    using UBAlignHelper = Gemv::helper::UBAlignHelper<ElementA>;

    using TensorCoord = layout::VectorLayout::TensorCoord;

    using FT_REDUCE_TYPE = Gemv::helper::FT_REDUCE_TYPE;

    using TileFaultSumBmax = Gemv::Tile::TileFaultSum<ArchTag, FT_REDUCE_TYPE::MAX, BType_, XType_>;
    using TileFaultSumBSum = Gemv::Tile::TileFaultSum<ArchTag, FT_REDUCE_TYPE::SUM, BType_, XType_>;
    using TileFaultSumARed = Gemv::Tile::TileFaultSum<ArchTag, FT_REDUCE_TYPE::SUM_MAX_MIXED, AType_, YType_>;
    using TileVmulsforMean = Gemv::Tile::TileVmuls<ArchTag, YType_>;
    // using TileFaultSumA = 

    // Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::A_B_MIXED
    static constexpr FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE = Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::A_B_MIXED;
    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR;


    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t Abuf_SIZE_ = 128 * 1024;
    static constexpr uint32_t YSumbuf_SIZE_ = 16 * 1024;
    static constexpr uint32_t YMaxbuf_forB_SIZE_ = 8 * 1024;
    static constexpr uint32_t YMaxbuf_forA_SIZE_ = 8 * 1024;
    static constexpr uint32_t workspace_SIZE_ = 32 * 1024;

    static_assert(UBTileShapeforB::N == L1TileShape::N,
        "The situation where the basic Tile of UB and L1 for MMA differ on the n axes is not supported yet");

    static_assert((UBBlockShapeforB::N % UBTileShapeforB::N) == 0,
        "The situation where the basic Tile of UB and L1 for MMA differ on the n axes is not supported yet");

    static_assert(std::is_same_v<LayoutA, LayoutB>,
        "The LayoutA and LayoutB of Gemm should be consistent.");

    static_assert(std::is_same_v<ElementA, ElementB>,
        "The ElementA and ElementB of Gemm should be consistent.");


    CATLASS_DEVICE
    BlockFTSumNoSplitK() {}

    /// Construct
    CATLASS_DEVICE
    BlockFTSumNoSplitK(Arch::Resource<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbAOffset = UBufAddrStart;
        uint32_t UbYSumOffset = UBufAddrStart + Abuf_SIZE_;

        uint32_t UbYMaxOffsetforA = UBufAddrStart + Abuf_SIZE_ + YSumbuf_SIZE_;
        uint32_t UbYMaxOffsetforB = UBufAddrStart + Abuf_SIZE_ + YSumbuf_SIZE_ + YMaxbuf_forA_SIZE_;
        uint32_t UbWOffset = UBufAddrStart + Abuf_SIZE_ + YSumbuf_SIZE_ + YMaxbuf_forA_SIZE_ + YMaxbuf_forB_SIZE_;
        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SIZE_ / 2));
            UbYSumTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYSumOffset + i * (YSumbuf_SIZE_ / 2));

            UbYMaxTensorforAList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbYMaxOffsetforA + i * (YMaxbuf_forA_SIZE_ / 2));
            UbYMaxTensorforBList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbYMaxOffsetforB + i * (YMaxbuf_forB_SIZE_ / 2));

            UbWTensorList[i] =
                resource.ubBuf.template GetBufferByByte<ElementA>(UbWOffset + i * (workspace_SIZE_ / 2));

            // Assign event ID for each stages
            UbInAEventList[i] = i;
            UbOutSumEventList[i] = i;
            UbOutMaxEventforAList[i] = i + STAGES;
            UbOutMaxEventforBList[i] = i * STAGES * 2;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutSumEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutMaxEventforAList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutMaxEventforBList[i]);
        }
    }

    /// Construct
    CATLASS_DEVICE
    BlockFTSumNoSplitK(Arch::ResourceAIV<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbAOffset = UBufAddrStart;
        uint32_t UbYSumOffset = UBufAddrStart + Abuf_SIZE_;

        uint32_t UbYMaxOffsetforA = UBufAddrStart + Abuf_SIZE_ + YSumbuf_SIZE_;
        uint32_t UbYMaxOffsetforB = UBufAddrStart + Abuf_SIZE_ + YSumbuf_SIZE_ + YMaxbuf_forA_SIZE_;
        uint32_t UbWOffset = UBufAddrStart + Abuf_SIZE_ + YSumbuf_SIZE_ + YMaxbuf_forA_SIZE_ + YMaxbuf_forB_SIZE_;
        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SIZE_ / 2));
            UbYSumTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYSumOffset + i * (YSumbuf_SIZE_ / 2));

            UbYMaxTensorforAList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbYMaxOffsetforA + i * (YMaxbuf_forA_SIZE_ / 2));
            UbYMaxTensorforBList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbYMaxOffsetforB + i * (YMaxbuf_forB_SIZE_ / 2));

            UbWTensorList[i] =
                resource.ubBuf.template GetBufferByByte<ElementA>(UbWOffset + i * (workspace_SIZE_ / 2));

            // Assign event ID for each stages
            UbInAEventList[i] = i;
            UbOutSumEventList[i] = i;
            UbOutMaxEventforAList[i] = i + STAGES;
            UbOutMaxEventforBList[i] = i * STAGES * 2;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutSumEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutMaxEventforAList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutMaxEventforBList[i]);
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockFTSumNoSplitK()
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(UbOutSumEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(UbOutMaxEventforAList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(UbOutMaxEventforBList[i]);
        }
    }


    CATLASS_DEVICE
    void MaxRed(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementX> const &gmZMax,
        LayoutX const &layoutX,
        GemvCoord const &actualShape)
    {
        TileMRound = RoundUp(UBTileShapeforB::M, UBAlignHelper::ALIGN);
        TileNRound = RoundUp(UBTileShapeforB::N, UBAlignHelper::ALIGN);
        BlockMRound = RoundUp(UBBlockShapeforB::M, UBAlignHelper::ALIGN);
        BlockNRound = RoundUp(UBBlockShapeforB::N, UBAlignHelper::ALIGN);

        // uint32_t strideACol, strideARow;
        // uint32_t strideOut;
        
        
        uint32_t strideACol = layoutA.stride(1) * TileNRound;
        uint32_t strideARow = layoutA.stride(0) * TileMRound;

        uint32_t strideOut = layoutA.stride(0);

        uint32_t NloopBlock = 1;
        uint32_t MBLoopIdx = 0;

        uint32_t MloopBlock = 1;
        uint32_t NBLoopIdx = 0;

        m_actual_total = (actualShape.m() < BlockMRound) ? actualShape.m() : BlockMRound;
        n_actual_total = (actualShape.n() < BlockNRound) ? actualShape.n() : BlockNRound;

        // ;
        uint32_t splitNnum = (n_actual_total + TileNRound - 1) / TileNRound;
        uint32_t tileMnum = (m_actual_total + TileMRound - 1) / TileMRound;

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
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutMaxEventforBList[UbOutListId]));
            auto UbYMaxTensor = UbYMaxTensorforBList[UbOutListId];
             
            AscendC::Duplicate<ElementX>(UbYMaxTensor, (ElementX)0.0, m_actual_total);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutMaxEventforBList[UbOutListId]));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutMaxEventforBList[UbOutListId]));

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
                CATLASS_DEVICE
                void operator()(
                    AscendC::LocalTensor<ElementY> dstTensor,
                    AscendC::LocalTensor<ElementA> srcTensor_m,
                    AscendC::LocalTensor<ElementA> temp,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
                )
                */
                tileFaultSumBmax(
                        UbYMaxTile[TileY_col_offset],
                        UbATensorList[UbInListId],
                        UbWTensorList[UbInListId],
                        layoutComputeInUb,
                        layoutTileCompute);
                
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
                UbInListId = UbInListIdNext;
            }

            // uint32_t ubTileOutOffset = TileY_Row_offset;
            auto layoutDstY = layoutX.GetTileLayout(TensorCoord(y_actual_total));
            auto layoutComputeInUb = layoutX.GetTileLayout(TensorCoord(y_actual_total));

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutMaxEventforBList[UbOutListId]));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutMaxEventforBList[UbOutListId]));

            vecCopyUbToGmforBMax(gmZMax[TileY_Row_offset], UbYMaxTensorforBList[UbOutListId], layoutDstY, layoutComputeInUb);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutMaxEventforBList[UbOutListId]));
            UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
        }
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementY> const &gmZSum, 
        AscendC::GlobalTensor<ElementA> const &gmZMax,
        LayoutY const &layoutY,
        GemvCoord const &actualShape)
    {
        TileMRound = RoundUp(UBTileShapeforA::M, UBAlignHelper::ALIGN);
        TileNRound = RoundUp(UBTileShapeforA::N, UBAlignHelper::ALIGN);

        uint32_t strideACol = layoutA.stride(1) * TileNRound;
        uint32_t strideARow = layoutA.stride(0) * TileMRound;

        uint32_t strideOut = 1;
        // layoutA.stride(0);

        uint32_t NloopBlock = 1;
        uint32_t MBLoopIdx = 0;

        uint32_t MloopBlock = 1;
        uint32_t NBLoopIdx = 0;

        m_actual_total = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
        n_actual_total = actualShape.n();

        ElementY A_row_scale_ratio = (ElementY)(1.0f / (n_actual_total * 1.0f));

        uint32_t splitNnum = (n_actual_total + TileNRound - 1) / TileNRound;
        uint32_t tileMnum = (m_actual_total + TileMRound - 1) / TileMRound;

        uint32_t Nloop = splitNnum;
        uint32_t Mloop = tileMnum;

        y_actual_total = m_actual_total;
        x_actual_total = n_actual_total;

        uint32_t A_row_offset = MBLoopIdx * TileMRound;
        uint32_t A_col_offset = NBLoopIdx * TileNRound;
        uint32_t A_block_offset = A_row_offset * layoutA.stride(0) + A_col_offset * layoutA.stride(1);

        m_actual = m_actual_total;
        // (m_actual_total < TileMRound) ? m_actual_total : TileMRound;
        n_actual = (n_actual_total < TileNRound) ? n_actual_total : TileNRound;

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
        auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
        auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
        matrixCopyGmToUb(UbATensorList[UbInListId], gmA, layoutAInUb, layoutTileA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutSumEventList[UbOutListId]));
        auto UbYSumTensor = UbYSumTensorList[UbOutListId];
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutMaxEventforAList[UbOutListId]));
        auto UbYMaxTensor = UbYMaxTensorforAList[UbOutListId];
             
        AscendC::Duplicate<ElementY>(UbYSumTensor, (ElementY)0.0, m_actual_total);
        AscendC::Duplicate<ElementA>(UbYMaxTensor, (ElementA)0.0, m_actual_total);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutSumEventList[UbOutListId]));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutSumEventList[UbOutListId]));

        // uint32_t TileY_Row_offset = NLoopIdx * strideOut;

        // main loop
        for (uint32_t NLoopIdx = 0; NLoopIdx < Nloop; NLoopIdx++) {
            n_actual = (NLoopIdx == Nloop - 1) ? (n_actual_total - NLoopIdx * TileNRound) : TileNRound;
            m_actual = m_actual_total;
            uint32_t TileA_Col_offset = NLoopIdx * strideACol;

            y_actual = m_actual;
            x_actual = n_actual;

            uint32_t TileA_Row_offset = 0;
            uint32_t TileY_col_offset = 0;

            uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;

            if(NLoopIdx < (Nloop - 1)) {
                uint32_t NLoopIdxNext = NLoopIdx + 1;
                uint32_t m_actual_next = m_actual_total;
                uint32_t n_actual_next = (NLoopIdxNext == Nloop - 1) ? (n_actual_total - NLoopIdxNext * TileNRound) : TileNRound;
                            
                uint32_t y_actual_next = m_actual_next;
                uint32_t x_actual_next = n_actual_next;

                uint32_t TileA_Row_offset_next = 0;
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
            CATLASS_DEVICE
            void operator()(
                AscendC::LocalTensor<ElementY> dstTensorSum,
                AscendC::LocalTensor<ElementA> dstTensorMax,
                AscendC::LocalTensor<ElementA> srcTensor_m,
                AscendC::LocalTensor<ElementA> sum_workspace,
                LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
            )
            */
            tileFaultSumARed(UbYSumTensor,
                UbYMaxTensor,
                UbATensorList[UbInListId],
                UbWTensorList[UbInListId],
                layoutComputeInUb,
                layoutTileCompute);
                
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
            UbInListId = UbInListIdNext;
        }

        AscendC::PipeBarrier<PIPE_V>();

        tileVmulsforMean(UbYSumTensor, UbYSumTensor, A_row_scale_ratio, m_actual_total);
        
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutSumEventList[UbOutListId]));

        // uint32_t ubTileOutOffset = TileY_Row_offset;
        auto layoutDstY = layoutY.GetTileLayout(TensorCoord(m_actual_total));
        auto layoutComputeInUb = layoutY.GetTileLayout(TensorCoord(m_actual_total));
      
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutSumEventList[UbOutListId]));

        vecCopyUbToGmforAMean(gmZSum, UbYSumTensorList[UbOutListId], layoutDstY, layoutComputeInUb);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutSumEventList[UbOutListId]));

        vecCopyUbToGmforAMax(gmZMax, UbYMaxTensorforAList[UbOutListId], layoutDstY, layoutComputeInUb);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutMaxEventforAList[UbOutListId]));
        UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> UbATensorList[STAGES];
    AscendC::LocalTensor<ElementY> UbYSumTensorList[STAGES];
    AscendC::LocalTensor<ElementA> UbYMaxTensorforAList[STAGES];
    AscendC::LocalTensor<ElementX> UbYMaxTensorforBList[STAGES];
    AscendC::LocalTensor<ElementA> UbWTensorList[STAGES];

    // Multi-stage event id list
    int32_t UbInAEventList[STAGES];
    int32_t UbOutSumEventList[STAGES];
    int32_t UbOutMaxEventforAList[STAGES];
    int32_t UbOutMaxEventforBList[STAGES];

    // The id of current stage
    uint32_t UbOutListId{0};
    uint32_t UbInListId{0};

    uint32_t m_actual, n_actual, x_actual, y_actual;
    uint32_t m_actual_total, n_actual_total, x_actual_total, y_actual_total;
    uint32_t TileMRound, TileNRound;
    uint32_t BlockMRound, BlockNRound;
    

    TileFaultSumBmax tileFaultSumBmax;
    TileFaultSumARed tileFaultSumARed;
    TileFaultSumBSum tileFaultSumBSum;

    TileVmulsforMean tileVmulsforMean;

    MatrixCopyGmToUb matrixCopyGmToUb;
    VecCopyGmToUb vecCopyGmToUb;
    VecCopyUbToGmforAMax vecCopyUbToGmforAMax;
    VecCopyUbToGmforAMean vecCopyUbToGmforAMean;
    VecCopyUbToGmforBMax vecCopyUbToGmforBMax;

};

} // namespace Catlass::Gemv::Block

#endif // CATLASS_GEMV_BLOCK_BLOCK_GEMV_AIV_HPP
