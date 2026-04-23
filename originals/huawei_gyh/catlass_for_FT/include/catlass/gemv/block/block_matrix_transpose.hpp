#ifndef CATLASS_GEMV_BLOCK_BLOCK_MATRIX_TRANSPOSE_HPP
#define CATLASS_GEMV_BLOCK_BLOCK_MATRIX_TRANSPOSE_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemv_coord.hpp"
#include "catlass/gemv/helper.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
// include/catlass/gemv/tile/tile_matrix_transpose.hpp
#include "catlass/gemv/tile/tile_matrix_transpose.hpp"
// #include "catlass/gemv/tile/tile_vmad.hpp"
// #include "catlass/gemv/tile/tile_fault_sum.hpp"

namespace Catlass::Gemv::Block {

template <
    class UBTileShape_,
    class UBTileTailShape_,
    class UBBlockShape_,
    class UBBlockTailShape_,
    class AType_,
    class YType_,
    class BiasType_,
    class TileCopy_,
    class TileMatrixTranspose_
>
struct BlockMatrixTranspose <
    Gemm::GemvAtlasA2,
    UBTileShape_,
    UBTileTailShape_,
    UBBlockShape_,
    UBBlockTailShape_,
    AType_,
    YType_,
    BiasType_,
    TileCopy_,
    TileMatrixTranspose_
> {
public:
    // Type Aliases
    using DispatchPolicy = Gemm::GemvAtlasA2;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using UBTileShape = UBTileShape_;
    using UBBlockShape = UBBlockShape_;

    using UBTileTailShape = UBTileTailShape_;
    using UBBlockTailShape = UBBlockTailShape_;

    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;

    using ElementY = typename YType_::Element;
    using LayoutY = typename YType_::Layout;

    using TileMatrixTranspose = TileMatrixTranspose_;

    // // the function of aiv
    // using MatrixCopyGmToUb = Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;
    // using MatrixCopyUbToGm = Gemv::Tile::MatrixCopyUBToGm<ArchTag, YType>;

    using MatrixCopyGmToUb = typename TileCopy_::MatrixCopyGmToUb;
    using MatrixCopyUbToGm = typename TileCopy_::MatrixCopyUbToGm;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    using UBAlignHelper = Gemv::helper::UBAlignHelper<ElementA>;
    using UBTransposeAlignHelper = Gemv::helper::UBTransposeAlignHelper<ElementA>;

    using TensorCoord = layout::VectorLayout::TensorCoord;

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t Abuf_SIZE_ = 64 * 1024;
    static constexpr uint32_t Shared_buf_SIZE_ = 64 * 1024;
    static constexpr uint32_t Ybuf_SIZE_ = 64 * 1024;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t BASE_C_UNIT = ((Abuf_SIZE_ / STAGES) / sizeof(ElementA)) / UBTileShape::M;

    static constexpr uint32_t BASE_HW_SIZE = UBTransposeAlignHelper::BLK_ALIGN * UBTransposeAlignHelper::H_ALIGN;

    static_assert(BASE_C_UNIT >= ELE_NUM_PER_C0,
        "The situation where the basic Tile of UB on C_UNIT smaller than 32 Byte is not supported yet");
    
    static_assert(UBTileShape::M % UBTransposeAlignHelper::BLK_ALIGN == 0,
        "UBTileShape::M must be multiple of UBTransposeAlignHelper::BLK_ALIGN");

    static_assert(UBTileShape::N % UBTransposeAlignHelper::C_ALIGN == 0,
        "UBTileShape::N must be multiple of UBTransposeAlignHelper::C_ALIGN");
    
    static_assert(UBTileShape::N <= UBTransposeAlignHelper::C_UNIT_UPPER_LIMIT,
        "UBTileShape::N must be lower than UBTransposeAlignHelper::C_UNIT_UPPER_LIMIT");
    
    static_assert(UBTileShape::M % UBTransposeAlignHelper::H_ALIGN == 0,
        "UBTileShape::N must be multiple of UBTransposeAlignHelper::H_ALIGN");
    
    static_assert(UBTileShape::M % BASE_HW_SIZE == 0,
        "UBTileShape::N must be multiple of BASE_HW_SIZE");

    static_assert(UBTileTailShape::M % UBTransposeAlignHelper::BLK_ALIGN == 0,
        "UBTileShape::M must be multiple of UBTransposeAlignHelper::BLK_ALIGN");
    
    static_assert(UBTileTailShape::M % UBTransposeAlignHelper::H_ALIGN == 0,
        "UBTileShape::N must be multiple of UBTransposeAlignHelper::H_ALIGN");
    
    static_assert(UBTileTailShape::M % BASE_HW_SIZE == 0,
        "UBTileShape::N must be multiple of BASE_HW_SIZE");

    static_assert(UBBlockTailShape::N == UBTileTailShape::N,
        "The situation where the basic Tile of UB for Tail Transpose differ on the n axes is not supported yet");

    static_assert((UBBlockShape::N % UBTileShape::N) == 0,
        "UBBlockShape::N must be multiple of UBTileShape::N");
    
    static_assert((UBBlockShape::M % UBTileShape::M) == 0,
        "UBBlockShape::M must be multiple of UBTileShape::M");
    
    static_assert((UBBlockShape::M % UBTileShape::M) == 0,
        "UBBlockShape::M must be multiple of UBTileShape::M");



    CATLASS_DEVICE
    BlockMatrixTranspose() {}

    /// Construct
    CATLASS_DEVICE
    BlockMatrixTranspose(Arch::Resource<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbAOffset = UBufAddrStart;
        uint32_t UbSharedBufOffset = UBufAddrStart + Abuf_SIZE_;
        
        uint32_t UbYOffset = UBufAddrStart + Abuf_SIZE_ + Shared_buf_SIZE_;

        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SIZE_ / STAGES));
            UbYTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset + i * (Ybuf_SIZE_ / STAGES));
            SharedTmpBufferList[i] = resource.ubBuf.template GetBufferByByte<uint8_t>(UbSharedBufOffset + i * (Shared_buf_SIZE_ / STAGES));
            // WorkspaceList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbWorkSpaceOffset + i * (Workspace_SIZE_ / STAGES));

            // Assign event ID for each stages
            UbInAEventList[i] = i;
            UbOutEventList[i] = i + STAGES;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutEventList[i]);
        }
    }

    /// Construct
    CATLASS_DEVICE
    BlockMatrixTranspose(Arch::ResourceAIV<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbAOffset = UBufAddrStart;
        uint32_t UbSharedBufOffset = UBufAddrStart + Abuf_SIZE_;
        
        uint32_t UbYOffset = UBufAddrStart + Abuf_SIZE_ + Shared_buf_SIZE_;

        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SIZE_ / STAGES));
            UbYTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset + i * (Ybuf_SIZE_ / STAGES));
            SharedTmpBufferList[i] = resource.ubBuf.template GetBufferByByte<uint8_t>(UbSharedBufOffset + i * (Shared_buf_SIZE_ / STAGES));
            // WorkspaceList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbWorkSpaceOffset + i * (Workspace_SIZE_ / STAGES));

            // Assign event ID for each stages
            UbInAEventList[i] = i;
            UbOutEventList[i] = i + STAGES;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutEventList[i]);
        }
    }


    /// Destructor
    CATLASS_DEVICE
    ~BlockMatrixTranspose()
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(UbOutEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutEventList[i]);
        }
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmA,
        AscendC::GlobalTensor<ElementA> const &gmNextBlockA,
        LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementY> const &gmY, LayoutY const &layoutY,
        GemvCoord const &actualShape, GemvCoord const &actualShapeNext,
        bool isFirstBlock, bool hasNextBlock)
    {
        TileMRound = RoundUp(UBTileShape::M, BASE_HW_SIZE);
        TileNRound = RoundUp(UBTileShape::N, UBTransposeAlignHelper::C_ALIGN);

        BlockMRound = RoundUp(UBBlockShape::M, BASE_HW_SIZE);
        BlockNRound = RoundUp(UBBlockShape::N, UBTransposeAlignHelper::C_ALIGN);

        strideAInCol = layoutA.stride(1) * TileNRound;
        strideAInRow = layoutA.stride(0) * TileMRound;

        strideYOutCol = layoutY.stride(1) * TileMRound;
        strideYOutRow = layoutY.stride(0) * TileNRound;

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

        uint32_t Y_row_offset = NBLoopIdx * BlockNRound;
        uint32_t Y_col_offset = MBLoopIdx * BlockMRound;
        uint32_t Y_block_offset = Y_row_offset * layoutY.stride(0) + Y_col_offset * layoutY.stride(1);

        m_actual = (m_actual_total < TileMRound) ? m_actual_total : TileMRound;
        n_actual = (n_actual_total < TileNRound) ? n_actual_total : TileNRound;

        if(isFirstBlock){
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbOutListId]));
            auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
            auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
            matrixCopyGmToUb(UbATensorList[UbOutListId], gmA[A_block_offset], layoutAInUb, layoutTileA);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbOutListId]));

            // AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListId]));
            // auto UbYTransedTensor = UbYTensorList[UbOutListId];
            // AscendC::Duplicate<ElementY>(UbYTransedTensor, (ElementY)0.0, n_actual * TileMRound);
            // // AscendC::PipeBarrier<PIPE_V>();
            // AscendC::SetFlag<AscendC::HardEvent::V_V>((event_t)(UbOutEventList[UbOutListId]));
        }

        for (uint32_t NLoopIdx = 0; NLoopIdx < Nloop; NLoopIdx++) {
            n_actual = (NLoopIdx == Nloop - 1) ? (n_actual_total - NLoopIdx * TileNRound) : TileNRound;
            uint32_t A_col_offset = NLoopIdx * strideAInCol;
            uint32_t Y_row_offset = NLoopIdx * strideYOutRow;

            for(uint32_t MLoopIdx = 0; MLoopIdx < Mloop; MLoopIdx++){
                m_actual = (MLoopIdx == Mloop - 1) ? (m_actual_total - MLoopIdx * TileMRound) : TileMRound;
                uint32_t A_row_offset = MLoopIdx * strideAInRow;
                uint32_t A_tile_offset = A_block_offset + A_row_offset + A_col_offset;

                uint32_t Y_col_offset = MLoopIdx * strideYOutCol;
                uint32_t Y_tile_offset = Y_block_offset + Y_row_offset + Y_col_offset;

                uint32_t UbOutListIdNext = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;

                if(MLoopIdx < (Mloop -1)){
                    uint32_t NLoopIdxNext = NLoopIdx;
                    uint32_t MLoopIdxNext = MLoopIdx + 1;
                    uint32_t m_actual_next = (MLoopIdxNext == (Mloop - 1)) ? (m_actual_total - MLoopIdxNext * TileMRound) : TileMRound;
                    uint32_t n_actual_next = n_actual;

                    uint32_t A_col_offset_next = A_col_offset;
                    uint32_t Y_row_offset_next = Y_row_offset;

                    uint32_t A_row_offset_next = MLoopIdxNext * strideAInRow;
                    uint32_t A_tile_offset_next = A_block_offset + A_row_offset_next + A_col_offset_next;
                    
                    uint32_t Y_col_offset_next = MLoopIdxNext * strideYOutCol;
                    uint32_t Y_tile_offset_next = Y_block_offset + Y_row_offset_next + Y_col_offset_next;

                    // Pre-load next tile A into UB
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbOutListIdNext]));
                    auto layoutAInUbNext = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    auto layoutTileANext = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                    matrixCopyGmToUb(UbATensorList[UbOutListIdNext], gmA[A_tile_offset_next], layoutAInUbNext, layoutTileANext);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbOutListIdNext]));

                    // AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListIdNext]));
                    // auto UbYTransedTensorNext = UbYTensorList[UbOutListIdNext];
                    // AscendC::Duplicate<ElementY>(UbYTransedTensorNext, (ElementY)0.0, n_actual_next * TileMRound);
                    // // AscendC::PipeBarrier<PIPE_V>();
                    // AscendC::SetFlag<AscendC::HardEvent::V_V>((event_t)(UbOutEventList[UbOutListIdNext]));
                }else if(MLoopIdx == (Mloop -1) && (NLoopIdx < (Nloop -1))){
                    uint32_t NLoopIdxNext = NLoopIdx + 1;
                    uint32_t MLoopIdxNext = 0;
                    uint32_t m_actual_next = (MLoopIdxNext == (Mloop - 1)) ? (m_actual_total - MLoopIdxNext * TileMRound) : TileMRound;
                    uint32_t n_actual_next = (NLoopIdxNext == (Nloop - 1)) ? (n_actual_total - NLoopIdxNext * TileNRound) : TileNRound;
                    
                    uint32_t A_col_offset_next = NLoopIdxNext * strideAInCol;
                    uint32_t Y_row_offset_next = NLoopIdxNext * strideYOutRow;

                    uint32_t A_row_offset_next = MLoopIdxNext * strideAInRow;
                    uint32_t A_tile_offset_next = A_block_offset + A_row_offset_next + A_col_offset_next;
                    
                    uint32_t Y_col_offset_next = MLoopIdxNext * strideYOutCol;
                    uint32_t Y_tile_offset_next = Y_block_offset + Y_row_offset_next + Y_col_offset_next;

                    // Pre-load next tile A into UB
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbOutListIdNext]));
                    auto layoutAInUbNext = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    auto layoutTileANext = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                    matrixCopyGmToUb(UbATensorList[UbOutListIdNext], gmA[A_tile_offset_next], layoutAInUbNext, layoutTileANext);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbOutListIdNext]));

                    // AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListIdNext]));
                    // auto UbYTransedTensorNext = UbYTensorList[UbOutListIdNext];
                    // AscendC::Duplicate<ElementY>(UbYTransedTensorNext, (ElementY)0.0, n_actual_next * TileMRound);
                    // // AscendC::PipeBarrier<PIPE_V>();
                    // AscendC::SetFlag<AscendC::HardEvent::V_V>((event_t)(UbOutEventList[UbOutListIdNext]));
                }else if((MLoopIdx == (Mloop -1)) && (NLoopIdx == (Nloop -1)) && hasNextBlock){
                    uint32_t m_actual_total_next = (actualShapeNext.m() < BlockMRound) ? actualShapeNext.m() : BlockMRound;
                    uint32_t n_actual_total_next = (actualShapeNext.n() < BlockNRound) ? actualShapeNext.n() : BlockNRound;

                    uint32_t NLoopIdxNext = 0;
                    uint32_t MLoopIdxNext = 0;

                    uint32_t m_actual_next = (m_actual_total_next < TileMRound) ? m_actual_total_next : TileMRound;
                    uint32_t n_actual_next = (n_actual_total_next < TileNRound) ? n_actual_total_next : TileNRound;
                    
                    uint32_t A_col_offset_next = NLoopIdxNext * strideAInCol;
                    uint32_t Y_row_offset_next = NLoopIdxNext * strideYOutRow;

                    uint32_t A_row_offset_next = MLoopIdxNext * strideAInRow;
                    uint32_t A_tile_offset_next = A_row_offset_next + A_col_offset_next;
                    
                    uint32_t Y_col_offset_next = MLoopIdxNext * strideYOutCol;
                    uint32_t Y_tile_offset_next = Y_row_offset_next + Y_col_offset_next;

                    // Pre-load next tile A into UB
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbOutListIdNext]));
                    auto layoutAInUbNext = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    auto layoutTileANext = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                    matrixCopyGmToUb(UbATensorList[UbOutListIdNext], gmNextBlockA[A_tile_offset_next], layoutAInUbNext, layoutTileANext);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbOutListIdNext]));

                    // AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListIdNext]));
                    // auto UbYTransedTensorNext = UbYTensorList[UbOutListIdNext];
                    // AscendC::Duplicate<ElementY>(UbYTransedTensorNext, (ElementY)0.0, n_actual_next * TileMRound);
                    // // AscendC::PipeBarrier<PIPE_V>();
                    // AscendC::SetFlag<AscendC::HardEvent::V_V>((event_t)(UbOutEventList[UbOutListIdNext]));
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbOutListId]));
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListId]));

                auto layoutComputeInUb = LayoutY::template MakeLayoutInUb<ElementY>(MakeCoord(TileNRound, TileMRound));
                auto layoutTileCompute = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));

                /*
                CATLASS_DEVICE
                void operator()(
                    AscendC::LocalTensor<ElementY> dstTensor,
                    AscendC::LocalTensor<ElementA> srcTensor_m,
                    AscendC::LocalTensor<ElementA> temp_workspace,
                    AscendC::LocalTensor<uint8_t> sharedTmpBuffer,
                    LayoutDst const &layoutDst, 
                    LayoutSrc const &layoutSrc
                )
                */
                // WorkspaceList[UbOutListId],
                tileMatrixTranspose(
                    UbYTensorList[UbOutListId],
                    UbATensorList[UbOutListId],
                    SharedTmpBufferList[UbOutListId],
                    layoutComputeInUb, layoutTileCompute);
                
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbOutListId]));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));
                

                uint32_t aligned_m_actual = RoundUp(m_actual, UBTransposeAlignHelper::BLK_ALIGN);
                aligned_m_actual = RoundUp(aligned_m_actual, UBTransposeAlignHelper::H_ALIGN);

                auto layoutYInUb = LayoutY::template MakeLayoutInUb<ElementY>(MakeCoord(n_actual, aligned_m_actual));
                auto layoutDstY = layoutY.GetTileLayout(MakeCoord(n_actual, m_actual));


                /*
                CATLASS_DEVICE
                void operator()(
                    AscendC::GlobalTensor<Element> const &dstTensor,
                    AscendC::LocalTensor<Element> const &srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
                */

                // vecCopyUbToGm(gmZ[TileY_Row_offset], UbYTensorList[UbOutListId], layoutDstY, layoutComputeInUb);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));
                matrixCopyUbToGm(gmY[Y_tile_offset], UbYTensorList[UbOutListId], layoutDstY, layoutYInUb);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)UbOutEventList[UbOutListId]);
                UbOutListId = UbOutListIdNext;
            }
        }
    }

    CATLASS_DEVICE
    void tail_op(
        AscendC::GlobalTensor<ElementA> const &gmA,
        AscendC::GlobalTensor<ElementA> const &gmNextBlockA,
        LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementY> const &gmY, LayoutY const &layoutY,
        GemvCoord const &actualShape, GemvCoord const &actualShapeNext,
        bool isFirstBlock, bool hasNextBlock)
    {
        TileMRound = RoundUp(UBTileTailShape::M, BASE_HW_SIZE);
        TileNRound = RoundUp(UBTileTailShape::N, UBTransposeAlignHelper::C_ALIGN);

        BlockMRound = RoundUp(UBBlockTailShape::M, BASE_HW_SIZE);
        BlockNRound = RoundUp(UBBlockTailShape::N, UBTransposeAlignHelper::C_ALIGN);

        strideAInCol = layoutA.stride(1) * TileNRound;
        strideAInRow = layoutA.stride(0) * TileMRound;

        strideYOutCol = layoutY.stride(1) * TileMRound;
        strideYOutRow = layoutY.stride(0) * TileNRound;

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

        uint32_t Y_row_offset = NBLoopIdx * BlockNRound;
        uint32_t Y_col_offset = MBLoopIdx * BlockMRound;
        uint32_t Y_block_offset = Y_row_offset * layoutY.stride(0) + Y_col_offset * layoutY.stride(1);

        m_actual = (m_actual_total < TileMRound) ? m_actual_total : TileMRound;
        n_actual = (n_actual_total < TileNRound) ? n_actual_total : TileNRound;

        if(isFirstBlock){
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbOutListId]));
            auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
            auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
            matrixCopyGmToUb(UbATensorList[UbOutListId], gmA[A_block_offset], layoutAInUb, layoutTileA);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbOutListId]));

            // AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListId]));
            // auto UbYTransedTensor = UbYTensorList[UbOutListId];
            // AscendC::Duplicate<ElementY>(UbYTransedTensor, (ElementY)0.0, n_actual * TileMRound);
            // // AscendC::PipeBarrier<PIPE_V>();
            // AscendC::SetFlag<AscendC::HardEvent::V_V>((event_t)(UbOutEventList[UbOutListId]));
        }

        for (uint32_t NLoopIdx = 0; NLoopIdx < Nloop; NLoopIdx++) {
            n_actual = (NLoopIdx == Nloop - 1) ? (n_actual_total - NLoopIdx * TileNRound) : TileNRound;
            uint32_t A_col_offset = NLoopIdx * strideAInCol;
            uint32_t Y_row_offset = NLoopIdx * strideYOutRow;

            for(uint32_t MLoopIdx = 0; MLoopIdx < Mloop; MLoopIdx++){
                m_actual = (MLoopIdx == Mloop - 1) ? (m_actual_total - MLoopIdx * TileMRound) : TileMRound;
                uint32_t A_row_offset = MLoopIdx * strideAInRow;
                uint32_t A_tile_offset = A_block_offset + A_row_offset + A_col_offset;

                uint32_t Y_col_offset = MLoopIdx * strideYOutCol;
                uint32_t Y_tile_offset = Y_block_offset + Y_row_offset + Y_col_offset;

                uint32_t UbOutListIdNext = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;

                if(MLoopIdx < (Mloop -1)){
                    uint32_t NLoopIdxNext = NLoopIdx;
                    uint32_t MLoopIdxNext = MLoopIdx + 1;
                    uint32_t m_actual_next = (MLoopIdxNext == (Mloop - 1)) ? (m_actual_total - MLoopIdxNext * TileMRound) : TileMRound;
                    uint32_t n_actual_next = n_actual;

                    uint32_t A_col_offset_next = A_col_offset;
                    uint32_t Y_row_offset_next = Y_row_offset;

                    uint32_t A_row_offset_next = MLoopIdxNext * strideAInRow;
                    uint32_t A_tile_offset_next = A_block_offset + A_row_offset_next + A_col_offset_next;
                    
                    uint32_t Y_col_offset_next = MLoopIdxNext * strideYOutCol;
                    uint32_t Y_tile_offset_next = Y_block_offset + Y_row_offset_next + Y_col_offset_next;

                    // Pre-load next tile A into UB
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbOutListIdNext]));
                    auto layoutAInUbNext = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    auto layoutTileANext = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                    matrixCopyGmToUb(UbATensorList[UbOutListIdNext], gmA[A_tile_offset_next], layoutAInUbNext, layoutTileANext);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbOutListIdNext]));

                    // AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListIdNext]));
                    // auto UbYTransedTensorNext = UbYTensorList[UbOutListIdNext];
                    // AscendC::Duplicate<ElementY>(UbYTransedTensorNext, (ElementY)0.0, n_actual_next * TileMRound);
                    // // AscendC::PipeBarrier<PIPE_V>();
                    // AscendC::SetFlag<AscendC::HardEvent::V_V>((event_t)(UbOutEventList[UbOutListIdNext]));
                }else if(MLoopIdx == (Mloop -1) && (NLoopIdx < (Nloop -1))){
                    uint32_t NLoopIdxNext = NLoopIdx + 1;
                    uint32_t MLoopIdxNext = 0;
                    uint32_t m_actual_next = (MLoopIdxNext == (Mloop - 1)) ? (m_actual_total - MLoopIdxNext * TileMRound) : TileMRound;
                    uint32_t n_actual_next = (NLoopIdxNext == (Nloop - 1)) ? (n_actual_total - NLoopIdxNext * TileNRound) : TileNRound;
                    
                    uint32_t A_col_offset_next = NLoopIdxNext * strideAInCol;
                    uint32_t Y_row_offset_next = NLoopIdxNext * strideYOutRow;

                    uint32_t A_row_offset_next = MLoopIdxNext * strideAInRow;
                    uint32_t A_tile_offset_next = A_block_offset + A_row_offset_next + A_col_offset_next;
                    
                    uint32_t Y_col_offset_next = MLoopIdxNext * strideYOutCol;
                    uint32_t Y_tile_offset_next = Y_block_offset + Y_row_offset_next + Y_col_offset_next;

                    // Pre-load next tile A into UB
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbOutListIdNext]));
                    auto layoutAInUbNext = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    auto layoutTileANext = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                    matrixCopyGmToUb(UbATensorList[UbOutListIdNext], gmA[A_tile_offset_next], layoutAInUbNext, layoutTileANext);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbOutListIdNext]));

                    // AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListIdNext]));
                    // auto UbYTransedTensorNext = UbYTensorList[UbOutListIdNext];
                    // AscendC::Duplicate<ElementY>(UbYTransedTensorNext, (ElementY)0.0, n_actual_next * TileMRound);
                    // // AscendC::PipeBarrier<PIPE_V>();
                    // AscendC::SetFlag<AscendC::HardEvent::V_V>((event_t)(UbOutEventList[UbOutListIdNext]));
                }else if(MLoopIdx == (Mloop -1) && (NLoopIdx == (Nloop -1)) && hasNextBlock){
                    uint32_t m_actual_total_next = (actualShapeNext.m() < BlockMRound) ? actualShapeNext.m() : BlockMRound;
                    uint32_t n_actual_total_next = (actualShapeNext.n() < BlockNRound) ? actualShapeNext.n() : BlockNRound;

                    uint32_t NLoopIdxNext = 0;
                    uint32_t MLoopIdxNext = 0;

                    uint32_t m_actual_next = (m_actual_total_next < TileMRound) ? m_actual_total_next : TileMRound;
                    uint32_t n_actual_next = (n_actual_total_next < TileNRound) ? n_actual_total_next : TileNRound;
                    
                    uint32_t A_col_offset_next = NLoopIdxNext * strideAInCol;
                    uint32_t Y_row_offset_next = NLoopIdxNext * strideYOutRow;

                    uint32_t A_row_offset_next = MLoopIdxNext * strideAInRow;
                    uint32_t A_tile_offset_next = A_row_offset_next + A_col_offset_next;
                    
                    uint32_t Y_col_offset_next = MLoopIdxNext * strideYOutCol;
                    uint32_t Y_tile_offset_next = Y_row_offset_next + Y_col_offset_next;

                    // Pre-load next tile A into UB
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbOutListIdNext]));
                    auto layoutAInUbNext = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    auto layoutTileANext = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                    matrixCopyGmToUb(UbATensorList[UbOutListIdNext], gmNextBlockA[A_tile_offset_next], layoutAInUbNext, layoutTileANext);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbOutListIdNext]));

                    // AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListIdNext]));
                    // auto UbYTransedTensorNext = UbYTensorList[UbOutListIdNext];
                    // AscendC::Duplicate<ElementY>(UbYTransedTensorNext, (ElementY)0.0, n_actual_next * TileMRound);
                    // // AscendC::PipeBarrier<PIPE_V>();
                    // AscendC::SetFlag<AscendC::HardEvent::V_V>((event_t)(UbOutEventList[UbOutListIdNext]));
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbOutListId]));
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListId]));

                auto layoutComputeInUb = LayoutY::template MakeLayoutInUb<ElementY>(MakeCoord(TileNRound, TileMRound));
                auto layoutTileCompute = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));

                /*
                CATLASS_DEVICE
                void operator()(
                    AscendC::LocalTensor<ElementY> dstTensor,
                    AscendC::LocalTensor<ElementA> srcTensor_m,
                    AscendC::LocalTensor<ElementA> temp_workspace,
                    AscendC::LocalTensor<uint8_t> sharedTmpBuffer,
                    LayoutDst const &layoutDst, 
                    LayoutSrc const &layoutSrc
                )
                */
                // WorkspaceList[UbOutListId],
                tileMatrixTranspose(
                    UbYTensorList[UbOutListId],
                    UbATensorList[UbOutListId],
                    SharedTmpBufferList[UbOutListId],
                    layoutComputeInUb, layoutTileCompute);
                
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbOutListId]));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));
                

                uint32_t aligned_m_actual = RoundUp(m_actual, UBTransposeAlignHelper::BLK_ALIGN);
                aligned_m_actual = RoundUp(aligned_m_actual, UBTransposeAlignHelper::H_ALIGN);

                auto layoutYInUb = LayoutY::template MakeLayoutInUb<ElementY>(MakeCoord(n_actual, aligned_m_actual));
                auto layoutDstY = layoutY.GetTileLayout(MakeCoord(n_actual, m_actual));


                /*
                CATLASS_DEVICE
                void operator()(
                    AscendC::GlobalTensor<Element> const &dstTensor,
                    AscendC::LocalTensor<Element> const &srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
                */

                // vecCopyUbToGm(gmZ[TileY_Row_offset], UbYTensorList[UbOutListId], layoutDstY, layoutComputeInUb);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));
                matrixCopyUbToGm(gmY[Y_tile_offset], UbYTensorList[UbOutListId], layoutDstY, layoutYInUb);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)UbOutEventList[UbOutListId]);
                UbOutListId = UbOutListIdNext;
            }
        }
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> UbATensorList[STAGES];
    AscendC::LocalTensor<ElementY> UbYTensorList[STAGES];
    AscendC::LocalTensor<uint8_t> SharedTmpBufferList[STAGES];
    // AscendC::LocalTensor<ElementA> WorkspaceList[STAGES];

    // Multi-stage event id list
    int32_t UbInAEventList[STAGES];
    int32_t UbOutEventList[STAGES];

    // The id of current stage
    uint32_t UbOutListId{0};
    uint32_t UbInListId{0};

    uint32_t m_actual, n_actual, x_actual, y_actual;
    uint32_t m_actual_total, n_actual_total, x_actual_total, y_actual_total;

    uint32_t TileMRound, TileNRound;

    uint32_t BlockMRound, BlockNRound;

    uint32_t MatrixOffset;
    uint32_t strideAInCol, strideAInRow;
    uint32_t strideYOutCol, strideYOutRow;

    uint32_t splitNnum;
    uint32_t tileMnum;

    TileMatrixTranspose tileMatrixTranspose;

    MatrixCopyGmToUb matrixCopyGmToUb;
    MatrixCopyUbToGm matrixCopyUbToGm;
};

} // namespace Catlass::Gemv::Block

#endif // CATLASS_GEMV_BLOCK_BLOCK_GEMV_AIV_HPP
