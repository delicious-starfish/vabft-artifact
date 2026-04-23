#ifndef CATLASS_GEMV_BLOCK_BLOCK_MATRIX_VECTORIZED_ADD_HPP
#define CATLASS_GEMV_BLOCK_BLOCK_MATRIX_VECTORIZED_ADD_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemv_coord.hpp"
#include "catlass/gemv/helper.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
// include/catlass/gemv/tile/tile_vec_elem_add.hpp
#include "catlass/gemv/tile/tile_vec_elem_add.hpp"
// #include "catlass/gemv/tile/tile_vmad.hpp"
// #include "catlass/gemv/tile/tile_fault_sum.hpp"

namespace Catlass::Gemv::Block {

template <
    class UBTileShape_,
    class UBBlockShape_,
    class AType_,
    class YType_,
    class BiasType_,
    class TileCopy_,
    class TileVecAdd_
>
struct BlockMatrixAddVectorized <
    Gemm::GemvAtlasA2,
    UBTileShape_,
    UBBlockShape_,
    AType_,
    YType_,
    BiasType_,
    TileCopy_,
    TileVecAdd_
> {
public:
    // Type Aliases
    using DispatchPolicy = Gemm::GemvAtlasA2;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using UBTileShape = UBTileShape_;
    using UBBlockShape = UBBlockShape_;

    using TensorCoord = layout::VectorLayout::TensorCoord;

    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using LayoutVA = layout::VectorLayout;

    using ElementY = typename YType_::Element;
    using LayoutY = typename YType_::Layout;
    using LayoutVY = layout::VectorLayout;

    using TileVecAdd = TileVecAdd_;

    // // the function of aiv
    // using MatrixCopyGmToUb = Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;
    // using MatrixCopyUbToGm = Gemv::Tile::MatrixCopyUBToGm<ArchTag, YType>;

    using MatrixCopyGmToUb = typename TileCopy_::MatrixCopyGmToUb;
    using MatrixCopyUbToGm = typename TileCopy_::MatrixCopyUbToGm;

    using VecCopyGmToUbCommon = typename TileCopy_::VecCopyGmToUbCommon;
    using VecCopyGmToUbTail = typename TileCopy_::VecCopyGmToUbCommon;

    using VecCopyUbToGmCommon = typename TileCopy_::VecCopyUbToGmCommon;
    using VecCopyUbToGmTail = typename TileCopy_::VecCopyUbToGmTail;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    using UBAlignHelper = Gemv::helper::UBAlignHelper<ElementA>;

    // using TensorCoord = layout::VectorLayout::TensorCoord;

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t Abuf_SOURCE_SIZE_ = 64 * 1024;
    static constexpr uint32_t Ybuf_SIZE_ = 128 * 1024;

    static_assert((UBBlockShape::N % UBTileShape::N) == 0,
        "The situation where the basic Tile of UB for Reduce Add differ on the n axes is not supported yet");

    static_assert(std::is_same_v<ElementA, ElementY>,
        "The ElementA and ElementY of Gemm should be consistent.");


    CATLASS_DEVICE
    BlockMatrixAddVectorized() {}

    /// Construct
    CATLASS_DEVICE
    BlockMatrixAddVectorized(Arch::Resource<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbAOffset = UBufAddrStart;
        uint32_t UbYOffset = UBufAddrStart + Abuf_SOURCE_SIZE_;

        // UbA1Tensor = resource.ubBuf.template GetBufferByByte<ElementA>(UbA1Offset);
        // UbA2Tensor = resource.ubBuf.template GetBufferByByte<ElementA>(UbA2Offset);
        // UbYTensor = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset);

        // UbInA1Event = 0;
        // UbInA2Event = 1;
        // UbOutEvent = 2;

        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SOURCE_SIZE_ / 2));
            UbYTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset + i * (Ybuf_SIZE_ / 2));

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
    BlockMatrixAddVectorized(Arch::ResourceAIV<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbAOffset = UBufAddrStart;
        uint32_t UbYOffset = UBufAddrStart + Abuf_SOURCE_SIZE_;

        // UbA1Tensor = resource.ubBuf.template GetBufferByByte<ElementA>(UbA1Offset);
        // UbA2Tensor = resource.ubBuf.template GetBufferByByte<ElementA>(UbA2Offset);
        // UbYTensor = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset);

        // UbInA1Event = 0;
        // UbInA2Event = 1;
        // UbOutEvent = 2;

        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SOURCE_SIZE_ / 2));
            UbYTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset + i * (Ybuf_SIZE_ / 2));

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
    ~BlockMatrixAddVectorized()
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
        LayoutVA const &layoutVA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementY> const &gmY,
        LayoutVY const &layoutVY,
        uint32_t actual_data_num, uint32_t actual_data_num_next,
        uint32_t SplitKNum, bool isFirstBlock, bool hasNextBlock)
    {
        TileSizeRound = UBTileShape::M * UBTileShape::N;
        TileSizeRound = RoundUp(TileSizeRound, UBAlignHelper::ALIGN);

        BlockSizeRound = UBBlockShape::M * UBBlockShape::N;
        BlockSizeRound = RoundUp(BlockSizeRound, UBAlignHelper::ALIGN);

        len_actual_total = (actual_data_num < BlockSizeRound) ?  actual_data_num : BlockSizeRound;

        uint32_t TileNum = (len_actual_total + TileSizeRound - 1) / TileSizeRound;

        uint32_t NloopBlock = 1;
        uint32_t NBLoopIdx = 0;

        uint32_t Nloop = TileNum;

        uint32_t A_vec_offset = NBLoopIdx * BlockSizeRound;
        uint32_t Y_vec_offset = A_vec_offset;
        uint32_t A_block_offset = A_vec_offset;
        uint32_t Y_block_offset = Y_vec_offset;

        strideSlice = layoutA.shape(0) * layoutA.shape(1);

        if(isFirstBlock){
            len_actual = (len_actual_total < TileSizeRound) ? len_actual_total : TileSizeRound;
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)UbOutEventList[UbOutListId]);
            auto UbYTensor = UbYTensorList[UbOutListId];
            auto UbYTensorFirst = UbYTensor[0];
            auto UbYTensorSecond = UbYTensor[TileSizeRound];

            /*
            CATLASS_DEVICE
            void operator()(
                AscendC::LocalTensor<Element> dstTensor,
                AscendC::GlobalTensor<Element> srcTensor,
                uint32_t len
            ) 
            */
            vecCopyGmToUbCommon(UbYTensorFirst, gmA[A_block_offset], len_actual);
            vecCopyGmToUbCommon(UbYTensorSecond, gmA[A_block_offset + strideSlice], len_actual);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)UbOutEventList[UbOutListId]);
        }

        for(uint32_t loopIdx=0; loopIdx < Nloop; loopIdx++){

            len_actual = (loopIdx == (Nloop - 1)) ? (len_actual_total - loopIdx * TileSizeRound) : TileSizeRound;

            uint32_t UbOutListIdNext = ((UbOutListId + 1) < STAGES) ? (UbOutListId + 1) : 0;
            uint32_t Y_tile_offset = loopIdx * TileSizeRound;
            uint32_t A_tile_offset = Y_tile_offset;

            if(loopIdx < (Nloop - 1)){
                uint32_t loopIdNext = loopIdx + 1;
                uint32_t len_actual_next = (loopIdNext == (Nloop - 1)) ? (len_actual_total - loopIdNext * TileSizeRound) : TileSizeRound;
                uint32_t A_tile_offset_next = loopIdNext * TileSizeRound;
                uint32_t Y_tile_offset_next = A_tile_offset_next;

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)UbOutEventList[UbOutListIdNext]);
                auto UbYTensorNext = UbYTensorList[UbOutListIdNext];
                auto UbYTensorNextFirst = UbYTensorNext[0];
                auto UbYTensorNextSecond = UbYTensorNext[TileSizeRound];

                vecCopyGmToUbCommon(UbYTensorNextFirst, gmA[A_tile_offset_next], len_actual_next);
                vecCopyGmToUbCommon(UbYTensorNextSecond, gmA[A_tile_offset_next + strideSlice], len_actual_next);

                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)UbOutEventList[UbOutListIdNext]);

            }else if(hasNextBlock){
                uint32_t loopIdNext = 0;
                uint32_t len_actual_next = (actual_data_num_next < TileSizeRound) ? actual_data_num_next : TileSizeRound;
                uint32_t A_tile_offset_next = loopIdNext * TileSizeRound;
                uint32_t Y_tile_offset_next = A_tile_offset_next;

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)UbOutEventList[UbOutListIdNext]);
                auto UbYTensorNext = UbYTensorList[UbOutListIdNext];
                auto UbYTensorNextFirst = UbYTensorNext[0];
                auto UbYTensorNextSecond = UbYTensorNext[TileSizeRound];

                vecCopyGmToUbCommon(UbYTensorNextFirst, gmNextBlockA[A_tile_offset_next], len_actual_next);
                vecCopyGmToUbCommon(UbYTensorNextSecond, gmNextBlockA[A_tile_offset_next + strideSlice], len_actual_next);

                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)UbOutEventList[UbOutListIdNext]);
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)UbOutEventList[UbOutListId]);
            
            auto UbYTensor = UbYTensorList[UbOutListId];
            // auto UbYTensorFirst = UbYTensor[0];
            // auto UbYTensorSecond = UbYTensor[TileSizeRound];

            if(SplitKNum > 2){
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
                vecCopyGmToUbCommon(UbATensorList[UbInListId], gmA[A_tile_offset + 2 * strideSlice], len_actual);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
            }
            for(uint32_t sliceIdx=2; sliceIdx < SplitKNum; sliceIdx++){
                uint32_t UbInListIdNext = ((UbInListId + 1) < STAGES) ? (UbInListId + 1) : 0;

                if(sliceIdx < (SplitKNum - 1)){
                    uint32_t sliceIdxNext = sliceIdx + 1;
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                    vecCopyGmToUbCommon(UbATensorList[UbInListIdNext], gmA[A_tile_offset + sliceIdxNext * strideSlice], len_actual);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
                auto UbInATensor = UbATensorList[UbInListId];
                uint32_t offsetYInTile = UbInListId * TileSizeRound;
                auto UbYTensorInTile = UbYTensor[offsetYInTile];
                // layoutY.GetTileLayout(TensorCoord(m_actual_total));
                auto layoutComputeInUb = layoutVA.GetTileLayout(TensorCoord(len_actual));
                auto layoutTileCompute = layoutVA.GetTileLayout(TensorCoord(len_actual));
                /*
                CATLASS_DEVICE
                void operator()(
                    AscendC::LocalTensor<ElementY> dstTensor,
                    AscendC::LocalTensor<ElementA> srcTensor_m1,
                    AscendC::LocalTensor<ElementA> srcTensor_m2,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
                */
                tileVecAdd(
                    UbYTensorInTile,
                    UbYTensorInTile,
                    UbInATensor,
                    layoutComputeInUb, layoutTileCompute);

                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));

                UbInListId = UbInListIdNext;
            }

            AscendC::PipeBarrier<PIPE_V>();

            auto layoutComputeInUb = layoutVA.GetTileLayout(TensorCoord(len_actual));
            auto layoutDstY = layoutVY.GetTileLayout(TensorCoord(len_actual));
            auto layoutTileCompute = layoutVA.GetTileLayout(TensorCoord(len_actual));

            tileVecAdd(UbYTensor,
                UbYTensor[0],
                UbYTensor[TileSizeRound],
                layoutComputeInUb, layoutTileCompute
            );
            // AscendC::PipeBarrier<PIPE_V>();

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)UbOutEventList[UbOutListId]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)UbOutEventList[UbOutListId]);

            /*
            CATLASS_DEVICE
            void operator()(
                AscendC::GlobalTensor<Element> dstTensor,
                AscendC::LocalTensor<Element> srcTensor,
                layout::VectorLayout const &layoutDst,
                layout::VectorLayout const &layoutSrc
            )
            */

            vecCopyUbToGmCommon(gmY[Y_tile_offset],
                UbYTensor,
                layoutDstY,
                layoutComputeInUb);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)UbOutEventList[UbOutListId]);

            UbOutListId = UbOutListIdNext;
        }
    }

    CATLASS_DEVICE
    void op_with_tail(
        AscendC::GlobalTensor<ElementA> const &gmA,
        LayoutVA const &layoutVA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementY> const &gmY,
        LayoutVY const &layoutVY,
        uint32_t actual_data_num, uint32_t SplitKNum)
    {
        TileSizeRound = UBTileShape::M * UBTileShape::N;
        TileSizeRound = RoundUp(TileSizeRound, UBAlignHelper::ALIGN);

        BlockSizeRound = UBBlockShape::M * UBBlockShape::N;
        BlockSizeRound = RoundUp(BlockSizeRound, UBAlignHelper::ALIGN);

        len_actual_total = (actual_data_num < BlockSizeRound) ?  actual_data_num : BlockSizeRound;

        uint32_t TileNum = (len_actual_total + TileSizeRound - 1) / TileSizeRound;

        uint32_t NloopBlock = 1;
        uint32_t NBLoopIdx = 0;

        uint32_t Nloop = TileNum;

        uint32_t A_vec_offset = NBLoopIdx * BlockSizeRound;
        uint32_t Y_vec_offset = A_vec_offset;
        uint32_t A_block_offset = A_vec_offset;
        uint32_t Y_block_offset = Y_vec_offset;

        strideSlice = layoutA.shape(0) * layoutA.shape(1);

        if (len_actual_total < TileSizeRound){
            len_actual =  len_actual_total;
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)UbOutEventList[UbOutListId]);
            auto UbYTensor = UbYTensorList[UbOutListId];
            auto UbYTensorFirst = UbYTensor[0];
            auto UbYTensorSecond = UbYTensor[TileSizeRound];

            /*
            CATLASS_DEVICE
            void operator()(
                AscendC::LocalTensor<Element> dstTensor,
                AscendC::GlobalTensor<Element> srcTensor,
                uint32_t len
            ) 
            */
            vecCopyGmToUbTail(UbYTensorFirst, gmA[A_block_offset], len_actual);
            vecCopyGmToUbTail(UbYTensorSecond, gmA[A_block_offset + strideSlice], len_actual);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)UbOutEventList[UbOutListId]);
        }else{
            len_actual = TileSizeRound;
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)UbOutEventList[UbOutListId]);
            auto UbYTensor = UbYTensorList[UbOutListId];
            auto UbYTensorFirst = UbYTensor[0];
            auto UbYTensorSecond = UbYTensor[TileSizeRound];

            /*
            CATLASS_DEVICE
            void operator()(
                AscendC::LocalTensor<Element> dstTensor,
                AscendC::GlobalTensor<Element> srcTensor,
                uint32_t len
            ) 
            */
            vecCopyGmToUbCommon(UbYTensorFirst, gmA[A_block_offset], len_actual);
            vecCopyGmToUbCommon(UbYTensorSecond, gmA[A_block_offset + strideSlice], len_actual);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)UbOutEventList[UbOutListId]);
        }
        

        for(uint32_t loopIdx=0; loopIdx < Nloop; loopIdx++){

            len_actual = (loopIdx == (Nloop - 1)) ? (len_actual_total - loopIdx * TileSizeRound) : TileSizeRound;

            uint32_t UbOutListIdNext = ((UbOutListId + 1) < STAGES) ? (UbOutListId + 1) : 0;
            uint32_t Y_tile_offset = loopIdx * TileSizeRound;
            uint32_t A_tile_offset = Y_tile_offset;

            if(loopIdx < (Nloop - 2)){
                uint32_t loopIdNext = loopIdx + 1;
                uint32_t len_actual_next = TileSizeRound;
                uint32_t A_tile_offset_next = loopIdNext * TileSizeRound;
                uint32_t Y_tile_offset_next = A_tile_offset_next;

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)UbOutEventList[UbOutListIdNext]);
                auto UbYTensorNext = UbYTensorList[UbOutListIdNext];
                auto UbYTensorNextFirst = UbYTensorNext[0];
                auto UbYTensorNextSecond = UbYTensorNext[TileSizeRound];

                vecCopyGmToUbCommon(UbYTensorNextFirst, gmA[A_tile_offset_next], len_actual_next);
                vecCopyGmToUbCommon(UbYTensorNextSecond, gmA[A_tile_offset_next + strideSlice], len_actual_next);

                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)UbOutEventList[UbOutListIdNext]);

            }else if(loopIdx == (Nloop - 2)){
                uint32_t loopIdNext = loopIdx + 1;
                uint32_t len_actual_next = (len_actual_total - loopIdNext * TileSizeRound);
                uint32_t A_tile_offset_next = loopIdNext * TileSizeRound;
                uint32_t Y_tile_offset_next = A_tile_offset_next;

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)UbOutEventList[UbOutListIdNext]);
                auto UbYTensorNext = UbYTensorList[UbOutListIdNext];
                auto UbYTensorNextFirst = UbYTensorNext[0];
                auto UbYTensorNextSecond = UbYTensorNext[TileSizeRound];

                vecCopyGmToUbTail(UbYTensorNextFirst, gmA[A_tile_offset_next], len_actual_next);
                vecCopyGmToUbTail(UbYTensorNextSecond, gmA[A_tile_offset_next + strideSlice], len_actual_next);

                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)UbOutEventList[UbOutListIdNext]);
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)UbOutEventList[UbOutListId]);
            
            auto UbYTensor = UbYTensorList[UbOutListId];
            // auto UbYTensorFirst = UbYTensor[0];
            // auto UbYTensorSecond = UbYTensor[TileSizeRound];

            if(loopIdx < (Nloop - 1)){
                if(SplitKNum > 2){
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
                    vecCopyGmToUbCommon(UbATensorList[UbInListId], gmA[A_tile_offset + 2 * strideSlice], len_actual);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
                }

                for(uint32_t sliceIdx=2; sliceIdx < SplitKNum; sliceIdx++){
                    uint32_t UbInListIdNext = ((UbInListId + 1) < STAGES) ? (UbInListId + 1) : 0;

                    if(sliceIdx < (SplitKNum - 1)){
                        uint32_t sliceIdxNext = sliceIdx + 1;
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                        vecCopyGmToUbCommon(UbATensorList[UbInListIdNext], gmA[A_tile_offset + sliceIdxNext * strideSlice], len_actual);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
                    }

                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
                    auto UbInATensor = UbATensorList[UbInListId];
                    uint32_t offsetYInTile = UbInListId * TileSizeRound;
                    auto UbYTensorInTile = UbYTensor[offsetYInTile];
                    // layoutY.GetTileLayout(TensorCoord(m_actual_total));
                    auto layoutComputeInUb = layoutVA.GetTileLayout(TensorCoord(len_actual));
                    auto layoutTileCompute = layoutVA.GetTileLayout(TensorCoord(len_actual));
                    /*
                    CATLASS_DEVICE
                    void operator()(
                        AscendC::LocalTensor<ElementY> dstTensor,
                        AscendC::LocalTensor<ElementA> srcTensor_m1,
                        AscendC::LocalTensor<ElementA> srcTensor_m2,
                        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
                    */
                    tileVecAdd(
                        UbYTensorInTile,
                        UbYTensorInTile,
                        UbInATensor,
                        layoutComputeInUb, layoutTileCompute);

                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));

                    UbInListId = UbInListIdNext;
                }

            }else{

                if(SplitKNum > 2){
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
                    vecCopyGmToUbTail(UbATensorList[UbInListId], gmA[A_tile_offset + 2 * strideSlice], len_actual);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
                }

                for(uint32_t sliceIdx=2; sliceIdx < SplitKNum; sliceIdx++){
                    uint32_t UbInListIdNext = ((UbInListId + 1) < STAGES) ? (UbInListId + 1) : 0;

                    if(sliceIdx < (SplitKNum - 1)){
                        uint32_t sliceIdxNext = sliceIdx + 1;
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                        vecCopyGmToUbTail(UbATensorList[UbInListIdNext], gmA[A_tile_offset + sliceIdxNext * strideSlice], len_actual);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
                    }

                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
                    auto UbInATensor = UbATensorList[UbInListId];
                    uint32_t offsetYInTile = UbInListId * TileSizeRound;
                    auto UbYTensorInTile = UbYTensor[offsetYInTile];
                    // layoutY.GetTileLayout(TensorCoord(m_actual_total));
                    auto layoutComputeInUb = layoutVA.GetTileLayout(TensorCoord(len_actual));
                    auto layoutTileCompute = layoutVA.GetTileLayout(TensorCoord(len_actual));
                    /*
                    CATLASS_DEVICE
                    void operator()(
                        AscendC::LocalTensor<ElementY> dstTensor,
                        AscendC::LocalTensor<ElementA> srcTensor_m1,
                        AscendC::LocalTensor<ElementA> srcTensor_m2,
                        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
                    */
                    tileVecAdd(
                        UbYTensorInTile,
                        UbYTensorInTile,
                        UbInATensor,
                        layoutComputeInUb, layoutTileCompute);

                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));

                    UbInListId = UbInListIdNext;
                }
            }

            AscendC::PipeBarrier<PIPE_V>();

            auto layoutComputeInUb = layoutVA.GetTileLayout(TensorCoord(len_actual));
            auto layoutDstY = layoutVY.GetTileLayout(TensorCoord(len_actual));
            auto layoutTileCompute = layoutVA.GetTileLayout(TensorCoord(len_actual));

            tileVecAdd(UbYTensor,
                UbYTensor[0],
                UbYTensor[TileSizeRound],
                layoutComputeInUb, layoutTileCompute
            );
            // AscendC::PipeBarrier<PIPE_V>();

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)UbOutEventList[UbOutListId]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)UbOutEventList[UbOutListId]);

            /*
            CATLASS_DEVICE
            void operator()(
                AscendC::GlobalTensor<Element> dstTensor,
                AscendC::LocalTensor<Element> srcTensor,
                layout::VectorLayout const &layoutDst,
                layout::VectorLayout const &layoutSrc
            )
            */

            if(loopIdx < (Nloop - 1)){
                vecCopyUbToGmCommon(gmY[Y_tile_offset],
                    UbYTensor,
                    layoutDstY,
                    layoutComputeInUb);
            }else{
                vecCopyUbToGmTail(gmY[Y_tile_offset],
                    UbYTensor,
                    layoutDstY,
                    layoutComputeInUb);
            }
            
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)UbOutEventList[UbOutListId]);

            UbOutListId = UbOutListIdNext;
        }
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> UbATensorList[STAGES];
    AscendC::LocalTensor<ElementY> UbYTensorList[STAGES];

    // Multi-stage event id list
    int32_t UbInAEventList[STAGES];
    int32_t UbOutEventList[STAGES];

    // The id of current stage
    uint32_t UbOutListId{0};
    uint32_t UbInListId{0};

    uint32_t m_actual, n_actual, x_actual, y_actual;
    uint32_t len_actual;
    uint32_t len_actual_total;
    uint32_t m_actual_total, n_actual_total, x_actual_total, y_actual_total;

    uint32_t TileMRound, TileNRound;
    uint32_t TileSizeRound;

    uint32_t BlockMRound, BlockNRound;
    uint32_t BlockSizeRound;

    uint32_t strideACol, strideARow;
    uint32_t strideOut;
    uint32_t strideSlice;

    TileVecAdd tileVecAdd;

    MatrixCopyGmToUb matrixCopyGmToUb;
    MatrixCopyUbToGm matrixCopyUbToGm;

    VecCopyGmToUbCommon vecCopyGmToUbCommon;
    VecCopyGmToUbTail vecCopyGmToUbTail;

    VecCopyUbToGmCommon vecCopyUbToGmCommon;
    VecCopyUbToGmTail vecCopyUbToGmTail;
};

} // namespace Catlass::Gemv::Block

#endif // CATLASS_GEMV_BLOCK_BLOCK_MATRIX_VECTORIZED_ADD_HPP
