/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMV_BLOCK_BLOCK_GEMV_SLICE_AIV_NO_SPLIT_HPP
#define CATLASS_GEMV_BLOCK_BLOCK_GEMV_SLICE_AIV_NO_SPLIT_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemv_coord.hpp"
#include "catlass/gemv/helper.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/gemm/dispatch_policy.hpp"

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
    class TileVmad_
>
struct BlockFTGemvNoSplitK <
    Gemm::GemvAtlasA2,
    Gemv::helper::FT_THRESHOLD_ALGORITHM::AABFT,
    Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::NO_FUSED,
    Gemv::helper::FT_ENC_TYPE::RCE,
    Gemv::helper::FT_COMP_TYPE::RSUB,
    UBTileShape_,
    UBBlockShape_,
    L1TileShape_,
    AType_,
    XType_,
    YType_,
    BiasType_,
    TileCopy_,
    TileVmad_
> {
public:
    // Type Aliases
    using DispatchPolicy = Gemm::GemvAtlasA2;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using FT_AIV_PIPE_FUSE_TYPE = Gemv::helper::FT_AIV_PIPE_FUSE_TYPE;
    using UBTileShape = UBTileShape_;
    using UBBlockShape = UBBlockShape_;
    using L1TileShape = L1TileShape_;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementX = typename XType_::Element;
    using LayoutX = typename XType_::Layout;
    using ElementY = typename YType_::Element;
    using LayoutY = typename YType_::Layout;
    using TileVmad = TileVmad_;
    using VecCopyGmToUb = typename TileCopy_::VecCopyGmToUb;
    using VecCopyUbToGm = typename TileCopy_::VecCopyUbToGm;
    using MatrixCopyGmToUb = typename TileCopy_::MatrixCopyGmToUb;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    using UBAlignHelper = Gemv::helper::UBAlignHelper<ElementA>;
    using TensorCoord = layout::VectorLayout::TensorCoord;

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE = Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::NO_FUSED;
    static constexpr uint32_t Abuf_SIZE_ = 128 * 1024;
    static constexpr uint32_t Xbuf_SIZE_ = 16 * 1024;
    static constexpr uint32_t Ybuf_SIZE_ = 16 * 1024;
    static constexpr uint32_t workspace_SIZE_ = 32 * 1024;

    static_assert(L1TileShape::M == UBBlockShape::M,
        "The situation where the basic Tile of UB and L1 for MMA differ on the m axes is not supported yet");

    static_assert((UBBlockShape::N % UBTileShape::N) == 0,
        "The situation where the basic Tile of UB and L1 for MMA differ on the n axes is not supported yet");

    CATLASS_DEVICE
    BlockFTGemvNoSplitK() {}

    /// Construct
    CATLASS_DEVICE
    BlockFTGemvNoSplitK(Arch::Resource<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbAOffset = UBufAddrStart;
        uint32_t UbXOffset = UBufAddrStart + Abuf_SIZE_;
        uint32_t UbYOffset = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_;
        uint32_t UbWOffset = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_ + Ybuf_SIZE_;
        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SIZE_ / 2));
            UbXTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbXOffset + i * (Xbuf_SIZE_ / 2));
            UbYTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset + i * (Ybuf_SIZE_ / 2));
            UbWTensorList[i] =
                resource.ubBuf.template GetBufferByByte<ElementAccumulator>(UbWOffset + i * (workspace_SIZE_ / 2));

            // Assign event ID for each stages
            UbInAEventList[i] = i;
            UbInXEventList[i] = i + STAGES;
            UbOutEventList[i] = i;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutEventList[i]);
        }
    }

    /// Construct
    CATLASS_DEVICE
    BlockFTGemvNoSplitK(Arch::ResourceAIV<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbAOffset = UBufAddrStart;
        uint32_t UbXOffset = UBufAddrStart + Abuf_SIZE_;
        uint32_t UbYOffset = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_;
        uint32_t UbWOffset = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_ + Ybuf_SIZE_;
        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SIZE_ / 2));
            UbXTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbXOffset + i * (Xbuf_SIZE_ / 2));
            UbYTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset + i * (Ybuf_SIZE_ / 2));
            UbWTensorList[i] =
                resource.ubBuf.template GetBufferByByte<ElementAccumulator>(UbWOffset + i * (workspace_SIZE_ / 2));

            // Assign event ID for each stages
            UbInAEventList[i] = i;
            UbInXEventList[i] = i + STAGES;
            UbOutEventList[i] = i;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutEventList[i]);
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockFTGemvNoSplitK()
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(UbOutEventList[i]);
        }
    }

    // float alpha,
        // float beta

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementX> const &gmX, LayoutX const &layoutX,
        AscendC::GlobalTensor<ElementY> const &gmZ, LayoutY const &layoutY,
        GemvCoord const &actualShape, uint32_t aiv_part_num)
    {
        TileMRound = RoundUp(UBTileShape::M, UBAlignHelper::ALIGN);
        TileNRound = RoundUp(UBTileShape::N, UBAlignHelper::ALIGN);
        
        strideACol = layoutA.stride(1) * TileNRound;
        strideARow = layoutA.stride(0) * TileMRound;

        BlockMRound = RoundUp(UBBlockShape::M, UBAlignHelper::ALIGN);
        BlockNRound = RoundUp(UBBlockShape::N, UBAlignHelper::ALIGN);

        m_actual_total = (actualShape.m() < BlockMRound) ? actualShape.m() : BlockMRound;
        n_actual_total = actualShape.n();
        // uint32_t aiv_part_num = 1 * AscendC::GetTaskRation();

        m_actual_part = m_actual_total / aiv_part_num;

        uint32_t M_start_offset = AscendC::GetSubBlockIdx() * m_actual_part;

        if(AscendC::GetSubBlockIdx() == (aiv_part_num -1)) {
            m_actual_part = m_actual_total - M_start_offset;
        }

        uint32_t Nloop = CeilDiv(n_actual_total, TileNRound);
        uint32_t Mloop = CeilDiv(m_actual_part, TileMRound);



        for(uint32_t mLoopIdx = 0; mLoopIdx < Mloop; mLoopIdx++){

            m_actual = (mLoopIdx < (Mloop - 1)) ? TileMRound : m_actual_part - mLoopIdx * TileMRound;
            n_actual = (n_actual_total < TileNRound) ? n_actual_total : TileNRound;

            uint32_t mLoopOffset = mLoopIdx * TileMRound + M_start_offset;
            uint32_t A_row_offset = mLoopOffset;
            uint32_t A_col_offset = 0;
            uint32_t A_block_offset = A_row_offset * layoutA.stride(0) + A_col_offset * layoutA.stride(1);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListId]));
            auto UbYTensor = UbYTensorList[UbOutListId];
            AscendC::Duplicate<ElementY>(UbYTensor, (ElementY)0.0, m_actual);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventList[UbOutListId]));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventList[UbOutListId]));
            
            if(mLoopIdx == 0){
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbInListId]));
                vecCopyGmToUb(UbXTensorList[UbInListId], gmX, n_actual);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbInListId]));

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
                auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
                matrixCopyGmToUb(UbATensorList[UbInListId], gmA[A_block_offset], layoutAInUb, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
            }
            
            // main loop
            for (uint32_t nLoopIdx = 0; nLoopIdx < Nloop; nLoopIdx++) {
                m_actual = (mLoopIdx < (Mloop - 1)) ? TileMRound : m_actual_part - mLoopIdx * TileMRound;
                n_actual = (nLoopIdx == Nloop - 1) ? (n_actual_total - nLoopIdx * TileNRound) : TileNRound;
                y_actual = m_actual;
                x_actual = n_actual;

                uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;
                if (nLoopIdx < Nloop - 1) {
                    uint32_t nLoopIdxNext = nLoopIdx + 1;
                    uint32_t m_actual_next = m_actual;
                    uint32_t n_actual_next =
                        (nLoopIdxNext == Nloop - 1) ? (n_actual_total - nLoopIdxNext * TileNRound) : TileNRound;
                    uint32_t y_actual_next = m_actual_next;
                    uint32_t x_actual_next = n_actual_next;
                    // Get L1 tensor for next stage
                    auto matrixTensor = UbATensorList[UbInListIdNext];
                    auto vecTensor = UbXTensorList[UbInListIdNext];

                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbInListIdNext]));
                    vecCopyGmToUb(vecTensor, gmX[nLoopIdxNext * TileNRound], x_actual_next);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbInListIdNext]));
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                    auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                    matrixCopyGmToUb(matrixTensor, gmA[A_block_offset + nLoopIdxNext * strideACol], layoutAInUb, layoutTileA);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
                }else if ((nLoopIdx == Nloop -1) && (mLoopIdx < (Mloop -1))){
                    uint32_t mLoopIdxNext = mLoopIdx + 1;
                    uint32_t nLoopIdxNext = 0;
                    uint32_t m_actual_next = (mLoopIdxNext < (Mloop - 1)) ? TileMRound : m_actual_part - mLoopIdxNext * TileMRound;
                    uint32_t n_actual_next =
                        (nLoopIdxNext == Nloop - 1) ? (n_actual_total - nLoopIdxNext * TileNRound) : TileNRound;
                    uint32_t y_actual_next = m_actual_next;
                    uint32_t x_actual_next = n_actual_next;
                    uint32_t A_block_offset_next = A_block_offset + strideARow;
                    // Get L1 tensor for next stage
                    auto matrixTensor = UbATensorList[UbInListIdNext];
                    auto vecTensor = UbXTensorList[UbInListIdNext];

                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbInListIdNext]));
                    vecCopyGmToUb(vecTensor, gmX[nLoopIdxNext * TileNRound], x_actual_next);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbInListIdNext]));
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                    auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                    matrixCopyGmToUb(matrixTensor, gmA[A_block_offset_next + nLoopIdxNext * strideACol], layoutAInUb, layoutTileA);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
                    
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbInListId]));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
                auto layoutComputeInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileCompute = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
                auto UbYTensor = UbYTensorList[UbOutListId];
                tileVmad(UbYTensor,
                    UbXTensorList[UbInListId],
                    UbATensorList[UbInListId],
                    UbWTensorList[UbInListId],
                    layoutComputeInUb,
                    layoutTileCompute);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbInListId]));
                UbInListId = UbInListIdNext;
            }

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));
            auto layoutDstY = layoutY.GetTileLayout(TensorCoord(y_actual));
            auto layoutComputeInUb = layoutY.GetTileLayout(TensorCoord(y_actual));
            vecCopyUbToGm(gmZ[mLoopOffset], UbYTensorList[UbOutListId], layoutDstY, layoutComputeInUb);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListId]));
            UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;

        }
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> UbATensorList[STAGES];
    AscendC::LocalTensor<ElementX> UbXTensorList[STAGES];
    AscendC::LocalTensor<ElementY> UbYTensorList[STAGES];
    AscendC::LocalTensor<ElementAccumulator> UbWTensorList[STAGES];

    // Multi-stage event id list
    int32_t UbInAEventList[STAGES];
    int32_t UbInXEventList[STAGES];
    int32_t UbOutEventList[STAGES];

    // The id of current stage
    uint32_t UbOutListId{0};
    uint32_t UbInListId{0};

    uint32_t m_actual, n_actual, x_actual, y_actual;
    uint32_t m_actual_total, n_actual_total, x_actual_total, y_actual_total;
    uint32_t m_actual_part;
    uint32_t TileMRound, TileNRound;
    uint32_t BlockMRound, BlockNRound;
    uint32_t TaskSplit;
    uint32_t MatrixOffset;
    uint32_t strideARow, strideACol;
    uint32_t strideOut;

    TileVmad tileVmad;
    
    MatrixCopyGmToUb matrixCopyGmToUb;
    VecCopyGmToUb vecCopyGmToUb;
    VecCopyUbToGm vecCopyUbToGm;
};


template <
    class UBTileShape_,
    class UBBlockShape_,
    class L1TileShape_,
    class AType_,
    class CType_,
    class XType_,
    class YType_,
    class ZType_,
    class BiasType_,
    class TileCopy_,
    class TileVmad_,
    class TileThreCalc_,
    class TileVmuls_
>
struct BlockFTGemvNoSplitK <
    Gemm::GemvAtlasA2,
    Gemv::helper::FT_THRESHOLD_ALGORITHM::AABFT,
    Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::ABE_FUSED_THRE,
    Gemv::helper::FT_ENC_TYPE::RCE,
    Gemv::helper::FT_COMP_TYPE::RSUB,
    UBTileShape_,
    UBBlockShape_,
    L1TileShape_,
    AType_,
    CType_,
    XType_,
    YType_,
    ZType_,
    BiasType_,
    TileCopy_,
    TileVmad_,
    TileThreCalc_,
    TileVmuls_
> {
public:
    // Type Aliases
    using DispatchPolicy = Gemm::GemvAtlasA2;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using FT_ENC_TYPE = Catlass::Gemv::helper::FT_ENC_TYPE;
    using FT_COMP_TYPE = Catlass::Gemv::helper::FT_COMP_TYPE;
    using FT_AIV_PIPE_FUSE_TYPE = Catlass::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE;
    using FT_THRESHOLD_ALGORITHM = Catlass::Gemv::helper::FT_THRESHOLD_ALGORITHM;

    using UBTileShape = UBTileShape_;
    using UBBlockShape = UBBlockShape_;
    using L1TileShape = L1TileShape_;

    using ThreCalcUBTileShape = GemvShape<UBTileShape::M,L1TileShape::N>;
    using ThreCalcUBTileShapeTotal = GemvShape<L1TileShape::M,L1TileShape::N>;

    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;

    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;

    using ElementX = typename XType_::Element;
    using LayoutX = typename XType_::Layout;

    using ElementY = typename YType_::Element;
    using LayoutY = typename YType_::Layout;

    using ElementZ = typename ZType_::Element;
    using LayoutZ = typename ZType_::Layout;

    using TileVmad = TileVmad_;

    using TileThreCalc = TileThreCalc_;
    using TileVmuls = TileVmuls_;

    using VecCopyGmToUb = typename TileCopy_::VecCopyGmToUb;
    using VecCopyUbToGm = typename TileCopy_::VecCopyUbToGm;
    using MatrixCopyGmToUb = typename TileCopy_::MatrixCopyGmToUb;
    using MatrixCopyGmToUbforThre = typename TileCopy_::MatrixCopyGmToUbforThre;
    using VecCopyGmToUbInY = typename TileCopy_::VecCopyGmToUbInY;
    using VecCopyUbToGmZ = typename TileCopy_::VecCopyUbToGmZ;
    using VecCopyUbToGmforThre = typename TileCopy_::VecCopyUbToGmforThre;

    using ElementXoR = uint16_t;
    using ElementComp = int32_t;
    using ElementSub =  ElementY;

    using ElementWork = ElementY;

    using TileCompare = Gemv::Tile::TileFaultVcompare<FT_COMP_TYPE::RSUB, ArchTag, 
                                        ZType_, YType_, YType_>;
    
    static constexpr FT_COMP_TYPE COMP_TYPE = FT_COMP_TYPE::RSUB;
    static constexpr FT_ENC_TYPE ENC_TYPE = FT_ENC_TYPE::RCE;
    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = FT_THRESHOLD_ALGORITHM::AABFT;



    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    using UBAlignHelper = Gemv::helper::UBAlignHelper<ElementA>;
    using TensorCoord = layout::VectorLayout::TensorCoord;

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE = Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::NO_FUSED;
    static constexpr uint32_t Abuf_SIZE_ = 128 * 1024;
    static constexpr uint32_t Xbuf_SIZE_ = 12 * 1024;
    static constexpr uint32_t Zbuf_SIZE_ = 2 *1024;
    static constexpr uint32_t Ybuf_SIZE_ = 6 * 1024;
    static constexpr uint32_t Threbuf_SIZE_ = 6 * 1024;
    static constexpr uint32_t InCbuf_SIZE_ = 6 * 1024;
    static constexpr uint32_t workspace_SIZE_ = 32 * 1024;

    static_assert(L1TileShape::M == UBBlockShape::M,
        "The situation where the basic Tile of UB and L1 for MMA differ on the m axes is not supported yet");

    static_assert((UBBlockShape::N % UBTileShape::N) == 0,
        "The situation where the basic Tile of UB and L1 for MMA differ on the n axes is not supported yet");

    CATLASS_DEVICE
    BlockFTGemvNoSplitK() {}

    /// Construct
    CATLASS_DEVICE
    BlockFTGemvNoSplitK(Arch::Resource<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbAOffset = UBufAddrStart;
        uint32_t UbXOffset = UBufAddrStart + Abuf_SIZE_;
        uint32_t UbYOffset = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_;

        uint32_t UbInCOffset = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_ + Ybuf_SIZE_;
        uint32_t UbThreOffset = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_ + Ybuf_SIZE_ + InCbuf_SIZE_;
        uint32_t UbZOffset = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_ + Ybuf_SIZE_ + InCbuf_SIZE_ + Threbuf_SIZE_;

        uint32_t UbWOffset = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_ + Ybuf_SIZE_ + InCbuf_SIZE_ + Threbuf_SIZE_ + Zbuf_SIZE_;
        
        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SIZE_ / 2));
            UbXTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbXOffset + i * (Xbuf_SIZE_ / 2));
            
            UbYTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset + i * (Ybuf_SIZE_ / 2));

            UbWTensorList[i] =
                resource.ubBuf.template GetBufferByByte<ElementAccumulator>(UbWOffset + i * (workspace_SIZE_ / 2));

            UbCTensorList[i] = UbATensorList[i].template ReinterpretCast<ElementC>();
            UbInCTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbInCOffset + i * (InCbuf_SIZE_ / 2));
            UbWforThreTensorList[i] = UbWTensorList[i].template ReinterpretCast<ElementY>();
            UbThreTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbThreOffset + i * (Threbuf_SIZE_ / 2));
            UbZTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementZ>(UbZOffset + i * (Zbuf_SIZE_ / 2));


            // Assign event ID for each stages
            UbInAEventList[i] = i;
            UbInXEventList[i] = i + STAGES;

            UbInCEventList[i] = i + STAGES * 2;
            UbOutEventList[i] = i;
            UbOutZEventList[i] = i + STAGES;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInCEventList[i]);
            // AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutZEventList[i]);
        }
    }

    /// Construct
    CATLASS_DEVICE
    BlockFTGemvNoSplitK(Arch::ResourceAIV<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbAOffset = UBufAddrStart;
        uint32_t UbXOffset = UBufAddrStart + Abuf_SIZE_;
        uint32_t UbYOffset = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_;

        uint32_t UbInCOffset = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_ + Ybuf_SIZE_;
        uint32_t UbThreOffset = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_ + Ybuf_SIZE_ + InCbuf_SIZE_;
        uint32_t UbZOffset = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_ + Ybuf_SIZE_ + InCbuf_SIZE_ + Threbuf_SIZE_;
        uint32_t UbWOffset = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_ + Ybuf_SIZE_ + InCbuf_SIZE_ + Threbuf_SIZE_ + Zbuf_SIZE_;
        
        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SIZE_ / 2));
            UbXTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbXOffset + i * (Xbuf_SIZE_ / 2));
            
            UbYTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset + i * (Ybuf_SIZE_ / 2));

            UbWTensorList[i] =
                resource.ubBuf.template GetBufferByByte<ElementAccumulator>(UbWOffset + i * (workspace_SIZE_ / 2));

            UbCTensorList[i] = UbATensorList[i].template ReinterpretCast<ElementC>();
            UbInCTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbInCOffset + i * (InCbuf_SIZE_ / 2));
            UbWforThreTensorList[i] = UbWTensorList[i].template ReinterpretCast<ElementY>();
            UbThreTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbThreOffset + i * (Threbuf_SIZE_ / 2));
            UbZTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementZ>(UbZOffset + i * (Zbuf_SIZE_ / 2));


            // Assign event ID for each stages
            UbInAEventList[i] = i;
            UbInXEventList[i] = i + STAGES;

            UbInCEventList[i] = i + STAGES * 2;
            UbOutEventList[i] = i;
            UbOutZEventList[i] = i + STAGES;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInCEventList[i]);
            // AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutZEventList[i]);
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockFTGemvNoSplitK()
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInCEventList[i]);
            // AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(UbOutEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(UbOutZEventList[i]);
        }
    }

    // float alpha,
        // float beta

    CATLASS_DEVICE
    void normal_op(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementX> const &gmX, LayoutX const &layoutX,
        AscendC::GlobalTensor<ElementY> const &gmZ, LayoutY const &layoutY,
        GemvCoord const &actualShape, uint32_t aiv_part_num)
    {
        TileMRound = RoundUp(UBTileShape::M, UBAlignHelper::ALIGN);
        TileNRound = RoundUp(UBTileShape::N, UBAlignHelper::ALIGN);
        
        strideACol = layoutA.stride(1) * TileNRound;
        strideARow = layoutA.stride(0) * TileMRound;

        BlockMRound = RoundUp(UBBlockShape::M, UBAlignHelper::ALIGN);
        BlockNRound = RoundUp(UBBlockShape::N, UBAlignHelper::ALIGN);

        m_actual_total = (actualShape.m() < BlockMRound) ? actualShape.m() : BlockMRound;
        n_actual_total = actualShape.n();
        // uint32_t aiv_part_num = 1 * AscendC::GetTaskRation();

        m_actual_part = m_actual_total / aiv_part_num;

        uint32_t M_start_offset = AscendC::GetSubBlockIdx() * m_actual_part;

        if(AscendC::GetSubBlockIdx() == (aiv_part_num -1)) {
            m_actual_part = m_actual_total - M_start_offset;
        }

        uint32_t Nloop = CeilDiv(n_actual_total, TileNRound);
        uint32_t Mloop = CeilDiv(m_actual_part, TileMRound);



        for(uint32_t mLoopIdx = 0; mLoopIdx < Mloop; mLoopIdx++){

            m_actual = (mLoopIdx < (Mloop - 1)) ? TileMRound : m_actual_part - mLoopIdx * TileMRound;
            n_actual = (n_actual_total < TileNRound) ? n_actual_total : TileNRound;

            uint32_t mLoopOffset = mLoopIdx * TileMRound + M_start_offset;
            uint32_t A_row_offset = mLoopOffset;
            uint32_t A_col_offset = 0;
            uint32_t A_block_offset = A_row_offset * layoutA.stride(0) + A_col_offset * layoutA.stride(1);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutZEventList[UbOutListId]));
            auto UbYTensor = UbYTensorList[UbOutListId];
            AscendC::Duplicate<ElementY>(UbYTensor, (ElementY)0.0, m_actual);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutZEventList[UbOutListId]));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutZEventList[UbOutListId]));
            
            if(mLoopIdx == 0){
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbInListId]));
                vecCopyGmToUb(UbXTensorList[UbInListId], gmX, n_actual);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbInListId]));

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
                auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
                matrixCopyGmToUb(UbATensorList[UbInListId], gmA[A_block_offset], layoutAInUb, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
            }
            
            // main loop
            for (uint32_t nLoopIdx = 0; nLoopIdx < Nloop; nLoopIdx++) {
                m_actual = (mLoopIdx < (Mloop - 1)) ? TileMRound : m_actual_part - mLoopIdx * TileMRound;
                n_actual = (nLoopIdx == Nloop - 1) ? (n_actual_total - nLoopIdx * TileNRound) : TileNRound;
                y_actual = m_actual;
                x_actual = n_actual;

                uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;
                if (nLoopIdx < Nloop - 1) {
                    uint32_t nLoopIdxNext = nLoopIdx + 1;
                    uint32_t m_actual_next = m_actual;
                    uint32_t n_actual_next =
                        (nLoopIdxNext == Nloop - 1) ? (n_actual_total - nLoopIdxNext * TileNRound) : TileNRound;
                    uint32_t y_actual_next = m_actual_next;
                    uint32_t x_actual_next = n_actual_next;
                    // Get L1 tensor for next stage
                    auto matrixTensor = UbATensorList[UbInListIdNext];
                    auto vecTensor = UbXTensorList[UbInListIdNext];

                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbInListIdNext]));
                    vecCopyGmToUb(vecTensor, gmX[nLoopIdxNext * TileNRound], x_actual_next);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbInListIdNext]));
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                    auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                    matrixCopyGmToUb(matrixTensor, gmA[A_block_offset + nLoopIdxNext * strideACol], layoutAInUb, layoutTileA);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
                }else if ((nLoopIdx == Nloop -1) && (mLoopIdx < (Mloop -1))){
                    uint32_t mLoopIdxNext = mLoopIdx + 1;
                    uint32_t nLoopIdxNext = 0;
                    uint32_t m_actual_next = (mLoopIdxNext < (Mloop - 1)) ? TileMRound : m_actual_part - mLoopIdxNext * TileMRound;
                    uint32_t n_actual_next =
                        (nLoopIdxNext == Nloop - 1) ? (n_actual_total - nLoopIdxNext * TileNRound) : TileNRound;
                    uint32_t y_actual_next = m_actual_next;
                    uint32_t x_actual_next = n_actual_next;
                    uint32_t A_block_offset_next = A_block_offset + strideARow;
                    // Get L1 tensor for next stage
                    auto matrixTensor = UbATensorList[UbInListIdNext];
                    auto vecTensor = UbXTensorList[UbInListIdNext];

                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbInListIdNext]));
                    vecCopyGmToUb(vecTensor, gmX[nLoopIdxNext * TileNRound], x_actual_next);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbInListIdNext]));
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                    auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                    matrixCopyGmToUb(matrixTensor, gmA[A_block_offset_next + nLoopIdxNext * strideACol], layoutAInUb, layoutTileA);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
                    
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbInListId]));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
                auto layoutComputeInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileCompute = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
                auto UbYTensor = UbYTensorList[UbOutListId];
                tileVmad(UbYTensor,
                    UbXTensorList[UbInListId],
                    UbATensorList[UbInListId],
                    UbWTensorList[UbInListId],
                    layoutComputeInUb,
                    layoutTileCompute);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbInListId]));
                UbInListId = UbInListIdNext;
            }

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutZEventList[UbOutListId]));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutZEventList[UbOutListId]));
            auto layoutDstY = layoutY.GetTileLayout(TensorCoord(y_actual));
            auto layoutComputeInUb = layoutY.GetTileLayout(TensorCoord(y_actual));
            vecCopyUbToGm(gmZ[mLoopOffset], UbYTensorList[UbOutListId], layoutDstY, layoutComputeInUb);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutZEventList[UbOutListId]));
            UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;

        }
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementX> const &gmX, LayoutX const &layoutX,
        AscendC::GlobalTensor<ElementY> const &gmY, LayoutY const &layoutY,
        AscendC::GlobalTensor<ElementC> const &gmC, LayoutC const &layoutC,
        AscendC::GlobalTensor<ElementY> const &gmInC, LayoutY const &layoutInC,
        AscendC::GlobalTensor<ElementY> const &gmThreZ, LayoutY const &layoutThre,
        AscendC::GlobalTensor<ElementZ> const &gmCOMPZ, LayoutZ const &layoutZ,
        GemvCoord const &actualShape, GemvCoord const &ThreactualShape,
        ElementY alpha, bool outputThre, bool outputABE,
        uint32_t aiv_part_num)
    {
        TileMRound = RoundUp(UBTileShape::M, UBAlignHelper::ALIGN);
        TileNRound = RoundUp(UBTileShape::N, UBAlignHelper::ALIGN);

        ThreTileMRound = RoundUp(ThreCalcUBTileShape::M, UBAlignHelper::ALIGN);
        ThreTileNRound = RoundUp(ThreCalcUBTileShape::N, UBAlignHelper::ALIGN);
        

        BlockMRound = RoundUp(UBBlockShape::M, UBAlignHelper::ALIGN);
        BlockNRound = RoundUp(UBBlockShape::N, UBAlignHelper::ALIGN);

        ThreBlockMRound = RoundUp(ThreCalcUBTileShapeTotal::M, UBAlignHelper::ALIGN);
        ThreBlockNRound = RoundUp(ThreCalcUBTileShapeTotal::N, UBAlignHelper::ALIGN);

        strideACol = layoutA.stride(1) * TileNRound;
        strideARow = layoutA.stride(0) * TileMRound;

        strideCCol = layoutC.stride(1) * ThreTileNRound;
        strideCRow = layoutC.stride(0) * ThreTileMRound;

        m_actual_total = (actualShape.m() < BlockMRound) ? actualShape.m() : BlockMRound;
        n_actual_total = actualShape.n();

        
        thre_n_actual_total = (ThreactualShape.n() < ThreBlockNRound) ? ThreactualShape.n() : ThreBlockNRound;
        // uint32_t aiv_part_num = 1 * AscendC::GetTaskRation();

        m_actual_part = m_actual_total / aiv_part_num;

        uint32_t M_start_offset = AscendC::GetSubBlockIdx() * m_actual_part;
        uint32_t Z_start_offset = (M_start_offset + 8 - 1) / 8;

        if(AscendC::GetSubBlockIdx() == (aiv_part_num -1)) {
            m_actual_part = m_actual_total - M_start_offset;
        }

        uint32_t Mloop = CeilDiv(m_actual_part, TileMRound);

        uint32_t NloopV = CeilDiv(n_actual_total, TileNRound);
        uint32_t NloopT = CeilDiv(thre_n_actual_total, ThreTileNRound);

        dst_offset_ratio = Mloop;
        out_z_actual_total = (m_actual_part + 8 - 1) / 8;

        for(uint32_t mLoopIdx = 0; mLoopIdx < Mloop; mLoopIdx++){

            m_actual = (mLoopIdx < (Mloop - 1)) ? TileMRound : m_actual_part - mLoopIdx * TileMRound;
            n_actual = (n_actual_total < TileNRound) ? n_actual_total : TileNRound;

            thre_n_actual = (thre_n_actual_total < ThreTileNRound) ? thre_n_actual_total : ThreTileNRound;

            out_z_actual = (m_actual + 8 - 1) / 8;

            uint32_t mLoopOffset = mLoopIdx * TileMRound + M_start_offset;
            uint32_t mLoopOffset_for_z = (mLoopOffset + 8 - 1) / 8;

            uint32_t A_row_offset = mLoopOffset;
            uint32_t A_col_offset = 0;
            uint32_t A_block_offset = A_row_offset * layoutA.stride(0) + A_col_offset * layoutA.stride(1);

            uint32_t C_row_offset = mLoopOffset;
            uint32_t C_col_offset = 0;
            uint32_t C_block_offset = C_row_offset * layoutC.stride(0) + C_col_offset * layoutC.stride(1);

            
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutZEventList[UbOutListId]));
            auto UbYTensor = UbYTensorList[UbOutListId];
            auto UbThreTensor = UbThreTensorList[UbOutListId];

            AscendC::Duplicate<ElementY>(UbYTensor, (ElementY)0.0, m_actual);
            AscendC::Duplicate<ElementY>(UbThreTensor, (ElementY)0.0, m_actual);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInCEventList[UbOutListId]));
            vecCopyGmToUbInY(UbInCTensorList[UbOutListId], gmInC[mLoopOffset], m_actual);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInCEventList[UbOutListId]));
            
            if(mLoopIdx == 0){
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbInListId]));
                vecCopyGmToUb(UbXTensorList[UbInListId], gmX, n_actual);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbInListId]));

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
                auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
                matrixCopyGmToUb(UbATensorList[UbInListId], gmA[A_block_offset], layoutAInUb, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
            }
            
            // main loop
            for (uint32_t nLoopIdx = 0; nLoopIdx < NloopV; nLoopIdx++) {
                m_actual = (mLoopIdx < (Mloop - 1)) ? TileMRound : m_actual_part - mLoopIdx * TileMRound;
                n_actual = (nLoopIdx == NloopV - 1) ? (n_actual_total - nLoopIdx * TileNRound) : TileNRound;
                y_actual = m_actual;
                x_actual = n_actual;

                uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;
                if (nLoopIdx < NloopV - 1) {
                    uint32_t nLoopIdxNext = nLoopIdx + 1;
                    uint32_t m_actual_next = m_actual;
                    uint32_t n_actual_next =
                        (nLoopIdxNext == NloopV - 1) ? (n_actual_total - nLoopIdxNext * TileNRound) : TileNRound;
                    uint32_t y_actual_next = m_actual_next;
                    uint32_t x_actual_next = n_actual_next;
                    // Get L1 tensor for next stage
                    auto matrixTensor = UbATensorList[UbInListIdNext];
                    auto vecTensor = UbXTensorList[UbInListIdNext];

                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbInListIdNext]));
                    vecCopyGmToUb(vecTensor, gmX[nLoopIdxNext * TileNRound], x_actual_next);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbInListIdNext]));

                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                    auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                    matrixCopyGmToUb(matrixTensor, gmA[A_block_offset + nLoopIdxNext * strideACol], layoutAInUb, layoutTileA);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
                }else if (nLoopIdx == NloopV -1){
                    uint32_t mLoopIdxNext = mLoopIdx;
                    uint32_t nLoopIdxNext = 0;
                    uint32_t m_actual_next = m_actual;
                    uint32_t thre_n_actual_next =
                        (nLoopIdxNext == NloopT - 1) ? (thre_n_actual_total - nLoopIdxNext * ThreTileNRound) : ThreTileNRound;
                    uint32_t y_actual_next = m_actual_next;
                    uint32_t x_actual_next = thre_n_actual_next;
                    // Get L1 tensor for next stage
                    auto matrixTensor = UbCTensorList[UbInListIdNext];

                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                    auto layoutCInUb = layoutC.GetTileLayout(MakeCoord(ThreTileMRound, ThreTileNRound));
                    auto layoutTileC = layoutC.GetTileLayout(MakeCoord(m_actual_next, thre_n_actual_next));
                    matrixCopyGmToUbforThre(matrixTensor, gmC[C_block_offset + nLoopIdxNext * strideCCol], layoutCInUb, layoutTileC);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbInListId]));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
                auto layoutComputeInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileCompute = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
                // auto UbYTensor = UbYTensorList[UbOutListId];
                tileVmad(UbYTensor,
                    UbXTensorList[UbInListId],
                    UbATensorList[UbInListId],
                    UbWTensorList[UbInListId],
                    layoutComputeInUb,
                    layoutTileCompute);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbInListId]));
                UbInListId = UbInListIdNext;
            }

            AscendC::PipeBarrier<PIPE_ALL>();

            // UbThreTensor
            // auto UbThreTensor = UbThreTensorList[UbOutListId];

            for (uint32_t nLoopIdx = 0; nLoopIdx < NloopT; nLoopIdx++) {
                thre_n_actual = (nLoopIdx == NloopT - 1) ? (thre_n_actual_total - nLoopIdx * ThreTileNRound) : ThreTileNRound;
                y_actual = m_actual;
                x_actual = thre_n_actual;

                uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;
                if (nLoopIdx < NloopT - 1) {
                    uint32_t nLoopIdxNext = nLoopIdx + 1;
                    uint32_t m_actual_next = m_actual;
                    uint32_t thre_n_actual_next =
                        (nLoopIdxNext == NloopT - 1) ? (thre_n_actual_total - nLoopIdxNext * ThreTileNRound) : ThreTileNRound;
                    uint32_t y_actual_next = m_actual_next;
                    uint32_t x_actual_next = thre_n_actual_next;
                    // uint32_t row_start_next = mLoopIdx * TileMRound;
                    // Get L1 tensor for next stage
                    auto matrixTensor = UbCTensorList[UbInListIdNext];

                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                    auto layoutCInUb = layoutC.GetTileLayout(MakeCoord(ThreTileMRound, ThreTileNRound));
                    auto layoutTileC = layoutC.GetTileLayout(MakeCoord(m_actual_next, thre_n_actual_next));
                    matrixCopyGmToUbforThre(matrixTensor, gmC[C_block_offset + nLoopIdxNext * strideCCol], layoutCInUb, layoutTileC);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
                }else if ((nLoopIdx == NloopT - 1) && (mLoopIdx < (Mloop -1))){
                    uint32_t mLoopIdxNext = mLoopIdx + 1;
                    uint32_t nLoopIdxNext = 0;
                    uint32_t m_actual_next = (mLoopIdxNext < (Mloop - 1)) ? TileMRound : m_actual_part - mLoopIdxNext * TileMRound;
                    uint32_t n_actual_next =
                        (nLoopIdxNext == NloopV - 1) ? (n_actual_total - nLoopIdxNext * TileNRound) : TileNRound;
                    uint32_t y_actual_next = m_actual_next;
                    uint32_t x_actual_next = n_actual_next;
                    uint32_t A_block_offset_next = A_block_offset + strideARow;

                    // Get L1 tensor for next stage
                    auto matrixTensor = UbATensorList[UbInListIdNext];
                    auto vecTensor = UbXTensorList[UbInListIdNext];

                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbInListIdNext]));
                    vecCopyGmToUb(vecTensor, gmX[nLoopIdxNext * TileNRound], x_actual_next);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbInListIdNext]));

                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                    auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                    matrixCopyGmToUb(matrixTensor, gmA[A_block_offset_next + nLoopIdxNext * strideACol], layoutAInUb, layoutTileA);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
                auto layoutComputeInUb = layoutC.GetTileLayout(MakeCoord(ThreTileMRound, ThreTileNRound));
                auto layoutTileCompute = layoutC.GetTileLayout(MakeCoord(m_actual, n_actual));

                tileThreCalc(
                    UbThreTensor,
                    UbCTensorList[UbInListId],
                    UbWforThreTensorList[UbInListId],
                    layoutComputeInUb,
                    layoutTileCompute,
                    dst_offset_ratio);

                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
                // AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbInListId]));
                UbInListId = UbInListIdNext;
            }
            
            AscendC::PipeBarrier<PIPE_V>();

            // UbYTensorList[UbOutListId]
            tileVmuls(UbThreTensorList[UbOutListId], UbThreTensorList[UbOutListId], (ElementY)alpha, m_actual);
            
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInCEventList[UbOutListId]));

            auto layoutCompareInUb = layoutThre.GetTileLayout(MakeCoord(ThreTileMRound));
            auto layoutTileCompare = layoutThre.GetTileLayout(MakeCoord(m_actual));

            /*
            void operator()(
                AscendC::LocalTensor<ElementZ> dstTensor,
                AscendC::LocalTensor<ElementX> srcTensor_x,
                AscendC::LocalTensor<ElementY> srcTensor_y,
                AscendC::LocalTensor<ElementX> srcTensor_thre,
                LayoutDst const &layoutDst, LayoutSrc const &layoutSrc, ElementX threshold
            )
            */
            if(outputABE){
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));

                auto layoutDstY = layoutY.GetTileLayout(TensorCoord(m_actual));
                auto layoutOutInUb = layoutY.GetTileLayout(TensorCoord(m_actual));
            
                vecCopyUbToGm(gmY[mLoopOffset], UbYTensorList[UbOutListId], layoutDstY, layoutOutInUb);
                // AscendC::PipeBarrier<PIPE_ALL>();

                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListId]));
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListId]));
            }
            
            tileCompare(
                UbZTensorList[UbOutListId],
                UbYTensorList[UbOutListId],
                UbInCTensorList[UbOutListId],
                UbThreTensorList[UbOutListId],
                layoutCompareInUb, 
                layoutTileCompare, 
                (ElementY)0.002f
            );

            AscendC::PipeBarrier<PIPE_V>();

            if(outputThre){
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutZEventList[UbOutListId]));
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutZEventList[UbOutListId]));

                auto layoutDstYThre = layoutThre.GetTileLayout(TensorCoord(m_actual));
                auto layoutComputeThreInUb = layoutThre.GetTileLayout(TensorCoord(m_actual));
                vecCopyUbToGmforThre(gmThreZ[mLoopOffset], 
                    UbThreTensorList[UbOutListId], 
                    layoutDstYThre, 
                    layoutComputeThreInUb);
            }

            auto layoutDstZ = layoutZ.GetTileLayout(TensorCoord(out_z_actual));
            auto layoutComputeZInUb = layoutZ.GetTileLayout(TensorCoord(out_z_actual));
            vecCopyUbToGmZ(gmCOMPZ[mLoopOffset_for_z], UbZTensorList[UbOutListId], layoutDstZ, layoutComputeZInUb);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutZEventList[UbOutListId]));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInCEventList[UbOutListId]));

            UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;

        }
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> UbATensorList[STAGES]; // A为矩阵A
    AscendC::LocalTensor<ElementX> UbXTensorList[STAGES]; // X为向量BE
    AscendC::LocalTensor<ElementY> UbYTensorList[STAGES]; // Y为ABE向量

    AscendC::LocalTensor<ElementAccumulator> UbWTensorList[STAGES]; // W为计算ABE的工作空间

    AscendC::LocalTensor<ElementY> UbInCTensorList[STAGES]; // InC为CE结果向量
    AscendC::LocalTensor<ElementY> UbWforThreTensorList[STAGES]; // WforThre为计算阈值向量的工作空间
    AscendC::LocalTensor<ElementY> UbThreTensorList[STAGES]; // Thre为阈值向量
    AscendC::LocalTensor<ElementZ> UbZTensorList[STAGES]; // Z 为比较结果的向量
    AscendC::LocalTensor<ElementC> UbCTensorList[STAGES]; // C 为C矩阵，用来求阈值



    // Multi-stage event id list
    int32_t UbInAEventList[STAGES];
    int32_t UbInXEventList[STAGES];
    int32_t UbInCEventList[STAGES];
    int32_t UbOutEventList[STAGES];
    int32_t UbOutZEventList[STAGES];

    // The id of current stage
    uint32_t UbOutListId{0};
    uint32_t UbInListId{0};

    uint32_t m_actual, n_actual, x_actual, y_actual;
    uint32_t m_actual_total, n_actual_total, x_actual_total, y_actual_total;
    uint32_t m_actual_part;

    uint32_t thre_n_actual_total;
    uint32_t thre_n_actual;

    uint32_t dst_offset_ratio;
    uint32_t out_z_actual_total, out_z_actual;

    uint32_t TileMRound, TileNRound;
    uint32_t ThreTileMRound, ThreTileNRound;

    uint32_t BlockMRound, BlockNRound;
    uint32_t ThreBlockMRound, ThreBlockNRound;

    uint32_t TaskSplit;
    uint32_t MatrixOffset;
    uint32_t strideARow, strideACol;
    uint32_t strideCRow, strideCCol;
    uint32_t strideOut;

    TileVmad tileVmad;
    
    MatrixCopyGmToUb matrixCopyGmToUb;
    VecCopyGmToUb vecCopyGmToUb;
    VecCopyUbToGm vecCopyUbToGm;

    MatrixCopyGmToUbforThre matrixCopyGmToUbforThre; // 用来来数据做阈值计算
    VecCopyGmToUbInY vecCopyGmToUbInY; // 用来拉CE数据
    VecCopyUbToGmZ vecCopyUbToGmZ; // 用来输出比较结果

    VecCopyUbToGmforThre vecCopyUbToGmforThre; // 用来输出阈值结果

    TileThreCalc tileThreCalc;
    TileVmuls tileVmuls;

    // Tile Compare
    TileCompare tileCompare;


};

} // namespace Catlass::Gemv::Block

#endif // CATLASS_GEMV_BLOCK_BLOCK_GEMV_AIV_HPP
