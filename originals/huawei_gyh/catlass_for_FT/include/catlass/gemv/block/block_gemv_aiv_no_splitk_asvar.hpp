/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMV_BLOCK_BLOCK_GEMV_ASVAR_AIV_NO_SPLIT_HPP
#define CATLASS_GEMV_BLOCK_BLOCK_GEMV_ASVAR_AIV_NO_SPLIT_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemv_coord.hpp"
#include "catlass/gemv/helper.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemv/tile/tile_vmuls.hpp"

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
    Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR,
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
    class XType_,
    class YType_,
    class ZType_,
    class BiasType_,
    class TileCopy_,
    class TileVmad_,
    class TileThreCalc_,
    class TileVmuls_,
    class TileStdEst_>
struct BlockFTGemvNoSplitK <
    Gemm::GemvAtlasA2,
    Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR,
    Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::ABE_FUSED_THRE,
    Gemv::helper::FT_ENC_TYPE::RCE,
    Gemv::helper::FT_COMP_TYPE::RSUB,
    UBTileShape_,
    UBBlockShape_,
    L1TileShape_,
    AType_,
    XType_,
    YType_,
    ZType_,
    BiasType_,
    TileCopy_,
    TileVmad_,
    TileThreCalc_,
    TileVmuls_,
    TileStdEst_> 
{
public:
    // Type Aliases
    using DispatchPolicy = Gemm::GemvAtlasA2;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using FT_ENC_TYPE = Catlass::Gemv::helper::FT_ENC_TYPE;
    using FT_COMP_TYPE = Catlass::Gemv::helper::FT_COMP_TYPE;
    using FT_REDUCE_TYPE = Catlass::Gemv::helper::FT_REDUCE_TYPE;
    using FT_AIV_PIPE_FUSE_TYPE = Catlass::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE;
    using FT_THRESHOLD_ALGORITHM = Catlass::Gemv::helper::FT_THRESHOLD_ALGORITHM;

    using UBTileShape = UBTileShape_;
    using UBBlockShape = UBBlockShape_;
    using L1TileShape = L1TileShape_;

    using TileFaultMeanMax = Gemv::Tile::TileFaultSum<ArchTag, FT_REDUCE_TYPE::SUM_MAX_ABE, AType_, XType_>;

    using ThreCalcUBTileShape = GemvShape<UBTileShape::M,L1TileShape::N>;
    using ThreCalcUBTileShapeTotal = GemvShape<L1TileShape::M,L1TileShape::N>;

    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;

    using ElementX = typename XType_::Element;
    using LayoutX = typename XType_::Layout;

    using ElementY = typename YType_::Element;
    using LayoutY = typename YType_::Layout;

    using ElementZ = typename ZType_::Element;
    using LayoutZ = typename ZType_::Layout;

    using TileVmad = TileVmad_;

    using TileThreCalc = TileThreCalc_;
    using TileVmuls = TileVmuls_;
    using TileStdEst = TileStdEst_;

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

    /*
    using ThreCalcTileVmuls = Gemv::Tile::TileVmuls<typename ThreCalcDispatchPolicy::ArchTag, ZType>;
    */
    using TileVmulsforMean = Gemv::Tile::TileVmuls<ArchTag, XType_>;

    using TileCompare = Gemv::Tile::TileFaultVcompare<FT_COMP_TYPE::RSUB, ArchTag, 
                                        ZType_, YType_, YType_>;
    
    static constexpr FT_COMP_TYPE COMP_TYPE = FT_COMP_TYPE::RSUB;
    static constexpr FT_ENC_TYPE ENC_TYPE = FT_ENC_TYPE::RCE;
    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = FT_THRESHOLD_ALGORITHM::ASVAR;



    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    using UBAlignHelper = Gemv::helper::UBAlignHelper<ElementA>;
    using TensorCoord = layout::VectorLayout::TensorCoord;

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE = Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::NO_FUSED;
    static constexpr uint32_t Abuf_SIZE_ = 128 * 1024;
    static constexpr uint32_t Xbuf_SIZE_ = 8 * 1024;
    static constexpr uint32_t Ybuf_SIZE_ = 4 * 1024;

    static constexpr uint32_t sum_workspace_SIZE_ = 16 * 1024;
    static constexpr uint32_t max_workspace_SIZE_ = 16 * 1024;
    static constexpr uint32_t workspace_SIZE_ = 32 * 1024;

    static constexpr uint32_t Threbuf_SIZE_ = 4 * 1024;
    static constexpr uint32_t InCbuf_SIZE_ = 4 * 1024;
    static constexpr uint32_t Stdbuf_SIZE_ = 4 * 1024;

    static constexpr uint32_t Meanbuf_SIZE_ = 2 * 1024;
    static constexpr uint32_t Maxbuf_SIZE_ = 2 * 1024;
    static constexpr uint32_t Zbuf_SIZE_ = 2 *1024;

    static constexpr uint32_t ELE_NUM_PER_BLK_FOR_B = BYTE_PER_BLK / sizeof(ElementX);

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
        uint32_t UbStdOffset = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_ + Ybuf_SIZE_ + InCbuf_SIZE_ + Threbuf_SIZE_;
        uint32_t UbMeanOffset = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_ + Ybuf_SIZE_ + InCbuf_SIZE_ + Threbuf_SIZE_ + Stdbuf_SIZE_;
        uint32_t UbMaxOffset = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_ + Ybuf_SIZE_ + InCbuf_SIZE_ + Threbuf_SIZE_ + Stdbuf_SIZE_ + Meanbuf_SIZE_;
        uint32_t UbZOffset = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_ + Ybuf_SIZE_ + InCbuf_SIZE_ + Threbuf_SIZE_ + Stdbuf_SIZE_ + Meanbuf_SIZE_ + Maxbuf_SIZE_;
        

        uint32_t UbWOffset = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_ + Ybuf_SIZE_ + InCbuf_SIZE_ + Threbuf_SIZE_ + Stdbuf_SIZE_ + Meanbuf_SIZE_ + Maxbuf_SIZE_ + Zbuf_SIZE_;
        uint32_t UbWMaxOffset = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_ + Ybuf_SIZE_ + InCbuf_SIZE_ + Threbuf_SIZE_ + Stdbuf_SIZE_ + Meanbuf_SIZE_ + Maxbuf_SIZE_ + Zbuf_SIZE_ + sum_workspace_SIZE_;

        m_actual_single = (max_workspace_SIZE_ < sum_workspace_SIZE_) ? max_workspace_SIZE_ / (BYTE_PER_C0 * 8 * STAGES) : sum_workspace_SIZE_ / (BYTE_PER_C0 * 8 * STAGES);
        
        // Init buffers
        UbInMeanEvent = STAGES * 2;
        UbInMaxEvent = 1 + STAGES * 2;
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SIZE_ / 2));
            UbXTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbXOffset + i * (Xbuf_SIZE_ / 2));
            
            UbYTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset + i * (Ybuf_SIZE_ / 2));

            UbWTensorList[i] =
                resource.ubBuf.template GetBufferByByte<ElementAccumulator>(UbWOffset + i * (workspace_SIZE_ / 2));
            UbWMeanMaxTensorList[i] = UbWTensorList[i].template ReinterpretCast<ElementX>();
            // resource.ubBuf.template GetBufferByByte<ElementX>(UbWOffset + i *(sum_workspace_SIZE_ / 2));
            // UbWforMaxTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbWMaxOffset + i *(max_workspace_SIZE_ / 2));
            UbWforThreTensorList[i] = UbWTensorList[i].template ReinterpretCast<ElementY>();

            UbInCTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbInCOffset + i * (InCbuf_SIZE_ / 2));
            UbThreTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbThreOffset + i * (Threbuf_SIZE_ / 2));
            UbStdTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbStdOffset + i * (Stdbuf_SIZE_ / 2));
            UbMeanTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbMeanOffset + i * (Meanbuf_SIZE_ / 2));
            UbMaxTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbMaxOffset + i * (Maxbuf_SIZE_ / 2));
            
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

        AscendC::SetFlag<AscendC::HardEvent::V_S>(UbInMeanEvent);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(UbInMaxEvent);
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
        uint32_t UbStdOffset = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_ + Ybuf_SIZE_ + InCbuf_SIZE_ + Threbuf_SIZE_;
        uint32_t UbMeanOffset = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_ + Ybuf_SIZE_ + InCbuf_SIZE_ + Threbuf_SIZE_ + Stdbuf_SIZE_;
        uint32_t UbMaxOffset = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_ + Ybuf_SIZE_ + InCbuf_SIZE_ + Threbuf_SIZE_ + Stdbuf_SIZE_ + Meanbuf_SIZE_;
        uint32_t UbZOffset = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_ + Ybuf_SIZE_ + InCbuf_SIZE_ + Threbuf_SIZE_ + Stdbuf_SIZE_ + Meanbuf_SIZE_ + Maxbuf_SIZE_;
        

        uint32_t UbWOffset = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_ + Ybuf_SIZE_ + InCbuf_SIZE_ + Threbuf_SIZE_ + Stdbuf_SIZE_ + Meanbuf_SIZE_ + Maxbuf_SIZE_ + Zbuf_SIZE_;
        uint32_t UbWMaxOffset = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_ + Ybuf_SIZE_ + InCbuf_SIZE_ + Threbuf_SIZE_ + Stdbuf_SIZE_ + Meanbuf_SIZE_ + Maxbuf_SIZE_ + Zbuf_SIZE_ + sum_workspace_SIZE_;

        m_actual_single = (max_workspace_SIZE_ < sum_workspace_SIZE_) ? max_workspace_SIZE_ / (BYTE_PER_C0 * 8 * STAGES) : sum_workspace_SIZE_ / (BYTE_PER_C0 * 8 * STAGES);
        
        // Init buffers
        UbInMeanEvent = STAGES * 2;
        UbInMaxEvent = 1 + STAGES * 2;
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SIZE_ / 2));
            UbXTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbXOffset + i * (Xbuf_SIZE_ / 2));
            
            UbYTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset + i * (Ybuf_SIZE_ / 2));

            UbWTensorList[i] =
                resource.ubBuf.template GetBufferByByte<ElementAccumulator>(UbWOffset + i * (workspace_SIZE_ / 2));
            UbWMeanMaxTensorList[i] = UbWTensorList[i].template ReinterpretCast<ElementX>();
            // resource.ubBuf.template GetBufferByByte<ElementX>(UbWOffset + i *(sum_workspace_SIZE_ / 2));
            // UbWforMaxTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbWMaxOffset + i *(max_workspace_SIZE_ / 2));
            UbWforThreTensorList[i] = UbWTensorList[i].template ReinterpretCast<ElementY>();

            UbInCTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbInCOffset + i * (InCbuf_SIZE_ / 2));
            UbThreTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbThreOffset + i * (Threbuf_SIZE_ / 2));
            UbStdTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbStdOffset + i * (Stdbuf_SIZE_ / 2));
            UbMeanTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbMeanOffset + i * (Meanbuf_SIZE_ / 2));
            UbMaxTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbMaxOffset + i * (Maxbuf_SIZE_ / 2));
            
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

        AscendC::SetFlag<AscendC::HardEvent::V_S>(UbInMeanEvent);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(UbInMaxEvent);
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

        AscendC::WaitFlag<AscendC::HardEvent::V_S>(UbInMeanEvent);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(UbInMaxEvent);
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

    // AscendC::GlobalTensor<ElementX> const &gmBMean,
    // AscendC::GlobalTensor<ElementX> const &gmBMax, LayoutX const &layoutBforFT,
    // AscendC::GlobalTensor<ElementX> const &gmBMean,
        // AscendC::GlobalTensor<ElementX> const &gmBMax, LayoutX const &layoutBforFT,

    // float B_slice_mean, float B_slice_max, 
    // float B_slice_mean_abs,
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementX> const &gmX, LayoutX const &layoutX,
        AscendC::GlobalTensor<ElementY> const &gmY, LayoutY const &layoutY,
        AscendC::GlobalTensor<ElementX> const &gmBMean,
        AscendC::GlobalTensor<ElementX> const &gmBMax, LayoutX const &layoutBforFT,
        AscendC::GlobalTensor<ElementY> const &gmInC, LayoutY const &layoutInC,
        AscendC::GlobalTensor<ElementY> const &gmThreZ, LayoutY const &layoutThre,
        AscendC::GlobalTensor<ElementZ> const &gmCOMPZ, LayoutZ const &layoutZ,
        GemvCoord const &actualShape, 
        float A_row_scale_ratio,
        float std_est_A_row_ratio, 
        float std_est_B_ratio,
        float kn_ratio_factor, 
        float kn_sqrt_ratio_factor, 
        float k_sqrt_n_ratio_factor,
        float e_max,
        bool outputThre, bool outputABE,
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

        m_actual_total = (actualShape.m() < BlockMRound) ? actualShape.m() : BlockMRound;
        n_actual_total = actualShape.n();

        // uint32_t aiv_part_num = 1 * AscendC::GetTaskRation();

        m_actual_part = m_actual_total / aiv_part_num;

        uint32_t M_start_offset = AscendC::GetSubBlockIdx() * m_actual_part;
        uint32_t Z_start_offset = (M_start_offset + 8 - 1) / 8;

        if(AscendC::GetSubBlockIdx() == (aiv_part_num -1)) {
            m_actual_part = m_actual_total - M_start_offset;
        }

        uint32_t Mloop = CeilDiv(m_actual_part, TileMRound);

        uint32_t NloopV = CeilDiv(n_actual_total, TileNRound);
        

        dst_offset_ratio = Mloop;
        out_z_actual_total = (m_actual_part + 8 - 1) / 8;

        // auto BMeanTile = UbWforMeanTensorList[UbOutListId];
        AscendC::WaitFlag<AscendC::HardEvent::V_S>((event_t)(UbInMeanEvent));
        B_slice_mean = gmBMean.GetValue(0);

        AscendC::WaitFlag<AscendC::HardEvent::V_S>((event_t)(UbInMaxEvent));
        B_slice_max = gmBMax.GetValue(0);

        // AscendC::PipeBarrier<PIPE_ALL>();

        B_slice_mean_abs = (B_slice_mean < (ElementX)0.0) ? ((ElementX)0.0 - B_slice_mean) : B_slice_mean; 
        B_slice_std = (B_slice_max - B_slice_mean) * std_est_B_ratio;

        AscendC::SetFlag<AscendC::HardEvent::S_V>((event_t)(UbInMeanEvent));
        AscendC::SetFlag<AscendC::HardEvent::S_V>((event_t)(UbInMaxEvent));

        AscendC::WaitFlag<AscendC::HardEvent::S_V>((event_t)(UbInMaxEvent));
        AscendC::WaitFlag<AscendC::HardEvent::S_V>((event_t)(UbInMeanEvent));

        for(uint32_t mLoopIdx = 0; mLoopIdx < Mloop; mLoopIdx++){

            m_actual = (mLoopIdx < (Mloop - 1)) ? TileMRound : m_actual_part - mLoopIdx * TileMRound;
            n_actual = (n_actual_total < TileNRound) ? n_actual_total : TileNRound;

            m_actual_single = m_actual;

            out_z_actual = (m_actual + 8 - 1) / 8;

            uint32_t mLoopOffset = mLoopIdx * TileMRound + M_start_offset;
            uint32_t mLoopOffset_for_z = (mLoopOffset + 8 - 1) / 8;

            uint32_t A_row_offset = mLoopOffset;
            uint32_t A_col_offset = 0;
            uint32_t A_block_offset = A_row_offset * layoutA.stride(0) + A_col_offset * layoutA.stride(1);
            
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutZEventList[UbOutListId]));
            auto UbYTensor = UbYTensorList[UbOutListId];
            auto UbThreTensor = UbThreTensorList[UbOutListId];
            auto UbMeanTensor = UbMeanTensorList[UbOutListId];
            auto UbMaxTensor = UbMaxTensorList[UbOutListId];
            auto UbStdTensor = UbStdTensorList[UbOutListId];

            AscendC::Duplicate<ElementY>(UbYTensor, (ElementY)0.0, m_actual);
            AscendC::Duplicate<ElementY>(UbThreTensor, (ElementY)0.0, m_actual);
            AscendC::Duplicate<ElementX>(UbMeanTensor, (ElementX)0.0, m_actual);
            AscendC::Duplicate<ElementX>(UbMaxTensor, (ElementX)0.0, m_actual);
            AscendC::Duplicate<ElementY>(UbStdTensor, (ElementY)0.0, m_actual);

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
                }else if ((nLoopIdx == NloopV - 1) && (mLoopIdx < (Mloop -1))){
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

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbInListId]));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
                auto layoutComputeInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileCompute = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));

                /*
                CATLASS_DEVICE
                void operator()(
                    AscendC::LocalTensor<ElementY> dstTensorSum,
                    AscendC::LocalTensor<ElementY> dstTensorMax,
                    AscendC::LocalTensor<ElementA> srcTensor_m,
                    AscendC::LocalTensor<ElementA> sum_workspace,
                    AscendC::LocalTensor<ElementA> max_workspace,
                    LayoutDst const &layoutDst, 
                    LayoutSrc const &layoutSrc, 
                    uint32_t m_reduce_single
                )
                */

                /*
                CATLASS_DEVICE
                void operator()(
                    AscendC::LocalTensor<ElementY> dstTensorSum,
                    AscendC::LocalTensor<ElementY> dstTensorMax,
                    AscendC::LocalTensor<ElementA> srcTensor_m,
                    AscendC::LocalTensor<ElementA> tmp_workspace,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc, 
                    uint32_t m_reduce_single
                )
                */
                // tileFaultMeanMax(
                //     UbMeanTensor,
                //     UbMaxTensor,
                //     UbATensorList[UbInListId],
                //     UbWforMeanTensorList[UbInListId],
                //     UbWforMaxTensorList[UbInListId],
                //     layoutComputeInUb,
                //     layoutTileCompute,
                //     m_actual_single);

                tileFaultMeanMax(
                    UbMeanTensor,
                    UbMaxTensor,
                    UbATensorList[UbInListId],
                    UbWMeanMaxTensorList[UbInListId],
                    layoutComputeInUb,
                    layoutTileCompute,
                    m_actual);

                AscendC::PipeBarrier<PIPE_V>();

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

            AscendC::PipeBarrier<PIPE_V>();

            tileVmulsforMean(UbMeanTensor, UbMeanTensor, (ElementX)A_row_scale_ratio, m_actual);
            AscendC::PipeBarrier<PIPE_V>();
            /*
            CATLASS_DEVICE
            void operator()(
                AscendC::LocalTensor<ElementY> dstTensor,
                AscendC::LocalTensor<ElementX> srcMeanTensor,
                AscendC::LocalTensor<ElementX> srcMaxTensor,
                LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
                ElementY std_scale_factor
            )
            */

            auto layoutStdInUb = layoutThre.GetTileLayout(MakeCoord(m_actual));
            auto layoutTileStd = layoutThre.GetTileLayout(MakeCoord(m_actual));

            tileStdEst(UbStdTensor,UbMeanTensor,UbMaxTensor,layoutStdInUb,layoutTileStd,(ElementY)std_est_A_row_ratio);

            AscendC::PipeBarrier<PIPE_V>();

             /*
                计算阈值即可
            */

            /*
            CATLASS_DEVICE
            void operator()(
                AscendC::LocalTensor<ElementY> dstTensor,
                AscendC::LocalTensor<ElementX> srcMeanTensor,
                AscendC::LocalTensor<ElementY> srcStdTensor,
                AscendC::LocalTensor<ElementY> thre_workspace,
                LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
                ElementY kn_ratio_factor, ElementY kn_sqrt_ratio_factor,
                ElementY k_sqrt_n_ratio_factor, ElementY B_slice_mean_abs,
                ElementY B_slice_std, ElementY e_max)
            */

            auto layoutThreInUb = layoutThre.GetTileLayout(MakeCoord(m_actual));
            auto layoutTileThre = layoutThre.GetTileLayout(MakeCoord(m_actual));

            /*
            float A_row_scale_ratio,
            float std_est_A_row_ratio, 
            float std_est_B_ratio,
            float kn_ratio_factor, 
            float kn_sqrt_ratio_factor, 
            float k_sqrt_n_ratio_factor,
            float e_max,
            */
            tileThreCalc(
                UbThreTensor,
                UbMeanTensor,
                UbStdTensor,
                UbWforThreTensorList[UbOutListId],
                layoutThreInUb,
                layoutTileThre, 
                (ElementY)kn_ratio_factor,
                (ElementY)kn_sqrt_ratio_factor,
                (ElementY)k_sqrt_n_ratio_factor,
                (ElementY)B_slice_mean_abs,
                (ElementY)B_slice_std,
                (ElementY)e_max);

            AscendC::PipeBarrier<PIPE_V>();

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

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInCEventList[UbOutListId]));

            auto layoutCompareInUb = layoutThre.GetTileLayout(MakeCoord(ThreTileMRound));
            auto layoutTileCompare = layoutThre.GetTileLayout(MakeCoord(m_actual));
            
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

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutZEventList[UbOutListId]));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutZEventList[UbOutListId]));

            if(outputThre){
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
        AscendC::SetFlag<AscendC::HardEvent::V_S>((event_t)(UbInMeanEvent));
        AscendC::SetFlag<AscendC::HardEvent::V_S>((event_t)(UbInMaxEvent));
    }

    // AscendC::GlobalTensor<ElementZ> const &gmCOMPZ, LayoutZ const &layoutZ,
    // bool outputThre, bool outputABE,
    // AscendC::GlobalTensor<ElementY> const &gmInC, LayoutY const &layoutInC,
    CATLASS_DEVICE
    void gemv_fused_thre_op(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementX> const &gmX, LayoutX const &layoutX,
        AscendC::GlobalTensor<ElementY> const &gmY, LayoutY const &layoutY,
        AscendC::GlobalTensor<ElementX> const &gmBMean,
        AscendC::GlobalTensor<ElementX> const &gmBMax, LayoutX const &layoutBforFT,
        AscendC::GlobalTensor<ElementY> const &gmThreZ, LayoutY const &layoutThre,
        GemvCoord const &actualShape, 
        float A_row_scale_ratio,
        float std_est_A_row_ratio, 
        float std_est_B_ratio,
        float kn_ratio_factor, 
        float kn_sqrt_ratio_factor, 
        float k_sqrt_n_ratio_factor,
        float e_max,
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

        m_actual_total = (actualShape.m() < BlockMRound) ? actualShape.m() : BlockMRound;
        n_actual_total = actualShape.n();

        // uint32_t aiv_part_num = 1 * AscendC::GetTaskRation();

        m_actual_part = m_actual_total / aiv_part_num;

        uint32_t M_start_offset = AscendC::GetSubBlockIdx() * m_actual_part;
        uint32_t Z_start_offset = (M_start_offset + 8 - 1) / 8;

        if(AscendC::GetSubBlockIdx() == (aiv_part_num -1)) {
            m_actual_part = m_actual_total - M_start_offset;
        }

        uint32_t Mloop = CeilDiv(m_actual_part, TileMRound);

        uint32_t NloopV = CeilDiv(n_actual_total, TileNRound);
        
        // dst_offset_ratio = Mloop;
        // out_z_actual_total = (m_actual_part + 8 - 1) / 8;

        // auto BMeanTile = UbWforMeanTensorList[UbOutListId];
        AscendC::WaitFlag<AscendC::HardEvent::V_S>((event_t)(UbInMeanEvent));
        B_slice_mean = gmBMean.GetValue(0);

        AscendC::WaitFlag<AscendC::HardEvent::V_S>((event_t)(UbInMaxEvent));
        B_slice_max = gmBMax.GetValue(0);

        // AscendC::PipeBarrier<PIPE_ALL>();

        B_slice_mean_abs = (B_slice_mean < (ElementX)0.0) ? ((ElementX)0.0 - B_slice_mean) : B_slice_mean; 
        B_slice_std = (B_slice_max - B_slice_mean) * std_est_B_ratio;

        AscendC::SetFlag<AscendC::HardEvent::S_V>((event_t)(UbInMeanEvent));
        AscendC::SetFlag<AscendC::HardEvent::S_V>((event_t)(UbInMaxEvent));

        AscendC::WaitFlag<AscendC::HardEvent::S_V>((event_t)(UbInMaxEvent));
        AscendC::WaitFlag<AscendC::HardEvent::S_V>((event_t)(UbInMeanEvent));

        for(uint32_t mLoopIdx = 0; mLoopIdx < Mloop; mLoopIdx++){

            m_actual = (mLoopIdx < (Mloop - 1)) ? TileMRound : m_actual_part - mLoopIdx * TileMRound;
            n_actual = (n_actual_total < TileNRound) ? n_actual_total : TileNRound;

            m_actual_single = m_actual;

            // out_z_actual = (m_actual + 8 - 1) / 8;

            uint32_t mLoopOffset = mLoopIdx * TileMRound + M_start_offset;
            uint32_t mLoopOffset_for_z = (mLoopOffset + 8 - 1) / 8;

            uint32_t A_row_offset = mLoopOffset;
            uint32_t A_col_offset = 0;
            uint32_t A_block_offset = A_row_offset * layoutA.stride(0) + A_col_offset * layoutA.stride(1);
            
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListId]));
            // AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutZEventList[UbOutListId]));
            
            auto UbYTensor = UbYTensorList[UbOutListId];
            auto UbThreTensor = UbThreTensorList[UbOutListId];
            auto UbMeanTensor = UbMeanTensorList[UbOutListId];
            auto UbMaxTensor = UbMaxTensorList[UbOutListId];
            auto UbStdTensor = UbStdTensorList[UbOutListId];

            AscendC::Duplicate<ElementY>(UbYTensor, (ElementY)0.0, m_actual);
            AscendC::Duplicate<ElementY>(UbThreTensor, (ElementY)0.0, m_actual);
            AscendC::Duplicate<ElementX>(UbMeanTensor, (ElementX)0.0, m_actual);
            AscendC::Duplicate<ElementX>(UbMaxTensor, (ElementX)0.0, m_actual);
            AscendC::Duplicate<ElementY>(UbStdTensor, (ElementY)0.0, m_actual);

            AscendC::PipeBarrier<PIPE_V>();
            
            if(mLoopIdx == 0){
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbInListId]));
                vecCopyGmToUb(UbXTensorList[UbInListId], gmX, n_actual);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbInListId]));

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
                auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
                matrixCopyGmToUb(UbATensorList[UbInListId], gmA[A_block_offset], layoutAInUb, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));

                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbInListId]));
            }
            
            // // main loop
            // for (uint32_t nLoopIdx = 0; nLoopIdx < NloopV; nLoopIdx++) {
            //     m_actual = (mLoopIdx < (Mloop - 1)) ? TileMRound : m_actual_part - mLoopIdx * TileMRound;
            //     n_actual = (nLoopIdx == NloopV - 1) ? (n_actual_total - nLoopIdx * TileNRound) : TileNRound;
            //     y_actual = m_actual;
            //     x_actual = n_actual;

            //     uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;
            //     if (nLoopIdx < NloopV - 1) {
            //         uint32_t nLoopIdxNext = nLoopIdx + 1;
            //         uint32_t m_actual_next = m_actual;
            //         uint32_t n_actual_next =
            //             (nLoopIdxNext == NloopV - 1) ? (n_actual_total - nLoopIdxNext * TileNRound) : TileNRound;
            //         uint32_t y_actual_next = m_actual_next;
            //         uint32_t x_actual_next = n_actual_next;
            //         // Get L1 tensor for next stage
            //         auto matrixTensor = UbATensorList[UbInListIdNext];
            //         auto vecTensor = UbXTensorList[UbInListIdNext];

            //         AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbInListIdNext]));
            //         vecCopyGmToUb(vecTensor, gmX[nLoopIdxNext * TileNRound], x_actual_next);
            //         AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbInListIdNext]));

            //         AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
            //         auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
            //         auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
            //         matrixCopyGmToUb(matrixTensor, gmA[A_block_offset + nLoopIdxNext * strideACol], layoutAInUb, layoutTileA);
            //         AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
            //     }else if ((nLoopIdx == NloopV - 1) && (mLoopIdx < (Mloop -1))){
            //         uint32_t mLoopIdxNext = mLoopIdx + 1;
            //         uint32_t nLoopIdxNext = 0;
            //         uint32_t m_actual_next = (mLoopIdxNext < (Mloop - 1)) ? TileMRound : m_actual_part - mLoopIdxNext * TileMRound;
            //         uint32_t n_actual_next =
            //             (nLoopIdxNext == NloopV - 1) ? (n_actual_total - nLoopIdxNext * TileNRound) : TileNRound;
            //         uint32_t y_actual_next = m_actual_next;
            //         uint32_t x_actual_next = n_actual_next;
            //         uint32_t A_block_offset_next = A_block_offset + strideARow;

            //         // Get L1 tensor for next stage
            //         auto matrixTensor = UbATensorList[UbInListIdNext];
            //         auto vecTensor = UbXTensorList[UbInListIdNext];

            //         AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbInListIdNext]));
            //         vecCopyGmToUb(vecTensor, gmX[nLoopIdxNext * TileNRound], x_actual_next);
            //         AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbInListIdNext]));

            //         AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
            //         auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
            //         auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
            //         matrixCopyGmToUb(matrixTensor, gmA[A_block_offset_next + nLoopIdxNext * strideACol], layoutAInUb, layoutTileA);
            //         AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
            //     }

            //     AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbInListId]));
            //     AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
            //     auto layoutComputeInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
            //     auto layoutTileCompute = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));

            //     tileFaultMeanMax(
            //         UbMeanTensor,
            //         UbMaxTensor,
            //         UbATensorList[UbInListId],
            //         UbWMeanMaxTensorList[UbInListId],
            //         layoutComputeInUb,
            //         layoutTileCompute,
            //         m_actual);

            //     AscendC::PipeBarrier<PIPE_V>();

            //     auto UbYTensor = UbYTensorList[UbOutListId];
            //     tileVmad(UbYTensor,
            //         UbXTensorList[UbInListId],
            //         UbATensorList[UbInListId],
            //         UbWTensorList[UbInListId],
            //         layoutComputeInUb,
            //         layoutTileCompute);

            //     AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
            //     AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbInListId]));
            //     UbInListId = UbInListIdNext;
            // }

            AscendC::PipeBarrier<PIPE_V>();

            tileVmulsforMean(UbMeanTensor, UbMeanTensor, (ElementX)A_row_scale_ratio, m_actual);
            AscendC::PipeBarrier<PIPE_V>();

            auto layoutStdInUb = layoutThre.GetTileLayout(MakeCoord(m_actual));
            auto layoutTileStd = layoutThre.GetTileLayout(MakeCoord(m_actual));

            tileStdEst(UbStdTensor,UbMeanTensor,UbMaxTensor,layoutStdInUb,layoutTileStd,(ElementY)std_est_A_row_ratio);

            AscendC::PipeBarrier<PIPE_V>();

            //  /*
            //     计算阈值即可
            // */

            auto layoutThreInUb = layoutThre.GetTileLayout(MakeCoord(m_actual));
            auto layoutTileThre = layoutThre.GetTileLayout(MakeCoord(m_actual));

            tileThreCalc(
                UbThreTensor,
                UbMeanTensor,
                UbStdTensor,
                UbWforThreTensorList[UbOutListId],
                layoutThreInUb,
                layoutTileThre, 
                (ElementY)kn_ratio_factor,
                (ElementY)kn_sqrt_ratio_factor,
                (ElementY)k_sqrt_n_ratio_factor,
                (ElementY)B_slice_mean_abs,
                (ElementY)B_slice_std,
                (ElementY)e_max);

            AscendC::PipeBarrier<PIPE_V>();

            // AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));
            // AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));

            auto layoutDstY = layoutY.GetTileLayout(TensorCoord(m_actual));
            auto layoutOutInUb = layoutY.GetTileLayout(TensorCoord(m_actual));
            
            vecCopyUbToGm(gmY[mLoopOffset], UbYTensorList[UbOutListId], layoutDstY, layoutOutInUb);

            auto layoutDstYThre = layoutThre.GetTileLayout(TensorCoord(m_actual));
            auto layoutComputeThreInUb = layoutThre.GetTileLayout(TensorCoord(m_actual));
            vecCopyUbToGmforThre(gmThreZ[mLoopOffset], 
                UbThreTensorList[UbOutListId], 
                layoutDstYThre, 
                layoutComputeThreInUb);

            

            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListId]));
            UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;

        }
        AscendC::SetFlag<AscendC::HardEvent::V_S>((event_t)(UbInMeanEvent));
        AscendC::SetFlag<AscendC::HardEvent::V_S>((event_t)(UbInMaxEvent));
    }

    /*
    float A_row_scale_ratio,
        float std_est_A_row_ratio, 
        float std_est_B_ratio,
        float kn_ratio_factor, 
        float kn_sqrt_ratio_factor, 
        float k_sqrt_n_ratio_factor,
        float e_max,
        bool outputThre, bool outputABE,
    */
    CATLASS_DEVICE
    void comp_op(
        AscendC::GlobalTensor<ElementY> const &gmY, LayoutY const &layoutY,
        AscendC::GlobalTensor<ElementY> const &gmInC, LayoutY const &layoutInC,
        AscendC::GlobalTensor<ElementY> const &gmThreZ, LayoutY const &layoutThre,
        AscendC::GlobalTensor<ElementZ> const &gmCOMPZ, LayoutZ const &layoutZ,
        GemvCoord const &actualShape, 
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

        uint32_t m_actual_total_comp = (actualShape.m() < BlockMRound) ? actualShape.m() : BlockMRound;

        // uint32_t aiv_part_num = 1 * AscendC::GetTaskRation();

        uint32_t m_actual_part_comp = m_actual_total_comp / aiv_part_num;

        uint32_t M_start_offset = AscendC::GetSubBlockIdx() * m_actual_part_comp;
        uint32_t Z_start_offset = (M_start_offset + 8 - 1) / 8;

        if(AscendC::GetSubBlockIdx() == (aiv_part_num -1)) {
            m_actual_part_comp = m_actual_total_comp - M_start_offset;
        }

        uint32_t Mloop = CeilDiv(m_actual_part, TileMRound);

        uint32_t NloopV = CeilDiv(n_actual_total, TileNRound);
        

        out_z_actual_total = (m_actual_part_comp + 8 - 1) / 8;

        uint32_t m_actual_comp = m_actual;

        for(uint32_t mLoopIdx = 0; mLoopIdx < Mloop; mLoopIdx++){

            m_actual_comp = (mLoopIdx < (Mloop - 1)) ? TileMRound : m_actual_part_comp - mLoopIdx * TileMRound;

            out_z_actual = (m_actual_comp + 8 - 1) / 8;

            uint32_t mLoopOffset = mLoopIdx * TileMRound + M_start_offset;
            uint32_t mLoopOffset_for_z = (mLoopOffset + 8 - 1) / 8;
            
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutZEventList[UbZOutListId]));
            auto UbYTensor = UbYTensorList[UbZOutListId];
            auto UbThreTensor = UbThreTensorList[UbZOutListId];

            if(mLoopIdx == 0){
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInCEventList[UbZOutListId]));
                vecCopyGmToUbInY(UbInCTensorList[UbZOutListId], gmInC[mLoopOffset], m_actual_comp);
                vecCopyGmToUbInY(UbYTensor, gmY[mLoopOffset], m_actual_comp);
                vecCopyGmToUbInY(UbThreTensor, gmThreZ[mLoopOffset], m_actual_comp);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInCEventList[UbZOutListId]));
            }
            
            uint32_t UbZOutListIdNext = (UbZOutListId + 1 < STAGES) ? (UbZOutListId + 1) : 0;

            if(mLoopIdx < (Mloop - 1)){
                uint32_t mLoopIdxNext = mLoopIdx + 1;
                uint32_t mLoopOffsetNext = mLoopIdxNext * TileMRound + M_start_offset;

                uint32_t m_actual_next = (mLoopIdxNext < (Mloop - 1)) ? TileMRound : m_actual_part_comp - mLoopIdxNext * TileMRound;

                auto UbYTensorNext = UbYTensorList[UbZOutListIdNext];
                auto UbThreTensorNext = UbThreTensorList[UbZOutListIdNext];

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInCEventList[UbZOutListIdNext]));
                vecCopyGmToUbInY(UbInCTensorList[UbZOutListIdNext], gmInC[mLoopOffsetNext], m_actual_next);
                vecCopyGmToUbInY(UbYTensorNext, gmY[mLoopOffsetNext], m_actual_next);
                vecCopyGmToUbInY(UbThreTensorNext, gmThreZ[mLoopOffsetNext], m_actual_next);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInCEventList[UbZOutListIdNext]));
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInCEventList[UbZOutListId]));

            auto layoutCompareInUb = layoutThre.GetTileLayout(MakeCoord(ThreTileMRound));
            auto layoutTileCompare = layoutThre.GetTileLayout(MakeCoord(m_actual_comp));
            
            tileCompare(
                UbZTensorList[UbZOutListId],
                UbYTensorList[UbZOutListId],
                UbInCTensorList[UbZOutListId],
                UbThreTensorList[UbZOutListId],
                layoutCompareInUb, 
                layoutTileCompare, 
                (ElementY)0.002f
            );

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInCEventList[UbZOutListId]));

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutZEventList[UbZOutListId]));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutZEventList[UbZOutListId]));

            auto layoutDstZ = layoutZ.GetTileLayout(TensorCoord(out_z_actual));
            auto layoutComputeZInUb = layoutZ.GetTileLayout(TensorCoord(out_z_actual));
            vecCopyUbToGmZ(gmCOMPZ[mLoopOffset_for_z], UbZTensorList[UbZOutListId], layoutDstZ, layoutComputeZInUb);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutZEventList[UbZOutListId]));

            UbZOutListId = (UbZOutListId + 1 < STAGES) ? (UbZOutListId + 1) : 0;
        }
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> UbATensorList[STAGES]; // A为矩阵A
    AscendC::LocalTensor<ElementX> UbXTensorList[STAGES]; // X为向量BE
    AscendC::LocalTensor<ElementY> UbYTensorList[STAGES]; // Y为ABE向量

    AscendC::LocalTensor<ElementAccumulator> UbWTensorList[STAGES]; // W为计算ABE的工作空间
    AscendC::LocalTensor<ElementY> UbWforThreTensorList[STAGES]; // WforThre为计算阈值向量的工作空间
    // AscendC::LocalTensor<ElementX> UbWforMeanTensorList[STAGES]; // 用来计算求和的工作空间
    // AscendC::LocalTensor<ElementX> UbWforMaxTensorList[STAGES]; // 用来计算最大值的工作空间

    AscendC::LocalTensor<ElementX> UbWMeanMaxTensorList[STAGES]; 

    AscendC::LocalTensor<ElementY> UbInCTensorList[STAGES]; // InC为CE结果向量
    AscendC::LocalTensor<ElementY> UbThreTensorList[STAGES]; // Thre为阈值向量
    AscendC::LocalTensor<ElementY> UbStdTensorList[STAGES]; // 存储 A 的行std向量
    AscendC::LocalTensor<ElementX> UbMeanTensorList[STAGES]; // 存储 A 的行mean 向量
    AscendC::LocalTensor<ElementX> UbMaxTensorList[STAGES]; //  存储 A 的行max 向量
    AscendC::LocalTensor<ElementZ> UbZTensorList[STAGES]; // Z 为比较结果的向量

    // Multi-stage event id list
    int32_t UbInAEventList[STAGES];
    int32_t UbInXEventList[STAGES];
    int32_t UbInCEventList[STAGES];
    int32_t UbInMaxEvent;
    int32_t UbInMeanEvent;
    int32_t UbOutEventList[STAGES];
    int32_t UbOutZEventList[STAGES];

    // The id of current stage
    uint32_t UbOutListId{0};
    uint32_t UbZOutListId{0};
    uint32_t UbInListId{0};

    uint32_t m_actual_single{32};
    ElementX B_slice_mean, B_slice_max, B_slice_std, B_slice_mean_abs;

    uint32_t m_actual, n_actual, x_actual, y_actual;
    uint32_t m_actual_total, n_actual_total, x_actual_total, y_actual_total;
    uint32_t m_actual_part;

    // uint32_t thre_n_actual_total;
    // uint32_t thre_n_actual;

    uint32_t dst_offset_ratio;
    uint32_t out_z_actual_total, out_z_actual;

    uint32_t TileMRound, TileNRound;
    uint32_t ThreTileMRound, ThreTileNRound;

    uint32_t BlockMRound, BlockNRound;
    uint32_t ThreBlockMRound, ThreBlockNRound;

    uint32_t TaskSplit;
    uint32_t MatrixOffset;
    uint32_t strideARow, strideACol;
    // uint32_t strideCRow, strideCCol;
    uint32_t strideOut;

    TileVmad tileVmad;

    TileFaultMeanMax tileFaultMeanMax;
    
    MatrixCopyGmToUb matrixCopyGmToUb;
    VecCopyGmToUb vecCopyGmToUb;
    VecCopyUbToGm vecCopyUbToGm;

    MatrixCopyGmToUbforThre matrixCopyGmToUbforThre; // 用来来数据做阈值计算
    VecCopyGmToUbInY vecCopyGmToUbInY; // 用来拉CE数据
    VecCopyUbToGmZ vecCopyUbToGmZ; // 用来输出比较结果

    VecCopyUbToGmforThre vecCopyUbToGmforThre; // 用来输出阈值结果

    TileThreCalc tileThreCalc;
    TileVmuls tileVmuls;
    TileVmulsforMean tileVmulsforMean;

    // Tile Compare
    TileCompare tileCompare;
    TileStdEst tileStdEst;
};

} // namespace Catlass::Gemv::Block

#endif // CATLASS_GEMV_BLOCK_BLOCK_GEMV_AIV_HPP
