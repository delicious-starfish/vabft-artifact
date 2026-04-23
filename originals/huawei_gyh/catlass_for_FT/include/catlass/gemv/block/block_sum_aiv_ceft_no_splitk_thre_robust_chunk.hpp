/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMV_BLOCK_BLOCK_GEMV_ASVAR_CE_AIV_NO_SPLIT_HPP_SPEC_ROBUST_CHUNK
#define CATLASS_GEMV_BLOCK_BLOCK_GEMV_ASVAR_CE_AIV_NO_SPLIT_HPP_SPEC_ROBUST_CHUNK

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
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
    class UBTileShapeforC_,
    class UBTileShapeforA_,
    class UBBlockShape_,
    class L1TileShape_,
    class AType_,
    class CType_,
    class XType_,
    class YType_,
    class BiasType_,
    class TileCopy_,
    class TileFaultSumVmad_>
struct BlockFTGemvCENoSplitK <
    Gemm::GemvAtlasA2,
    Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
    Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::THRE_FUSED,
    Gemv::helper::FT_ENC_TYPE::RCE,
    Gemv::helper::FT_COMP_TYPE::RSUB,
    Gemv::helper::FT_ABE_TYPE::CENTRAL_BLOCK_CHUNK_32,
    UBTileShapeforC_,
    UBTileShapeforA_,
    UBBlockShape_,
    L1TileShape_,
    AType_,
    CType_,
    XType_,
    YType_,
    BiasType_,
    TileCopy_,
    TileFaultSumVmad_> 
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
    using FT_ABE_TYPE = Catlass::Gemv::helper::FT_ABE_TYPE;

    using UBTileShapeforC = UBTileShapeforC_;
    using UBTileShapeforA = UBTileShapeforA_;
    using UBBlockShape = UBBlockShape_;
    using L1TileShape = L1TileShape_;

    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;

    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;

    using ElementX = typename XType_::Element;
    using LayoutX = typename XType_::Layout;

    using ElementY = typename YType_::Element;
    using LayoutY = typename YType_::Layout;

    using LayoutVX = layout::VectorLayout;
    using LayoutVY = layout::VectorLayout;

    using TileFaultSumVmad = TileFaultSumVmad_;

    /*
    template <
    class ElementA,
    class ElementX,
    class ElementY>
    struct TileFaultSumVmad<Arch::AtlasA2,
        Gemv::helper::FT_REDUCE_TYPE::SUM_VMAD,
        Gemm::GemmType<ElementA, layout::RowMajor>,
        Gemm::GemmType<ElementX, layout::VectorLayout>,
        Gemm::GemmType<ElementY, layout::VectorLayout>,
        void>
    */

    using TileFaultSumCSum = Gemv::Tile::TileFaultSumVmad<ArchTag, 
        FT_REDUCE_TYPE::SUM_VMAD, CType_, XType_, YType_>;

    using VecCopyGmToUb = typename TileCopy_::VecCopyGmToUb;
    using VecCopyUbToGm = typename TileCopy_::VecCopyUbToGm;
    using MatrixCopyGmToUbforA = typename TileCopy_::MatrixCopyGmToUbforA;
    using MatrixCopyGmToUbforC = typename TileCopy_::MatrixCopyGmToUbforC;
    using VecCopyGmToUbInY = typename TileCopy_::VecCopyGmToUbInY;

    using MatrixCopyUbToGmforA = typename TileCopy_::MatrixCopyUbToGmforA;
    
    static constexpr FT_COMP_TYPE COMP_TYPE = FT_COMP_TYPE::RSUB;
    static constexpr FT_ENC_TYPE ENC_TYPE = FT_ENC_TYPE::RCE;
    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = FT_THRESHOLD_ALGORITHM::ASVAR;
    static constexpr FT_ABE_TYPE ABE_TYPE = FT_ABE_TYPE::CENTRAL_BLOCK_CHUNK_32;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    using UBAlignHelperforA = Gemv::helper::UBAlignHelper<ElementA>;
    using UBAlignHelperforC = Gemv::helper::UBAlignHelper<ElementC>;

    using TensorCoord = layout::VectorLayout::TensorCoord;

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE = Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::NO_FUSED;
    static constexpr uint32_t Cbuf_SIZE_ = 128 * 1024;

    static constexpr uint32_t workspace_SIZE_ = 48 * 1024;

    static constexpr uint32_t Ybuf_SIZE_R1_ = 6 * 1024;
    static constexpr uint32_t Ybuf_SIZE_R2_ = 6 * 1024;
    static constexpr uint32_t Xbuf_SIZE_ = 4 * 1024;

    static constexpr uint32_t ELE_NUM_PER_BLK_FOR_C = BYTE_PER_BLK / sizeof(ElementY);

    static_assert(L1TileShape::M == UBBlockShape::M,
        "The situation where the basic Tile of UB and L1 for MMA differ on the m axes is not supported yet");

    static_assert(L1TileShape::N == UBBlockShape::N,
        "The situation where the basic Tile of UB and L1 for MMA differ on the n axes is not supported yet");
    
    static_assert(UBTileShapeforA::M == UBTileShapeforC::M,
        "The situation where the basic Tile of UB for CR and A Cast differ on the m axes is not supported yet");

    static_assert((UBBlockShape::N % UBTileShapeforC::N) == 0,
        "The situation where the basic Tile of UB and L1 for MMA differ on the n axes is not supported yet");

    static_assert(UBBlockShape::M / UBTileShapeforC::M <= 2,
        "The situation where the basic Tile of UB In Total AICores and L1 for MMA differ on the m axes is not supported yet");

    CATLASS_DEVICE
    BlockFTGemvCENoSplitK() {}

    /// Construct
    CATLASS_DEVICE
    BlockFTGemvCENoSplitK(Arch::Resource<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbCOffset = UBufAddrStart;
        uint32_t UbXOffset = UBufAddrStart + Cbuf_SIZE_;
        uint32_t UbYOffset_R1 = UBufAddrStart + Cbuf_SIZE_ + Xbuf_SIZE_;
        uint32_t UbYOffset_R2 = UBufAddrStart + Cbuf_SIZE_ + Xbuf_SIZE_ + Ybuf_SIZE_R1_;
        uint32_t UbWOffset = UBufAddrStart + Cbuf_SIZE_ + Xbuf_SIZE_ + Ybuf_SIZE_R1_ + Ybuf_SIZE_R2_;

        // Init buffers

        UbOutTotalEvent = 3 * STAGES;

        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbCTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(UbCOffset + i * (Cbuf_SIZE_ / 2));
            UbAOutTensorList[i] = UbCTensorList[i].template ReinterpretCast<ElementY>();
            UbXTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbXOffset + i * (Xbuf_SIZE_ / 2));

            UbYR1TensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset_R1 + i * (Ybuf_SIZE_R1_ / 2));
            UbYR2TensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset_R2 + i * (Ybuf_SIZE_R2_ / 2));

            UbWTensorList[i] =
                resource.ubBuf.template GetBufferByByte<ElementAccumulator>(UbWOffset + i * (workspace_SIZE_ / 2));
            UbATensorList[i] = UbWTensorList[i].template ReinterpretCast<ElementA>();


            // Assign event ID for each stages
            UbInCREventList[i] = i;
            UbInACastEventList[i] = i + STAGES;
            UbInXEventList[i] = i + 2 * STAGES;

            UbOutCEventList[i] = i + 3 * STAGES;
            // UbOutAEventList[i] = i + STAGES;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInCREventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(UbInACastEventList[i]);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutCEventList[i]);
        }

        // AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutTotalEvent);
    }

    /// Construct
    CATLASS_DEVICE
    BlockFTGemvCENoSplitK(Arch::ResourceAIV<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbCOffset = UBufAddrStart;
        uint32_t UbXOffset = UBufAddrStart + Cbuf_SIZE_;
        uint32_t UbYOffset_R1 = UBufAddrStart + Cbuf_SIZE_ + Xbuf_SIZE_;
        uint32_t UbYOffset_R2 = UBufAddrStart + Cbuf_SIZE_ + Xbuf_SIZE_ + Ybuf_SIZE_R1_;
        uint32_t UbWOffset = UBufAddrStart + Cbuf_SIZE_ + Xbuf_SIZE_ + Ybuf_SIZE_R1_ + Ybuf_SIZE_R2_;

        // Init buffers

        UbOutTotalEvent = 3 * STAGES;

        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbCTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(UbCOffset + i * (Cbuf_SIZE_ / 2));
            UbAOutTensorList[i] = UbCTensorList[i].template ReinterpretCast<ElementY>();
            UbXTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbXOffset + i * (Xbuf_SIZE_ / 2));

            UbYR1TensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset_R1 + i * (Ybuf_SIZE_R1_ / 2));
            UbYR2TensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset_R2 + i * (Ybuf_SIZE_R2_ / 2));

            UbWTensorList[i] =
                resource.ubBuf.template GetBufferByByte<ElementAccumulator>(UbWOffset + i * (workspace_SIZE_ / 2));
            UbATensorList[i] = UbWTensorList[i].template ReinterpretCast<ElementA>();


            // Assign event ID for each stages
            UbInCREventList[i] = i;
            UbInACastEventList[i] = i + STAGES;
            UbInXEventList[i] = i + 2 * STAGES;

            UbOutCEventList[i] = i + 3 * STAGES;
            // UbOutAEventList[i] = i + STAGES;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInCREventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(UbInACastEventList[i]);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutCEventList[i]);
        }

        // AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutTotalEvent);
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockFTGemvCENoSplitK()
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInCREventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(UbInACastEventList[i]);
            
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(UbOutCEventList[i]);
        }
        // AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutTotalEvent);
    }
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementC> const &gmC, LayoutC const &layoutC,
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementX> const &gmXR2, LayoutVX const &layoutXforFT,
        AscendC::GlobalTensor<ElementY> const &gmYR1, 
        AscendC::GlobalTensor<ElementY> const &gmYR2, LayoutVY const &layoutY,
        GemvCoord const &actualShapeforC, GemvCoord const &actualShapeforA,
        uint32_t aiv_part_num,
        Catlass::Arch::CrossCoreFlagWithReverse<> & flagAicFinishStore)
    {
        // , AscendC::GlobalTensor<ElementY> const &gmABeOut
        TileMRound = RoundUp(UBTileShapeforC::M, UBAlignHelperforC::ALIGN);

        TileNRoundforC = RoundUp(UBTileShapeforC::N, UBAlignHelperforC::ALIGN);
        TileNRoundforA = RoundUp(UBTileShapeforA::N, UBAlignHelperforA::ALIGN);
        
        BlockMRound = RoundUp(UBBlockShape::M, UBAlignHelperforC::ALIGN);
        BlockNRoundforC = RoundUp(UBBlockShape::N, UBAlignHelperforC::ALIGN);

        strideCCol = layoutC.stride(1) * TileNRoundforC;
        strideCRow = layoutC.stride(0) * TileMRound;

        strideACol = layoutA.stride(1) * TileNRoundforA;
        strideARow = layoutA.stride(0) * TileMRound;

        m_actual_total_for_c = (actualShapeforC.m() < BlockMRound) ? actualShapeforC.m() : BlockMRound;
        n_actual_total_for_c = (actualShapeforC.n() < BlockNRoundforC) ? actualShapeforC.n() : BlockNRoundforC;

        m_actual_total_for_a = (actualShapeforA.m() < BlockMRound) ? actualShapeforA.m() : BlockMRound;
        n_actual_total_for_a = actualShapeforA.n();

        // uint32_t aiv_part_num = 1 * AscendC::GetTaskRation();

        m_actual_part_for_c = m_actual_total_for_c / aiv_part_num;
        m_actual_part_for_a = m_actual_total_for_a / aiv_part_num;

        uint32_t M_start_offset_for_c = AscendC::GetSubBlockIdx() * m_actual_part_for_c;
        uint32_t M_start_offset_for_a = AscendC::GetSubBlockIdx() * m_actual_part_for_a;

        if(AscendC::GetSubBlockIdx() == (aiv_part_num -1)) {
            m_actual_part_for_c = m_actual_total_for_c - M_start_offset_for_c;
            m_actual_part_for_a = m_actual_total_for_a - M_start_offset_for_a;
        }

        uint32_t MloopforC = CeilDiv(m_actual_part_for_c, TileMRound);
        uint32_t MloopforA = CeilDiv(m_actual_part_for_a, TileMRound);

        uint32_t NloopVforA = CeilDiv(n_actual_total_for_a, TileNRoundforA);
        uint32_t NloopVforC = CeilDiv(n_actual_total_for_c, TileNRoundforC);

        m_actual_for_c = (m_actual_part_for_c < TileMRound) ? m_actual_part_for_c : TileMRound;
        n_actual_for_c = (n_actual_total_for_c < TileNRoundforC) ? n_actual_total_for_c : TileNRoundforC;
        // uint32_t mLoopOffset =M_start_offset;

        uint32_t mLoopOffset = M_start_offset_for_c;

        uint32_t C_row_offset = mLoopOffset;
        uint32_t C_col_offset = 0;
        uint32_t C_block_offset = C_row_offset * layoutC.stride(0) + C_col_offset * layoutC.stride(1);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutCEventList[UbCOutListId]));
        auto UbYR1Tensor = UbYR1TensorList[UbCOutListId];
        auto UbYR2Tensor = UbYR2TensorList[UbCOutListId];

        AscendC::Duplicate<ElementY>(UbYR1Tensor, (ElementY)0.0, m_actual_for_c);
        AscendC::Duplicate<ElementY>(UbYR2Tensor, (ElementY)0.0, m_actual_for_c);

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutCEventList[UbCOutListId]));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutCEventList[UbCOutListId]));

        Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInCREventList[UbInCListId]));
        auto layoutCInUb = layoutC.GetTileLayout(MakeCoord(TileMRound, TileNRoundforC));
        auto layoutTileC = layoutC.GetTileLayout(MakeCoord(m_actual_for_c, n_actual_for_c));
        matrixCopyGmToUbforC(UbCTensorList[UbInCListId], gmC[C_block_offset], layoutCInUb, layoutTileC);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInCREventList[UbInCListId]));

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbInCListId]));
        vecCopyGmToUb(UbXTensorList[UbInCListId], gmXR2[(uint32_t)(C_col_offset * layoutC.stride(1))], n_actual_for_c);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbInCListId]));


        // main loop
        for (uint32_t nLoopIdx = 0; nLoopIdx < NloopVforC; nLoopIdx++) {
            m_actual_for_c = (m_actual_part_for_c < TileMRound) ? m_actual_part_for_c : TileMRound;
            n_actual_for_c = (nLoopIdx == NloopVforC - 1) ? (n_actual_total_for_c - nLoopIdx * TileNRoundforC) : TileNRoundforC;
            y_actual_for_c = m_actual_for_c;
            x_actual_for_c = n_actual_for_c;

            uint32_t UbInCListIdNext = (UbInCListId + 1 < STAGES) ? (UbInCListId + 1) : 0;

            if (nLoopIdx < NloopVforC - 1) {
                uint32_t nLoopIdxNext = nLoopIdx + 1;
                uint32_t m_actual_for_c_next = m_actual_for_c;
                uint32_t n_actual_for_c_next =
                    (nLoopIdxNext == NloopVforC - 1) ? (n_actual_total_for_c - nLoopIdxNext * TileNRoundforC) : TileNRoundforC;
                uint32_t y_actual_for_c_next = m_actual_for_c_next;
                uint32_t x_actual_for_c_next = n_actual_for_c_next;
                // Get L1 tensor for next stage
                auto matrixTensor = UbCTensorList[UbInCListIdNext];

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInCREventList[UbInCListIdNext]));
                auto layoutCInUb = layoutC.GetTileLayout(MakeCoord(TileMRound, TileNRoundforC));
                auto layoutTileC = layoutC.GetTileLayout(MakeCoord(m_actual_for_c_next, n_actual_for_c_next));
                matrixCopyGmToUbforC(matrixTensor, gmC[C_block_offset + nLoopIdxNext * strideCCol], layoutCInUb, layoutTileC);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInCREventList[UbInCListIdNext]));

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbInCListIdNext]));
                vecCopyGmToUb(UbXTensorList[UbInCListIdNext], gmXR2[(uint32_t)(nLoopIdxNext * strideCCol)], n_actual_for_c_next);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbInCListIdNext]));
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInCREventList[UbInCListId]));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbInCListId]));

            auto layoutComputeInUb = layoutC.GetTileLayout(MakeCoord(TileMRound, TileNRoundforC));
            auto layoutTileCompute = layoutC.GetTileLayout(MakeCoord(m_actual_for_c, n_actual_for_c));

            /*
            void operator()(
                AscendC::LocalTensor<ElementY> dstTensorVMAD,
                AscendC::LocalTensor<ElementY> dstTensorSUM,
                AscendC::LocalTensor<ElementX> srcTensor_v,
                AscendC::LocalTensor<ElementA> srcTensor_m,
                AscendC::LocalTensor<ElementAccumulator> vmad_workspace,
                LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
            )
            */
            tileFaultSumCSum(UbYR2Tensor,
                UbYR1Tensor,
                UbXTensorList[UbInCListId],
                UbCTensorList[UbInCListId], 
                UbWTensorList[UbInCListId],
                layoutComputeInUb,
                layoutTileCompute);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInCREventList[UbInCListId]));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbInCListId]));
            UbInCListId = UbInCListIdNext;
        }

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutCEventList[UbCOutListId]));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutCEventList[UbCOutListId]));

        auto layoutDstY = layoutY.GetTileLayout(TensorCoord(m_actual_for_c));
        auto layoutOutInUb = layoutY.GetTileLayout(TensorCoord(m_actual_for_c));
            
        vecCopyUbToGm(gmYR1[mLoopOffset], UbYR1TensorList[UbCOutListId], layoutDstY, layoutOutInUb);
        vecCopyUbToGm(gmYR2[mLoopOffset], UbYR2TensorList[UbCOutListId], layoutDstY, layoutOutInUb);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutCEventList[UbCOutListId]));

        UbCOutListId = (UbCOutListId + 1 < STAGES) ? (UbCOutListId + 1) : 0;
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementC> UbCTensorList[STAGES]; // 矩阵C的输入以及需要Cast时矩阵A cast的输出
    AscendC::LocalTensor<ElementY> UbAOutTensorList[STAGES]; // 需要Cast时矩阵A cast的输出，共享C矩阵的输入空间
    AscendC::LocalTensor<ElementA> UbATensorList[STAGES]; // 矩阵A的输入，在需要Cast时共享workspace
    
    AscendC::LocalTensor<ElementY> UbYR1TensorList[STAGES]; // YR1为Ce向量
    AscendC::LocalTensor<ElementY> UbYR2TensorList[STAGES]; // YR2为Cr2向量

    AscendC::LocalTensor<ElementX> UbXTensorList[STAGES]; // X为计算Cr2时输入的r2向量

    AscendC::LocalTensor<ElementY> UbWTensorList[STAGES]; // W为计算Cr2的工作空间

    // Multi-stage event id list
    int32_t UbInCREventList[STAGES]; // 矩阵C输入向量
    int32_t UbInACastEventList[STAGES];
    int32_t UbInXEventList[STAGES]; // X输入向量

    int32_t UbOutCEventList[STAGES];
    int32_t UbOutTotalEvent;

    // int32_t UbInMaxEvent;

    // The id of current stage
    uint32_t UbCOutListId{0};
    uint32_t UbInCListId{0};
    uint32_t UbInAListId{0};

    uint32_t m_actual_for_c, n_actual_for_c, x_actual_for_c, y_actual_for_c;
    uint32_t m_actual_for_a, n_actual_for_a, x_actual_for_a, y_actual_for_a;

    uint32_t m_actual_total_for_c, n_actual_total_for_c, x_actual_total_for_c, y_actual_total_for_c;
    uint32_t m_actual_total_for_a, n_actual_total_for_a, x_actual_total_for_a, y_actual_total_for_a;
    uint32_t m_actual_part_for_c, m_actual_part_for_a;

    // uint32_t thre_n_actual_total;
    // uint32_t thre_n_actual;

    uint32_t TileMRound, TileNRoundforA, TileNRoundforC;

    uint32_t BlockMRound, BlockNRoundforC;

    uint32_t TaskSplit;
    uint32_t MatrixOffset;
    uint32_t strideCRow, strideCCol;
    uint32_t strideARow, strideACol;

    uint32_t strideOut;
    
    MatrixCopyGmToUbforC matrixCopyGmToUbforC;
    MatrixCopyGmToUbforA matrixCopyGmToUbforA;
    VecCopyGmToUb vecCopyGmToUb;
    VecCopyUbToGm vecCopyUbToGm;

    VecCopyGmToUbInY vecCopyGmToUbInY; // 用来拉A聚合数据
    MatrixCopyUbToGmforA matrixCopyUbToGmforA; // 用来存A的Cast数据回GM

    TileFaultSumCSum tileFaultSumCSum;
};

} // namespace Catlass::Gemv::Block

#endif // CATLASS_GEMV_BLOCK_BLOCK_GEMV_AIV_HPP
