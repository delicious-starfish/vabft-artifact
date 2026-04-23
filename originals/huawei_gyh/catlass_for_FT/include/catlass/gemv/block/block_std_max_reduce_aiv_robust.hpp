/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMV_BLOCK_BLOCK_STD_MEAN_MAX_AIV_HPP_ROBUST
#define CATLASS_GEMV_BLOCK_BLOCK_STD_MEAN_MAX_AIV_HPP_ROBUST

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemv_coord.hpp"
#include "catlass/gemv/helper.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemv/tile/tile_reduce_mean_var.hpp"

namespace Catlass::Gemv::Block {

template <
    class UBTileShape_,
    class AType_,
    class XType_,
    class YType_,
    class BiasType_,
    class TileCopy_,
    class TileFaultSum_,
    class TileVmuls_
>
struct BlockSumMaxNoSplitK <
    Gemm::GemvAtlasA2,
    Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
    UBTileShape_,
    AType_,
    XType_,
    YType_,
    BiasType_,
    TileCopy_,
    TileFaultSum_,
    TileVmuls_
> {
public:
    // Type Aliases
    using DispatchPolicy = Gemm::GemvAtlasA2;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using UBTileShape = UBTileShape_;

    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;

    using ElementX = typename XType_::Element;
    using LayoutX = typename XType_::Layout;
    using ElementY = typename YType_::Element;
    using LayoutY = typename YType_::Layout;
    using TileFaultSum = TileFaultSum_;
    using TileVmuls = TileVmuls_;

    using FT_THRESHOLD_ALGORITHM = Gemv::helper::FT_THRESHOLD_ALGORITHM;
    using FT_REDUCE_TYPE = Gemv::helper::FT_REDUCE_TYPE;

    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST;

    using VecCopyGmToUb = typename TileCopy_::VecCopyGmToUb;
    using VecCopyUbToGm = typename TileCopy_::VecCopyUbToGm;
    using MatrixCopyGmToUb = typename TileCopy_::MatrixCopyGmToUb;
    using MatrixCopyGmToUbSimplingContinue = typename TileCopy_::MatrixCopyGmToUbSimplingContinue;
    using MatrixCopyGmToUbSimplingStrided = typename TileCopy_::MatrixCopyGmToUbSimplingStrided;
    using TileFaultMean = Gemv::Tile::TileFaultSum<ArchTag, FT_REDUCE_TYPE::SUM, AType_, YType_>;
    using TileFaultMax = Gemv::Tile::TileFaultSum<ArchTag, FT_REDUCE_TYPE::MAX, AType_, YType_>;

    using TileReduceMean = Gemv::Tile::TileReduce<
        ArchTag,
        FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
        FT_REDUCE_TYPE::MEAN_SQUARE,
        AType_, YType_, YType_, void>;

    using TileReduceVar = Gemv::Tile::TileReduce<
        ArchTag,
        FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
        FT_REDUCE_TYPE::VAR,
        AType_, YType_, YType_, void>;



    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    using UBAlignHelper = Gemv::helper::UBAlignHelper<ElementA>;
    using TensorCoord = layout::VectorLayout::TensorCoord;
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t Abuf_SIZE_ = 144 * 1024;
    // static constexpr uint32_t Xbuf_SIZE_ = 16 * 1024;
    static constexpr uint32_t Ybuf_SIZE_for_BVar_ = 8 * 1024;
    static constexpr uint32_t Ybuf_SIZE_for_BMean_ = 8 * 1024;
    static constexpr uint32_t workspace_SIZE_ = 32 * 1024;

    static constexpr uint32_t Abuf_SIZE_for_BMean_ = 48 * 1024;
    static constexpr uint32_t Abuf_SIZE_for_BMin_ = 48 * 1024;
    static constexpr uint32_t Abuf_SIZE_for_BMax_ = 48 * 1024;
    

    CATLASS_DEVICE
    BlockSumMaxNoSplitK() {}

    /// Construct
    CATLASS_DEVICE
    BlockSumMaxNoSplitK(Arch::Resource<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbAOffset = UBufAddrStart;
        uint32_t UbAOffsetforBMean = UBufAddrStart;
        uint32_t UbAOffsetforBMax = UBufAddrStart + Abuf_SIZE_for_BMean_;
        uint32_t UbAOffsetforBMin = UBufAddrStart + Abuf_SIZE_for_BMean_ + Abuf_SIZE_for_BMax_;
        
        uint32_t UbYOffsetforBMean = UBufAddrStart + Abuf_SIZE_;
        uint32_t UbYOffsetforBVar = UBufAddrStart + Abuf_SIZE_ + Ybuf_SIZE_for_BMean_;
        
        uint32_t UbWOffset = UBufAddrStart + Abuf_SIZE_ + Ybuf_SIZE_for_BMean_ + Ybuf_SIZE_for_BVar_;
        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            // UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SIZE_ / 2));
            UbBMeanTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffsetforBMean + i * (Abuf_SIZE_for_BMean_/ 2));
            UbBMaxTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffsetforBMax + i * (Abuf_SIZE_for_BMax_ / 2));
            UbBMinTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffsetforBMin + i * (Abuf_SIZE_for_BMin_ / 2));

            // UbYTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset + i * (Ybuf_SIZE_ / 2));
            UbYTensorforBMeanList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffsetforBMean + i * (Ybuf_SIZE_for_BMean_ / 2));
            UbYTensorforBVarList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffsetforBVar + i * (Ybuf_SIZE_for_BVar_ / 2));

            UbWTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbWOffset + i * (workspace_SIZE_ / 2));
            UbWTensorforRawList[i] = UbWTensorList[i].template ReinterpretCast<ElementA>();


            // Assign event ID for each stages
            UbInAEventList[i] = i;
            // UbInXEventList[i] = i + STAGES;
            UbOutEventList[i] = i;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            // AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutEventList[i]);
        }
    }

    /// Construct
    CATLASS_DEVICE
    BlockSumMaxNoSplitK(Arch::ResourceAIV<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbAOffset = UBufAddrStart;
        uint32_t UbAOffsetforBMean = UBufAddrStart;
        uint32_t UbAOffsetforBMax = UBufAddrStart + Abuf_SIZE_for_BMean_;
        uint32_t UbAOffsetforBMin = UBufAddrStart + Abuf_SIZE_for_BMean_ + Abuf_SIZE_for_BMax_;
        
        uint32_t UbYOffsetforBMean = UBufAddrStart + Abuf_SIZE_;
        uint32_t UbYOffsetforBVar = UBufAddrStart + Abuf_SIZE_ + Ybuf_SIZE_for_BMean_;
        
        uint32_t UbWOffset = UBufAddrStart + Abuf_SIZE_ + Ybuf_SIZE_for_BMean_ + Ybuf_SIZE_for_BVar_;
        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            // UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SIZE_ / 2));
            UbBMeanTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffsetforBMean + i * (Abuf_SIZE_for_BMean_/ 2));
            UbBMaxTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffsetforBMax + i * (Abuf_SIZE_for_BMax_ / 2));
            UbBMinTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffsetforBMin + i * (Abuf_SIZE_for_BMin_ / 2));

            // UbYTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset + i * (Ybuf_SIZE_ / 2));
            UbYTensorforBMeanList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffsetforBMean + i * (Ybuf_SIZE_for_BMean_ / 2));
            UbYTensorforBVarList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffsetforBVar + i * (Ybuf_SIZE_for_BVar_ / 2));

            UbWTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbWOffset + i * (workspace_SIZE_ / 2));
            UbWTensorforRawList[i] = UbWTensorList[i].template ReinterpretCast<ElementA>();
            

            // Assign event ID for each stages
            UbInAEventList[i] = i;
            // UbInXEventList[i] = i + STAGES;
            UbOutEventList[i] = i;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            // AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutEventList[i]);
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockSumMaxNoSplitK()
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            // AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(UbOutEventList[i]);
        }
    }

    // AscendC::GlobalTensor<ElementY> const &gmZ,
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementY> const &gmZ, LayoutY const &layoutY,
        GemvCoord const &actualShape,
        uint32_t NRealRound,
        float kn_scale_ratio)
    {

        TileMRound = RoundUp(UBTileShape::M, UBAlignHelper::ALIGN);
        TileNRound = RoundUp(UBTileShape::N, UBAlignHelper::ALIGN);
        // RoundUp(NRealRound, UBAlignHelper::ALIGN);
        strideA = layoutA.stride(1) * TileNRound;
        m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
        n_actual = (actualShape.n() < TileNRound) ? actualShape.n() : TileNRound;

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListId]));

        auto UbYTensor = UbYTensorforBMeanList[UbOutListId];

        AscendC::Duplicate<ElementY>(UbYTensor, (ElementY)0.0, m_actual);    
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventList[UbOutListId]));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventList[UbOutListId]));

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
        auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
        auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
        matrixCopyGmToUb(UbBMeanTensorList[UbInListId], gmA, layoutAInUb, layoutTileA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
        // main loop
        uint32_t Nloop = CeilDiv(actualShape.n(), TileNRound);
        for (uint32_t LoopIdx = 0; LoopIdx < Nloop; LoopIdx++) {
            m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
            n_actual = (LoopIdx == Nloop - 1) ? (actualShape.n() - LoopIdx * TileNRound) : TileNRound;
            y_actual = m_actual;
            x_actual = n_actual;

            uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;
            if (LoopIdx < Nloop - 1) {
                uint32_t LoopIdxNext = LoopIdx + 1;
                uint32_t m_actual_next = m_actual;
                uint32_t n_actual_next =
                    (LoopIdxNext == Nloop - 1) ? (actualShape.n() - LoopIdxNext * TileNRound) : TileNRound;
                uint32_t y_actual_next = m_actual_next;
                uint32_t x_actual_next = n_actual_next;
                // Get L1 tensor for next stage
                auto matrixTensor = UbBMeanTensorList[UbInListIdNext];
                
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                matrixCopyGmToUb(matrixTensor, gmA[LoopIdxNext * strideA], layoutAInUb, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
            auto layoutComputeInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
            auto layoutTileCompute = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
            
            /*
            void operator()(
            AscendC::LocalTensor<ElementY> dstTensor,
            AscendC::LocalTensor<ElementA> srcTensor_m,
            AscendC::LocalTensor<ElementAccumulator> temp,
            LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
            )
            */
            tileFaultMean(UbYTensorforBMeanList[UbOutListId],
                UbBMeanTensorList[UbInListId],
                UbWTensorforRawList[UbInListId],
                layoutComputeInUb,
                layoutTileCompute);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
            UbInListId = UbInListIdNext;
        }
        AscendC::PipeBarrier<PIPE_V>();

        tileVmuls(UbYTensorforBMeanList[UbOutListId], UbYTensorforBMeanList[UbOutListId], (ElementY)kn_scale_ratio, m_actual);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));
        auto layoutDstY = layoutY.GetTileLayout(TensorCoord(y_actual));
        auto layoutComputeInUb = layoutY.GetTileLayout(TensorCoord(y_actual));
        vecCopyUbToGm(gmZ, UbYTensorforBMeanList[UbOutListId], layoutDstY, layoutComputeInUb);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListId]));
        UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
    }

    // AscendC::GlobalTensor<ElementY> const &gmZ,
    CATLASS_DEVICE
    void RowMax(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementY> const &gmZ, LayoutY const &layoutY,
        GemvCoord const &actualShape,
        uint32_t NRealRound,
        float kn_scale_ratio)
    {

        TileMRound = RoundUp(UBTileShape::M, UBAlignHelper::ALIGN);
        TileNRound = NRealRound;
        // RoundUp(NRealRound, UBAlignHelper::ALIGN);
        strideA = layoutA.stride(1) * TileNRound;
        m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
        n_actual = (actualShape.n() < TileNRound) ? actualShape.n() : TileNRound;

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListId]));

        auto UbYTensor = UbYTensorforBMeanList[UbOutListId];

        AscendC::Duplicate<ElementY>(UbYTensor, (ElementY)0.0, m_actual);    
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventList[UbOutListId]));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventList[UbOutListId]));

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
        auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
        auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
        matrixCopyGmToUb(UbBMaxTensorList[UbInListId], gmA, layoutAInUb, layoutTileA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
        // main loop
        uint32_t Nloop = CeilDiv(actualShape.n(), TileNRound);
        for (uint32_t LoopIdx = 0; LoopIdx < Nloop; LoopIdx++) {
            m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
            n_actual = (LoopIdx == Nloop - 1) ? (actualShape.n() - LoopIdx * TileNRound) : TileNRound;
            y_actual = m_actual;
            x_actual = n_actual;

            uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;
            if (LoopIdx < Nloop - 1) {
                uint32_t LoopIdxNext = LoopIdx + 1;
                uint32_t m_actual_next = m_actual;
                uint32_t n_actual_next =
                    (LoopIdxNext == Nloop - 1) ? (actualShape.n() - LoopIdxNext * TileNRound) : TileNRound;
                uint32_t y_actual_next = m_actual_next;
                uint32_t x_actual_next = n_actual_next;
                // Get L1 tensor for next stage
                auto matrixTensor = UbBMaxTensorList[UbInListIdNext];
                
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                matrixCopyGmToUb(matrixTensor, gmA[LoopIdxNext * strideA], layoutAInUb, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
            auto layoutComputeInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
            auto layoutTileCompute = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
            
            /*
            void operator()(
            AscendC::LocalTensor<ElementY> dstTensor,
            AscendC::LocalTensor<ElementA> srcTensor_m,
            AscendC::LocalTensor<ElementAccumulator> temp,
            LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
            )
            */
            tileFaultMax(UbYTensorforBMeanList[UbOutListId],
                UbBMaxTensorList[UbInListId],
                UbWTensorforRawList[UbInListId],
                layoutComputeInUb,
                layoutTileCompute);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
            UbInListId = UbInListIdNext;
        }
        AscendC::PipeBarrier<PIPE_V>();

        // tileVmuls(UbYTensorList[UbOutListId], UbYTensorList[UbOutListId], (ElementY)kn_scale_ratio, m_actual);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));
        auto layoutDstY = layoutY.GetTileLayout(TensorCoord(y_actual));
        auto layoutComputeInUb = layoutY.GetTileLayout(TensorCoord(y_actual));
        vecCopyUbToGm(gmZ, UbYTensorforBMeanList[UbOutListId], layoutDstY, layoutComputeInUb);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListId]));
        UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
    }

    // AscendC::GlobalTensor<ElementY> const &gmZ,
    CATLASS_DEVICE
    void RowMeanAbsSquare(
        AscendC::GlobalTensor<ElementA> const &gmB, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementY> const &gmZMeanAbs,
        AscendC::GlobalTensor<ElementY> const &gmZMeanSquare, 
        LayoutX const &layoutX,
        GemvCoord const &actualShape,
        uint32_t NRealRound,
        float n_scale_ratio)
    {

        TileMRound = RoundUp(UBTileShape::M, UBAlignHelper::ALIGN);
        TileNRound = RoundUp(UBTileShape::N, UBAlignHelper::ALIGN);
        // RoundUp(NRealRound, UBAlignHelper::ALIGN);
        strideA = layoutA.stride(1) * TileNRound;
        m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
        n_actual = (actualShape.n() < TileNRound) ? actualShape.n() : TileNRound;

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListId]));

        auto UbYTensorforAbs = UbYTensorforBMeanList[UbOutListId];
        auto UbYTensorforSquare = UbYTensorforBVarList[UbOutListId];

        AscendC::Duplicate<ElementY>(UbYTensorforAbs, (ElementY)0.0, m_actual);
        AscendC::Duplicate<ElementY>(UbYTensorforSquare, (ElementY)0.0, m_actual);    
        AscendC::PipeBarrier<PIPE_V>();

        // AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventList[UbOutListId]));
        // AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventList[UbOutListId]));

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
        auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
        auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
        matrixCopyGmToUb(UbBMeanTensorList[UbInListId], gmB, layoutAInUb, layoutTileA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
        // main loop
        uint32_t Nloop = CeilDiv(actualShape.n(), TileNRound);
        for (uint32_t LoopIdx = 0; LoopIdx < Nloop; LoopIdx++) {
            m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
            n_actual = (LoopIdx == Nloop - 1) ? (actualShape.n() - LoopIdx * TileNRound) : TileNRound;
            y_actual = m_actual;
            x_actual = n_actual;

            uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;
            if (LoopIdx < Nloop - 1) {
                uint32_t LoopIdxNext = LoopIdx + 1;
                uint32_t m_actual_next = m_actual;
                uint32_t n_actual_next =
                    (LoopIdxNext == Nloop - 1) ? (actualShape.n() - LoopIdxNext * TileNRound) : TileNRound;
                uint32_t y_actual_next = m_actual_next;
                uint32_t x_actual_next = n_actual_next;
                // Get L1 tensor for next stage
                auto matrixTensor = UbBMeanTensorList[UbInListIdNext];
                
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                matrixCopyGmToUb(matrixTensor, gmB[LoopIdxNext * strideA], layoutAInUb, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
            auto layoutComputeInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
            auto layoutTileCompute = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
            
            /*
            CATLASS_DEVICE
            void operator()(
                AscendC::LocalTensor<ElementY> dstTensorMeanAbs,
                AscendC::LocalTensor<ElementY> dstTensorMeanSquare,
                AscendC::LocalTensor<ElementA> srcMeanTensor,
                AscendC::LocalTensor<ElementY> red_workspace,
                LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
                ElementA n_ratio_factor)
            */
            tileReduceMean(UbYTensorforBMeanList[UbOutListId],
                UbYTensorforBVarList[UbOutListId],
                UbBMeanTensorList[UbInListId],
                UbWTensorList[UbInListId],
                layoutComputeInUb,
                layoutTileCompute,
                (ElementA)n_scale_ratio);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
            UbInListId = UbInListIdNext;
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));

        auto layoutDstY = layoutX.GetTileLayout(TensorCoord(y_actual));
        auto layoutComputeInUb = layoutX.GetTileLayout(TensorCoord(y_actual));

        vecCopyUbToGm(gmZMeanAbs, UbYTensorforBMeanList[UbOutListId], layoutDstY, layoutComputeInUb);
        vecCopyUbToGm(gmZMeanSquare, UbYTensorforBVarList[UbOutListId], layoutDstY, layoutComputeInUb);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListId]));
        UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
    }

    // AscendC::GlobalTensor<ElementY> const &gmZ,
    CATLASS_DEVICE
    void RowMeanAbsSquareWithSimpleCon(
        AscendC::GlobalTensor<ElementA> const &gmB, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementY> const &gmZMeanAbs,
        AscendC::GlobalTensor<ElementY> const &gmZMeanSquare, 
        LayoutX const &layoutX,
        GemvCoord const &actualShape,
        uint32_t NRealRound,
        float n_scale_ratio, uint32_t simpling_stride)
    {

        TileMRound = RoundUp(UBTileShape::M, UBAlignHelper::ALIGN);
        TileMRoundSimpling = TileMRound;

        TileNRound = RoundUp(UBTileShape::N, UBAlignHelper::ALIGN);
        TileNRoundSimpling = (simpling_stride < 2) ? TileNRound : (TileNRound / simpling_stride);
        TileNRoundSimpling = RoundUp(TileNRoundSimpling, UBAlignHelper::ALIGN);

        uint32_t simpling_stride_round = (simpling_stride < 2) ? 1 : simpling_stride;
        // TileNRound / TileNRoundSimpling;
        TileNRound = TileNRoundSimpling * simpling_stride_round;

        // RoundUp(NRealRound, UBAlignHelper::ALIGN);
        strideA = layoutA.stride(1) * TileNRound;
        m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
        n_actual = (actualShape.n() < TileNRound) ? actualShape.n() : TileNRound;

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListId]));

        auto UbYTensorforAbs = UbYTensorforBMeanList[UbOutListId];
        auto UbYTensorforSquare = UbYTensorforBVarList[UbOutListId];

        AscendC::Duplicate<ElementY>(UbYTensorforAbs, (ElementY)0.0, m_actual);
        AscendC::Duplicate<ElementY>(UbYTensorforSquare, (ElementY)0.0, m_actual);    
        AscendC::PipeBarrier<PIPE_V>();

        // AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventList[UbOutListId]));
        // AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventList[UbOutListId]));

        /*
        CATLASS_DEVICE
        void operator()(
            AscendC::LocalTensor<Element> dstTensor,
            AscendC::GlobalTensor<Element> srcTensor,
            LayoutDst const &layoutDst, 
            LayoutSrc const &layoutSrc, uint32_t simpling_stride)
        */

        bool using_simpling = (n_actual > TileNRoundSimpling) ? true : false;
        auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRoundSimpling));
        auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
        
        if(using_simpling && (simpling_stride_round > 1)){
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
            matrixCopyGmToUbSimplingC(UbBMeanTensorList[UbInListId], gmB, 
                layoutAInUb, layoutTileA, simpling_stride_round);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
        }else{
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
            matrixCopyGmToUb(UbBMeanTensorList[UbInListId], gmB, layoutAInUb, layoutTileA);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
        }
        
        // main loop
        uint32_t Nloop = CeilDiv(actualShape.n(), TileNRound);
        for (uint32_t LoopIdx = 0; LoopIdx < Nloop; LoopIdx++) {
            m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
            n_actual = (LoopIdx == Nloop - 1) ? (actualShape.n() - LoopIdx * TileNRound) : TileNRound;
            n_actual_local = (n_actual > TileNRoundSimpling) ? (n_actual / simpling_stride_round) : n_actual;

            y_actual = m_actual;
            x_actual = n_actual;

            uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;
            if (LoopIdx < Nloop - 1) {
                uint32_t LoopIdxNext = LoopIdx + 1;
                uint32_t m_actual_next = m_actual;
                uint32_t n_actual_next =
                    (LoopIdxNext == Nloop - 1) ? (actualShape.n() - LoopIdxNext * TileNRound) : TileNRound;

                bool using_simpling = (n_actual_next > TileNRoundSimpling) ? true : false;

                uint32_t y_actual_next = m_actual_next;
                uint32_t x_actual_next = n_actual_next;

                auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRoundSimpling));
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));

                // Get L1 tensor for next stage
                auto matrixTensor = UbBMeanTensorList[UbInListIdNext];
                
                if(using_simpling && (simpling_stride_round > 1)){
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                    matrixCopyGmToUbSimplingC(matrixTensor, gmB[LoopIdxNext * strideA], 
                        layoutAInUb, layoutTileA, simpling_stride_round);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
                }else{
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                    matrixCopyGmToUb(matrixTensor, gmB[LoopIdxNext * strideA], 
                        layoutAInUb, layoutTileA);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
                }
                
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
            auto layoutComputeInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRoundSimpling));
            auto layoutTileCompute = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual_local));
            
            /*
            CATLASS_DEVICE
            void operator()(
                AscendC::LocalTensor<ElementY> dstTensorMeanAbs,
                AscendC::LocalTensor<ElementY> dstTensorMeanSquare,
                AscendC::LocalTensor<ElementA> srcMeanTensor,
                AscendC::LocalTensor<ElementY> red_workspace,
                LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
                ElementA n_ratio_factor)
            */
            tileReduceMean(UbYTensorforBMeanList[UbOutListId],
                UbYTensorforBVarList[UbOutListId],
                UbBMeanTensorList[UbInListId],
                UbWTensorList[UbInListId],
                layoutComputeInUb,
                layoutTileCompute,
                (ElementA)n_scale_ratio);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
            UbInListId = UbInListIdNext;
        }

        AscendC::PipeBarrier<PIPE_V>();

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Muls(const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& srcLocal, 
            const T& scalarValue, 
            const int32_t& calCount)
        */

        AscendC::Muls(UbYTensorforBMeanList[UbOutListId], 
            UbYTensorforBMeanList[UbOutListId], 
            (ElementY)(1.0f*simpling_stride_round), y_actual);

        AscendC::Muls(UbYTensorforBVarList[UbOutListId],
            UbYTensorforBVarList[UbOutListId], 
            (ElementY)(1.0f*simpling_stride_round), y_actual);

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));

        auto layoutDstY = layoutX.GetTileLayout(TensorCoord(y_actual));
        auto layoutComputeInUb = layoutX.GetTileLayout(TensorCoord(y_actual));

        vecCopyUbToGm(gmZMeanAbs, UbYTensorforBMeanList[UbOutListId], layoutDstY, layoutComputeInUb);
        vecCopyUbToGm(gmZMeanSquare, UbYTensorforBVarList[UbOutListId], layoutDstY, layoutComputeInUb);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListId]));
        UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
    }

    // AscendC::GlobalTensor<ElementY> const &gmZ,
    CATLASS_DEVICE
    void RowMeanAbsSquareWithSimpleStride(
        AscendC::GlobalTensor<ElementA> const &gmB, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementY> const &gmZMeanAbs,
        AscendC::GlobalTensor<ElementY> const &gmZMeanSquare, 
        LayoutX const &layoutX,
        GemvCoord const &actualShape,
        uint32_t NRealRound,
        float n_scale_ratio, 
        uint32_t simpling_stride, uint32_t stride_unit)
    {

        /*
        CATLASS_DEVICE
        void operator()(
            AscendC::LocalTensor<Element> dstTensor,
            AscendC::GlobalTensor<Element> srcTensor,
            LayoutDst const &layoutDst, 
            LayoutSrc const &layoutSrc, uint32_t simpling_stride, uint32_t stride_unit)
        */

        TileMRound = RoundUp(UBTileShape::M, UBAlignHelper::ALIGN);
        TileMRoundSimpling = TileMRound;

        TileNRound = RoundUp(UBTileShape::N, UBAlignHelper::ALIGN);
        uint32_t stride_unit_aligned = RoundUp(stride_unit, UBAlignHelper::ALIGN);
        uint32_t stride_chunk_aligned = stride_unit_aligned * simpling_stride;
        uint32_t stride_chunk_num_raw = TileNRound / stride_chunk_aligned;
        uint32_t stride_chunk_num = stride_chunk_num_raw;
        
        if(stride_chunk_num_raw < 1){
            stride_unit_aligned = TileNRound / simpling_stride;
            stride_unit_aligned = RoundUp(stride_unit_aligned, UBAlignHelper::ALIGN);
            stride_chunk_num = 1;
        }

        TileNRound = stride_chunk_num * stride_unit_aligned * simpling_stride;
        TileNRoundSimpling = (simpling_stride < 2) ? TileNRound : stride_chunk_num * stride_unit_aligned;

        uint32_t simpling_stride_round = (simpling_stride < 2) ? 1 : simpling_stride;

        // RoundUp(NRealRound, UBAlignHelper::ALIGN);
        strideA = layoutA.stride(1) * TileNRound;
        m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
        n_actual = (actualShape.n() < TileNRound) ? actualShape.n() : TileNRound;

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListId]));

        auto UbYTensorforAbs = UbYTensorforBMeanList[UbOutListId];
        auto UbYTensorforSquare = UbYTensorforBVarList[UbOutListId];

        AscendC::Duplicate<ElementY>(UbYTensorforAbs, (ElementY)0.0, m_actual);
        AscendC::Duplicate<ElementY>(UbYTensorforSquare, (ElementY)0.0, m_actual);    
        AscendC::PipeBarrier<PIPE_V>();

        // AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventList[UbOutListId]));
        // AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventList[UbOutListId]));

        /*
        CATLASS_DEVICE
        void operator()(
            AscendC::LocalTensor<Element> dstTensor,
            AscendC::GlobalTensor<Element> srcTensor,
            LayoutDst const &layoutDst, 
            LayoutSrc const &layoutSrc, uint32_t simpling_stride)
        */

        bool using_simpling = (n_actual > TileNRoundSimpling) ? true : false;
        auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRoundSimpling));
        auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
        
        if(using_simpling && (simpling_stride_round > 1)){
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
            matrixCopyGmToUbSimplingS(UbBMeanTensorList[UbInListId], gmB, 
                layoutAInUb, layoutTileA, 
                simpling_stride_round, stride_unit_aligned);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
        }else{
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
            matrixCopyGmToUb(UbBMeanTensorList[UbInListId], gmB, layoutAInUb, layoutTileA);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
        }
        
        // main loop
        uint32_t Nloop = CeilDiv(actualShape.n(), TileNRound);
        for (uint32_t LoopIdx = 0; LoopIdx < Nloop; LoopIdx++) {
            m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
            n_actual = (LoopIdx == Nloop - 1) ? (actualShape.n() - LoopIdx * TileNRound) : TileNRound;

            if(LoopIdx < (Nloop - 1)){
                n_actual_local = TileNRoundSimpling;
            }else if(n_actual <= TileNRoundSimpling){
                n_actual_local = n_actual;
            }else{
                uint32_t last_chunk_num = n_actual / (stride_unit_aligned * simpling_stride);
                uint32_t last_remain_actual = n_actual % (stride_unit_aligned * simpling_stride);

                n_actual_local = last_chunk_num * stride_unit_aligned;
                uint32_t n_actual_tile = (last_remain_actual < stride_unit_aligned) ? last_remain_actual : stride_unit_aligned;
                
                n_actual_local = n_actual_local + n_actual_tile;
            }

            y_actual = m_actual;
            x_actual = n_actual;

            uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;
            if (LoopIdx < Nloop - 1) {
                uint32_t LoopIdxNext = LoopIdx + 1;
                uint32_t m_actual_next = m_actual;
                uint32_t n_actual_next =
                    (LoopIdxNext == Nloop - 1) ? (actualShape.n() - LoopIdxNext * TileNRound) : TileNRound;

                bool using_simpling = (n_actual_next > TileNRoundSimpling) ? true : false;

                uint32_t y_actual_next = m_actual_next;
                uint32_t x_actual_next = n_actual_next;

                auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRoundSimpling));
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));

                // Get L1 tensor for next stage
                auto matrixTensor = UbBMeanTensorList[UbInListIdNext];
                
                if(using_simpling && (simpling_stride_round > 1)){
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                    matrixCopyGmToUbSimplingS(matrixTensor, gmB[LoopIdxNext * strideA], 
                        layoutAInUb, layoutTileA, 
                        simpling_stride_round, stride_unit_aligned);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
                }else{
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                    matrixCopyGmToUb(matrixTensor, gmB[LoopIdxNext * strideA], 
                        layoutAInUb, layoutTileA);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
                }
                
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
            auto layoutComputeInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRoundSimpling));
            auto layoutTileCompute = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual_local));
            
            /*
            CATLASS_DEVICE
            void operator()(
                AscendC::LocalTensor<ElementY> dstTensorMeanAbs,
                AscendC::LocalTensor<ElementY> dstTensorMeanSquare,
                AscendC::LocalTensor<ElementA> srcMeanTensor,
                AscendC::LocalTensor<ElementY> red_workspace,
                LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
                ElementA n_ratio_factor)
            */
            tileReduceMean(UbYTensorforBMeanList[UbOutListId],
                UbYTensorforBVarList[UbOutListId],
                UbBMeanTensorList[UbInListId],
                UbWTensorList[UbInListId],
                layoutComputeInUb,
                layoutTileCompute,
                (ElementA)n_scale_ratio);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
            UbInListId = UbInListIdNext;
        }

        AscendC::PipeBarrier<PIPE_V>();

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Muls(const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& srcLocal, 
            const T& scalarValue, 
            const int32_t& calCount)
        */

        AscendC::Muls(UbYTensorforBMeanList[UbOutListId], 
            UbYTensorforBMeanList[UbOutListId], 
            (ElementY)(1.0f*simpling_stride_round), y_actual);

        AscendC::Muls(UbYTensorforBVarList[UbOutListId],
            UbYTensorforBVarList[UbOutListId], 
            (ElementY)(1.0f*simpling_stride_round), y_actual);

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));

        auto layoutDstY = layoutX.GetTileLayout(TensorCoord(y_actual));
        auto layoutComputeInUb = layoutX.GetTileLayout(TensorCoord(y_actual));

        vecCopyUbToGm(gmZMeanAbs, UbYTensorforBMeanList[UbOutListId], layoutDstY, layoutComputeInUb);
        vecCopyUbToGm(gmZMeanSquare, UbYTensorforBVarList[UbOutListId], layoutDstY, layoutComputeInUb);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListId]));
        UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
    }

    // AscendC::GlobalTensor<ElementY> const &gmZ,
    CATLASS_DEVICE
    void RowVariance(
        AscendC::GlobalTensor<ElementA> const &gmBMean,
        AscendC::GlobalTensor<ElementA> const &gmBMax,
        AscendC::GlobalTensor<ElementA> const &gmBMin, 
        LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementY> const &gmZVar, 
        LayoutX const &layoutX,
        GemvCoord const &actualShape,
        uint32_t NRealRound,
        float n_scale_ratio)
    {

        TileMRound = RoundUp(UBTileShape::M, UBAlignHelper::ALIGN);
        TileNRound = RoundUp(UBTileShape::N, UBAlignHelper::ALIGN);
        // RoundUp(NRealRound, UBAlignHelper::ALIGN);
        strideA = layoutA.stride(1) * TileNRound;
        m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
        n_actual = (actualShape.n() < TileNRound) ? actualShape.n() : TileNRound;

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListId]));

        auto UbYTensorforVar = UbYTensorforBVarList[UbOutListId];

        AscendC::Duplicate<ElementY>(UbYTensorforVar, (ElementY)0.0, m_actual);    
        AscendC::PipeBarrier<PIPE_V>();

        // AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventList[UbOutListId]));
        // AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventList[UbOutListId]));

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
        auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
        auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));

        matrixCopyGmToUb(UbBMeanTensorList[UbInListId], gmBMean, layoutAInUb, layoutTileA);
        matrixCopyGmToUb(UbBMaxTensorList[UbInListId], gmBMax, layoutAInUb, layoutTileA);
        matrixCopyGmToUb(UbBMinTensorList[UbInListId], gmBMin, layoutAInUb, layoutTileA);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
        // main loop
        uint32_t Nloop = CeilDiv(actualShape.n(), TileNRound);
        for (uint32_t LoopIdx = 0; LoopIdx < Nloop; LoopIdx++) {
            m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
            n_actual = (LoopIdx == Nloop - 1) ? (actualShape.n() - LoopIdx * TileNRound) : TileNRound;
            y_actual = m_actual;
            x_actual = n_actual;

            uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;
            if (LoopIdx < Nloop - 1) {
                uint32_t LoopIdxNext = LoopIdx + 1;
                uint32_t m_actual_next = m_actual;
                uint32_t n_actual_next =
                    (LoopIdxNext == Nloop - 1) ? (actualShape.n() - LoopIdxNext * TileNRound) : TileNRound;
                uint32_t y_actual_next = m_actual_next;
                uint32_t x_actual_next = n_actual_next;
                // Get L1 tensor for next stage
                auto matrixTensorforMean = UbBMeanTensorList[UbInListIdNext];
                auto matrixTensorforMax = UbBMaxTensorList[UbInListIdNext];
                auto matrixTensorforMin = UbBMinTensorList[UbInListIdNext];
                
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                matrixCopyGmToUb(matrixTensorforMean, gmBMean[LoopIdxNext * strideA], layoutAInUb, layoutTileA);
                matrixCopyGmToUb(matrixTensorforMax, gmBMax[LoopIdxNext * strideA], layoutAInUb, layoutTileA);
                matrixCopyGmToUb(matrixTensorforMin, gmBMin[LoopIdxNext * strideA], layoutAInUb, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
            auto layoutComputeInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
            auto layoutTileCompute = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
            
            /*
            CATLASS_DEVICE
            void operator()(
                AscendC::LocalTensor<ElementY> dstTensorVar,
                AscendC::LocalTensor<ElementA> srcMeanTensor,
                AscendC::LocalTensor<ElementA> srcMaxTensor,
                AscendC::LocalTensor<ElementA> srcMinTensor,
                AscendC::LocalTensor<ElementY> red_workspace,
                LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
                ElementA n_ratio_factor)
            */
            tileReduceVar(UbYTensorforBVarList[UbOutListId],
                UbBMeanTensorList[UbInListId],
                UbBMaxTensorList[UbInListId],
                UbBMinTensorList[UbInListId],
                UbWTensorList[UbInListId],
                layoutComputeInUb,
                layoutTileCompute,
                (ElementA)n_scale_ratio);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
            UbInListId = UbInListIdNext;
        }

        AscendC::PipeBarrier<PIPE_V>();
        /*
        template <typename T>
        __aicore__ inline void Sqrt(const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& srcLocal, const int32_t& calCount)
        */
        AscendC::Abs(UbYTensorforBVarList[UbOutListId], UbYTensorforBVarList[UbOutListId], m_actual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sqrt(UbYTensorforBVarList[UbOutListId], UbYTensorforBVarList[UbOutListId], m_actual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));

        auto layoutDstY = layoutX.GetTileLayout(TensorCoord(y_actual));
        auto layoutComputeInUb = layoutX.GetTileLayout(TensorCoord(y_actual));

        vecCopyUbToGm(gmZVar, UbYTensorforBVarList[UbOutListId], layoutDstY, layoutComputeInUb);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListId]));
        UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
    }

protected:
    // Multi-stage tensors list
    // AscendC::LocalTensor<ElementA> UbATensorList[STAGES];

    AscendC::LocalTensor<ElementA> UbBMeanTensorList[STAGES];
    AscendC::LocalTensor<ElementA> UbBMaxTensorList[STAGES];
    AscendC::LocalTensor<ElementA> UbBMinTensorList[STAGES];

    AscendC::LocalTensor<ElementY> UbYTensorforBVarList[STAGES];
    AscendC::LocalTensor<ElementY> UbYTensorforBMeanList[STAGES];

    // AscendC::LocalTensor<ElementX> UbXTensorList[STAGES];
    // AscendC::LocalTensor<ElementY> UbYTensorList[STAGES];
    AscendC::LocalTensor<ElementY> UbWTensorList[STAGES];
    AscendC::LocalTensor<ElementA> UbWTensorforRawList[STAGES];

    // Multi-stage event id list
    int32_t UbInAEventList[STAGES];
    // int32_t UbInXEventList[STAGES];
    int32_t UbOutEventList[STAGES];

    // The id of current stage
    uint32_t UbOutListId{0};
    uint32_t UbInListId{0};

    uint32_t m_actual, n_actual, x_actual, y_actual, n_actual_local;
    uint32_t TileMRound, TileNRound;
    uint32_t TileMRoundSimpling, TileNRoundSimpling;
    uint32_t strideA;

    TileFaultMean tileFaultMean;
    TileFaultMax tileFaultMax;
    TileVmuls tileVmuls;
    MatrixCopyGmToUb matrixCopyGmToUb;
    VecCopyGmToUb vecCopyGmToUb;
    VecCopyUbToGm vecCopyUbToGm;

    MatrixCopyGmToUbSimplingContinue matrixCopyGmToUbSimplingC;
    MatrixCopyGmToUbSimplingStrided matrixCopyGmToUbSimplingS;

    TileReduceMean tileReduceMean;
    TileReduceVar tileReduceVar;
};

} // namespace Catlass::Gemv::Block

#endif // CATLASS_GEMV_BLOCK_BLOCK_GEMV_AIV_HPP
