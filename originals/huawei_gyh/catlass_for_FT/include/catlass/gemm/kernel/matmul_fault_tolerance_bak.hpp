/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_MATMUL_FAULT_TOLERANCE_HPP
#define CATLASS_GEMM_KERNEL_MATMUL_FAULT_TOLERANCE_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemv_coord.hpp"
#include "catlass/gemv/helper.hpp"

namespace Catlass::Gemm::Kernel {

template <
    class BlockMmad_,
    class BlockSumGemvB_,
    class BlockSumGemvC_,
    class BlockScheduler_
>
class MatmulFaultTolerance {
public:
    using BlockMmad = BlockMmad_;
    using BlockSumGemvB = BlockSumGemvB_;
    using BlockSumGemvC = BlockSumGemvC_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    using ElementX = typename BlockSumGemvB::ElementX;
    using LayoutX = typename BlockSumGemvB::LayoutX;
    using ElementY = typename BlockSumGemvB::ElementY;
    using LayoutY = typename BlockSumGemvB::LayoutY;
    using UBTileShapeB = typename BlockSumGemvB::UBTileShape;
    using UBTileShapeC = typename BlockSumGemvC::UBTileShape;
    using UBAlignHelper = Gemv::helper::UBAlignHelper<ElementA>;

    using ElementD = ElementX;

    using BlockScheduler = BlockScheduler_;

    static_assert(std::is_same_v<typename BlockSumGemvB::ElementA, ElementA> &&
        std::is_same_v<typename BlockSumGemvB::ElementA, ElementA> &&
        std::is_same_v<ElementA, ElementB>,
        "The AType of Gemv and AType and BType of Mmad should be consistent.");

    static_assert(std::is_same_v<ElementX, ElementY> && std::is_same_v<ElementX, ElementD>,
        "The XType and YType of Gemv and DType should be consistent.");

    static_assert(std::is_same_v<LayoutA, LayoutB>,
        "The LayoutA and LayoutB of Gemv should be consistent.");

    enum class AivCore {
        AIV0 = 0,
        AIV1
    };

    enum class CheckSumType {
        NO_CHECKSUM = 0,
        ROW_CHECKSUM
    };

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrD;
        CheckSumType checkSumType;
        GM_ADDR ptrWorkspace;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const &problemShape_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrC_, LayoutC const &layoutC_,
            GM_ADDR ptrD_, int32_t checkSumType_, GM_ADDR ptrWorkspace_
        ) : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_),
            ptrC(ptrC_), layoutC(layoutC_), ptrD(ptrD_), checkSumType(static_cast<CheckSumType>(checkSumType_)), ptrWorkspace(ptrWorkspace_) {}
    };

    struct Arguments {
        GemmCoord problemShape;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
        GM_ADDR ptrD;
        int32_t checkSumType;
        uint32_t blockNum;
    };

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return args.blockNum * Catlass::Arch::MAX_REVERSE_DEPTH * L1TileShape::M * L1TileShape::N * sizeof(ElementC);
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        GemmCoord problemShape = args.problemShape;
        uint32_t m = problemShape.m();
        uint32_t n = problemShape.n();
        uint32_t k = problemShape.k();
        LayoutA layoutA{m, k};
        LayoutB layoutB{k, n};
        LayoutC layoutC{m, n};
        Params params{problemShape, args.ptrA, layoutA, args.ptrB, layoutB, args.ptrC, layoutC, args.ptrD, args.checkSumType, workspace};
        return params;
    }

    // Methods
    CATLASS_DEVICE
    MatmulFaultTolerance() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);
        AscendC::GlobalTensor<ElementC> gmWorkspace;
        gmWorkspace.SetGlobalBuffer((__gm__ ElementC *)params.ptrWorkspace + AscendC::GetBlockIdx() * Catlass::Arch::MAX_REVERSE_DEPTH * L1TileShape::M * L1TileShape::N);

        uint32_t gmCSplitKBufferIdx = 0;

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            // Compute block location
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            // Compute initial location in logical coordinates
            MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
            MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
            MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
            int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
            int64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
            int64_t gmOffsetC = params.layoutC.GetOffset(offsetC);

            // Compute block-scoped matrix multiply-add
            blockMmad(
                gmA[gmOffsetA], params.layoutA,
                gmB[gmOffsetB], params.layoutB,
                gmC[gmOffsetC], params.layoutC,
                gmWorkspace, gmCSplitKBufferIdx,
                actualBlockShape, flagAicFinishStore);
        }
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        AscendC::SetAtomicNone();

        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);
        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer((__gm__ ElementD *)params.ptrD);

        // Get aicore information
        uint32_t aivIndex = AscendC::GetBlockIdx();
        uint32_t aicoreIndex = aivIndex / AscendC::GetSubBlockNum();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t aivNum = aicoreNum * AscendC::GetSubBlockNum();
        AivCore aivCore = static_cast<AivCore>(AscendC::GetSubBlockIdx());

        AscendC::GlobalTensor<ElementC> gmWorkspace;
        gmWorkspace.SetGlobalBuffer((__gm__ ElementC *)params.ptrWorkspace + aicoreIndex * Catlass::Arch::MAX_REVERSE_DEPTH * L1TileShape::M * L1TileShape::N);

        MatrixCoord loopsMN = matmulBlockScheduler.loopsMN;
        uint32_t maxMN = max(loopsMN.row(), loopsMN.column());

        uint32_t UBKRound = RoundUp(params.problemShape.k(), UBAlignHelper::ALIGN);
        uint32_t kTileCount = CeilDiv<L1TileShape::K>(params.problemShape.k());

        if(params.checkSumType == CheckSumType::ROW_CHECKSUM) {
            if(aivCore == AivCore::AIV0) {
                uint32_t UBTileMRound = RoundUp(UBTileShapeC::M, UBAlignHelper::ALIGN);
                uint32_t UBTileNRound = RoundUp(UBTileShapeC::N, UBAlignHelper::ALIGN);
                LayoutC layoutCSplitKInGm = {L1TileShape::M, L1TileShape::N};

                BlockSumGemvC blockSumGemv(resource);

                for(uint32_t loopIdx = aivIndex; loopIdx < loopsMN.column(); loopIdx += aivNum) {
                    uint32_t nActual = (loopIdx == (loopsMN.column() - 1)) ?
                        (params.problemShape.n() - loopIdx * L1TileShape::N) : L1TileShape::N;

                    uint32_t Nsum = CeilDiv(params.problemShape.k(), UBTileMRound);
                    for(uint32_t sumId = 0; sumId < Nsum; ++sumId) {
                        MatrixCoord offsetB{sumId * UBTileMRound, loopIdx * L1TileShape::N};
                        int64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
                        int64_t gmOffsetBE = loopIdx * UBKRound + sumId * UBTileMRound;
                        uint32_t kActual = (sumId == Nsum - 1) ? (params.problemShape.k() - sumId * UBTileMRound) : UBTileMRound;
                        GemvCoord actualBlockShape = GemvCoord{kActual, nActual};
                        layout::VectorLayout layoutBE{kActual};
                        blockSumGemv.rowSum(gmB[gmOffsetB], params.layoutB, gmD[gmOffsetBE], layoutBE, actualBlockShape);
                    }  
                }

                AscendC::SyncAll<true>();

                uint32_t gmCSplitKBufferIdx = 0;

                for (uint32_t loopIdx = aicoreIndex; loopIdx < coreLoops; loopIdx += aicoreNum) {
                    // Compute block location
                    GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                    GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

                    for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; kLoopIdx++) {
                        MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
                        int64_t gmOffsetC = params.layoutC.GetOffset(offsetC);
                        int64_t gmOffsetCE = loopsMN.column() * UBKRound + coreLoops * kTileCount * L1TileShape::M + (loopIdx * kTileCount + kLoopIdx) * L1TileShape::M;
                        layout::VectorLayout layoutCE{actualBlockShape.m()};
                        GemvCoord gemvActualBlockShape = GemvCoord{actualBlockShape.m(), actualBlockShape.n()};

                        // Synchronize cross core
                        Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);
                        blockSumGemv.rowSum(gmWorkspace[gmCSplitKBufferIdx * L1TileShape::M * L1TileShape::N], layoutCSplitKInGm, gmD[gmOffsetCE], layoutCE, gemvActualBlockShape);
                        gmCSplitKBufferIdx = (gmCSplitKBufferIdx + 1) % Catlass::Arch::MAX_REVERSE_DEPTH;
                    }
                }
            } else if(aivCore == AivCore::AIV1) {
                uint32_t UBTileMRound = RoundUp(UBTileShapeB::M, UBAlignHelper::ALIGN);
                uint32_t UBTileNRound = RoundUp(UBTileShapeB::N, UBAlignHelper::ALIGN);
                BlockSumGemvB blockSumGemv(resource);

                for(uint32_t loopIdx = aivIndex; loopIdx < loopsMN.column(); loopIdx += aivNum) {
                    uint32_t nActual = (loopIdx == (loopsMN.column() - 1)) ?
                        (params.problemShape.n() - loopIdx * L1TileShape::N) : L1TileShape::N;

                    uint32_t Nsum = CeilDiv(params.problemShape.k(), UBTileMRound);
                    for(uint32_t sumId = 0; sumId < Nsum; ++sumId) {
                        MatrixCoord offsetB{sumId * UBTileMRound, loopIdx * L1TileShape::N};
                        int64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
                        int64_t gmOffsetBE = loopIdx * UBKRound + sumId * UBTileMRound;
                        uint32_t kActual = (sumId == Nsum - 1) ? (params.problemShape.k() - sumId * UBTileMRound) : UBTileMRound;
                        GemvCoord actualBlockShape = GemvCoord{kActual, nActual};
                        layout::VectorLayout layoutBE{kActual};
                        blockSumGemv.rowSum(gmB[gmOffsetB], params.layoutB, gmD[gmOffsetBE], layoutBE, actualBlockShape);
                    }  
                }

                AscendC::SyncAll<true>();

                // Loop through the epilogue calculations of each basic block
                for (uint32_t loopIdx = aicoreIndex; loopIdx < coreLoops; loopIdx += aicoreNum) {
                    // Compute block location
                    GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                    GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
            
                    for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; kLoopIdx++) {
                        // Compute initial location in logical coordinates
                        MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, kLoopIdx * L1TileShape::K};
                        int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
                        uint32_t gmoffsetBE = blockCoord.n() * UBKRound + kLoopIdx * L1TileShape::K;
                        uint32_t gmoffsetABE = loopsMN.column() * UBKRound + (loopIdx * kTileCount + kLoopIdx) * L1TileShape::M;
                        layout::VectorLayout layoutBE{L1TileShape::K};
                        layout::VectorLayout layoutABE{L1TileShape::M};
                        uint32_t kActual = (kLoopIdx == kTileCount - 1) ? (params.problemShape.k() - kLoopIdx * L1TileShape::K) : L1TileShape::K;
                        GemvCoord gemvActualBlockShape = GemvCoord{actualBlockShape.m(), kActual};

                        // Synchronize cross core useless
                        Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);
                        blockSumGemv(gmA[gmOffsetA], params.layoutA, gmD[gmoffsetBE], layoutBE, gmD[gmoffsetABE], layoutABE, gemvActualBlockShape);
                    }
                }
            }
        }

    }

private:
    // ID used for inter-core synchronization
    static constexpr Arch::FlagID FLAG_AIC_FINISH_STORE = 0;
    static constexpr Arch::FlagID RV_FLAG_AIC_FINISH_STORE = 1;
    Arch::CrossCoreFlagWithReverse<> flagAicFinishStore{FLAG_AIC_FINISH_STORE, RV_FLAG_AIC_FINISH_STORE};

    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_MATMUL_FAULT_TOLERANCE_HPP
