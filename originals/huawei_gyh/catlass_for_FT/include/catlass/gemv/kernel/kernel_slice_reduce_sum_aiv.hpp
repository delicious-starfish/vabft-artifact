/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMV_KERNEL_SLICESUM_TEST_AIV_HPP
#define CATLASS_GEMV_KERNEL_SLICESUM_TEST_AIV_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemv_coord.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include <cmath>

namespace Catlass::Gemv::Kernel {

// tmeplate for gemv kernle, Compute z = αAx + βy
template <
    class BlockSliceSum_,
    class BlockEpilogue_
>
class KernelSliceSumAiv {
public:
    using BlockSliceSum = BlockSliceSum_;
    using ArchTag = typename BlockSliceSum::ArchTag;
    using UBTileShape = typename BlockSliceSum::UBTileShape;
    using ElementA = typename BlockSliceSum::ElementA;
    using LayoutA = typename BlockSliceSum::LayoutA;
    using ElementX = typename BlockSliceSum::ElementY;
    using LayoutX = typename BlockSliceSum::LayoutY;
    using ElementY = typename BlockSliceSum::ElementY;
    using LayoutY = typename BlockSliceSum::LayoutY;
    using ElementAccumulator = typename BlockSliceSum::ElementAccumulator;

    using L1TileShape = GemmShape<128, 256, 128>;

    using LayoutCOMPX = Catlass::layout::VectorLayout;

    using LayoutYforFT = LayoutA;

    using ElementSliceIn = ElementY;
    using LayoutSliceIn = LayoutYforFT;

    using ElementSliceOut = ElementY;
    using LayoutSliceOut = Catlass::layout::VectorLayout;

    using SliceSumUBTileShape = typename BlockSliceSum::UBTileShape;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemGemmShape;
        GemvCoord problemShape;
        Catlass::GemvCoord problemSliceShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrZ;
        LayoutY layoutY;
        ElementY alpha;
        uint32_t SplitNnum;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        /*
        GM_ADDR ptrX_,LayoutX layoutX_,
             GM_ADDR ptrY_,
        */
        // ,uint32_t split_
        // ptrX(ptrX_),layoutX(layoutX_),
        // ,beta(beta_),split(split_)

        CATLASS_HOST_DEVICE
        Params(
            Catlass::GemmCoord const &problemGemmShape_,
            GemvCoord const &problemShape_,  
            Catlass::GemvCoord const &problemSliceShape_,
            GM_ADDR ptrA_, LayoutA layoutA_, 
            GM_ADDR ptrZ_, LayoutY layoutY_, ElementY alpha_, 
            uint32_t SplitNnum_)
            : problemGemmShape(problemGemmShape_), problemShape(problemShape_), 
            problemSliceShape(problemSliceShape_),
            ptrA(ptrA_), layoutA(layoutA_),
            ptrZ(ptrZ_),layoutY(layoutY_),alpha(alpha_), SplitNnum(SplitNnum_){}
    };

    //TODO: add arguments
    struct Arguments {
        Catlass::GemmCoord problemGemmShape;
        GemvCoord problemShape;
        GM_ADDR ptrA;
        GM_ADDR ptrZ;
        float rounding_exponent;
        float beta;   
    };

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return sizeof(ElementY) * args.problemShape.m();
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        Catlass::GemmCoord problemGemmShape = args.problemGemmShape;
        Catlass::GemvCoord problemShape = args.problemShape;

        uint32_t SplitNnum = ((args.problemGemmShape.n() + L1TileShape::N - 1) / L1TileShape::N);
        
        uint32_t m = problemShape.m();
        uint32_t n = problemShape.n();

        uint32_t m2 = problemGemmShape.m();
        uint32_t n2 = problemGemmShape.n();
        uint32_t k2 = problemGemmShape.k();

        Catlass::GemvCoord problemCompShape{1,(m+n)};
        Catlass::GemvCoord problemSliceShape{SplitNnum, m2};

        LayoutA layoutA{SplitNnum, m2};
        LayoutY layoutX{n};
        LayoutY layoutY{m};

        float input_exponent = (args.rounding_exponent < 0.0f) ? args.rounding_exponent : (0.0 - args.rounding_exponent);

        float rounding_error = std::pow(2.0f,input_exponent);

        float row_sqrt = 1.0f;

        if(args.beta < 1.0f){
            row_sqrt = std::sqrt(n*1.0f);
        }else{
            row_sqrt = args.beta;
        }

        ElementY alpha = static_cast<ElementY>(row_sqrt * rounding_error);
        // float round_base = 2.0f;

        // float rounding_error = std::pow(rou);

        /*
        // Data members
        GemmCoord problemGemmShape;
        GemvCoord problemShape;
        Catlass::GemvCoord problemSliceShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrZ;
        LayoutY layoutY;
        ElementY alpha;
        uint32_t SplitNnum;
        */

        /*
        Params(
            Catlass::GemmCoord const &problemGemmShape_,
            GemvCoord const &problemShape_,  
            Catlass::GemvCoord const &problemSliceShape_,
            GM_ADDR ptrA_, LayoutA layoutA_, 
            GM_ADDR ptrZ_, LayoutY layoutY_, ElementY alpha_, 
            uint32_t SplitNnum_)
            : problemGemmShape(problemGemmShape_), problemShape(problemShape_), 
            problemSliceShape(problemSliceShape_),
            ptrA(ptrA_), layoutA(layoutA_),
            ptrZ(ptrZ_),layoutY(layoutY_),alpha(alpha_), SplitNnum(SplitNnum_)
        */

        Params params{problemGemmShape,
            problemShape,
            problemSliceShape,
            args.ptrA,
            layoutA,
            args.ptrZ,
            layoutY,
            alpha,
            SplitNnum
        };

        return params;
    }

    // Methods
    CATLASS_DEVICE
    KernelSliceSumAiv() {}

    CATLASS_DEVICE
    void SliceNSum_op(Params const &params)
    {
        AscendC::SetAtomicNone();
        // Arch::Resource<ArchTag> resource;
        BlockSliceSum blockSliceSum(resource);
        uint32_t align = Catlass::BYTE_PER_C0 / sizeof(ElementSliceIn);
        uint32_t maxmPerBlock_round = RoundUp(SliceSumUBTileShape::M, align);
        uint32_t maxnPerBlock_round = RoundUp(SliceSumUBTileShape::N, align);
        uint32_t split = 1;

        LayoutYforFT layoutSliceIn{params.SplitNnum, params.problemGemmShape.m()};
        LayoutCOMPX layoutSliceOut{params.problemGemmShape.m()};

        // add split k
        uint32_t M_Split = params.SplitNnum;
        // RoundDown(, params.split) / params.split;
        uint32_t Nloopnum = CeilDiv(params.problemSliceShape.n(), maxnPerBlock_round);
        int32_t loopnum;

        loopnum = Nloopnum;

        uint32_t offset_matrix;
        uint32_t offset_vector_out;
        uint32_t offset_vector_in = 0;
        
        // Represent the full gm
        AscendC::GlobalTensor<ElementSliceIn> gmIn;
        gmIn.SetGlobalBuffer((__gm__ ElementSliceIn *)params.ptrA);

        AscendC::GlobalTensor<ElementSliceOut> gmOut;
        gmOut.SetGlobalBuffer((__gm__ ElementSliceOut *)params.ptrZ);
        uint32_t aiv_num = AscendC::GetBlockNum() * AscendC::GetTaskRation();
        for (uint32_t loop_id = 0; loop_id < loopnum; loop_id++) {
            uint32_t aiv_id = AscendC::GetBlockIdx();
            if (loop_id % aiv_num != aiv_id)
                continue;
            uint32_t n_actual = ((int32_t)loop_id > (int32_t)(loopnum - split - 1))
                                        ? params.problemSliceShape.n() - ((loop_id / split) * maxnPerBlock_round)
                                        : maxnPerBlock_round;
            uint32_t m_actual = params.problemSliceShape.m();

            if constexpr (std::is_same_v<LayoutYforFT, Catlass::layout::ColumnMajor>) {
                offset_matrix = loop_id * maxnPerBlock_round * params.problemSliceShape.m();
                offset_vector_out = (loop_id / split) * maxnPerBlock_round;

                m_actual = params.problemSliceShape.m();

            } else {
                offset_matrix = loop_id * maxnPerBlock_round;
                offset_vector_out = loop_id * maxnPerBlock_round;
            }
            Catlass::GemvCoord actualBlockShape = Catlass::GemvCoord{m_actual, n_actual};

            /*
            void operator()(AscendC::GlobalTensor<ElementA> const &gmA, 
            LayoutA const &layoutA,
            AscendC::GlobalTensor<ElementY> const &gmZ, 
            LayoutY const &layoutY, GemvCoord const &actualShape)
            */

            blockSliceSum(gmIn[offset_matrix], layoutSliceIn,
                gmOut[offset_vector_out], layoutSliceOut,
                actualBlockShape);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params) {};

    /// Executes one Matmul
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params) {}

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        SliceNSum_op(params);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    // ID used for inter-core synchronization
    static constexpr Catlass::Arch::FlagID FLAG_AIC_FINISH_STORE = 0;
    static constexpr Catlass::Arch::FlagID RV_FLAG_AIC_FINISH_STORE = 1;
    Catlass::Arch::CrossCoreFlagWithReverse<> flagAicFinishStore{FLAG_AIC_FINISH_STORE,RV_FLAG_AIC_FINISH_STORE};
    Catlass::Arch::Resource<ArchTag> resource;
};

}

#endif // CATLASS_GEMV_KERNEL_SLICESUM_TEST_AIV_HPP