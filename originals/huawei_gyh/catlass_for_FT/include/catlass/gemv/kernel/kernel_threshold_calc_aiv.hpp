/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMV_KERNEL_THRESHOLD_CALC_AIV_HPP
#define CATLASS_GEMV_KERNEL_THRESHOLD_CALC_AIV_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemv_coord.hpp"
#include <cmath>

namespace Catlass::Gemv::Kernel {

// tmeplate for gemv kernle, Compute z = αAx + βy
template <
    class BlockThresholdCalc_,
    class BlockEpilogue_
>
class KernelThreCalcAiv {
public:
    using BlockThresholdCalc = BlockThresholdCalc_;
    using ArchTag = typename BlockThresholdCalc::ArchTag;
    using UBTileShape = typename BlockThresholdCalc::UBTileShape;
    using ElementA = typename BlockThresholdCalc::ElementA;
    using LayoutA = typename BlockThresholdCalc::LayoutA;
    using ElementX = typename BlockThresholdCalc::ElementX;
    using LayoutX = typename BlockThresholdCalc::LayoutX;
    using ElementY = typename BlockThresholdCalc::ElementY;
    using LayoutY = typename BlockThresholdCalc::LayoutY;
    using ElementAccumulator = typename BlockThresholdCalc::ElementAccumulator;


    /// Parameters structure
    struct Params {
        // Data members
        GemvCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        // GM_ADDR ptrX;
        // LayoutX layoutX;
        // GM_ADDR ptrY;
        GM_ADDR ptrZ;
        LayoutY layoutY;
        ElementY alpha;
        // float beta;
        // uint32_t split;

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
        Params(GemvCoord const &problemShape_,  GM_ADDR ptrA_, LayoutA layoutA_, 
            GM_ADDR ptrZ_, LayoutY layoutY_, ElementY alpha_)
            : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_),
            ptrZ(ptrZ_),layoutY(layoutY_),alpha(alpha_) {}
    };

    //TODO: add arguments
    // GM_ADDR ptrX;
    //  GM_ADDR ptrY;
    // uint32_t split;
    struct Arguments {
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
        GemvCoord problemShape = args.problemShape;
        uint32_t m = problemShape.m();
        uint32_t n = problemShape.n();
        LayoutA layoutA{m, n};
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
        Params(GemvCoord const &problemShape_,  GM_ADDR ptrA_, LayoutA layoutA_, 
            GM_ADDR ptrZ_, LayoutY layoutY_, ElementY alpha_)

            args.alpha,
            args.beta,
            args.split
        */

        Params params{problemShape,
            args.ptrA,
            layoutA,
            args.ptrZ,
            layoutY,
            alpha
        };

        return params;
    }

    // Methods
    CATLASS_DEVICE
    KernelThreCalcAiv() {}

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
        AscendC::SetAtomicNone();
        Arch::Resource<ArchTag> resource;
        BlockThresholdCalc blockThreCalc(resource);
        uint32_t align = BYTE_PER_C0 / sizeof(ElementA);
        uint32_t maxmPerBlock_round = RoundUp(UBTileShape::M, align);
        uint32_t maxnPerBlock_round = RoundUp(UBTileShape::N, align);
        uint32_t split = 1;

        // add split k
        uint32_t N_Split = params.problemShape.n();
        // RoundDown(, params.split) / params.split;
        uint32_t Mloopnum = CeilDiv(params.problemShape.m(), maxmPerBlock_round);
        int32_t loopnum;
        // float Realbeta = params.alpha;
        if constexpr (std::is_same_v<LayoutA, Catlass::layout::ColumnMajor>) {
            loopnum = Mloopnum * split;
            // Realbeta = params.alpha;
        } else {
            loopnum = Mloopnum;
        }

        uint32_t offset_matrix;
        uint32_t offset_vector_out;
        uint32_t offset_vector_in = 0;

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        
        // AscendC::GlobalTensor<ElementX> gmX;
        // gmX.SetGlobalBuffer((__gm__ ElementX *)params.ptrX);
        // AscendC::GlobalTensor<ElementY> gmY;
        // gmY.SetGlobalBuffer((__gm__ ElementY *)params.ptrY);
        AscendC::GlobalTensor<ElementY> gmZ;
        gmZ.SetGlobalBuffer((__gm__ ElementY *)params.ptrZ);
        uint32_t aiv_num = AscendC::GetBlockNum() * AscendC::GetTaskRation();
        for (uint32_t loop_id = 0; loop_id < loopnum; loop_id++) {
            uint32_t aiv_id = AscendC::GetBlockIdx();
            if (loop_id % aiv_num != aiv_id)
                continue;
            uint32_t m_actual = ((int32_t)loop_id > (int32_t)(loopnum - split - 1))
                                        ? params.problemShape.m() - ((loop_id / split) * maxmPerBlock_round)
                                        : maxmPerBlock_round;
            uint32_t n_actual = params.problemShape.n();

            if constexpr (std::is_same_v<LayoutA, Catlass::layout::ColumnMajor>) {
                offset_matrix = (loop_id % split) * N_Split * params.problemShape.m() +
                                (loop_id / split) * maxmPerBlock_round;
                offset_vector_out = (loop_id / split) * maxmPerBlock_round;
                offset_vector_in = (loop_id % split) * N_Split;

                if ((loop_id % split) == split - 1) {
                    n_actual = params.problemShape.n() - N_Split * (split - 1);
                } else {
                    n_actual = N_Split;
                }
            } else {
                offset_matrix = loop_id * maxmPerBlock_round * params.problemShape.n();
                offset_vector_out = loop_id * maxmPerBlock_round;
            }
            GemvCoord actualBlockShape = GemvCoord{m_actual, n_actual};

            // float realbeta = (loop_id % split == 0) ? Realbeta : 0.0f;

            /*
            (AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
            AscendC::GlobalTensor<ElementY> const &gmZ, LayoutY const &layoutY,
            GemvCoord const &actualShape,ElementY alpha)

            gmX[offset_vector_in], params.layoutX,
            gmY[offset_vector_out]
            // realbeta
            */

            blockThreCalc(gmA[offset_matrix], params.layoutA,
                gmZ[offset_vector_out], params.layoutY,
                actualBlockShape, params.alpha);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }
};

}

#endif // CATLASS_GEMV_KERNLE_GEMV_AIV_HPP