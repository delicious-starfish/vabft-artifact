/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMV_KERNLE_LARGE_COMPARE_HPP
#define CATLASS_GEMV_KERNLE_LARGE_COMPARE_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemv_coord.hpp"
#include "catlass/gemv/helper.hpp"


namespace Catlass::Gemv::Kernel {

// tmeplate for gemv kernle, Compute z = αAx + βy
template <
    class BlockCompare_,
    class BlockEpilogue_
>
class KernelCompareAiv {
public:
    using BlockCompare = BlockCompare_;
    using ArchTag = typename BlockCompare::ArchTag;
    using COMPUBTileShape = typename BlockCompare::UBTileShape;

    using ElementCOMPX = typename BlockCompare::ElementX;
    using LayoutCOMPX = typename BlockCompare::LayoutX;

    using ElementCOMPY = typename BlockCompare::ElementY;
    using LayoutCOMPY = typename BlockCompare::LayoutY;

    using ElementCOMPZ = typename BlockCompare::ElementZ;
    using LayoutCOMPZ = typename BlockCompare::LayoutZ;

    // using LayoutW = Catlass::layout::VectorLayout;
    using FT_COMP_TYPE = Catlass::Gemv::helper::FT_COMP_TYPE;
    

    using ElementWork = typename std::conditional<
        (BlockCompare::COMP_TYPE == FT_COMP_TYPE::XOR),
        uint16_t,
        typename std::conditional<(BlockCompare::COMP_TYPE == FT_COMP_TYPE::COMPARE), int32_t, ElementCOMPX>::type
    >::type;
    // using ElementAccumulator = typename BlockGemv::ElementAccumulator;

    /// Parameters structure
    // LayoutW layoutWorkspace;
    struct Params {
        // Data members
        GemvCoord problemCompShape;
        GM_ADDR ptrInputX;
        LayoutCOMPX layoutInputX;
        GM_ADDR ptrInputY;
        LayoutCOMPY layoutInputY;
        GM_ADDR ptrOutputZ;
        LayoutCOMPZ layoutOutputZ;
        GM_ADDR ptrWorkspace;
        uint32_t UbNum;
        bool OutputWorkspace;
        ElementCOMPX threshold;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        // LayoutW layoutWorkspace_, layoutWorkspace(layoutWorkspace_),
        CATLASS_HOST_DEVICE
        Params(GemvCoord const &problemCompShape_,  GM_ADDR ptrInputX_, 
            LayoutCOMPX layoutInputX_,  GM_ADDR ptrInputY_,
            LayoutCOMPY layoutInputY_,
            GM_ADDR ptrOutputZ_, 
            LayoutCOMPZ layoutOutputZ_, 
            GM_ADDR ptrWorkspace_,
            uint32_t UbNum_,
            bool OutputWorkspace_,
            ElementCOMPX threshold_)
            : problemCompShape(problemCompShape_), ptrInputX(ptrInputX_), layoutInputX(layoutInputX_), 
              ptrInputY(ptrInputY_), layoutInputY(layoutInputY_),
              ptrOutputZ(ptrOutputZ_),layoutOutputZ(layoutOutputZ_),
              ptrWorkspace(ptrWorkspace_),
              UbNum(UbNum_),
              OutputWorkspace(OutputWorkspace_),
              threshold(threshold_) {}
    };

    //TODO: add arguments
    struct Arguments {
        GemvCoord problemCompShape;
        GM_ADDR ptrInputX;
        GM_ADDR ptrInputY;
        GM_ADDR ptrOutputZ;
        uint32_t UbNum;
        bool OutputWorkspace;
        ElementCOMPX threshold;
    };

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        // uint32_t total_elements = args.problemCompShape.m() * args.problemCompShape.n();
        // uint32_t total_input_bytes = total_elements * sizeof(ElementCOMPX);
        // uint32_t total_workspace_bytes = RoundUp(total_input_bytes, static_cast<uint32_t>(sizeof(ElementWork)));
        // uint32_t total_workspace_elements = total_workspace_bytes / sizeof(ElementWork);
        // sizeof(ElementWork) * total_workspace_elements
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *ptrWorkspace)
    {
        GemvCoord problemCompShape = args.problemCompShape;
        uint32_t m = problemCompShape.m();
        uint32_t n = problemCompShape.n();
        uint32_t total_input_elements = m * n;
        
        uint32_t total_input_bytes = total_input_elements * sizeof(ElementCOMPX);
        uint32_t total_output_elements = (total_input_elements + 8 - 1) / 8;
        uint32_t total_workspace_bytes = RoundUp(total_input_bytes, static_cast<uint32_t>(sizeof(ElementWork)));
        uint32_t total_workspace_elements = total_workspace_bytes / sizeof(ElementWork);
        // printf("Total input bytes: %d",total_input_bytes);

        LayoutCOMPX layoutX{total_input_elements};
        LayoutCOMPY layoutY{total_input_elements};
        LayoutCOMPZ layoutZ{total_output_elements};
        // LayoutW layoutW{total_workspace_elements};

        /*
        (GemvCoord const &problemCompShape_,  GM_ADDR ptrInputX_, 
            LayoutCOMPX layoutInputX_,  GM_ADDR ptrInputY_,
            LayoutCOMPY layoutInputY_,
            GM_ADDR ptrOutputZ_, 
            LayoutCOMPZ layoutOutputZ_, 
            GM_ADDR ptrWorkspace_,
            LayoutW layoutWorkspace_,
            uint32_t UbNum_,
            bool OutputWorkspace_,
            ElementCOMPX threshold_)
        */
        //  layoutW, 

        Params params{problemCompShape,  
            args.ptrInputX, layoutX,
            args.ptrInputY, layoutY,
            args.ptrOutputZ, layoutZ,
            ptrWorkspace, args.UbNum,
            args.OutputWorkspace,
            args.threshold};

        return params;
    }

    // Methods
    CATLASS_DEVICE
    KernelCompareAiv() {}

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

        BlockCompare blockCompare(resource);

        uint32_t align = BYTE_PER_BLK / sizeof(ElementCOMPX);
        uint32_t total_block_elements = COMPUBTileShape::M * COMPUBTileShape::N * params.UbNum;
        uint32_t maxPerBlock_round = RoundUp(total_block_elements, align);

        // uint32_t maxPerBlock_work = maxPerBlock_round * sizeof(ElementCOMPX) / sizeof(ElementWork);
        uint32_t maxPerBlock_out = maxPerBlock_round / 8;

        uint32_t M = params.problemCompShape.m();
        uint32_t N = params.problemCompShape.n();
        uint32_t total_input_elements = M * N;
        
        uint32_t total_input_bytes = total_input_elements * sizeof(ElementCOMPX);
        uint32_t total_output_elements = (total_input_elements + 8-1)/8;
        uint32_t total_workspace_bytes = RoundUp(total_input_bytes, static_cast<uint32_t>(sizeof(ElementWork)));

        uint32_t total_workspace_elements = total_workspace_bytes / sizeof(ElementWork);

        // add split k
        uint32_t loopnum = CeilDiv(total_input_elements, maxPerBlock_round);

        uint32_t offset_vector_in_x = 0;
        uint32_t offset_vector_in_y = 0;
        uint32_t offset_vector_out = 0;
        uint32_t offset_vector_workspace = 0;

        // Represent the full gm
        AscendC::GlobalTensor<ElementCOMPX> gmX;
        gmX.SetGlobalBuffer((__gm__ ElementCOMPX *)params.ptrInputX);
        AscendC::GlobalTensor<ElementCOMPY> gmY;
        gmY.SetGlobalBuffer((__gm__ ElementCOMPY *)params.ptrInputY);
        // AscendC::GlobalTensor<ElementWork> gmWork;
        // gmWork.SetGlobalBuffer((__gm__ ElementWork *)params.ptrWorkspace);
        AscendC::GlobalTensor<ElementCOMPZ> gmZ;
        gmZ.SetGlobalBuffer((__gm__ ElementCOMPZ *)params.ptrOutputZ);

        bool isFirstBlock = true;
        bool hasNextBlock = false;
        uint32_t aiv_num = AscendC::GetBlockNum() * AscendC::GetTaskRation();
        uint32_t aiv_id = AscendC::GetBlockIdx();

        for (uint32_t loop_id = aiv_id; loop_id < loopnum; loop_id+=aiv_num) {
            
            if (loop_id % aiv_num != aiv_id)
                continue;
            
            // if(loop_id == aiv_id){
            //     isFirstBlock = true;
            // }else{
            //     isFirstBlock = false;
            // }

            uint32_t InputGmBlockIdx = loop_id;
            uint32_t input_element_actual = (InputGmBlockIdx == loopnum - 1) ? (total_input_elements - InputGmBlockIdx * maxPerBlock_round) : maxPerBlock_round;

            int64_t gmOffsetX = InputGmBlockIdx * maxPerBlock_round;
            int64_t gmOffsetY = InputGmBlockIdx * maxPerBlock_round;
            // int64_t gmOffsetWork = InputGmBlockIdx * maxPerBlock_work;
            int64_t gmOffsetZ = InputGmBlockIdx * maxPerBlock_out;

            

            GemvCoord actualBlockShape = GemvCoord{1, input_element_actual};

            uint32_t InputNextGmBlockIdx;
            int64_t gmOffsetNextX;
            int64_t gmOffsetNextY;
            // int64_t gmOffsetNextWork;
            int64_t gmOffsetNextZ;
            GemvCoord nextActualBlockShape;

            if((loop_id + aiv_num) < loopnum){
                // hasNextBlock = true;
                uint32_t nextLoopIdx = loop_id + aiv_num;
                InputNextGmBlockIdx = nextLoopIdx;

                uint32_t input_element_actual_next = 
                    (InputNextGmBlockIdx == loopnum - 1) ? (total_input_elements - InputNextGmBlockIdx * maxPerBlock_round) : maxPerBlock_round;
                
                nextActualBlockShape = GemvCoord(1, input_element_actual_next);
                
                gmOffsetNextX = InputNextGmBlockIdx * maxPerBlock_round;
                gmOffsetNextY = InputNextGmBlockIdx * maxPerBlock_round;
                // gmOffsetNextWork = InputNextGmBlockIdx * maxPerBlock_work;
                gmOffsetNextZ = InputNextGmBlockIdx * maxPerBlock_out;
            }

            // gmWork[gmOffsetWork], params.layoutWorkspace,
            blockCompare(gmX[gmOffsetX], params.layoutInputX,
                         gmY[gmOffsetY], params.layoutInputY,
                         gmX[gmOffsetNextX], gmY[gmOffsetNextY],
                         gmZ[gmOffsetZ],params.layoutOutputZ,
                         actualBlockShape, nextActualBlockShape, isFirstBlock, 
                         hasNextBlock, params.OutputWorkspace, params.threshold);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }
};

}

#endif // CATLASS_GEMV_KERNLE_LARGE_COMPARE_HPP