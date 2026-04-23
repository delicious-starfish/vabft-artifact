#ifndef CATLASS_GEMM_KERNEL_MATMUL_EPILOGUE_HPP_SELF_PRELOAD
#define CATLASS_GEMM_KERNEL_MATMUL_EPILOGUE_HPP_SELF_PRELOAD

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace CubeSelf::Gemm::Kernel{
    // Template for matmul add kernel. Compute D = A * B + X
template <
    class BlockMmad_,
    // class BlockEpilogue_,
    class BlockScheduler_
>
class MatmulEpiloguePreload{
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;

    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;

    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;

    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    // using BlockEpilogue = BlockEpilogue_;

    using ElementD = ElementC;
    // typename BlockEpilogue::ElementD;
    using LayoutD = LayoutC;
    // typename BlockEpilogue::LayoutD;

    // using EpilogueParams = typename BlockEpilogue::Params;

    using BlockScheduler = BlockScheduler_;

    static_assert(std::is_same_v<ElementD, ElementC> &&
        std::is_same_v<LayoutD, LayoutC>,
        "The CType of Mmad and Epilogue should be consistent.");
    
    /// Parameters structure
    struct Params {
        // Data members
        Catlass::GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrWorkspace;
        // EpilogueParams epilogueParams;
        GM_ADDR ptrC;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {};

        // EpilogueParams const &epilogueParams_,
        // epilogueParams(epilogueParams_),
        CATLASS_HOST_DEVICE
        Params(
            Catlass::GemmCoord const &problemShape_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrWorkspace_, GM_ADDR ptrC_
        ) : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_), 
            ptrWorkspace(ptrWorkspace_), ptrC(ptrC_) {} 
    };

    struct Arguments {
        Catlass::GemmCoord problemShape;
        size_t elementSize;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
    };

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return args.elementSize * args.problemShape.m() * args.problemShape.n();
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        Catlass::GemmCoord problemShape = args.problemShape;

        uint32_t m = problemShape.m();
        uint32_t n = problemShape.n();
        uint32_t k = problemShape.k();
        LayoutA layoutA{m, k};
        LayoutB layoutB{k, n};
        LayoutC layoutC{m, n};
        // typename BlockEpilogue::Params epilogueParams{args.ptrC, layoutC, args.ptrC, layoutC};
        // epilogueParams, 
        /*
        Params(
            Catlass::GemmCoord const &problemShape_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrWorkspace_, GM_ADDR ptrC_
        ) : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), 
            ptrB(ptrB_), layoutB(layoutB_), 
            ptrWorkspace(ptrWorkspace_), ptrC(ptrC_) 
        */

        Params params{problemShape, args.ptrA, layoutA, 
            args.ptrB, layoutB, workspace, args.ptrC};

        return params;
    }

    // Methods
    CATLASS_DEVICE
    MatmulEpiloguePreload() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    template<>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        BlockScheduler matmulBlockScheduler(params.problemShape, 
            Catlass::MakeCoord(L1TileShape::M,L1TileShape::N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();


        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);

        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);

        AscendC::GlobalTensor<ElementC> gmC;
        // ptrWorkspace
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        /*
        GetBlockIdx:
        I. 功能说明
        获取当前核的index，用于代码内部的多核逻辑控制及多核偏移量计算等。

        II. 函数原型
        __aicore__ inline int64_t GetBlockIdx()
        
        III. 参数说明: 无

        IV. 返回值
        当前核的index，index的范围为[0, 用户配置的block_dim数量 - 1]

        V. 支持的型号
        Atlas 训练系列产品
        Atlas 推理系列产品AI Core
        Atlas 推理系列产品Vector Core
        Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件
        Atlas A3 训练系列产品/Atlas A3 推理系列产品
        Atlas 200I/500 A2 推理产品

        VI. 约束说明
        GetBlockIdx为一个系统内置函数，返回当前核的index。

        VII. 调用示例
        #include "kernel_operator.h"
        constexpr int32_t SINGLE_CORE_OFFSET = 256;
        class KernelGetBlockIdx {
        public:
            __aicore__ inline KernelGetBlockIdx () {}
            __aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm)
            {
                // 根据index对每个核进行地址偏移
                src0Global.SetGlobalBuffer((__gm__ float*)src0Gm + AscendC::GetBlockIdx() * SINGLE_CORE_OFFSET);
                src1Global.SetGlobalBuffer((__gm__ float*)src1Gm + AscendC::GetBlockIdx() * SINGLE_CORE_OFFSET);
                dstGlobal.SetGlobalBuffer((__gm__ float*)dstGm + AscendC::GetBlockIdx() * SINGLE_CORE_OFFSET);
                pipe.InitBuffer(inQueueSrc0, 1, 256 * sizeof(float));
                pipe.InitBuffer(inQueueSrc1, 1, 256 * sizeof(float));
                pipe.InitBuffer(selMask, 1, 256);
                pipe.InitBuffer(outQueueDst, 1, 256 * sizeof(float));
            }
            ......
        };
        */

        /*
        GetBlockNum:
        
        I. 功能说明
        获取当前任务配置的核数，用于代码内部的多核逻辑控制等。

        II. 函数原型
        __aicore__ inline int64_t GetBlockNum()
        
        III. 参数说明: 无

        IV. 返回值
        当前任务配置的核数。

        V. 支持的型号
        Atlas 训练系列产品
        Atlas 推理系列产品AI Core
        Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件
        Atlas A3 训练系列产品/Atlas A3 推理系列产品
        Atlas 200I/500 A2 推理产品

        VI. 约束说明: 无。

        VII. 调用示例
        #include "kernel_operator.h"
        // 在核内做简单的tiling计算时使用block_num，复杂tiling建议在host侧完成
        __aicore__ inline void InitTilingParam(int32_t& totalSize, int32_t& loopSize)
        {
            loopSize = totalSize / AscendC::GetBlockNum();
        };
        */

        Catlass::layout::RowMajor layoutC(params.problemShape.m(), params.problemShape.n());
        // 共24个核，以其核的编号作为起始loop循环的位置，每次处理的循环编号为间隔核的数量
        // 此处,对于AIV 而言，GetBlockIdx()获取的是其 AIV core 的 Block ID，对于AIC而言，获取的是AIC core 的BLOCK ID
        // GetBlockNum(): 获取的是AI Core的数量，或者说是AIC 与 AIV 组合的数量，一个AIC 对应多个AIV， 往往获得的值等于使用的AIC的数量
        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            // Compute block location
            Catlass::GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            Catlass::GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            // Compute initial location in logical coordinates
            Catlass::MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
            Catlass::MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
            Catlass::MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
            
            int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
            int64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
            int64_t gmOffsetC = layoutC.GetOffset(offsetC);

            bool isFirstBlock = (loopIdx == AscendC::GetBlockIdx());

            bool hasNextBlock = false;
            uint64_t gmOffsetNextA = gmOffsetA;
            uint64_t gmOffsetNextB = gmOffsetB;
            uint64_t gmOffsetNextC = gmOffsetC;
            Catlass::GemmCoord nextActualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            uint32_t loopIdxNext = loopIdx + AscendC::GetBlockNum();

            if(loopIdxNext < coreLoops){
                hasNextBlock = true;
                Catlass::GemmCoord blockCoordNext = matmulBlockScheduler.GetBlockCoord(loopIdxNext);
                nextActualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoordNext);

                // Compute initial location in logical coordinates for next Block
                Catlass::MatrixCoord offsetNextA{blockCoordNext.m() * L1TileShape::M, blockCoordNext.k() * L1TileShape::K};
                Catlass::MatrixCoord offsetNextB{blockCoordNext.k() * L1TileShape::K, blockCoordNext.n() * L1TileShape::N};
                Catlass::MatrixCoord OffsetNextC{blockCoordNext.m() * L1TileShape::M, blockCoordNext.n() * L1TileShape::N};

                gmOffsetNextA = params.layoutA.GetOffset(offsetNextA);
                gmOffsetNextB = params.layoutB.GetOffset(offsetNextB);
                gmOffsetNextC = layoutC.GetOffset(OffsetNextC);
            }

            // Compute block-scoped matrix multiply-add
            // blockMmad(
            //     gmA[gmOffsetA], params.layoutA,
            //     gmB[gmOffsetB], params.layoutB,
            //     gmC[gmOffsetC], layoutC,
            //     actualBlockShape);

            /*
            CATLASS_DEVICE
            void operator()(
                AscendC::GlobalTensor<ElementA> const & gmA, 
                AscendC::GlobalTensor<ElementA> const & gmNextA,
                LayoutA const &layoutA,
                AscendC::GlobalTensor<ElementB> const & gmB, 
                AscendC::GlobalTensor<ElementB> const & gmNextB,
                LayoutB const &layoutB,
                AscendC::GlobalTensor<ElementC> const & gmC, LayoutC const &layoutC,
                Catlass::GemmCoord const &actualShape, 
                Catlass::GemmCoord const &actualShapeNext,
                bool isFirstBlock, bool hasNextBlock
            )
            */
            
            blockMmad(gmA[gmOffsetA], gmA[gmOffsetNextA], params.layoutA,
                gmB[gmOffsetB], gmB[gmOffsetNextB], params.layoutB,
                gmC[gmOffsetC], layoutC, 
                actualBlockShape, 
                nextActualBlockShape, 
                isFirstBlock, hasNextBlock); 
            // 通知相应 AIV core，MMAD计算已经完成了，结果已经写入了GM 
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
        }

        /*
        硬件流水类型：

        */

        /*
        PipeBarrier():
        I. 功能说明
        阻塞相同流水，具有数据依赖的相同流水之间需要插入此同步。

        II. 函数原型
        template <pipe_t pipe>
        __aicore__ inline void PipeBarrier()

        III. 参数说明
                        表1 模板参数说明
        参数名                                      描述
        pipe                            模板参数，表示阻塞的流水类别。
                                        支持的流水参考硬件流水类型。
                                        如果不关注流水类别，希望阻塞所有流水，
                                        可以传入PIPE_ALL。
        
        IV. 返回值: 无

        V. 约束说明
        Scalar流水之间的同步由硬件自动保证，调用PipeBarrier<PIPE_S>()会引发硬件错误。

        VI. 调用示例
        如下示例，Mul指令的输入dst0Local是Add指令的输出，两个矢量运算指令产生依赖，
        需要插入PipeBarrier保证两条指令的执行顺序。

        注：仅作为示例参考，开启自动同步的情况下，编译器自动插入PIPE_V同步，
        无需开发者手动插入。

        图1 Mul指令和Add指令是串行关系，必须等待Add指令执行完成后，才能执行Mul指令。

        Add(dst0Local, src0Local, src1Local, 512);
                |
              PIPE_V
                |
                v
        Mul(dst1Local, dst0Local, src2Local, 512);

        代码实例：
        AscendC::LocalTensor<half> src0Local;
        AscendC::LocalTensor<half> src1Local;
        AscendC::LocalTensor<half> src2Local;
        AscendC::LocalTensor<half> dst0Local;
        AscendC::LocalTensor<half> dst1Local;

        AscendC::Add(dst0Local, src0Local, src1Local, 512);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(dst1Local, dst0Local, src2Local, 512);     
        */
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template<>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        BlockScheduler matmulBlockScheduler(params.problemShape, Catlass::MakeCoord(L1TileShape::M, L1TileShape::N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        // BlockEpilogue blockEpilogue(resource,params.epilogueParams);

        // Represent the full gm
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrWorkspace);
        
        Catlass::layout::RowMajor layoutC(params.problemShape.m(), params.problemShape.n());

        // Get AICore Information
        // AIV 与 AIC的组合中，多个AIV core 对应一个AIC core，因此
        // 此处,对于AIV 而言，获取的是其 AIV core 的 Block ID，对于AIC而言，获取的是AIC core 的BLOCK ID
        // 因此此处：AscendC::GetBlockIdx() / AscendC::GetSubBlockNum(); 是获取当前AIV BLOCK ID对应的 AIC core的 BLOCK ID
        uint32_t aicoreIndex = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t subcoreIndex = AscendC::GetSubBlockIdx();

        /*
        GetSubeBlockNum:
        I. 功能说明
        获取单个 AI Core上Vector核的数量。或者说获取单个AIC-AIV组合中AIV 核的数量

        II. 函数原型
        __aicore__ inline int64_t GetSubBlockNum()
        
        III. 参数说明: 无
        
        IV. 返回值
        返回int64类型的Vector核数量。

        V. 支持的型号
        Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件
        Atlas A3 训练系列产品/Atlas A3 推理系列产品

        VI. 约束说明: 无
        */

        /*
        GetSubBlockIdx:
        I. 功能说明
        获取单个 AI Core 上某个 Vector 核的ID。

        II. 函数原型
        __aicore__ inline int64_t GetSubBlockIdx()
        
        III. 参数说明: 无

        IV. 返回值
        返回int64类型的Vector核ID。

        V. 支持的型号
        Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件
        Atlas A3 训练系列产品/Atlas A3 推理系列产品

        VI. 约束说明: 无

        VII. 调用示例
        int64_t subBlockID = AscendC::GetSubBlockIdx();
        */

        // Loop through the epilogue calculations of each basic block
        Catlass::GemmCoord blockShape = L1TileShape::ToCoord();

        for (uint32_t loopIdx = aicoreIndex; loopIdx < coreLoops; loopIdx += aicoreNum) {
            // Compute block location
            Catlass::GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            Catlass::GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            // Get the data and layout of C under the current basic block
            auto gmBlockC = gmC[layoutC.GetOffset(blockCoord.GetCoordMN() * blockShape.GetCoordMN())];
            auto layoutBlockC = layoutC.GetTileLayout(actualBlockShape.GetCoordMN());
            // Synchronize cross core
            // 等待相应 AIC core 的MMAD计算完成，可以从GM 中获取数据进行计算
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);
            // 进行收尾操作
            // Actual calculatioin logic for performing block-scoped epilogue
            // blockEpilogue(blockShape, blockCoord, actualBlockShape, gmBlockC, layoutBlockC);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }


private:
    // ID used for inter-core synchronization
    static constexpr Catlass::Arch::FlagID FLAG_AIC_FINISH_STORE = 0;
    static constexpr Catlass::Arch::FlagID RV_FLAG_AIC_FINISH_STORE = 1;
    Catlass::Arch::CrossCoreFlagWithReverse<> flagAicFinishStore{FLAG_AIC_FINISH_STORE,RV_FLAG_AIC_FINISH_STORE};
    Catlass::Arch::Resource<ArchTag> resource;
};

} // namespace CubeSelf::Gemm::Kernel

#endif