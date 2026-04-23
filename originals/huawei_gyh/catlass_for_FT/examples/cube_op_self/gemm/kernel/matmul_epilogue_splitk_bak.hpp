#ifndef CATLASS_GEMM_KERNEL_MATMUL_EPILOGUE_SPLIT_K_SELF_HPP
#define CATLASS_GEMM_KERNEL_MATMUL_EPILOGUE_SPLIT_K_SELF_HPP

#include <cmath>
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace CubeSelf::Gemm::Kernel{

// Template for Matmul kernel. Compute C = A * B
template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_,
    class BlockReduceAdd_
>
class MatmulAsVarSplitK{
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
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using BlockScheduler = BlockScheduler_;
    using BlockReduceAdd = BlockReduceAdd_;

    using SliceReduceUBTileShape = typename BlockReduceAdd::UBTileShape;
    using SliceReduceUBBlockShape = typename BlockReduceAdd::UBBlockShape;

    using LayoutVA = Catlass::layout::VectorLayout;
    using LayoutVY = Catlass::layout::VectorLayout;

    /// Parameters structure
    struct Params {
        // Data members
        Catlass::GemmCoord problemGemmShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrWorkspace;
        // uint32_t splitkFactor = 1;
        uint32_t SplitKNum = 1;
        uint32_t SplitKNumLimit = 8;
        uint32_t actualKSliceSize[3];

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(Catlass::GemmCoord const &problemGemmShape_,
            GM_ADDR ptrA_, LayoutA layoutA_, 
            GM_ADDR ptrB_, LayoutB layoutB_,
            GM_ADDR ptrC_, LayoutB layoutC_,
            GM_ADDR ptrWorkspace_, 
            uint32_t SplitKNum_, uint32_t SplitKNumLimit_,
            const uint32_t (&actualKSliceSize_)[3]
        ) : problemGemmShape(problemGemmShape_), ptrA(ptrA_), layoutA(layoutA_), 
            ptrB(ptrB_), layoutB(layoutB_),
            ptrC(ptrC_), layoutC(layoutC_), 
            ptrWorkspace(ptrWorkspace_), 
            SplitKNum(SplitKNum_), SplitKNumLimit(SplitKNumLimit_) {
                for (int i = 0; i < 3; ++i) {
                    this->actualKSliceSize[i] = actualKSliceSize_[i];
                }
            }
    };

    struct Arguments {
        Catlass::GemmCoord problemGemmShape;
        uint32_t aicCoreNum;
        size_t workspaceElementSize;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
        uint32_t SliceKUnit;
        uint32_t SplitKNumLimit;
    };

    static uint32_t GetSplitkFactorRaw(uint32_t m, uint32_t n, uint32_t k, uint32_t aicCoreNum){
        uint32_t maxSplitkFactor;
        if (k <= 1024){
            // When k is less than or equal to 1024, it can be divided into at most 2 parts.
            // k 小于等于 1024 时，最多分成两份
            maxSplitkFactor = 2;
        } else if (k <= 2048){
            // When k is less than or equal to 2048, it can be divided into at most 4 parts.
            // k 小于2048时最多分为4份
            maxSplitkFactor = 4;
        } else if (k <= 4096){
            // When k is less than or equal to 4096, it can be divided into at most 8 parts.
            // k 小于2048时最多分为8份
            maxSplitkFactor = 8;
        } else {
            // else it can be divided into at most 16 parts.
            // 在K较大的矩阵下也最多分为16份
            maxSplitkFactor = 16;
        }

        uint32_t splitkFactor = 1;

        // 每个L1Tile的规模，为调度的基本单位
        uint32_t m0 = L1TileShape::M;
        uint32_t n0 = L1TileShape::N;
        uint32_t k0 = L1TileShape::K;

        // 每个K Slice内执行的Block的数量
        uint32_t baseTilesCount = CeilDiv(m, m0) * CeilDiv(n, n0);
        // 对于小矩阵而言，如果一个K Slice对应的Block小于aicCore的数量，那么为了最大化并行度，让所有的核均执行一部分K Slice是最好的选择
        // 所以，让全部的核均去分摊一个Slice，但最大不超过SplitkFactor的上限
        splitkFactor = std::min(aicCoreNum / baseTilesCount, maxSplitkFactor);
        // 然而当矩阵规模较大，aicCoreNum小于一个Slice上Block的数量时，要防止splitkFactor小于1：
        // Prevent the split factor form being less than 1
        splitkFactor = std::max(splitkFactor, static_cast<uint32_t>(1));
        if(baseTilesCount < aicCoreNum){
            // 对于小矩阵而言，如果不对K进行划分无法填满所有的AIC Core，那么分K来分摊进行，目标是最大化并行度：
            while (splitkFactor + 1 <= maxSplitkFactor &&
                CeilDiv(baseTilesCount * splitkFactor, aicCoreNum) >=
                CeilDiv(baseTilesCount, aicCoreNum) * splitkFactor) {
                // (1) 不能超过最大的分配数量
                // (2) 若划分可以增加实际执行的轮数，即使得核间负载的均衡度上升的话，则增加划分的数量
                splitkFactor += 1;
            }
        }
        // 此外，需要保证划分后每次执行的Slice在K的维度上要比单个L1Tile中K的大小大，防止增加过多迭代，出现碎片化且无法对齐等问题
        // Ensure that splitkFactor is less than the number of base tiels in the k direction.
        splitkFactor = std::min(CeilDiv(k, k0), splitkFactor);
        // 最后，上面的切分都是发生在M，N较小的情况下的，当M,N较大的情况下，为了提升cache的利用率，增大平行度，也要对K进行切分
        // If k is very large, splitting k can lead to better cache utilization.
        // If k is greater than 8192.
        if (k > 8192) {
            // split the k direction into at least 2 parts.
            splitkFactor = std::max(splitkFactor, static_cast<uint32_t>(2));
        }
        // If k is greater than 32768.
        if (k > 32768) {
            // split the k direction into at least 4 parts.
            splitkFactor = std::max(splitkFactor, static_cast<uint32_t>(4));
        }
        return splitkFactor;
    }

    static uint32_t GetSplitkFactorForABFT(uint32_t k, uint32_t SliceKUnit, uint32_t SplitKNumLimit)
    {
        uint32_t maxSplitkFactor;

        if (k <= 1024) {
            // When k is less than or equal to 1024, it can be divided into at most 2 parts.
            maxSplitkFactor = 1;
        } else if (k <= 2048) {
            // When k is less than or equal to 2048, it can be divided into at most 4 parts.
            maxSplitkFactor = 2;
        } else if (k <= 4096) {
            // When k is less than or equal to 4096, it can be divided into at most 8 parts.
            maxSplitkFactor = 4;
        } else {
            // else it can be divided into at most 16 parts.
            maxSplitkFactor = 8;
        }

        maxSplitkFactor = (maxSplitkFactor <= SplitKNumLimit) ? maxSplitkFactor : SplitKNumLimit;

        maxSplitkFactor = (maxSplitkFactor <= 1) ? 1 : maxSplitkFactor;

        uint32_t splitkFactor = 1;

        if(k <= SliceKUnit){

            splitkFactor = 1;

        }else{

            uint32_t TileKNumUpper = ((SliceKUnit + L1TileShape::K - 1) / L1TileShape::K);
            uint32_t SliceKUnitRound = TileKNumUpper * L1TileShape::K;
            uint32_t SliceKUnitHalf = (SliceKUnitRound + 1) / 2;
            uint32_t SliceKNumDown = k / SliceKUnitRound;
            uint32_t SliceKRemain = k % SliceKUnitRound;
            if(SliceKNumDown < 1){
                splitkFactor = 1;
            }else if(SliceKRemain <= SliceKUnitHalf){
                splitkFactor = SliceKNumDown;
            }else{
                splitkFactor = SliceKNumDown + 1;
            }

            splitkFactor = (splitkFactor <= maxSplitkFactor) ? splitkFactor : maxSplitkFactor;

        }

        return splitkFactor;
    }

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        /*
        GetSplitkFactorForABFT(uint32_t k, uint32_t SliceKUnit, uint32_t SplitKNumLimit)
        */
        return args.workspaceElementSize * args.problemGemmShape.m() * args.problemGemmShape.n() *
            GetSplitkFactorForABFT(args.problemGemmShape.k(), args.SliceKUnit, args.SplitKNumLimit);
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        LayoutA layoutA{args.problemGemmShape.m(), args.problemGemmShape.k()};
        LayoutB layoutB{args.problemGemmShape.k(), args.problemGemmShape.n()};
        LayoutC layoutC{args.problemGemmShape.m(), args.problemGemmShape.n()};
        /*
        GetSplitkFactor(args.problemShape.m(),
                args.problemShape.n(),
                args.problemShape.k(),
                args.aicCoreNum)
        */
        uint32_t SplitKNum = GetSplitkFactorForABFT(args.problemGemmShape.k(), args.SliceKUnit, args.SplitKNumLimit);
        uint32_t SplitKNumLimit = args.SplitKNumLimit;

        uint32_t actualKSliceSize[3];

        uint32_t TileKNum = (args.problemGemmShape.k() + L1TileShape::K - 1) / L1TileShape::K;
        if(TileKNum % SplitKNum == 0){
            actualKSliceSize[0] = (TileKNum / SplitKNum) * L1TileShape::K;
            actualKSliceSize[1] = (TileKNum / SplitKNum) * L1TileShape::K;
        }else{
            actualKSliceSize[0] = (TileKNum / SplitKNum + 1) * L1TileShape::K;
            actualKSliceSize[1] = (TileKNum / SplitKNum) * L1TileShape::K;
        }
        actualKSliceSize[2] = args.problemGemmShape.k() - (((TileKNum % SplitKNum) + (TileKNum / SplitKNum) * (SplitKNum - 1)) * L1TileShape::K);
        /*
        Params(Catlass::GemmCoord const &problemGemmShape_,
            GM_ADDR ptrA_, LayoutA layoutA_, 
            GM_ADDR ptrB_, LayoutB layoutB_,
            GM_ADDR ptrC_, LayoutB layoutC_,
            GM_ADDR ptrWorkspace_, 
            uint32_t SplitKNum_, uint32_t SplitKNumLimit_,
            const uint32_t (&actualKSliceSize_)[3]
        ) : problemGemmShape(problemGemmShape_), ptrA(ptrA_), layoutA(layoutA_), 
            ptrB(ptrB_), layoutB(layoutB_),
            ptrC(ptrC_), layoutC(layoutC_), 
            ptrWorkspace(ptrWorkspace_), 
            SplitKNum(SplitKNum_), SplitKNumLimit(SplitKNumLimit_) 
        */
        Params params{
            args.problemGemmShape,
            args.ptrA, layoutA,
            args.ptrB, layoutB,
            args.ptrC, layoutC,
            workspace,
            SplitKNum, SplitKNumLimit,
            actualKSliceSize
        };

        // printf("SplitKNum: %d\n", params.SplitKNum);
        // printf("SplitKNumLimit: %d\n", params.SplitKNumLimit);
        return params;
    }

    // Methods
    CATLASS_DEVICE
    MatmulAsVarSplitK() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    /// Executes one Matmul
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        BlockScheduler matmulBlockScheduler(params.problemGemmShape,
            Catlass::GemmCoord(L1TileShape::M, L1TileShape::N, L1TileShape::K), params.SplitKNum);
        
        uint32_t coreLoops  = matmulBlockScheduler.GetCoreLoops();

        // Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);

        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);

        // 若进行分K，则需要在workspace中存放数据，否则不需要进行最后的累加
        AscendC::GlobalTensor<ElementC> gmC;
        if(params.SplitKNum >= 2){
            gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrWorkspace);
        }else{
            gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);
        }

        uint64_t MatrixElementSize = static_cast<uint64_t>(params.problemGemmShape.m()) * static_cast<uint64_t>(params.problemGemmShape.n());

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            // Compute block location
            uint32_t KSliceIdx = matmulBlockScheduler.GetSplitkSliceIdx(loopIdx);
            Catlass::GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            Catlass::GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(
                blockCoord, KSliceIdx);

            // Compute initial location in logical coordinates
            Catlass::MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
            Catlass::MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
            Catlass::MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};

            uint64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
            uint64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
            uint64_t gmOffsetC = params.layoutC.GetOffset(offsetC) + MatrixElementSize * static_cast<uint64_t>(KSliceIdx);
            
            bool isFirstBlock = (loopIdx == AscendC::GetBlockIdx());
            // 
            bool hasNextBlock = false;
            uint64_t gmOffsetNextA = gmOffsetA;
            uint64_t gmOffsetNextB = gmOffsetB;
            uint64_t gmOffsetNextC = gmOffsetC;
            Catlass::GemmCoord nextActualBlockShape = Catlass::GemmCoord{actualBlockShape.m(), actualBlockShape.n(), actualBlockShape.k()};
            uint32_t loopIdxNext = loopIdx + AscendC::GetBlockNum();

            if(loopIdxNext < coreLoops){
                hasNextBlock = true;
                uint32_t KSliceIdxNext = matmulBlockScheduler.GetSplitkSliceIdx(loopIdxNext);
                Catlass::GemmCoord blockCoordNext = matmulBlockScheduler.GetBlockCoord(loopIdxNext);
                Catlass::GemmCoord nextActualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoordNext, KSliceIdxNext);

                // Compute initial location in logical coordinates
                Catlass::MatrixCoord offsetNextA{blockCoordNext.m() * L1TileShape::M, blockCoordNext.k() * L1TileShape::K};
                Catlass::MatrixCoord offsetNextB{blockCoordNext.k() * L1TileShape::K, blockCoordNext.n() * L1TileShape::N};
                Catlass::MatrixCoord OffsetNextC{blockCoordNext.m() * L1TileShape::M, blockCoordNext.n() * L1TileShape::N};

                gmOffsetNextA = params.layoutA.GetOffset(offsetNextA);
                gmOffsetNextB = params.layoutB.GetOffset(offsetNextB);
                gmOffsetNextC = params.layoutC.GetOffset(OffsetNextC) + MatrixElementSize * static_cast<uint64_t>(KSliceIdxNext);
            }

            /*
                /// Perform a block-scoped matrix multiply-accumulate
                CATLASS_DEVICE
                void operator()(
                    AscendC::GlobalTensor<ElementA> const & gmA, 
                    AscendC::GlobalTensor<ElementA> const & gmNextA,
                    LayoutA const &layoutA,
                    AscendC::GlobalTensor<ElementB> const & gmB, 
                    AscendC::GlobalTensor<ElementB> const & gmNextB,
                    LayoutB const &layoutB,
                    AscendC::GlobalTensor<ElementC> const & gmC, LayoutC const &layoutC,
                    Catlass::GemmCoord const &actualShape, Catlass::GemmCoord const &actualShapeNext,
                    bool isFirstBlock, bool hasNextBlock
            */
            // Compute block-scoped matrix multiply-add
            // blockMmad(gmA[gmOffsetA], params.layoutA,
            //           gmB[gmOffsetB], params.layoutB,
            //           gmC[gmOffsetC], params.layoutC,
            //           actualBlockShape);

            blockMmad(gmA[gmOffsetA], gmA[gmOffsetNextA], params.layoutA,
                gmB[gmOffsetB], gmB[gmOffsetNextB], params.layoutB,
                gmC[gmOffsetC], params.layoutC, 
                actualBlockShape, nextActualBlockShape, isFirstBlock, hasNextBlock);
        }

        Catlass::Arch::CrossCoreBarrierAIC<0x0, PIPE_FIX>();

        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);

        AscendC::PipeBarrier<PIPE_ALL>();    
    }

    CATLASS_DEVICE
    void Reduce_Add_on_AIV(Params const &params)
    {
        AscendC::SetAtomicNone();
        // Arch::Resource<ArchTag> resource;

        // Represent the full gm

        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t aivNum = aicoreNum * AscendC::GetTaskRation();

        // BlockScheduler matmulBlockScheduler(params.problemGemmShape, Catlass::MakeCoord(L1TileShape::M,L1TileShape::N));
        // uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();
        // uint32_t aivNum = aicoreNum * AscendC::GetSubBlockNum();
        // AscendC::printf("%zu\n",AscendC::GetBlockNum());
        uint32_t aivIndex = AscendC::GetBlockIdx();
        uint32_t aicoreIndex = aivIndex / AscendC::GetSubBlockNum();

        uint32_t align = Catlass::BYTE_PER_C0 / sizeof(ElementC);
        // uint32_t aicoreIndex = aivIndex / AscendC::GetTaskRation();

        AscendC::GlobalTensor<ElementC> gmCout;
        gmCout.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);
        AscendC::GlobalTensor<ElementC> gmCInSlice;
        gmCInSlice.SetGlobalBuffer((__gm__ ElementC *)params.ptrWorkspace);

        // Get aicore information
        uint32_t UBTileSizeRound = SliceReduceUBTileShape::M * SliceReduceUBTileShape::N;
        UBTileSizeRound = RoundUp(UBTileSizeRound, align);

        uint32_t UBBlockSizeRound =  SliceReduceUBBlockShape::M * SliceReduceUBBlockShape::N;
        UBBlockSizeRound = RoundUp(UBBlockSizeRound, align);

        uint32_t MatrixSize = params.problemGemmShape.m() * params.problemGemmShape.n();
        
        uint32_t loopsNum = CeilDiv(MatrixSize, UBBlockSizeRound);
        
        BlockReduceAdd blockSliceRedAdd(resource);

        int64_t OffsetInSliceInit = 0;

        LayoutVY layoutRedOut{MatrixSize};
        LayoutVA layoutSliceIn{MatrixSize};

        uint32_t tail_remain = (MatrixSize % align);

        if(tail_remain < 1){
            for(uint32_t loopId = aivIndex; loopId < loopsNum; loopId += aivNum) {

                bool isFirstBlock = (loopId == aivIndex);
                bool hasNextBlock = false;
            
                uint32_t len_actual = (loopId == (loopsNum - 1)) ?
                        (MatrixSize - loopId * UBBlockSizeRound) : UBBlockSizeRound;

                // Catlass::MatrixCoord offsetSliceIn{loopIdM * UBTileMRound, loopIdN * UBTileNRound};
                // Catlass::MatrixCoord offsetABEOut{loopIdM * UBTileMRound, loopIdN * UBTileNRound};

                int64_t OffsetSliceFirst = static_cast<int64_t>(loopId) * static_cast<int64_t>(UBBlockSizeRound); 
                int64_t gmOffsetInSlice = OffsetInSliceInit + OffsetSliceFirst; 

                uint32_t actual_data_num = len_actual;
                
                uint32_t loopIdNext = loopId + aivNum;
                uint32_t actual_data_num_next = actual_data_num;

                int64_t OffsetSliceFirstNext = OffsetSliceFirst;
                int64_t gmOffsetInSliceNext = gmOffsetInSlice;
                if(loopIdNext < loopsNum){
                    hasNextBlock = true;
                    uint32_t len_actual_next = (loopIdNext == (loopsNum - 1)) ? (MatrixSize - loopIdNext * UBBlockSizeRound) : UBBlockSizeRound;

                    OffsetSliceFirstNext = static_cast<int64_t>(loopIdNext) * static_cast<int64_t>(UBBlockSizeRound);
                    gmOffsetInSliceNext = OffsetInSliceInit + OffsetSliceFirstNext;

                    actual_data_num_next = len_actual_next;
                }

                /*
                CATLASS_DEVICE
                void operator()(
                    AscendC::GlobalTensor<ElementA> const &gmA,
                    AscendC::GlobalTensor<ElementA> const &gmNextBlockA,
                    LayoutVA const &layoutVA, LayoutA const &layoutA
                    AscendC::GlobalTensor<ElementY> const &gmY,
                    LayoutVY const &layoutVY,
                    uint32_t actual_data_num, uint32_t actual_data_num_next,
                    uint32_t SplitKNum, bool isFirstBlock, bool hasNextBlock)
                */

                blockSliceRedAdd(gmCInSlice[gmOffsetInSlice], 
                    gmCInSlice[gmOffsetInSliceNext],
                    layoutSliceIn, params.layoutC,
                    gmCout[gmOffsetInSlice], layoutRedOut,
                    actual_data_num, actual_data_num_next,
                    params.SplitKNum, isFirstBlock, hasNextBlock);
            
            }
        }else{

            for(uint32_t loopId = aivIndex; loopId < loopsNum; loopId += aivNum) {

                bool isFirstBlock = (loopId == aivIndex);
                bool hasNextBlock = false;

                /*
                (loopId == (loopsNum - 2)) ?
                        (MatrixSize - loopId * UBBlockSizeRound) : 
                */
                uint32_t len_actual = (loopId == (loopsNum - 1)) ?
                        (MatrixSize - loopId * UBBlockSizeRound) : UBBlockSizeRound;

                // Catlass::MatrixCoord offsetSliceIn{loopIdM * UBTileMRound, loopIdN * UBTileNRound};
                // Catlass::MatrixCoord offsetABEOut{loopIdM * UBTileMRound, loopIdN * UBTileNRound};

                int64_t OffsetSliceFirst = static_cast<int64_t>(loopId) * static_cast<int64_t>(UBBlockSizeRound); 
                int64_t gmOffsetInSlice = OffsetInSliceInit + OffsetSliceFirst; 

                uint32_t actual_data_num = len_actual;
                
                uint32_t loopIdNext = loopId + aivNum;
                uint32_t actual_data_num_next = actual_data_num;

                int64_t OffsetSliceFirstNext = OffsetSliceFirst;
                int64_t gmOffsetInSliceNext = gmOffsetInSlice;
                if(loopIdNext < (loopsNum-1)){
                    hasNextBlock = true;
                    uint32_t len_actual_next = UBBlockSizeRound;

                    /*
                    (loopIdNext == (loopsNum - 1)) ? (MatrixSize - loopIdNext * UBBlockSizeRound) : 
                    */

                    OffsetSliceFirstNext = static_cast<int64_t>(loopIdNext) * static_cast<int64_t>(UBBlockSizeRound);
                    gmOffsetInSliceNext = OffsetInSliceInit + OffsetSliceFirstNext;

                    actual_data_num_next = len_actual_next;
                }

                if(loopId < (loopsNum-1)){
                    /*
                    CATLASS_DEVICE
                    void operator()(
                        AscendC::GlobalTensor<ElementA> const &gmA,
                        AscendC::GlobalTensor<ElementA> const &gmNextBlockA,
                        LayoutVA const &layoutVA, LayoutA const &layoutA
                        AscendC::GlobalTensor<ElementY> const &gmY,
                        LayoutVY const &layoutVY,
                        uint32_t actual_data_num, uint32_t actual_data_num_next,
                        uint32_t SplitKNum, bool isFirstBlock, bool hasNextBlock)
                    */

                    blockSliceRedAdd(gmCInSlice[gmOffsetInSlice], 
                        gmCInSlice[gmOffsetInSliceNext],
                        layoutSliceIn, params.layoutC,
                        gmCout[gmOffsetInSlice], layoutRedOut,
                        actual_data_num, actual_data_num_next,
                        params.SplitKNum, isFirstBlock, hasNextBlock);  
                }else{
                    /*
                    CATLASS_DEVICE
                    void op_with_tail(
                        AscendC::GlobalTensor<ElementA> const &gmA,
                        LayoutVA const &layoutVA, LayoutA const &layoutA
                        AscendC::GlobalTensor<ElementY> const &gmY,
                        LayoutVY const &layoutVY,
                        uint32_t actual_data_num, uint32_t SplitKNum)
                    */
                    blockSliceRedAdd.op_with_tail(
                        gmCInSlice[gmOffsetInSlice],
                        layoutSliceIn, params.layoutC,
                        gmCout[gmOffsetInSlice], layoutRedOut, actual_data_num,
                        params.SplitKNum);
                }
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params){

        Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);
        if(params.SplitKNum >= 2){
            Reduce_Add_on_AIV(params);
        }
        
        Catlass::Arch::CrossCoreBarrierAIV<0x0, PIPE_V>();
        Catlass::Arch::CrossCoreBarrierAIV<0x0, PIPE_MTE3>();
    }

private:
    // ID used for inter-core synchronization
    static constexpr Catlass::Arch::FlagID FLAG_AIC_FINISH_STORE = 0;
    static constexpr Catlass::Arch::FlagID RV_FLAG_AIC_FINISH_STORE = 1;
    Catlass::Arch::CrossCoreFlagWithReverse<> flagAicFinishStore{FLAG_AIC_FINISH_STORE,RV_FLAG_AIC_FINISH_STORE};
    Catlass::Arch::Resource<ArchTag> resource;
};

} // namespace CubeSelf::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_MATMUL_EPILOGUE_SPLIT_K_SELF_HPP