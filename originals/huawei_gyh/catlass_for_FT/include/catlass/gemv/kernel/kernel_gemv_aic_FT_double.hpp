#ifndef CATLASS_GEMV_KERNLE_GEMV_AIC_FT_DOUBLE_HPP
#define CATLASS_GEMV_KERNLE_GEMV_AIC_FT_DOUBLE_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemv/helper.hpp"
#include "catlass/gemv_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Gemv::Kernel{

// tmeplate for gemv kernle, Compute z = αAx + βy
template<
    class BlockGemv_,
    class BlockEpilogue_
>
class KernelGemvFTDoubleAic {
public:
    using BlockGemv = BlockGemv_;
    using ArchTag = typename BlockGemv::ArchTag;
    using L1TileShape = typename BlockGemv::L1TileShape;
    using L0TileShape = typename BlockGemv::L0TileShape;

    using ElementX = typename BlockGemv::ElementX;
    using LayoutX = typename BlockGemv::LayoutX;

    using ElementA = typename BlockGemv::ElementA;
    using LayoutA = typename BlockGemv::LayoutA;

    using LayoutACol = typename std::conditional<
        std::is_same<LayoutA, Catlass::layout::RowMajor>::value,
        Catlass::layout::ColumnMajor,
        Catlass::layout::RowMajor>::type;
    
    using AColType = Gemm::GemmType<ElementA, LayoutACol>;

    using ElementY = typename BlockGemv::ElementY;
    using LayoutY = typename BlockGemv::LayoutY;

    using BlockEpilogue = BlockEpilogue_;

    // using ElementZ = typename BlockEpilogue::ElementZ;
    using LayoutZ = Catlass::layout::VectorLayout;
    // typename BlockEpilogue::LayoutZ;
    // using EpilogueParams = typename BlockEpilogue::Params;

    using FT_ENC_TYPE = Gemv::helper::FT_ENC_TYPE;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;
    
    struct Params {
        // Data members
        GemvCoord problemShape;
        GemvCoord problemShapeCol;
        GM_ADDR ptrX;
        LayoutX layoutX;
        GM_ADDR ptrA;
        LayoutA layoutA;
        LayoutACol layoutACol;
        GM_ADDR ptrWorkspace;
        // EpilogueParams epilogueParamsRow;
        // EpilogueParams epilogueParamsCol;
        FT_ENC_TYPE enc_type;
        GM_ADDR ptrZRow;
        GM_ADDR ptrZCol;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}
        /*
        EpilogueParams const &epilogueParamsRow_, 
        EpilogueParams const &epilogueParamsCol_,

        epilogueParamsRow(epilogueParamsRow_),
        epilogueParamsCol(epilogueParamsCol_),
        */
        CATLASS_HOST_DEVICE
        Params(GemvCoord const &problemShape_,GemvCoord const &problemShapeCol_, 
            GM_ADDR ptrX_, LayoutX layoutX_, 
            GM_ADDR ptrA_, LayoutA layoutA_, LayoutACol layoutACol_,
            GM_ADDR ptrWorkspace_, 
            FT_ENC_TYPE enc_type_, GM_ADDR ptrZRow_, GM_ADDR ptrZCol_)
            : problemShape(problemShape_), problemShapeCol(problemShapeCol_),
              ptrX(ptrX_), layoutX(layoutX_), ptrA(ptrA_),
              layoutA(layoutA_), layoutACol(layoutACol_),
              ptrWorkspace(ptrWorkspace_), 
              enc_type(enc_type_), ptrZRow(ptrZRow_), ptrZCol(ptrZCol_) {}
    };

    struct Arguments {
        GemvCoord problemShape;
        // ElementY alpha;
        // ElementY beta;
        size_t elementSize;
        GM_ADDR ptrX;
        GM_ADDR ptrA;
        GM_ADDR ptrZRow;
        GM_ADDR ptrZCol;
        FT_ENC_TYPE enc_type;
    };

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        // if(args.enc_type == FT_ENC_TYPE::ETC){
        //     return args.elementSize * args.problemShape.n();
        // } else if(args.enc_type == FT_ENC_TYPE::CE){
        //     return args.elementSize * args.problemShape.m();
        // } else{
        //     return args.elementSize * (args.problemShape.m() + args.problemShape.n());
        // }
        return args.elementSize * (args.problemShape.m() + args.problemShape.n());
        
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        GemvCoord problemShape = args.problemShape;
        uint32_t m = problemShape.m();
        uint32_t n = problemShape.n();

        GemvCoord problemShapeCol{n, m};
        problemShapeCol.m() = n;
        problemShapeCol.n() = m;

        // LayoutX layoutX{n};
        // LayoutZ layoutZ{m};
        LayoutA layoutA{m, n};
        LayoutACol layoutACol{n, m};

        LayoutX layoutX{m+n};

        LayoutZ layoutZRow{m};
        // typename BlockEpilogue::Params epilogueParamsRow{args.alpha, args.beta, args.ptrZRow, layoutZRow, args.ptrZRow, layoutZRow};

        LayoutZ layoutZCol{n};
        // typename BlockEpilogue::Params epilogueParamsCol{args.alpha, args.beta, args.ptrZCol, layoutZCol, args.ptrZCol, layoutZCol};

        Params params{problemShape, problemShapeCol, 
            args.ptrX, layoutX, args.ptrA, layoutA,
            layoutACol,
            workspace, 
            args.enc_type, args.ptrZRow, args.ptrZCol};
        
        // epilogueParamsRow, 
        // epilogueParamsCol,

        return params;
    }

    // Methods
    CATLASS_DEVICE
    KernelGemvFTDoubleAic() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const& params);

    CATLASS_DEVICE
    void canonical_op(Params const& params)
    {
        BlockGemv blockGemv(resource);
        // Represent the full gm
        AscendC::GlobalTensor<ElementX> gmX;
        gmX.SetGlobalBuffer((__gm__ ElementX *)params.ptrX);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);

        AscendC::GlobalTensor<ElementY> gmY;
        gmY.SetGlobalBuffer((__gm__ ElementY *)params.ptrZRow);

        layout::RowMajor layoutY(1, params.problemShape.m());

        uint32_t maxMPerBlock = L1TileShape::M;
        uint32_t maxNPerBlock = L1TileShape::N;
        uint32_t M = params.problemShape.m();
        uint32_t N = params.problemShape.n();
        LayoutX layoutX{N};

        uint32_t MLoops = CeilDiv(M, maxMPerBlock);
        uint32_t coreLoops = MLoops;
        uint32_t singleIdx = 0;

        static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
        static constexpr uint32_t L0C_TILE_SIZE = L0TileShape::M * L0TileShape::N;
        static constexpr uint32_t L0C_TILE_NUM = L0C_SIZE / L0C_TILE_SIZE / sizeof(ElementAccumulator);

        #pragma unroll
        for (uint32_t i = 0; i < L0C_TILE_NUM; i++) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>((event_t)i);
        }

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            // Compute Block location
            uint32_t MGmBlockIdx = loopIdx;
            uint32_t MGmActual = (MGmBlockIdx == MLoops - 1) ? (M - MGmBlockIdx * maxMPerBlock) : maxMPerBlock;
            uint32_t NGmActual = N;
            int64_t gmOffsetX;
            int64_t gmOffsetA;
            int64_t gmOffsetY;
            int64_t gmOffsetNextX;
            int64_t gmOffsetNextA;
            int64_t gmOffsetNextY;

            if constexpr (std::is_same<LayoutA, Catlass::layout::RowMajor>::value) {
                gmOffsetX = 0;
                gmOffsetA = MGmBlockIdx * maxMPerBlock * params.layoutA.stride(0);

                gmOffsetY = MGmBlockIdx * maxMPerBlock;
            } else {
                gmOffsetX = 0;
                gmOffsetA = MGmBlockIdx * maxMPerBlock;
                gmOffsetY = MGmBlockIdx * maxMPerBlock;
            }

            bool isFirstBlock = (loopIdx == AscendC::GetBlockIdx());
            bool hasNextBlock = false;
            uint32_t MNextGmBlockIdx;
            GemvCoord nextActualBlockShape;
            if (loopIdx + AscendC::GetBlockNum() < coreLoops) {
                hasNextBlock = true;
                uint32_t nextLoopIdx = loopIdx + AscendC::GetBlockNum();
                MNextGmBlockIdx = nextLoopIdx;
                uint32_t MNextGmActual =
                    (MNextGmBlockIdx == MLoops - 1) ? (M - MNextGmBlockIdx * maxMPerBlock) : maxMPerBlock;
                uint32_t NNextGmActual = N;
                nextActualBlockShape = GemvCoord(MNextGmActual, NNextGmActual);
            }

            if constexpr (std::is_same<LayoutA, Catlass::layout::RowMajor>::value) {
                gmOffsetNextX = 0;
                gmOffsetNextA = MNextGmBlockIdx * maxMPerBlock * params.layoutA.stride(0);

                gmOffsetNextY = MNextGmBlockIdx * maxMPerBlock;
            } else {
                gmOffsetNextX = 0;
                gmOffsetNextA = MNextGmBlockIdx * maxMPerBlock;
                gmOffsetNextY = MNextGmBlockIdx * maxMPerBlock;
            }

            GemvCoord actualBlockShape = GemvCoord(MGmActual, NGmActual);

            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((event_t)singleIdx % L0C_TILE_NUM);

            
            // Compute block-scoped matrix multiply-add
            blockGemv(gmX[gmOffsetX], layoutX,
                      gmA[gmOffsetA], params.layoutA,
                      gmY[gmOffsetY], layoutY,
                      gmX[gmOffsetNextX],
                      gmA[gmOffsetNextA],
                      actualBlockShape,
                      nextActualBlockShape,
                      isFirstBlock,
                      hasNextBlock,
                      singleIdx);

            Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>((event_t)singleIdx % L0C_TILE_NUM);

            singleIdx++;
        }

        #pragma unroll
        for (uint32_t i = 0; i < L0C_TILE_NUM; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((event_t)i);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_DEVICE
    void CE_op(Params const& params)
    {
        BlockGemv blockGemv(resource);
        // Represent the full gm
        AscendC::GlobalTensor<ElementX> gmX;
        gmX.SetGlobalBuffer((__gm__ ElementX *)params.ptrX);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);

        AscendC::GlobalTensor<ElementY> gmY;
        gmY.SetGlobalBuffer((__gm__ ElementY *)params.ptrZRow);

        layout::RowMajor layoutY(1, params.problemShape.m());

        uint32_t maxMPerBlock = L1TileShape::M;
        uint32_t maxNPerBlock = L1TileShape::N;
        uint32_t M = params.problemShape.m();
        uint32_t N = params.problemShape.n();
        LayoutX layoutX{N};

        uint32_t MLoops = CeilDiv(M, maxMPerBlock);
        uint32_t coreLoops = MLoops;
        uint32_t singleIdx = 0;

        static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
        static constexpr uint32_t L0C_TILE_SIZE = L0TileShape::M * L0TileShape::N;
        static constexpr uint32_t L0C_TILE_NUM = L0C_SIZE / L0C_TILE_SIZE / sizeof(ElementAccumulator);

        #pragma unroll
        for (uint32_t i = 0; i < L0C_TILE_NUM; i++) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>((event_t)i);
        }

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            // Compute Block location
            uint32_t MGmBlockIdx = loopIdx;
            uint32_t MGmActual = (MGmBlockIdx == MLoops - 1) ? (M - MGmBlockIdx * maxMPerBlock) : maxMPerBlock;
            uint32_t NGmActual = N;
            int64_t gmOffsetX;
            int64_t gmOffsetA;
            int64_t gmOffsetY;
            int64_t gmOffsetNextX;
            int64_t gmOffsetNextA;
            int64_t gmOffsetNextY;

            if constexpr (std::is_same<LayoutA, Catlass::layout::RowMajor>::value) {
                gmOffsetX = 0;
                gmOffsetA = MGmBlockIdx * maxMPerBlock * params.layoutA.stride(0);

                gmOffsetY = MGmBlockIdx * maxMPerBlock;
            } else {
                gmOffsetX = 0;
                gmOffsetA = MGmBlockIdx * maxMPerBlock;
                gmOffsetY = MGmBlockIdx * maxMPerBlock;
            }

            bool isFirstBlock = (loopIdx == AscendC::GetBlockIdx());
            bool hasNextBlock = false;
            uint32_t MNextGmBlockIdx;
            GemvCoord nextActualBlockShape;
            if (loopIdx + AscendC::GetBlockNum() < coreLoops) {
                hasNextBlock = true;
                uint32_t nextLoopIdx = loopIdx + AscendC::GetBlockNum();
                MNextGmBlockIdx = nextLoopIdx;
                uint32_t MNextGmActual =
                    (MNextGmBlockIdx == MLoops - 1) ? (M - MNextGmBlockIdx * maxMPerBlock) : maxMPerBlock;
                uint32_t NNextGmActual = N;
                nextActualBlockShape = GemvCoord(MNextGmActual, NNextGmActual);
            }

            if constexpr (std::is_same<LayoutA, Catlass::layout::RowMajor>::value) {
                gmOffsetNextX = 0;
                gmOffsetNextA = MNextGmBlockIdx * maxMPerBlock * params.layoutA.stride(0);

                gmOffsetNextY = MNextGmBlockIdx * maxMPerBlock;
            } else {
                gmOffsetNextX = 0;
                gmOffsetNextA = MNextGmBlockIdx * maxMPerBlock;
                gmOffsetNextY = MNextGmBlockIdx * maxMPerBlock;
            }

            GemvCoord actualBlockShape = GemvCoord(MGmActual, NGmActual);

            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((event_t)singleIdx % L0C_TILE_NUM);

            // Compute block-scoped matrix multiply-add
            blockGemv.RowSum(gmX[gmOffsetX], layoutX,
                      gmA[gmOffsetA], params.layoutA,
                      gmY[gmOffsetY], layoutY,
                      gmX[gmOffsetNextX],
                      gmA[gmOffsetNextA],
                      actualBlockShape,
                      nextActualBlockShape,
                      isFirstBlock,
                      hasNextBlock,
                      singleIdx);

            Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>((event_t)singleIdx % L0C_TILE_NUM);

            singleIdx++;
        }

        #pragma unroll
        for (uint32_t i = 0; i < L0C_TILE_NUM; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((event_t)i);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // CATLASS_DEVICE
    // void CE_op(Params const& params)
    // {
    //     BlockGemv blockGemv(resource);
    //     // Represent the full gm
    //     // AscendC::GlobalTensor<ElementX> gmX;
    //     // gmX.SetGlobalBuffer((__gm__ ElementX *)params.ptrX);

    //     AscendC::GlobalTensor<ElementA> gmA;
    //     gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);

    //     AscendC::GlobalTensor<ElementY> gmY;
    //     gmY.SetGlobalBuffer((__gm__ ElementY *)params.ptrWorkspace);

    //     layout::RowMajor layoutY(1, params.problemShape.m());

    //     uint32_t maxMPerBlock = L1TileShape::M;
    //     uint32_t maxNPerBlock = L1TileShape::N;
    //     uint32_t M = params.problemShape.m();
    //     uint32_t N = params.problemShape.n();

    //     uint32_t MLoops = CeilDiv(M, maxMPerBlock);
    //     uint32_t coreLoops = MLoops;
    //     uint32_t singleIdx = 0;

    //     static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
    //     static constexpr uint32_t L0C_TILE_SIZE = L0TileShape::M * L0TileShape::N;
    //     static constexpr uint32_t L0C_TILE_NUM = L0C_SIZE / L0C_TILE_SIZE / sizeof(ElementAccumulator);

    //     #pragma unroll
    //     for (uint32_t i = 0; i < L0C_TILE_NUM; i++) {
    //         AscendC::SetFlag<AscendC::HardEvent::FIX_M>((event_t)i);
    //     }

    //     blockGemv.FillScalarInLocal(AscendC::GetBlockIdx(), static_cast<ElementX>(1.0),FT_ENC_TYPE::CE);
    //     AscendC::PipeBarrier<PIPE_ALL>();

    //     for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
    //         // Compute Block location
    //         uint32_t MGmBlockIdx = loopIdx;
    //         uint32_t MGmActual = (MGmBlockIdx == MLoops - 1) ? (M - MGmBlockIdx * maxMPerBlock) : maxMPerBlock;
    //         uint32_t NGmActual = N;
    //         // int64_t gmOffsetX;
    //         int64_t gmOffsetA;
    //         int64_t gmOffsetY;
    //         // int64_t gmOffsetNextX;
    //         int64_t gmOffsetNextA;
    //         int64_t gmOffsetNextY;

    //         if constexpr (std::is_same<LayoutA, Catlass::layout::RowMajor>::value) {
    //             // gmOffsetX = 0;
    //             gmOffsetA = MGmBlockIdx * maxMPerBlock * params.layoutA.stride(0);

    //             gmOffsetY = MGmBlockIdx * maxMPerBlock;
    //         } else {
    //             // gmOffsetX = 0;
    //             gmOffsetA = MGmBlockIdx * maxMPerBlock;
    //             gmOffsetY = MGmBlockIdx * maxMPerBlock;
    //         }

    //         bool isFirstBlock = (loopIdx == AscendC::GetBlockIdx());
    //         bool hasNextBlock = false;
    //         uint32_t MNextGmBlockIdx;
    //         GemvCoord nextActualBlockShape;
    //         if (loopIdx + AscendC::GetBlockNum() < coreLoops) {
    //             hasNextBlock = true;
    //             uint32_t nextLoopIdx = loopIdx + AscendC::GetBlockNum();
    //             MNextGmBlockIdx = nextLoopIdx;
    //             uint32_t MNextGmActual =
    //                 (MNextGmBlockIdx == MLoops - 1) ? (M - MNextGmBlockIdx * maxMPerBlock) : maxMPerBlock;
    //             uint32_t NNextGmActual = N;
    //             nextActualBlockShape = GemvCoord(MNextGmActual, NNextGmActual);
    //         }

    //         if constexpr (std::is_same<LayoutA, Catlass::layout::RowMajor>::value) {
    //             // gmOffsetNextX = 0;
    //             gmOffsetNextA = MNextGmBlockIdx * maxMPerBlock * params.layoutA.stride(0);

    //             gmOffsetNextY = MNextGmBlockIdx * maxMPerBlock;
    //         } else {
    //             // gmOffsetNextX = 0;
    //             gmOffsetNextA = MNextGmBlockIdx * maxMPerBlock;
    //             gmOffsetNextY = MNextGmBlockIdx * maxMPerBlock;
    //         }

    //         GemvCoord actualBlockShape = GemvCoord(MGmActual, NGmActual);

    //         AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((event_t)singleIdx % L0C_TILE_NUM);

    //         // Compute block-scoped matrix multiply-add
    //         blockGemv.RowSum(
    //                   gmA[gmOffsetA], params.layoutA,
    //                   gmY[gmOffsetY], layoutY,
    //                   gmA[gmOffsetNextA],
    //                   actualBlockShape,
    //                   nextActualBlockShape,
    //                   isFirstBlock,
    //                   hasNextBlock,
    //                   singleIdx);
                      
    //         Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
    //         AscendC::SetFlag<AscendC::HardEvent::FIX_M>((event_t)singleIdx % L0C_TILE_NUM);

    //         singleIdx++;
    //     }

    //     #pragma unroll
    //     for (uint32_t i = 0; i < L0C_TILE_NUM; i++) {
    //         AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((event_t)i);
    //     }

    //     AscendC::PipeBarrier<PIPE_ALL>();
    // }

    CATLASS_DEVICE
    void ETC_op(Params const& params)
    {
        BlockGemv blockGemv(resource);
        // Represent the full gm
        AscendC::GlobalTensor<ElementX> gmX;
        gmX.SetGlobalBuffer((__gm__ ElementX *)params.ptrX);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);

        AscendC::GlobalTensor<ElementY> gmY;
        gmY.SetGlobalBuffer((__gm__ ElementY *)params.ptrZCol);

        // AscendC::GlobalTensor<ElementY> gmZ;
        // gmZ.SetGlobalBuffer((__gm__ ElementY *)params.ptrZCol);


        layout::RowMajor layoutY(1, params.problemShapeCol.m());
        

        uint32_t maxMPerBlock = L1TileShape::M;
        uint32_t maxNPerBlock = L1TileShape::N;

        uint32_t M = params.problemShapeCol.m();
        uint32_t N = params.problemShapeCol.n();

        LayoutX layoutX{N};

        uint32_t MLoops = CeilDiv(M, maxMPerBlock);
        uint32_t coreLoops = MLoops;
        uint32_t singleIdx = 0;

        static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
        static constexpr uint32_t L0C_TILE_SIZE = L0TileShape::M * L0TileShape::N;
        static constexpr uint32_t L0C_TILE_NUM = L0C_SIZE / L0C_TILE_SIZE / sizeof(ElementAccumulator);

        #pragma unroll
        for (uint32_t i = 0; i < L0C_TILE_NUM; i++) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>((event_t)i);
        }

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            // Compute Block location
            uint32_t MGmBlockIdx = loopIdx;
            uint32_t MGmActual = (MGmBlockIdx == MLoops - 1) ? (M - MGmBlockIdx * maxMPerBlock) : maxMPerBlock;
            uint32_t NGmActual = N;
            int64_t gmOffsetX;
            int64_t gmOffsetA;
            int64_t gmOffsetY;
            int64_t gmOffsetNextX;
            int64_t gmOffsetNextA;
            int64_t gmOffsetNextY;

            if constexpr (std::is_same<LayoutACol, Catlass::layout::RowMajor>::value) {
                gmOffsetX = 0;
                gmOffsetA = MGmBlockIdx * maxMPerBlock * params.layoutACol.stride(0);

                gmOffsetY = MGmBlockIdx * maxMPerBlock;
            } else {
                gmOffsetX = 0;
                gmOffsetA = MGmBlockIdx * maxMPerBlock;
                gmOffsetY = MGmBlockIdx * maxMPerBlock;
            }

            bool isFirstBlock = (loopIdx == AscendC::GetBlockIdx());
            bool hasNextBlock = false;
            uint32_t MNextGmBlockIdx;
            GemvCoord nextActualBlockShape;
            if (loopIdx + AscendC::GetBlockNum() < coreLoops) {
                hasNextBlock = true;
                uint32_t nextLoopIdx = loopIdx + AscendC::GetBlockNum();
                MNextGmBlockIdx = nextLoopIdx;
                uint32_t MNextGmActual =
                    (MNextGmBlockIdx == MLoops - 1) ? (M - MNextGmBlockIdx * maxMPerBlock) : maxMPerBlock;
                uint32_t NNextGmActual = N;
                nextActualBlockShape = GemvCoord(MNextGmActual, NNextGmActual);
            }

            if constexpr (std::is_same<LayoutACol, Catlass::layout::RowMajor>::value) {
                gmOffsetNextX = 0;
                gmOffsetNextA = MNextGmBlockIdx * maxMPerBlock * params.layoutACol.stride(0);

                gmOffsetNextY = MNextGmBlockIdx * maxMPerBlock;
            } else {
                gmOffsetNextX = 0;
                gmOffsetNextA = MNextGmBlockIdx * maxMPerBlock;
                gmOffsetNextY = MNextGmBlockIdx * maxMPerBlock;
            }

            GemvCoord actualBlockShape = GemvCoord(MGmActual, NGmActual);

            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((event_t)singleIdx % L0C_TILE_NUM);

            // Compute block-scoped matrix multiply-add
            // gmY[params.problemShape.m() + gmOffsetY]
            // params.problemShape.m() + 
            blockGemv.ColSum(gmX[gmOffsetX], layoutX,
                      gmA[gmOffsetA], params.layoutACol,
                      gmY[gmOffsetY], layoutY,
                      gmX[gmOffsetNextX],
                      gmA[gmOffsetNextA],
                      actualBlockShape,
                      nextActualBlockShape,
                      isFirstBlock,
                      hasNextBlock,
                      singleIdx);

            Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>((event_t)singleIdx % L0C_TILE_NUM);

            singleIdx++;
        }

        #pragma unroll
        for (uint32_t i = 0; i < L0C_TILE_NUM; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((event_t)i);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    
    
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const& params)
    {
        if(params.enc_type == FT_ENC_TYPE::NO){
            canonical_op(params);
        }else if(params.enc_type == FT_ENC_TYPE::CE) {
            CE_op(params);
        }else if(params.enc_type == FT_ENC_TYPE::ETC) {
            ETC_op(params);
        }else if(params.enc_type == FT_ENC_TYPE::BOTHC) {
            CE_op(params);
            // device 级别的同步
            AscendC::SyncAll<true>();
            ETC_op(params);
        }
    }

    CATLASS_DEVICE
    void EpilogueOp(Params const& params)
    {
        // BlockEpilogue blockEpilogue(resource, params.epilogueParamsRow);

        // Represent the full gm
        AscendC::GlobalTensor<ElementY> gmY;
        gmY.SetGlobalBuffer((__gm__ ElementY *)params.ptrWorkspace);

        layout::VectorLayout layoutY(params.problemShape.m());

        // Get aicore information
        uint32_t aicoreIndex = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t subcoreIndex = AscendC::GetSubBlockIdx();

        uint32_t maxMPerBlock = L1TileShape::M;
        uint32_t M = params.problemShape.m();
        uint32_t MLoops = CeilDiv(M, maxMPerBlock);
        uint32_t coreLoops = MLoops;

        // Loop through the epilogue calculations of each basic block
        layout::VectorLayout::TensorCoord blockShape{L1TileShape::M};

        for (uint32_t loopIdx = aicoreIndex; loopIdx < coreLoops; loopIdx += aicoreNum) {
            // Compute block location
            layout::VectorLayout::TensorCoord blockCoord{loopIdx};
            uint32_t MGmActual = (loopIdx == coreLoops) ? M - loopIdx * maxMPerBlock : maxMPerBlock;

            layout::VectorLayout::TensorCoord actualBlockShape{MGmActual};

            // Get the offset
            layout::VectorLayout::TensorCoord blockOffset = blockCoord * blockShape;

            // Get the data and layout of y under the current basic block
            auto gmBlockY = gmY[layoutY.GetOffset(blockOffset)];
            auto layoutBlockY = layoutY.GetTileLayout(actualBlockShape);

            // Synchronize cross core
            Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);

            // Actual calculatioin logic for performing block-scoped epilogue
            // blockEpilogue(blockOffset, actualBlockShape, gmBlockY, layoutBlockY);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_DEVICE
    void EpilogueCE(Params const& params)
    {
        // BlockEpilogue blockEpilogue(resource, params.epilogueParamsRow);

        // Represent the full gm
        AscendC::GlobalTensor<ElementY> gmY;
        gmY.SetGlobalBuffer((__gm__ ElementY *)params.ptrWorkspace);

        layout::VectorLayout layoutY(params.problemShape.m());

        // Get aicore information
        uint32_t aicoreIndex = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t subcoreIndex = AscendC::GetSubBlockIdx();

        uint32_t maxMPerBlock = L1TileShape::M;
        uint32_t M = params.problemShape.m();
        uint32_t MLoops = CeilDiv(M, maxMPerBlock);
        uint32_t coreLoops = MLoops;

        // Loop through the epilogue calculations of each basic block
        layout::VectorLayout::TensorCoord blockShape{L1TileShape::M};

        for (uint32_t loopIdx = aicoreIndex; loopIdx < coreLoops; loopIdx += aicoreNum) {
            // Compute block location
            layout::VectorLayout::TensorCoord blockCoord{loopIdx};
            uint32_t MGmActual = (loopIdx == coreLoops) ? M - loopIdx * maxMPerBlock : maxMPerBlock;

            layout::VectorLayout::TensorCoord actualBlockShape{MGmActual};

            // Get the offset
            layout::VectorLayout::TensorCoord blockOffset = blockCoord * blockShape;

            // Get the data and layout of y under the current basic block
            auto gmBlockY = gmY[layoutY.GetOffset(blockOffset)];
            auto layoutBlockY = layoutY.GetTileLayout(actualBlockShape);

            // Synchronize cross core
            Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);

            // Actual calculatioin logic for performing block-scoped epilogue
            // blockEpilogue(blockOffset, actualBlockShape, gmBlockY, layoutBlockY);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_DEVICE
    void EpilogueETC(Params const& params)
    {
        // BlockEpilogue blockEpilogue(resource, params.epilogueParamsCol);

        // Represent the full gm
        AscendC::GlobalTensor<ElementY> gmY;
        gmY.SetGlobalBuffer((__gm__ ElementY *)params.ptrWorkspace);

        layout::VectorLayout layoutY(params.problemShapeCol.m());

        // Get aicore information
        uint32_t aicoreIndex = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t subcoreIndex = AscendC::GetSubBlockIdx();

        uint32_t maxMPerBlock = L1TileShape::M;
        uint32_t M = params.problemShapeCol.m();
        uint32_t MLoops = CeilDiv(M, maxMPerBlock);
        uint32_t coreLoops = MLoops;

        // Loop through the epilogue calculations of each basic block
        layout::VectorLayout::TensorCoord blockShape{L1TileShape::M};

        for (uint32_t loopIdx = aicoreIndex; loopIdx < coreLoops; loopIdx += aicoreNum) {
            // Compute block location
            layout::VectorLayout::TensorCoord blockCoord{loopIdx};
            uint32_t MGmActual = (loopIdx == coreLoops) ? M - loopIdx * maxMPerBlock : maxMPerBlock;

            layout::VectorLayout::TensorCoord actualBlockShape{MGmActual};

            // Get the offset
            layout::VectorLayout::TensorCoord blockOffset = blockCoord * blockShape;

            // Get the data and layout of y under the current basic block
            auto gmBlockY = gmY[params.problemShape.m() + layoutY.GetOffset(blockOffset)];
            auto layoutBlockY = layoutY.GetTileLayout(actualBlockShape);

            // Synchronize cross core
            Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);

            // Actual calculatioin logic for performing block-scoped epilogue
            // blockEpilogue(blockOffset, actualBlockShape, gmBlockY, layoutBlockY);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // 尝试将ABE/ETAB的逻辑实现在这里，此外还要实现二者的比较，建议使用信号量驱动，写成流水线
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const& params)
    {
        if(params.enc_type == FT_ENC_TYPE::NO){
            EpilogueOp(params);
        }else if(params.enc_type == FT_ENC_TYPE::CE) {
            EpilogueCE(params);
        }else if(params.enc_type == FT_ENC_TYPE::ETC) {
            EpilogueETC(params);
        }else if(params.enc_type == FT_ENC_TYPE::BOTHC) {
            EpilogueCE(params);
            // device 级别的同步
            // AscendC::SyncAll<true>();
            Catlass::Arch::CrossCoreBarrierAIV<0x0, PIPE_V>();
            Catlass::Arch::CrossCoreBarrierAIV<0x0, PIPE_MTE3>();
            EpilogueETC(params);
        }
    }

private:
    // ID used for inter-core synchronization
    static constexpr Arch::FlagID FLAG_AIC_FINISH_STORE = 0;
    static constexpr Arch::FlagID RV_FLAG_AIC_FINISH_STORE = 1;
    Arch::CrossCoreFlagWithReverse<> flagAicFinishStore{FLAG_AIC_FINISH_STORE, RV_FLAG_AIC_FINISH_STORE};
    Arch::Resource<ArchTag> resource;

    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 0;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
};

}  // namespace Catlass::Gemv::kernel

#endif  // CATLASS_GEMV_KERNLE_GEMV_AIC_FT_HPP

