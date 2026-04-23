#ifndef CATLASS_GEMV_KERNLE_GEMV_FT_DOUBLE_HPP_TOTAL_AIV
#define CATLASS_GEMV_KERNLE_GEMV_FT_DOUBLE_HPP_TOTAL_AIV

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


template<
    class BlockSumGemv_,
    class BlockCompare_
>
class KernelGemvFTDoubleTotalAiv {
public:
    using BlockSumGemv = BlockSumGemv_;
    using BlockCompare = BlockCompare_;
    using ArchTag = typename BlockSumGemv::ArchTag;

    using ElementA = typename BlockSumGemv::ElementA;
    using LayoutA = typename BlockSumGemv::LayoutA;
    using ElementB = typename BlockSumGemv::ElementA;
    using LayoutB = typename BlockSumGemv::LayoutA;

    using LayoutACol = typename std::conditional<
        std::is_same<LayoutA, Catlass::layout::RowMajor>::value,
        Catlass::layout::ColumnMajor,
        Catlass::layout::RowMajor>::type;

    using LayoutBCol = typename std::conditional<
        std::is_same<LayoutB, Catlass::layout::RowMajor>::value,
        Catlass::layout::ColumnMajor,
        Catlass::layout::RowMajor>::type;

    using ElementX = typename BlockSumGemv::ElementX;
    using LayoutX = typename BlockSumGemv::LayoutX;

    using ElementC = typename BlockSumGemv::ElementA;
    using LayoutC = typename BlockSumGemv::LayoutA;

    using LayoutCCol = typename std::conditional<
        std::is_same<LayoutC, Catlass::layout::RowMajor>::value,
        Catlass::layout::ColumnMajor,
        Catlass::layout::RowMajor>::type;
    
    using CColType = Gemm::GemmType<ElementC, LayoutCCol>;

    using ElementZ = typename BlockSumGemv::ElementY;

    using ElementCOMPX = ElementZ;
    using LayoutCOMPX = Catlass::layout::VectorLayout;


    using ElementY = typename BlockSumGemv::ElementY;
    using LayoutY = typename BlockSumGemv::LayoutY;

    using ElementCOMPY = ElementZ;
    using LayoutCOMPY = Catlass::layout::VectorLayout;

    using UBTileShape = typename BlockSumGemv::UBTileShape;
    using UBAlignHelper = Gemv::helper::UBAlignHelper<ElementA>;
    using COMPUBTileShape = typename BlockCompare::UBTileShape;

    // using BlockEpilogue = BlockEpilogue_;
    using FT_ENC_TYPE = Gemv::helper::FT_ENC_TYPE;
    using FT_COMP_TYPE = Catlass::Gemv::helper::FT_COMP_TYPE;

    using ElementCOMPZ = typename BlockCompare::ElementZ;
    using LayoutCOMPZ = Catlass::layout::VectorLayout;

    using ElementWork = typename std::conditional<
        (BlockCompare::COMP_TYPE == FT_COMP_TYPE::XOR),
        uint16_t,
        typename std::conditional<(BlockCompare::COMP_TYPE == FT_COMP_TYPE::COMPARE), int32_t, ElementCOMPX>::type>::type;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementC, ElementX>::ElementAccumulator;
    
    // EpilogueParams epilogueParamsRow;
    //  EpilogueParams epilogueParamsCol;

    static_assert(std::is_same_v<LayoutA, LayoutB>,
        "The LayoutA and LayoutB of Gemm should be consistent.");

    enum class AivCore {
        AIV0 = 0,
        AIV1
    };

    struct Params {
        // Data members
        GemmCoord problemGemmShape;
        GemvCoord problemShape;
        GemvCoord problemShapeCol;
        GemvCoord problemCompShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        LayoutACol layoutACol;
        GM_ADDR ptrB;
        LayoutB layoutB;
        LayoutBCol layoutBCol;
        GM_ADDR ptrC;
        LayoutC layoutC;
        LayoutCCol layoutCCol;
        GM_ADDR ptrX;
        LayoutX layoutX;
        GM_ADDR ptrWorkspace;
        FT_ENC_TYPE enc_type;
        GM_ADDR ptrZRow;
        GM_ADDR ptrZCol;
        GM_ADDR ptrZRow2; 
        GM_ADDR ptrZCol2;
        GM_ADDR ptrCOMPZRow;
        LayoutCOMPZ layoutCOMPZRow;
        GM_ADDR ptrCOMPZCol;
        LayoutCOMPZ layoutCOMPZCol;
        LayoutCOMPX layoutCOMPX;
        LayoutCOMPY layoutCOMPY;
        uint32_t UbNum;
        bool OutputWorkspace;
        ElementCOMPX threshold;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemGemmShape_, GemvCoord const &problemShape_, 
            GemvCoord const &problemShapeCol_, GemvCoord const &problemCompShape_,
            GM_ADDR ptrA_, LayoutA layoutA_, LayoutACol layoutACol_,
            GM_ADDR ptrB_, LayoutB layoutB_, LayoutBCol layoutBCol_,
            GM_ADDR ptrC_, LayoutC layoutC_, LayoutCCol layoutCCol_,
            GM_ADDR ptrX_, LayoutX layoutX_, 
            GM_ADDR ptrWorkspace_, FT_ENC_TYPE enc_type_, 
            GM_ADDR ptrZRow_, GM_ADDR ptrZCol_, GM_ADDR ptrZRow2_, GM_ADDR ptrZCol2_,
            GM_ADDR ptrCOMPZRow_, LayoutCOMPZ layoutCOMPZRow_, 
            GM_ADDR ptrCOMPZCol_, LayoutCOMPZ layoutCOMPZCol_,
            LayoutCOMPX layoutCOMPX_, LayoutCOMPY layoutCOMPY_,
            uint32_t UbNum_, bool OutputWorkspace_, ElementCOMPX threshold_)
            : problemGemmShape(problemGemmShape_), problemShape(problemShape_),
              problemShapeCol(problemShapeCol_), problemCompShape(problemCompShape_),
              ptrA(ptrA_), layoutA(layoutA_), layoutACol(layoutACol_),
              ptrB(ptrB_), layoutB(layoutB_), layoutBCol(layoutBCol_), ptrC(ptrC_), layoutC(layoutC_),
              layoutCCol(layoutCCol_), ptrX(ptrX_), layoutX(layoutX_), 
              ptrWorkspace(ptrWorkspace_), enc_type(enc_type_), ptrZRow(ptrZRow_), ptrZCol(ptrZCol_),
              ptrZRow2(ptrZRow2_), ptrZCol2(ptrZCol2_), 
              ptrCOMPZRow(ptrCOMPZRow_), layoutCOMPZRow(layoutCOMPZRow_),
              ptrCOMPZCol(ptrCOMPZCol_), layoutCOMPZCol(layoutCOMPZCol_),
              layoutCOMPX(layoutCOMPX_), layoutCOMPY(layoutCOMPY_),
              UbNum(UbNum_), OutputWorkspace(OutputWorkspace_), threshold(threshold_) {}

    };

    struct Arguments {
        GemmCoord problemGemmShape;
        GemvCoord problemShape;
        size_t elementSize;
        GM_ADDR ptrX;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
        GM_ADDR ptrZRow;
        GM_ADDR ptrZCol;
        GM_ADDR ptrZRow2;
        GM_ADDR ptrZCol2;
        GM_ADDR ptrCOMPZRow;
        GM_ADDR ptrCOMPZCol;
        FT_ENC_TYPE enc_type;
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
        return sizeof(ElementY) * (args.problemGemmShape.m() + args.problemGemmShape.k());
        
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        GemvCoord problemShape = args.problemShape;
        uint32_t m = problemShape.m();
        uint32_t n = problemShape.n();

        GemvCoord problemCompShape{1,(m+n)};

        uint32_t total_input_elements = m + n;
        
        uint32_t total_input_bytes = total_input_elements * sizeof(ElementCOMPX);
        uint32_t total_output_elements = (m + 8 - 1) / 8 + (n + 8 - 1) / 8;
        uint32_t row_output_elements = (m + 8 - 1) / 8;
        uint32_t col_output_elements = (n + 8 - 1) / 8;
        // printf("Total input bytes: %d",total_input_bytes);

        LayoutCOMPX layoutCOMPX{total_input_elements};
        LayoutCOMPY layoutCOMPY{total_input_elements};

        LayoutCOMPZ layoutCOMPZ{total_output_elements};

        LayoutCOMPZ layoutCOMPZRow{row_output_elements};
        LayoutCOMPZ layoutCOMPZCol{col_output_elements};

        GemvCoord problemShapeCol{n, m};
        problemShapeCol.m() = n;
        problemShapeCol.n() = m;

        LayoutC layoutC{m, n};
        LayoutCCol layoutCCol{n, m};

        
        
        GemmCoord problemGemmShape = args.problemGemmShape;
        uint32_t m2 = problemGemmShape.m();
        uint32_t n2 = problemGemmShape.n();
        uint32_t k2 = problemGemmShape.k();
        LayoutA layoutA{m2, k2};
        LayoutACol layoutACol{k2, m2};
        LayoutB layoutB{k2, n2};
        LayoutBCol layoutBCol{n2, k2};
        uint32_t xlen = (m2 > n2) ? m2 : n2;
        LayoutX layoutX{xlen};

        Params params{problemGemmShape, problemShape, problemShapeCol, problemCompShape,
            args.ptrA, layoutA, layoutACol,
            args.ptrB, layoutB, layoutBCol,
            args.ptrC, layoutC, layoutCCol,
            args.ptrX, layoutX,
            workspace, args.enc_type, 
            args.ptrZRow, args.ptrZCol, args.ptrZRow2, args.ptrZCol2,
            args.ptrCOMPZRow, layoutCOMPZRow,
            args.ptrCOMPZCol, layoutCOMPZCol,
            layoutCOMPX, layoutCOMPY, 
            args.UbNum, args.OutputWorkspace,
            args.threshold};
    
        return params;
    }

    // Methods
    CATLASS_DEVICE
    KernelGemvFTDoubleTotalAiv() {
    }

    
    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params) {};

    // Executes one Matmul
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params) {}

    CATLASS_DEVICE
    void CE_op_aiv(Params const &params)
    {
        AscendC::SetAtomicNone();
        // Arch::Resource<ArchTag> resource;

        // Represent the full gm

        uint32_t aivIndex = AscendC::GetBlockIdx();
        // uint32_t aicoreIndex = aivIndex / AscendC::GetTaskRation();

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);
        AscendC::GlobalTensor<ElementX> gmX;
        gmX.SetGlobalBuffer((__gm__ ElementX *)params.ptrX);
        AscendC::GlobalTensor<ElementZ> gmZ;
        gmZ.SetGlobalBuffer((__gm__ ElementZ *)params.ptrZRow);

        // Get aicore information
        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetTaskRation();

        uint32_t UBTileKRound = RoundUp(UBTileShape::K, UBAlignHelper::ALIGN);
        uint32_t UBTileMRound = RoundUp(UBTileShape::M, UBAlignHelper::ALIGN);
        uint32_t UBTileNRound = RoundUp(UBTileShape::N, UBAlignHelper::ALIGN);
        //uint32_t UBTileKRound = 1;
        //uint32_t UBTileMRound = 1;

        uint32_t loopsNum = CeilDiv(params.problemGemmShape.m(), UBTileMRound);
        //uint32_t loopsNum = params.problemGemmShape.k();

        BlockSumGemv blockSumGemv(resource);

        float alpha{1.0};
        float beta{0.0};

        for(uint32_t loopId = 0; loopId < loopsNum; ++loopId) {
            uint32_t aivId = AscendC::GetBlockIdx();
            if (loopId % aivNum != aivId) continue;

            uint32_t mActual = ((int32_t)loopId == (int32_t)(loopsNum - 1)) ?
                params.problemGemmShape.m() - loopId * UBTileMRound : UBTileMRound;
            //uint32_t kActual = 1;
            uint32_t nActual = params.problemGemmShape.n();

            int64_t gmOffsetC = loopId * UBTileMRound * params.problemGemmShape.n();
            int64_t gmOffsetE = 0;
            int64_t gmOffsetCE = loopId * UBTileMRound;

            GemvCoord actualBlockShape = GemvCoord{mActual, nActual};
            layout::VectorLayout layoutCE{mActual};
            layout::VectorLayout layoutE{nActual};

            

            blockSumGemv(gmC[gmOffsetC], params.layoutC,
                gmX[gmOffsetE], layoutE,
                gmX[gmOffsetE], layoutCE,
                gmZ[gmOffsetCE],
                actualBlockShape,
                alpha, beta);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_DEVICE
    void ABE_op(Params const &params)
    {
        AscendC::SetAtomicNone();
        // Arch::Resource<ArchTag> resource;

        // Represent the full gm
        // Get aicore information
        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetTaskRation();
        // AscendC::printf("%zu\n",AscendC::GetBlockNum());
        uint32_t aivIndex = AscendC::GetBlockIdx();
        // uint32_t aicoreIndex = aivIndex / AscendC::GetTaskRation();

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementX> gmX;
        gmX.SetGlobalBuffer((__gm__ ElementX *)params.ptrX);
        AscendC::GlobalTensor<ElementZ> gmZ;
        gmZ.SetGlobalBuffer((__gm__ ElementZ *)params.ptrZRow2);
        AscendC::GlobalTensor<ElementY> gmY;
        gmY.SetGlobalBuffer((__gm__ ElementY *)params.ptrWorkspace);


        uint32_t UBTileKRound = RoundUp(UBTileShape::K, UBAlignHelper::ALIGN);
        uint32_t UBTileMRound = RoundUp(UBTileShape::M, UBAlignHelper::ALIGN);
        //uint32_t UBTileKRound = 1;
        //uint32_t UBTileMRound = 1;

        uint32_t loopsNum = CeilDiv(params.problemGemmShape.k(), UBTileKRound);
        //uint32_t loopsNum = params.problemGemmShape.k();

        BlockSumGemv blockSumGemv(resource);

        float alpha{1.0};
        float beta{0.0};

        for(uint32_t loopId = 0; loopId < loopsNum; ++loopId) {
            uint32_t aivId = AscendC::GetBlockIdx();
            if (loopId % aivNum != aivId) continue;

            uint32_t kActual = ((int32_t)loopId == (int32_t)(loopsNum - 1)) ?
                params.problemGemmShape.k() - loopId * UBTileKRound : UBTileKRound;
            //uint32_t kActual = 1;
            uint32_t nActual = params.problemGemmShape.n();

            int64_t gmOffsetB = loopId * UBTileKRound * params.problemGemmShape.n();
            int64_t gmOffsetE = 0;
            int64_t gmOffsetBE = loopId * UBTileKRound;

            GemvCoord actualBlockShape = GemvCoord{kActual, nActual};
            layout::VectorLayout layoutBE{kActual};
            layout::VectorLayout layoutE{nActual};

            blockSumGemv(gmB[gmOffsetB], params.layoutB,
                gmX[gmOffsetE], layoutE,
                gmX[gmOffsetE], layoutBE,
                gmY[gmOffsetBE],
                actualBlockShape,
                alpha, beta);
        }

        // AscendC::SyncAll<true>();

        Arch::CrossCoreBarrierAIV<0x0, PIPE_V>();
        // Arch::CrossCoreBarrierAIV<0x0, PIPE_MTE3>();

        loopsNum = CeilDiv(params.problemGemmShape.m(), UBTileMRound);
        for(uint32_t loopId = 0; loopId < loopsNum; ++loopId) {
            uint32_t aivId = AscendC::GetBlockIdx();
            if (loopId % aivNum != aivId) continue;

            uint32_t mActual = ((int32_t)loopId == (int32_t)(loopsNum - 1)) ?
                params.problemGemmShape.m() - loopId * UBTileMRound : UBTileMRound;
            //uint32_t mActual = 1;
            uint32_t kActual = params.problemGemmShape.k();

            GemvCoord actualBlockShape = GemvCoord{mActual, kActual};

            int64_t gmOffsetA = loopId * UBTileMRound * params.problemGemmShape.k();
            int64_t gmOffsetBE = 0;
            int64_t gmOffsetABE = loopId * UBTileMRound;

            layout::VectorLayout layoutBE{kActual};
            layout::VectorLayout layoutABE{mActual};

            blockSumGemv(gmA[gmOffsetA], params.layoutA,
                gmY[gmOffsetBE], layoutBE,
                gmY[gmOffsetBE], layoutABE,
                gmZ[gmOffsetABE],
                actualBlockShape,
                alpha, beta);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_DEVICE
    void ETC_op_aiv(Params const &params)
    {
        AscendC::SetAtomicNone();

        // Represent the full gm
        // Arch::Resource<ArchTag> resource;
        BlockSumGemv blockSumGemv(resource);

        uint32_t aivIndex = AscendC::GetBlockIdx();
        // uint32_t aicoreIndex = aivIndex / AscendC::GetTaskRation();

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);
        AscendC::GlobalTensor<ElementX> gmX;
        gmX.SetGlobalBuffer((__gm__ ElementX *)params.ptrX);
        AscendC::GlobalTensor<ElementZ> gmZ;
        gmZ.SetGlobalBuffer((__gm__ ElementZ *)params.ptrZCol);

        // Get aicore information
        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetTaskRation();

        uint32_t UBTileKRound = RoundUp(UBTileShape::K, UBAlignHelper::ALIGN);
        uint32_t UBTileMRound = RoundUp(UBTileShape::M, UBAlignHelper::ALIGN);
        uint32_t UBTileNRound = RoundUp(UBTileShape::N, UBAlignHelper::ALIGN);
        //uint32_t UBTileKRound = 1;
        //uint32_t UBTileMRound = 1;

        uint32_t loopsNum = CeilDiv(params.problemGemmShape.n(), UBTileNRound);
        //uint32_t loopsNum = params.problemGemmShape.k();

        

        float alpha{1.0};
        float beta{0.0};

        for(uint32_t loopId = 0; loopId < loopsNum; ++loopId) {
            uint32_t aivId = AscendC::GetBlockIdx();
            if (loopId % aivNum != aivId) continue;

            uint32_t nActual = ((int32_t)loopId == (int32_t)(loopsNum - 1)) ?
                params.problemGemmShape.n() - loopId * UBTileNRound : UBTileNRound;
            //uint32_t kActual = 1;
            uint32_t mActual = params.problemGemmShape.m();

            int64_t gmOffsetC = loopId * UBTileNRound;
            int64_t gmOffsetET = 0;
            int64_t gmOffsetETC = loopId * UBTileNRound;

            GemvCoord actualBlockShape = GemvCoord{nActual, mActual};
            layout::VectorLayout layoutETC{nActual};
            layout::VectorLayout layoutET{mActual};

            blockSumGemv.GemvCol(gmC[gmOffsetC], params.layoutCCol,
                gmX[gmOffsetET], layoutET,
                gmX[gmOffsetET], layoutETC,
                gmZ[gmOffsetETC],
                actualBlockShape,
                alpha, beta);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }


    CATLASS_DEVICE
    void ETAB_op(Params const &params)
    {
        AscendC::SetAtomicNone();

        // Represent the full gm
        // Arch::Resource<ArchTag> resource;

        uint32_t aivIndex = AscendC::GetBlockIdx();
        // uint32_t aicoreIndex = aivIndex / AscendC::GetTaskRation();

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementX> gmX;
        gmX.SetGlobalBuffer((__gm__ ElementX *)params.ptrX);
        AscendC::GlobalTensor<ElementZ> gmZ;
        gmZ.SetGlobalBuffer((__gm__ ElementZ *)params.ptrZCol2);
        AscendC::GlobalTensor<ElementY> gmY;
        gmY.SetGlobalBuffer((__gm__ ElementY *)params.ptrWorkspace);

        // Get aicore information
        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetTaskRation();

        uint32_t UBTileKRound = RoundUp(UBTileShape::K, UBAlignHelper::ALIGN);
        uint32_t UBTileMRound = RoundUp(UBTileShape::M, UBAlignHelper::ALIGN);
        uint32_t UBTileNRound = RoundUp(UBTileShape::N, UBAlignHelper::ALIGN);
        //uint32_t UBTileKRound = 1;
        //uint32_t UBTileMRound = 1;

        uint32_t loopsNum = CeilDiv(params.problemGemmShape.k(), UBTileKRound);
        //uint32_t loopsNum = params.problemGemmShape.k();

        BlockSumGemv blockSumGemv(resource);

        float alpha{1.0};
        float beta{0.0};

        for(uint32_t loopId = 0; loopId < loopsNum; ++loopId) {
            uint32_t aivId = AscendC::GetBlockIdx();
            if (loopId % aivNum != aivId) continue;

            uint32_t kActual = ((int32_t)loopId == (int32_t)(loopsNum - 1)) ?
                params.problemGemmShape.k() - loopId * UBTileKRound : UBTileKRound;
            //uint32_t kActual = 1;
            uint32_t mActual = params.problemGemmShape.m();

            int64_t gmOffsetA = loopId * UBTileKRound;
            int64_t gmOffsetET = 0;
            int64_t gmOffsetETA = loopId * UBTileKRound;

            GemvCoord actualBlockShape = GemvCoord{kActual, mActual};
            layout::VectorLayout layoutETA{kActual};
            layout::VectorLayout layoutET{mActual};


            blockSumGemv.GemvCol(gmA[gmOffsetA], params.layoutACol,
                gmX[gmOffsetET], layoutET,
                gmX[gmOffsetET], layoutETA,
                gmY[gmOffsetETA],
                actualBlockShape,
                alpha, beta);
        }

        // AscendC::SyncAll<true>();
        Arch::CrossCoreBarrierAIV<0x0, PIPE_V>();
        // Arch::CrossCoreBarrierAIV<0x0, PIPE_MTE3>();

        //AscendC::PipeBarrier<PIPE_ALL>();

        //loopsNum = params.problemGemmShape.m();
        loopsNum = CeilDiv(params.problemGemmShape.n(), UBTileNRound);
        for(uint32_t loopId = 0; loopId < loopsNum; ++loopId) {
            uint32_t aivId = AscendC::GetBlockIdx();
            if (loopId % aivNum != aivId) continue;

            uint32_t nActual = ((int32_t)loopId == (int32_t)(loopsNum - 1)) ?
                params.problemGemmShape.n() - loopId * UBTileNRound : UBTileNRound;
            //uint32_t mActual = 1;
            uint32_t kActual = params.problemGemmShape.k();

            GemvCoord actualBlockShape = GemvCoord{nActual, kActual};

            int64_t gmOffsetB = loopId * UBTileNRound;
            int64_t gmOffsetETA = 0;
            int64_t gmOffsetETAB = loopId * UBTileNRound;

            layout::VectorLayout layoutETA{kActual};
            layout::VectorLayout layoutETAB{nActual};

            blockSumGemv.GemvCol(gmB[gmOffsetB], params.layoutBCol,
                gmY[gmOffsetETA], layoutETA,
                gmY[gmOffsetETA], layoutETAB,
                gmZ[gmOffsetETAB],
                actualBlockShape,
                alpha, beta);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_DEVICE
    void COMP_op(Params const &params, uint32_t element_num, GM_ADDR ptrInputX, GM_ADDR ptrInputY, GM_ADDR ptrOutputCOMP)
    {
        AscendC::SetAtomicNone();
        // Arch::Resource<ArchTag> resource;

        BlockCompare blockCompare(resource);

        uint32_t align = BYTE_PER_BLK / sizeof(ElementCOMPX);
        uint32_t total_block_elements = COMPUBTileShape::M * COMPUBTileShape::N * params.UbNum;
        uint32_t maxPerBlock_round = RoundUp(total_block_elements, align);

        // uint32_t maxPerBlock_work = maxPerBlock_round * sizeof(ElementCOMPX) / sizeof(ElementWork);
        uint32_t maxPerBlock_out = maxPerBlock_round / 8;

        uint32_t total_input_elements = element_num;
        
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
        gmX.SetGlobalBuffer((__gm__ ElementCOMPX *)ptrInputX);
        AscendC::GlobalTensor<ElementCOMPY> gmY;
        gmY.SetGlobalBuffer((__gm__ ElementCOMPY *)ptrInputY);
        // AscendC::GlobalTensor<ElementWork> gmWork;
        // gmWork.SetGlobalBuffer((__gm__ ElementWork *)params.ptrWorkspace);
        AscendC::GlobalTensor<ElementCOMPZ> gmZ;
        gmZ.SetGlobalBuffer((__gm__ ElementCOMPZ *)ptrOutputCOMP);

        bool isFirstBlock = true;
        bool hasNextBlock = false;
        uint32_t aiv_num = AscendC::GetBlockNum() * AscendC::GetTaskRation();
        uint32_t aiv_id = AscendC::GetBlockIdx();

        for (uint32_t loop_id = aiv_id; loop_id < loopnum; loop_id+=aiv_num) {
            
            if (loop_id % aiv_num != aiv_id)
                continue;

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

            LayoutCOMPX layoutInputX{element_num};
            LayoutCOMPY layoutInputY{element_num};
            LayoutCOMPZ layoutOutputZ{total_output_elements};

            // gmWork[gmOffsetWork], params.layoutWorkspace,
            blockCompare(gmX[gmOffsetX], layoutInputX,
                         gmY[gmOffsetY], layoutInputY,
                         gmX[gmOffsetNextX], gmY[gmOffsetNextY],
                         gmZ[gmOffsetZ],layoutOutputZ,
                         actualBlockShape, nextActualBlockShape, isFirstBlock, 
                         hasNextBlock, params.OutputWorkspace, params.threshold);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params){

        if(params.enc_type == FT_ENC_TYPE::NO){
            // EpilogueOp(params);
        }else if(params.enc_type == FT_ENC_TYPE::CE) {
            ABE_op(params);
            // device 级别的同步
            // AscendC::SyncAll<true>();
            Arch::CrossCoreBarrierAIV<0x0, PIPE_V>();
            // Arch::CrossCoreBarrierAIV<0x0, PIPE_MTE3>();
            CE_op_aiv(params);
        }else if(params.enc_type == FT_ENC_TYPE::ETC) {
            ETAB_op(params);
            // device 级别的同步
            // AscendC::SyncAll<true>();

            Arch::CrossCoreBarrierAIV<0x0, PIPE_V>();
            // Arch::CrossCoreBarrierAIV<0x0, PIPE_MTE3>();
            ETC_op_aiv(params);
        }else if(params.enc_type == FT_ENC_TYPE::BOTHC) {
            ABE_op(params);
            Arch::CrossCoreBarrierAIV<0x0, PIPE_V>();
            // Arch::CrossCoreBarrierAIV<0x0, PIPE_MTE3>();
            CE_op_aiv(params);
            //device 级别的同步
            // AscendC::SyncAll<true>();
            Arch::CrossCoreBarrierAIV<0x0, PIPE_V>();
            // Arch::CrossCoreBarrierAIV<0x0, PIPE_MTE3>();
            ETAB_op(params);
            Arch::CrossCoreBarrierAIV<0x0, PIPE_V>();
            // Arch::CrossCoreBarrierAIV<0x0, PIPE_MTE3>();
            ETC_op_aiv(params);
        }
        // AscendC::SyncAll<true>();
        Arch::CrossCoreBarrierAIV<0x0, PIPE_V>();
        // Arch::CrossCoreBarrierAIV<0x0, PIPE_MTE3>();

        uint32_t m = params.problemShape.m();
        uint32_t n = params.problemShape.n();

        if(params.enc_type == FT_ENC_TYPE::CE){
            COMP_op(params, m, params.ptrZRow, params.ptrZRow2, params.ptrCOMPZRow);
            
        }else if(params.enc_type == FT_ENC_TYPE::ETC){
            COMP_op(params, n, params.ptrZCol, params.ptrZCol2, params.ptrCOMPZCol);
        }else if(params.enc_type == FT_ENC_TYPE::BOTHC){
            COMP_op(params, m, params.ptrZRow, params.ptrZRow2, params.ptrCOMPZRow);
            // AscendC::SyncAll<true>();
            Arch::CrossCoreBarrierAIV<0x0, PIPE_V>();
            // Arch::CrossCoreBarrierAIV<0x0, PIPE_MTE3>();
            COMP_op(params, n, params.ptrZCol, params.ptrZCol2, params.ptrCOMPZCol);
            // AscendC::SyncAll<true>();
        } 
        // AscendC::SyncAll<true>(); 
        Arch::CrossCoreBarrierAIV<0x0, PIPE_V>();
        // Arch::CrossCoreBarrierAIV<0x0, PIPE_MTE3>();
    }


private:
    Arch::ResourceAIV<ArchTag> resource;
//     // ID used for inter-core synchronization
    // static constexpr Arch::FlagID FLAG_AIC_FINISH_STORE = 0;
    // static constexpr Arch::FlagID RV_FLAG_AIC_FINISH_STORE = 1;
    // Arch::CrossCoreFlagWithReverse<> flagAicFinishStore{FLAG_AIC_FINISH_STORE, RV_FLAG_AIC_FINISH_STORE};
    // Arch::Resource<ArchTag> resource;

//     static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 0;
//     Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
};

}  // namespace Catlass::Gemv::kernel

#endif  // CATLASS_GEMV_KERNLE_GEMV_AIC_FT_HPP

