#ifndef CATLASS_GEMV_KERNLE_GEMV_FT_DOUBLE_THRESHOLD_COMPUTE_HPP
#define CATLASS_GEMV_KERNLE_GEMV_FT_DOUBLE_THRESHOLD_COMPUTE_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemv/helper.hpp"
#include "catlass/gemv_coord.hpp"
#include "catlass/matrix_coord.hpp"
// #include "catlass/epilogue/tile/tile_copy.hpp"
// #include "catlass/gemv/tile/tile_copy.hpp"


namespace Catlass::Gemv::Kernel{


template<
    class BlockGemv_,
    class BlockSumGemv_,
    class BlockCompare_,
    class BlockCompareRaw_,
    class BlockThreCalc_,
    class BlockEpilogue_
>
class KernelGemvFTDoubleThre {
public:
    using BlockGemv = BlockGemv_;
    using BlockSumGemv = BlockSumGemv_;
    using BlockCompare = BlockCompare_;
    using BlockThreCalc = BlockThreCalc_;

    using ArchTag = typename BlockGemv::ArchTag;
    using L1TileShapeC = typename BlockGemv::L1TileShape;
    using L0TileShapeC = typename BlockGemv::L0TileShape;

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

    using ElementX = typename BlockGemv::ElementX;
    using LayoutX = typename BlockGemv::LayoutX;



    using ElementC = typename BlockGemv::ElementA;
    using LayoutC = typename BlockGemv::LayoutA;

    using LayoutCCol = typename std::conditional<
        std::is_same<LayoutC, Catlass::layout::RowMajor>::value,
        Catlass::layout::ColumnMajor,
        Catlass::layout::RowMajor>::type;
    
    using CColType = Gemm::GemmType<ElementC, LayoutCCol>;

    using ElementZ = typename BlockGemv::ElementY;
    using LayoutZ = typename BlockGemv::LayoutY;

    using ElementCOMPX = ElementZ;
    using LayoutCOMPX = Catlass::layout::VectorLayout;


    using ElementY = typename BlockSumGemv::ElementY;
    using LayoutY = typename BlockSumGemv::LayoutY;

    using ElementCOMPY = ElementZ;
    using LayoutCOMPY = Catlass::layout::VectorLayout;

    using UBTileShape = typename BlockSumGemv::UBTileShape;
    using ThreCalcUBTileShape = typename BlockThreCalc::UBTileShape;

    using UBAlignHelper = Gemv::helper::UBAlignHelper<ElementA>;
    using COMPUBTileShape = typename BlockCompare::UBTileShape;

    using BlockEpilogue = BlockEpilogue_;
    using FT_ENC_TYPE = Gemv::helper::FT_ENC_TYPE;
    using FT_COMP_TYPE = Catlass::Gemv::helper::FT_COMP_TYPE;

    using ElementCOMPZ = typename BlockCompare::ElementZ;
    using LayoutCOMPZ = Catlass::layout::VectorLayout;

    using COMPZType = Gemm::GemmType<uint8_t, LayoutCOMPZ>;

    using OutCType = Gemm::GemmType<ElementC, LayoutC>;

    using ElementWork = typename std::conditional<
        (BlockCompare::COMP_TYPE == FT_COMP_TYPE::XOR),
        uint16_t,
        typename std::conditional<(BlockCompare::COMP_TYPE == FT_COMP_TYPE::COMPARE), int32_t, ElementCOMPX>::type>::type;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementC, ElementX>::ElementAccumulator;

    using BlockCompareRaw = BlockCompareRaw_;

    using FT_RCE_THRE_TYPE = Catlass::Gemv::helper::FT_RCE_THRE_TYPE;

    static_assert(std::is_same_v<LayoutA, LayoutB>,
        "The LayoutA and LayoutB of Gemm should be consistent.");

    enum class AivCore {
        AIV0 = 0,
        AIV1
    };

    /*
    GemmCoord problemGemmShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        LayoutACol layoutACol;
        GM_ADDR ptrB;
        LayoutB layoutB;
        LayoutBCol layoutBCol;
        GM_ADDR ptrX;
        LayoutX layoutX;
        GM_ADDR ptrZRow2; // 最终 列和/行和 checksum 与 eTC; Ce sum 结果的存放位置
        GM_ADDR ptrZCol2;
        GM_ADDR ptrWorkspace; // 中间暂存MMAD 输出结果的GM，这些结果要通过Ce/eTC 求行和/列和以便和ABe/eTAB 作比较，以检测校验soft silent errors.
        FT_ENC_TYPE enc_type;
    */
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
        GM_ADDR ptrXV;
        LayoutX layoutXV;
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
        GM_ADDR ptrThreZ;
        LayoutCOMPX layoutThre;
        ElementZ rounding_alpha;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemGemmShape_, GemvCoord const &problemShape_, 
            GemvCoord const &problemShapeCol_, GemvCoord const &problemCompShape_,
            GM_ADDR ptrA_, LayoutA layoutA_, LayoutACol layoutACol_,
            GM_ADDR ptrB_, LayoutB layoutB_, LayoutBCol layoutBCol_,
            GM_ADDR ptrC_, LayoutC layoutC_, LayoutCCol layoutCCol_,
            GM_ADDR ptrX_, LayoutX layoutX_, GM_ADDR ptrXV_, LayoutX layoutXV_,
            GM_ADDR ptrWorkspace_, FT_ENC_TYPE enc_type_, 
            GM_ADDR ptrZRow_, GM_ADDR ptrZCol_, GM_ADDR ptrZRow2_, GM_ADDR ptrZCol2_,
            GM_ADDR ptrCOMPZRow_, LayoutCOMPZ layoutCOMPZRow_, 
            GM_ADDR ptrCOMPZCol_, LayoutCOMPZ layoutCOMPZCol_,
            LayoutCOMPX layoutCOMPX_, LayoutCOMPY layoutCOMPY_,
            uint32_t UbNum_, bool OutputWorkspace_, ElementCOMPX threshold_,
            GM_ADDR ptrThreZ_, LayoutCOMPX layoutThre_, ElementZ rounding_alpha_)
            : problemGemmShape(problemGemmShape_), problemShape(problemShape_),
              problemShapeCol(problemShapeCol_), problemCompShape(problemCompShape_),
              ptrA(ptrA_), layoutA(layoutA_), layoutACol(layoutACol_),
              ptrB(ptrB_), layoutB(layoutB_), layoutBCol(layoutBCol_), ptrC(ptrC_), layoutC(layoutC_),
              layoutCCol(layoutCCol_), ptrX(ptrX_), layoutX(layoutX_), ptrXV(ptrXV_), layoutXV(layoutXV_),
              ptrWorkspace(ptrWorkspace_), enc_type(enc_type_), ptrZRow(ptrZRow_), ptrZCol(ptrZCol_),
              ptrZRow2(ptrZRow2_), ptrZCol2(ptrZCol2_), 
              ptrCOMPZRow(ptrCOMPZRow_), layoutCOMPZRow(layoutCOMPZRow_),
              ptrCOMPZCol(ptrCOMPZCol_), layoutCOMPZCol(layoutCOMPZCol_),
              layoutCOMPX(layoutCOMPX_), layoutCOMPY(layoutCOMPY_),
              UbNum(UbNum_), OutputWorkspace(OutputWorkspace_),threshold(threshold_), 
              ptrThreZ(ptrThreZ_), layoutThre(layoutThre_), 
              rounding_alpha(rounding_alpha_) {}
    };

    /*
    GemmCoord problemGemmShape;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrXV;
        GM_ADDR ptrZRow2; // 最终 列和/行和 checksum 与 eTC; Ce sum 结果的存放位置
        GM_ADDR ptrZCol2;
        uint32_t blockNum;
        FT_ENC_TYPE enc_type;
    */
    struct Arguments {
        GemmCoord problemGemmShape;
        GemvCoord problemShape;
        size_t elementSize;
        GM_ADDR ptrX;
        GM_ADDR ptrXV;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
        GM_ADDR ptrZRow;
        GM_ADDR ptrZCol;
        GM_ADDR ptrZRow2;
        GM_ADDR ptrZCol2;
        GM_ADDR ptrCOMPZRow;
        GM_ADDR ptrCOMPZCol;
        GM_ADDR ptrThreZ;
        FT_ENC_TYPE enc_type;
        uint32_t UbNum;
        bool OutputWorkspace;
        ElementCOMPX threshold;
        float rounding_exponent;
        float size_beta;
        FT_RCE_THRE_TYPE rce_thre_type;

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
        // return args.elementSize * (args.problemShape.m() + args.problemShape.n());
        //  + sizeof(ElementZ) * args.problemShape.m()
        return (sizeof(ElementY) * (args.problemGemmShape.m() + args.problemGemmShape.k()));
        
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


        // LayoutX layoutX{n};
        // LayoutZ layoutZ{m};
        LayoutC layoutC{m, n};
        LayoutCCol layoutCCol{n, m};

        LayoutX layoutX{m+n};

        LayoutZ layoutZRow{m};
        // typename BlockEpilogue::Params epilogueParamsRow{args.alpha, args.beta, args.ptrZRow, layoutZRow, args.ptrZRow, layoutZRow};

        LayoutZ layoutZCol{n};

        LayoutCOMPX layoutThre{m};
        // typename BlockEpilogue::Params epilogueParamsCol{args.alpha, args.beta, args.ptrZCol, layoutZCol, args.ptrZCol, layoutZCol};

        // epilogueParamsRow, 
            // epilogueParamsCol,
        
        GemmCoord problemGemmShape = args.problemGemmShape;
        uint32_t m2 = problemGemmShape.m();
        uint32_t n2 = problemGemmShape.n();
        uint32_t k2 = problemGemmShape.k();
        LayoutA layoutA{m2, k2};
        LayoutACol layoutACol{k2, m2};
        LayoutB layoutB{k2, n2};
        LayoutBCol layoutBCol{n2, k2};
        uint32_t xlen = (m2 > n2) ? m2 : n2;
        LayoutX layoutXV{xlen};

        /*
        float rounding_exponent;
        float size_beta;
        */

        float input_exponent = (args.rounding_exponent < 0.0f) ? args.rounding_exponent : (0.0 - args.rounding_exponent);

        float rounding_error = std::pow(2.0f,input_exponent);

        float row_sqrt = 1.0f;

        if(args.size_beta < 1.0f){
            row_sqrt = std::sqrt(n*1.0f);
        }else{
            row_sqrt = args.size_beta;
        }

        
        ElementZ rounding_alpha = static_cast<ElementZ>(row_sqrt * rounding_error);
        
        if(args.rce_thre_type == FT_RCE_THRE_TYPE::ROUND_WITH_ACC){
            float acc_rounding_error = std::pow(2.0f, -23.0f);
            float acc_scaling_factor = 1.0f * n*(n+1)*(2*n+1) / 48.0f;
            acc_scaling_factor = std::sqrt(acc_scaling_factor);
            rounding_alpha = static_cast<ElementZ>(row_sqrt * rounding_error + acc_rounding_error * acc_scaling_factor); 
        }

        Params params{problemGemmShape, problemShape, problemShapeCol, problemCompShape,
            args.ptrA, layoutA, layoutACol,
            args.ptrB, layoutB, layoutBCol,
            args.ptrC, layoutC, layoutCCol,
            args.ptrX, layoutX,
            args.ptrXV, layoutXV,
            workspace, args.enc_type, 
            args.ptrZRow, args.ptrZCol, args.ptrZRow2, args.ptrZCol2,
            args.ptrCOMPZRow, layoutCOMPZRow,
            args.ptrCOMPZCol, layoutCOMPZCol,
            layoutCOMPX, layoutCOMPY, 
            args.UbNum, args.OutputWorkspace, args.threshold, 
            args.ptrThreZ, layoutThre, rounding_alpha};
    
        return params;
    }

    // Methods
    CATLASS_DEVICE
    KernelGemvFTDoubleThre() {}

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

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        AscendC::GlobalTensor<ElementZ> gmZ;
        gmZ.SetGlobalBuffer((__gm__ ElementZ *)params.ptrZRow);

        layout::RowMajor layoutY(1, params.problemShape.m());

        uint32_t maxMPerBlock = L1TileShapeC::M;
        uint32_t maxNPerBlock = L1TileShapeC::N;
        uint32_t M = params.problemShape.m();
        uint32_t N = params.problemShape.n();
        LayoutX layoutX{N};

        uint32_t MLoops = CeilDiv(M, maxMPerBlock);
        uint32_t coreLoops = MLoops;
        uint32_t singleIdx = 0;

        static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
        static constexpr uint32_t L0C_TILE_SIZE = L0TileShapeC::M * L0TileShapeC::N;
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
            int64_t gmOffsetC;
            int64_t gmOffsetY;
            int64_t gmOffsetNextX;
            int64_t gmOffsetNextC;
            int64_t gmOffsetNextY;

            if constexpr (std::is_same<LayoutC, Catlass::layout::RowMajor>::value) {
                gmOffsetX = 0;
                gmOffsetC = MGmBlockIdx * maxMPerBlock * params.layoutC.stride(0);

                gmOffsetY = MGmBlockIdx * maxMPerBlock;
            } else {
                gmOffsetX = 0;
                gmOffsetC = MGmBlockIdx * maxMPerBlock;
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

            if constexpr (std::is_same<LayoutC, Catlass::layout::RowMajor>::value) {
                gmOffsetNextX = 0;
                gmOffsetNextC = MNextGmBlockIdx * maxMPerBlock * params.layoutC.stride(0);

                gmOffsetNextY = MNextGmBlockIdx * maxMPerBlock;
            } else {
                gmOffsetNextX = 0;
                gmOffsetNextC = MNextGmBlockIdx * maxMPerBlock;
                gmOffsetNextY = MNextGmBlockIdx * maxMPerBlock;
            }

            GemvCoord actualBlockShape = GemvCoord(MGmActual, NGmActual);

            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((event_t)singleIdx % L0C_TILE_NUM);

            
            // Compute block-scoped matrix multiply-add
            blockGemv(gmX[gmOffsetX], layoutX,
                      gmC[gmOffsetC], params.layoutC,
                      gmZ[gmOffsetY], layoutY,
                      gmX[gmOffsetNextX],
                      gmC[gmOffsetNextC],
                      actualBlockShape,
                      nextActualBlockShape,
                      isFirstBlock,
                      hasNextBlock,
                      singleIdx);

            // Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
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

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        AscendC::GlobalTensor<ElementZ> gmZ;
        gmZ.SetGlobalBuffer((__gm__ ElementZ *)params.ptrZRow);

        layout::RowMajor layoutY(1, params.problemShape.m());

        uint32_t maxMPerBlock = L1TileShapeC::M;
        uint32_t maxNPerBlock = L1TileShapeC::N;
        uint32_t M = params.problemShape.m();
        uint32_t N = params.problemShape.n();
        LayoutX layoutX{N};

        uint32_t MLoops = CeilDiv(M, maxMPerBlock);
        uint32_t coreLoops = MLoops;
        uint32_t singleIdx = 0;

        static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
        static constexpr uint32_t L0C_TILE_SIZE = L0TileShapeC::M * L0TileShapeC::N;
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
            int64_t gmOffsetC;
            int64_t gmOffsetY;
            int64_t gmOffsetNextX;
            int64_t gmOffsetNextC;
            int64_t gmOffsetNextY;

            if constexpr (std::is_same<LayoutC, Catlass::layout::RowMajor>::value) {
                gmOffsetX = 0;
                gmOffsetC = MGmBlockIdx * maxMPerBlock * params.layoutC.stride(0);

                gmOffsetY = MGmBlockIdx * maxMPerBlock;
            } else {
                gmOffsetX = 0;
                gmOffsetC = MGmBlockIdx * maxMPerBlock;
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

            if constexpr (std::is_same<LayoutC, Catlass::layout::RowMajor>::value) {
                gmOffsetNextX = 0;
                gmOffsetNextC = MNextGmBlockIdx * maxMPerBlock * params.layoutC.stride(0);

                gmOffsetNextY = MNextGmBlockIdx * maxMPerBlock;
            } else {
                gmOffsetNextX = 0;
                gmOffsetNextC = MNextGmBlockIdx * maxMPerBlock;
                gmOffsetNextY = MNextGmBlockIdx * maxMPerBlock;
            }

            GemvCoord actualBlockShape = GemvCoord(MGmActual, NGmActual);

            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((event_t)singleIdx % L0C_TILE_NUM);

            // Compute block-scoped matrix multiply-add
            blockGemv.RowSum(gmX[gmOffsetX], layoutX,
                      gmC[gmOffsetC], params.layoutC,
                      gmZ[gmOffsetY], layoutY,
                      gmX[gmOffsetNextX],
                      gmC[gmOffsetNextC],
                      actualBlockShape,
                      nextActualBlockShape,
                      isFirstBlock,
                      hasNextBlock,
                      singleIdx);

            // Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
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
    void ETC_op(Params const& params)
    {
        BlockGemv blockGemv(resource);
        // Represent the full gm
        AscendC::GlobalTensor<ElementX> gmX;
        gmX.SetGlobalBuffer((__gm__ ElementX *)params.ptrX);

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        // AscendC::GlobalTensor<ElementY> gmY;
        // gmY.SetGlobalBuffer((__gm__ ElementY *)params.ptrWorkspace);

        AscendC::GlobalTensor<ElementZ> gmZ;
        gmZ.SetGlobalBuffer((__gm__ ElementZ *)params.ptrZCol);


        layout::RowMajor layoutY(1, params.problemShapeCol.m());
        

        uint32_t maxMPerBlock = L1TileShapeC::M;
        uint32_t maxNPerBlock = L1TileShapeC::N;

        uint32_t M = params.problemShapeCol.m();
        uint32_t N = params.problemShapeCol.n();

        LayoutX layoutX{N};

        uint32_t MLoops = CeilDiv(M, maxMPerBlock);
        uint32_t coreLoops = MLoops;
        uint32_t singleIdx = 0;

        static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
        static constexpr uint32_t L0C_TILE_SIZE = L0TileShapeC::M * L0TileShapeC::N;
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
            int64_t gmOffsetC;
            int64_t gmOffsetY;
            int64_t gmOffsetNextX;
            int64_t gmOffsetNextC;
            int64_t gmOffsetNextY;

            if constexpr (std::is_same<LayoutCCol, Catlass::layout::RowMajor>::value) {
                gmOffsetX = 0;
                gmOffsetC = MGmBlockIdx * maxMPerBlock * params.layoutCCol.stride(0);

                gmOffsetY = MGmBlockIdx * maxMPerBlock;
            } else {
                gmOffsetX = 0;
                gmOffsetC = MGmBlockIdx * maxMPerBlock;
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

            if constexpr (std::is_same<LayoutCCol, Catlass::layout::RowMajor>::value) {
                gmOffsetNextX = 0;
                gmOffsetNextC = MNextGmBlockIdx * maxMPerBlock * params.layoutCCol.stride(0);

                gmOffsetNextY = MNextGmBlockIdx * maxMPerBlock;
            } else {
                gmOffsetNextX = 0;
                gmOffsetNextC = MNextGmBlockIdx * maxMPerBlock;
                gmOffsetNextY = MNextGmBlockIdx * maxMPerBlock;
            }

            GemvCoord actualBlockShape = GemvCoord(MGmActual, NGmActual);

            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((event_t)singleIdx % L0C_TILE_NUM);

            // Compute block-scoped matrix multiply-add
            // gmY[params.problemShape.m() + gmOffsetY]
            // params.problemShape.m() + 
            blockGemv.ColSum(gmX[gmOffsetX], layoutX,
                      gmC[gmOffsetC], params.layoutCCol,
                      gmZ[gmOffsetY], layoutY,
                      gmX[gmOffsetNextX],
                      gmC[gmOffsetNextC],
                      actualBlockShape,
                      nextActualBlockShape,
                      isFirstBlock,
                      hasNextBlock,
                      singleIdx);

            // Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
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
        }else if(params.enc_type == FT_ENC_TYPE::RCE){
            CE_op(params);
        }
        AscendC::SyncAll<true>();
        // AscendC::printf("AIC DONE\n");
        Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
    }

    CATLASS_DEVICE
    void ABE_op(Params const &params)
    {
        // AscendC::SetAtomicNone();

        // Represent the full gm

        uint32_t aivIndex = AscendC::GetBlockIdx();
        uint32_t aicoreIndex = aivIndex / AscendC::GetSubBlockNum();

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementX> gmX;
        gmX.SetGlobalBuffer((__gm__ ElementX *)params.ptrXV);
        AscendC::GlobalTensor<ElementZ> gmZ;
        gmZ.SetGlobalBuffer((__gm__ ElementZ *)params.ptrZRow2);
        AscendC::GlobalTensor<ElementY> gmY;
        gmY.SetGlobalBuffer((__gm__ ElementY *)params.ptrWorkspace);


        // Get aicore information
        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();

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

            /*blockSumGemv.rowSum(gmB[gmOffsetB], layoutB,
                gmD[gmOffsetBE], layoutBE,
                actualBlockShape, Gemv::helper::FT_ENC_TYPE::BE);*/

            blockSumGemv(gmB[gmOffsetB], params.layoutB,
                gmX[gmOffsetE], layoutE,
                gmX[gmOffsetE], layoutBE,
                gmY[gmOffsetBE],
                actualBlockShape,
                alpha, beta);
        }

        AscendC::SyncAll<true>();

        //AscendC::PipeBarrier<PIPE_ALL>();

        //loopsNum = params.problemGemmShape.m();
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
    void ETAB_op(Params const &params)
    {
        AscendC::SetAtomicNone();

        // Represent the full gm

        uint32_t aivIndex = AscendC::GetBlockIdx();
        uint32_t aicoreIndex = aivIndex / AscendC::GetSubBlockNum();

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementX> gmX;
        gmX.SetGlobalBuffer((__gm__ ElementX *)params.ptrXV);
        AscendC::GlobalTensor<ElementZ> gmZ;
        gmZ.SetGlobalBuffer((__gm__ ElementZ *)params.ptrZCol2);
        AscendC::GlobalTensor<ElementY> gmY;
        gmY.SetGlobalBuffer((__gm__ ElementY *)params.ptrWorkspace);

        // Get aicore information
        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();

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

            /*blockSumGemv.rowSum(gmB[gmOffsetB], layoutB,
                gmD[gmOffsetBE], layoutBE,
                actualBlockShape, Gemv::helper::FT_ENC_TYPE::BE);*/

            blockSumGemv.GemvCol(gmA[gmOffsetA], params.layoutACol,
                gmX[gmOffsetET], layoutET,
                gmX[gmOffsetET], layoutETA,
                gmY[gmOffsetETA],
                actualBlockShape,
                alpha, beta);
        }

        AscendC::SyncAll<true>();

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
    void EpilogueOp(Params const& params)
    {
        // BlockEpilogue blockEpilogue(resource, params.epilogueParamsRow);

        // Represent the full gm
        AscendC::GlobalTensor<ElementZ> gmY;
        gmY.SetGlobalBuffer((__gm__ ElementZ *)params.ptrWorkspace);

        layout::VectorLayout layoutY(params.problemShape.m());

        // Get aicore information
        uint32_t aicoreIndex = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t subcoreIndex = AscendC::GetSubBlockIdx();

        uint32_t maxMPerBlock = L1TileShapeC::M;
        uint32_t M = params.problemShape.m();
        uint32_t MLoops = CeilDiv(M, maxMPerBlock);
        uint32_t coreLoops = MLoops;

        // Loop through the epilogue calculations of each basic block
        layout::VectorLayout::TensorCoord blockShape{L1TileShapeC::M};

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
            // Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);

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
        AscendC::GlobalTensor<ElementZ> gmY;
        gmY.SetGlobalBuffer((__gm__ ElementZ *)params.ptrWorkspace);

        layout::VectorLayout layoutY(params.problemShape.m());

        // Get aicore information
        uint32_t aicoreIndex = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t subcoreIndex = AscendC::GetSubBlockIdx();

        uint32_t maxMPerBlock = L1TileShapeC::M;
        uint32_t M = params.problemShape.m();
        uint32_t MLoops = CeilDiv(M, maxMPerBlock);
        uint32_t coreLoops = MLoops;

        // Loop through the epilogue calculations of each basic block
        layout::VectorLayout::TensorCoord blockShape{L1TileShapeC::M};

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
            // Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);

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
        AscendC::GlobalTensor<ElementZ> gmY;
        gmY.SetGlobalBuffer((__gm__ ElementZ *)params.ptrWorkspace);

        layout::VectorLayout layoutY(params.problemShapeCol.m());

        // Get aicore information
        uint32_t aicoreIndex = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t subcoreIndex = AscendC::GetSubBlockIdx();

        uint32_t maxMPerBlock = L1TileShapeC::M;
        uint32_t M = params.problemShapeCol.m();
        uint32_t MLoops = CeilDiv(M, maxMPerBlock);
        uint32_t coreLoops = MLoops;

        // Loop through the epilogue calculations of each basic block
        layout::VectorLayout::TensorCoord blockShape{L1TileShapeC::M};

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
            // Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);

            // Actual calculatioin logic for performing block-scoped epilogue
            // blockEpilogue(blockOffset, actualBlockShape, gmBlockY, layoutBlockY);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // 尝试将ABE/ETAB的逻辑实现在这里，此外还要实现二者的比较，建议使用信号量驱动，写成流水线
    // template <>
    // CATLASS_DEVICE
    // void operator()<AscendC::AIV>(Params const& params)
    // {
    //     if(params.enc_type == FT_ENC_TYPE::NO){
    //         EpilogueOp(params);
    //     }else if(params.enc_type == FT_ENC_TYPE::CE) {
    //         EpilogueCE(params);
    //     }else if(params.enc_type == FT_ENC_TYPE::ETC) {
    //         EpilogueETC(params);
    //     }else if(params.enc_type == FT_ENC_TYPE::BOTHC) {
    //         EpilogueCE(params);
    //         // device 级别的同步
    //         AscendC::SyncAll<true>();
    //         EpilogueETC(params);
    //     }
    // }

    CATLASS_DEVICE
    void Threshold_op(Params const &params)
    {
        AscendC::SetAtomicNone();
        // Arch::Resource<ArchTag> resource;
        BlockThreCalc blockThreCalc(resource);
        uint32_t align = BYTE_PER_C0 / sizeof(ElementC);
        uint32_t maxmPerBlock_round = RoundUp(ThreCalcUBTileShape::M, align);
        uint32_t maxnPerBlock_round = RoundUp(ThreCalcUBTileShape::N, align);
        uint32_t split = 1;

        // add split k
        uint32_t N_Split = params.problemShape.n();
        // RoundDown(, params.split) / params.split;
        uint32_t Mloopnum = CeilDiv(params.problemShape.m(), maxmPerBlock_round);
        int32_t loopnum;
        // float Realbeta = params.alpha;
        if constexpr (std::is_same_v<LayoutC, Catlass::layout::ColumnMajor>) {
            loopnum = Mloopnum * split;
            // Realbeta = params.alpha;
        } else {
            loopnum = Mloopnum;
        }

        uint32_t offset_matrix;
        uint32_t offset_vector_out;
        uint32_t offset_vector_in = 0;

        // uint32_t total_workspace_offset = (params.problemShape.m() + params.problemGemmShape.k())*sizeof(ElementY) / sizeof(ElementZ);
        // Represent the full gm
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        AscendC::GlobalTensor<ElementZ> gmT;
        gmT.SetGlobalBuffer((__gm__ ElementZ *)params.ptrThreZ);
        uint32_t aiv_num = AscendC::GetBlockNum() * AscendC::GetTaskRation();
        for (uint32_t loop_id = 0; loop_id < loopnum; loop_id++) {
            uint32_t aiv_id = AscendC::GetBlockIdx();
            if (loop_id % aiv_num != aiv_id)
                continue;
            uint32_t m_actual = ((int32_t)loop_id > (int32_t)(loopnum - split - 1))
                                        ? params.problemShape.m() - ((loop_id / split) * maxmPerBlock_round)
                                        : maxmPerBlock_round;
            uint32_t n_actual = params.problemShape.n();

            if constexpr (std::is_same_v<LayoutC, Catlass::layout::ColumnMajor>) {
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
            // +total_workspace_offset
            // params.rounding_alpha

            blockThreCalc(gmC[offset_matrix], params.layoutC,
                gmT[offset_vector_out], params.layoutThre,
                actualBlockShape, params.rounding_alpha);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_DEVICE
    void COMP_op(Params const &params, uint32_t element_num, GM_ADDR ptrInputX, GM_ADDR ptrInputY, GM_ADDR ptrOutputCOMP)
    {
        AscendC::SetAtomicNone();
        // Arch::Resource<ArchTag> resource;

        BlockCompareRaw blockCompare(resource);

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


    CATLASS_DEVICE
    void COMP_op_with_thre_vector(Params const &params, 
        uint32_t element_num, GM_ADDR ptrInputX, GM_ADDR ptrInputY, 
        GM_ADDR ptrOutputCOMP)
    {
        AscendC::SetAtomicNone();
        // Arch::Resource<ArchTag> resource;

        BlockCompare blockCompare(resource);

        uint32_t align = BYTE_PER_BLK / sizeof(ElementCOMPX);
        uint32_t total_block_elements = COMPUBTileShape::M * COMPUBTileShape::N * params.UbNum;
        uint32_t maxPerBlock_round = RoundUp(total_block_elements, align);

        // uint32_t total_workspace_offset = (params.problemShape.m() + params.problemGemmShape.k())*sizeof(ElementY) / sizeof(ElementZ);

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

        AscendC::GlobalTensor<ElementZ> gmW;
        gmW.SetGlobalBuffer((__gm__ ElementZ *)params.ptrThreZ);

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
            int64_t gmOffsetW = InputGmBlockIdx * maxPerBlock_round;

            // int64_t gmOffsetWork = InputGmBlockIdx * maxPerBlock_work;
            int64_t gmOffsetZ = InputGmBlockIdx * maxPerBlock_out;

            GemvCoord actualBlockShape = GemvCoord{1, input_element_actual};

            uint32_t InputNextGmBlockIdx;
            int64_t gmOffsetNextX;
            int64_t gmOffsetNextY;
            int64_t gmOffsetNextW;

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
                gmOffsetNextW = InputNextGmBlockIdx * maxPerBlock_round;

                // gmOffsetNextWork = InputNextGmBlockIdx * maxPerBlock_work;
                gmOffsetNextZ = InputNextGmBlockIdx * maxPerBlock_out;
            }

            LayoutCOMPX layoutInputX{element_num};
            LayoutCOMPY layoutInputY{element_num};
            LayoutCOMPX layoutInputW{element_num};
            
            LayoutCOMPZ layoutOutputZ{total_output_elements};

            // gmWork[gmOffsetWork], params.layoutWorkspace,
            blockCompare(gmX[gmOffsetX], layoutInputX,
                         gmY[gmOffsetY], layoutInputY,
                         gmW[gmOffsetW], layoutInputW,
                         gmX[gmOffsetNextX], gmY[gmOffsetNextY],
                         gmW[gmOffsetNextW],
                         gmZ[gmOffsetZ], layoutOutputZ,
                         actualBlockShape, nextActualBlockShape, isFirstBlock, 
                         hasNextBlock, params.OutputWorkspace, params.threshold);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params){
        if(params.enc_type == FT_ENC_TYPE::NO){
            EpilogueOp(params);
        }else if(params.enc_type == FT_ENC_TYPE::CE) {
            ABE_op(params);
        }else if(params.enc_type == FT_ENC_TYPE::ETC) {
            ETAB_op(params);
        }else if(params.enc_type == FT_ENC_TYPE::BOTHC) {
            ABE_op(params);
            // device 级别的同步
            AscendC::SyncAll<true>();
            ETAB_op(params);
        }else if(params.enc_type == FT_ENC_TYPE::RCE){
            ABE_op(params);
            AscendC::SyncAll<true>();
            Threshold_op(params);
        }
        AscendC::SyncAll<true>();

        Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);

        uint32_t m = params.problemShape.m();
        uint32_t n = params.problemShape.n();

        // GM_ADDR ptrZRow;
        // GM_ADDR ptrZCol;
        // GM_ADDR ptrZRow2; 
        // GM_ADDR ptrZCol2;
        // GM_ADDR ptrCOMPZRow;
        // LayoutCOMPZ layoutCOMPZRow;
        // GM_ADDR ptrCOMPZCol;

        if(params.enc_type == FT_ENC_TYPE::CE){
            COMP_op(params, m, params.ptrZRow, params.ptrZRow2, params.ptrCOMPZRow);
            AscendC::SyncAll<true>();
        }else if(params.enc_type == FT_ENC_TYPE::ETC){
            COMP_op(params, n, params.ptrZCol, params.ptrZCol2, params.ptrCOMPZCol);
            AscendC::SyncAll<true>();
        }else if(params.enc_type == FT_ENC_TYPE::BOTHC){
            COMP_op(params, m, params.ptrZRow, params.ptrZRow2, params.ptrCOMPZRow);
            AscendC::SyncAll<true>();
            COMP_op(params, n, params.ptrZCol, params.ptrZCol2, params.ptrCOMPZCol);
            AscendC::SyncAll<true>();
        }else if(params.enc_type == FT_ENC_TYPE::RCE){
            /*
            COMP_op_with_thre_vector(Params const &params, 
        uint32_t element_num, GM_ADDR ptrInputX, GM_ADDR ptrInputY, 
        GM_ADDR ptrOutputCOMP)
            */
            COMP_op_with_thre_vector(params, m, params.ptrZRow, params.ptrZRow2, params.ptrCOMPZRow);  
            AscendC::SyncAll<true>();          
        }  
    }


private:
    // ID used for inter-core synchronization
    static constexpr Arch::FlagID FLAG_AIC_FINISH_STORE = 8;
    static constexpr Arch::FlagID RV_FLAG_AIC_FINISH_STORE = 9;
    Arch::CrossCoreFlagWithReverse<> flagAicFinishStore{FLAG_AIC_FINISH_STORE, RV_FLAG_AIC_FINISH_STORE};
    Arch::Resource<ArchTag> resource;

    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 0;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
};

}  // namespace Catlass::Gemv::kernel

#endif  // CATLASS_GEMV_KERNLE_GEMV_AIC_FT_HPP

