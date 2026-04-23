#ifndef CATLASS_GEMM_KERNEL_MATMUL_FT_THRESHOLD_SPLITK_MIXED_HPP_AIC
#define CATLASS_GEMM_KERNEL_MATMUL_FT_THRESHOLD_SPLITK_MIXED_HPP_AIC

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemv/helper.hpp"
#include "catlass/gemv_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include <cmath>

// class BlockEpilogue_,
// class BlockGemv_,
// class BlockCompare_,
// class BlockThreCalc_
// class BlockCompareRaw_
// class BlockFTSum_,
// class BlockFTGemv_,
// class BlockMeanMax_,
// class BlockSumGemv_,
namespace CubeSelf::Gemm::Kernel{
    // Template for matmul add kernel. Compute D = A * B + X
template <
    class BlockMmad_,
    class BlockScheduler_,
    class BlockMatrixAdd_,
    class BlockSliceSum_,
    class BlockMeanMax_
>
class MatmulFTSplitAICMixed {
public:
    using BlockMmad = BlockMmad_;
    using BlockMatrixAdd = BlockMatrixAdd_;
    // using BlockGemv = BlockGemv_;
    using BlockSliceSum = BlockSliceSum_;
    // using BlockSumGemv = BlockSumGemv_;
    // using BlockThreCalc = BlockThreCalc_;

    // using BlockFTSum = BlockFTSum_;
    // using BlockFTGemv = BlockFTGemv_;

    using BlockMeanMax = BlockMeanMax_;

    // using BlockEpilogue = BlockEpilogue_;
    using FT_ENC_TYPE = Catlass::Gemv::helper::FT_ENC_TYPE;
    using FT_COMP_TYPE = Catlass::Gemv::helper::FT_COMP_TYPE;

    using FT_AIV_PIPE_FUSE_TYPE = Catlass::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE;
    using FT_THRESHOLD_ALGORITHM = Catlass::Gemv::helper::FT_THRESHOLD_ALGORITHM;

    using FT_RCE_THRE_TYPE = Catlass::Gemv::helper::FT_RCE_THRE_TYPE;
    
    // static const FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE = BlockFTGemv::FUSE_TYPE;
    static const FT_THRESHOLD_ALGORITHM ALGO_TYPE = FT_THRESHOLD_ALGORITHM::ASVAR;

    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using L0TileShape = typename BlockMmad::L0TileShape;

    using L0TileShapeforFT = typename BlockMmad::L0TileShapeforFT;

    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;

    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;

    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    using LayoutACol = typename std::conditional<
        std::is_same<LayoutA, Catlass::layout::RowMajor>::value,
        Catlass::layout::ColumnMajor,
        Catlass::layout::RowMajor>::type;

    using LayoutBCol = typename std::conditional<
        std::is_same<LayoutB, Catlass::layout::RowMajor>::value,
        Catlass::layout::ColumnMajor,
        Catlass::layout::RowMajor>::type;
    
    using ElementXforFT = typename BlockMmad::ElementX;

    // using ElementX = typename BlockFTSum::ElementX;
    // using LayoutX = typename BlockFTSum::LayoutX;

    using LayoutX = typename BlockMmad::LayoutX;
    using ElementY = typename BlockMmad::ElementY;
    // typename BlockFTGemv::ElementX;
    using LayoutY = Catlass::layout::VectorLayout;
    // typename BlockFTGemv::LayoutX;
    using LayoutYforFT = typename BlockMmad::LayoutY;

    using LayoutCCol = typename std::conditional<
        std::is_same<LayoutC, Catlass::layout::RowMajor>::value,
        Catlass::layout::ColumnMajor,
        Catlass::layout::RowMajor>::type;
    
    using CColType = Catlass::Gemm::GemmType<ElementC, LayoutCCol>;

    using ElementAccumulator = 
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementXforFT, ElementXforFT>::ElementAccumulator;

    using ElementZ = typename BlockMmad::ElementY;
    // typename BlockFTGemv::ElementY;
    using LayoutZ = LayoutY;
    // typename BlockFTGemv::LayoutY;

    using ElementZforBRed = ElementZ;

    using ElementCOMPX = ElementZ;
    using LayoutCOMPX = Catlass::layout::VectorLayout;

    using ElementCOMPY = ElementZ;
    using LayoutCOMPY = Catlass::layout::VectorLayout;

    using ElementSliceIn = ElementZ;
    using LayoutSliceIn = LayoutYforFT;

    using ElementSliceOut = ElementZ;
    using LayoutSliceOut = LayoutYforFT;
    // Catlass::layout::VectorLayout;

    // using UBTileShape = typename BlockSumGemv::UBTileShape;

    // using UBTileShapeBE = typename BlockFTSum::UBTileShape;
    // using UBBlockShapeBE = typename BlockFTSum::UBBlockShape;

    // using UBTileShapeABE = typename BlockFTGemv::UBTileShape;
    // using UBBlockShapeABE = typename BlockFTGemv::UBBlockShape;

    using UBTileShapeBReduce = typename BlockMeanMax::UBTileShape;

    using UBTileShapeCEReduce = typename BlockSliceSum::UBTileShape;
    using UBTileShapeMatAdd = typename BlockMatrixAdd::UBTileShape;
    using UBBlockShapeMatAdd = typename BlockMatrixAdd::UBBlockShape;

    using UBAlignHelper = Catlass::Gemv::helper::UBAlignHelper<ElementA>;
    // using COMPUBTileShape = typename BlockThreCalc::UBTileShapeTotal;

    // using ThreCalcUBBlockShape = typename BlockThreCalc::UBTileShapeTotal;
    // using ThreCalcUBTileShape = typename BlockThreCalc::UBTileShape;

    // using COMPUBTileShape = typename BlockFTGemv::ThreCalcUBTileShapeTotal;

    // using ThreCalcUBBlockShape = typename BlockFTGemv::ThreCalcUBTileShapeTotal;
    // using ThreCalcUBTileShape = typename BlockFTGemv::ThreCalcUBTileShape;

    using SliceSumUBTileShape = typename BlockSliceSum::UBTileShape;

    
    // using ElementCOMPZ = typename BlockThreCalc::ElementZ;
    // using ElementCOMPZ = typename BlockFTGemv::ElementZ;
    using LayoutCOMPZ = Catlass::layout::VectorLayout;

    // using ElementWork = typename std::conditional<
    //     (BlockThreCalc::COMP_TYPE == FT_COMP_TYPE::XOR),
    //     uint16_t,
    //     typename std::conditional<(BlockThreCalc::COMP_TYPE == FT_COMP_TYPE::COMPARE), int32_t, ElementCOMPX>::type>::type;

    // using ElementWork = typename std::conditional<
    //     (BlockFTGemv::COMP_TYPE == FT_COMP_TYPE::XOR),
    //     uint16_t,
    //     typename std::conditional<(BlockFTGemv::COMP_TYPE == FT_COMP_TYPE::COMPARE), int32_t, ElementCOMPX>::type>::type;

    using BlockScheduler = BlockScheduler_;

    // using BlockCompareRaw = BlockCompareRaw_;

    // static_assert(std::is_same_v<typename BlockSumGemv::ElementA, ElementA> &&
    //     std::is_same_v<typename BlockSumGemv::LayoutA, LayoutA>,
    //     "The AType of Mmad and GEMV should be consistent.");
    
    // static_assert(std::is_same_v<typename BlockSumGemv::ElementB, ElementB> &&
    //     std::is_same_v<typename BlockSumGemv::LayoutB, LayoutB>,
    //     "The AType of Mmad and GEMV should be consistent.");
    static_assert(std::is_same_v<LayoutA, LayoutB>,
        "The LayoutA and LayoutB of Gemm should be consistent.");

    enum class AivCore {
        AIV0 = 0,
        AIV1
    };

    
    
    /// Parameters structure
    struct Params {
        // Data members
        Catlass::GemmCoord problemGemmShape; 
        Catlass::GemvCoord problemShape; 
        Catlass::GemvCoord problemShapeCol;
        Catlass::GemvCoord problemCompShape;
        Catlass::GemvCoord problemSliceShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        LayoutACol layoutACol;
        GM_ADDR ptrB;
        LayoutB layoutB;
        LayoutBCol layoutBCol;
        GM_ADDR ptrC;
        GM_ADDR ptrCout;
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
        GM_ADDR ptrBMean;
        GM_ADDR ptrBMax;
        GM_ADDR ptrThreZ;
        LayoutCOMPX layoutThre;
        ElementZ rounding_alpha;
        float e_max;
        float std_est_A_row_ratio;
        float A_row_scale_ratio;
        float std_est_ratios[2];
        float kn_ratios[2];
        float kn_scale_ratios[2];
        float kn_sqrt_ratios[2];
        float k_sqrt_n_ratios[2];
        uint32_t SplitNnum;
        uint32_t SplitBReduceM;
        uint32_t SplitBReduceN;
        uint32_t SplitCEReduceM;
        uint32_t SplitCEReduceN;
        bool outputThre;
        bool outputABE;
        uint32_t k_FT_tile_real;
        uint32_t ft_k_interval_real;
        uint32_t kFTRoundCountReal;
        uint32_t slice_k_num_limit;
        // GM_ADDR ptrWorkspace;
        // EpilogueParams epilogueParams;
        // GM_ADDR ptrC;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {};

        CATLASS_HOST_DEVICE
        Params(
            Catlass::GemmCoord const &problemGemmShape_,
            Catlass::GemvCoord const &problemShape_,
            Catlass::GemvCoord const &problemShapeCol_,
            Catlass::GemvCoord const &problemCompShape_,
            Catlass::GemvCoord const &problemSliceShape_,
            GM_ADDR ptrA_, LayoutA layoutA_, LayoutACol layoutACol_,
            GM_ADDR ptrB_, LayoutB layoutB_, LayoutBCol layoutBCol_,
            GM_ADDR ptrC_, GM_ADDR ptrCout_,
            LayoutC layoutC_, LayoutCCol layoutCCol_,
            GM_ADDR ptrX_, LayoutX layoutX_, GM_ADDR ptrXV_, LayoutX layoutXV_,
            GM_ADDR ptrWorkspace_,
            FT_ENC_TYPE enc_type_, GM_ADDR ptrZRow_, GM_ADDR ptrZCol_,
            GM_ADDR ptrZRow2_, GM_ADDR ptrZCol2_, GM_ADDR ptrCOMPZRow_,
            LayoutCOMPZ layoutCOMPZRow_, GM_ADDR ptrCOMPZCol_,
            LayoutCOMPZ layoutCOMPZCol_, LayoutCOMPX layoutCOMPX_,
            LayoutCOMPY layoutCOMPY_, uint32_t UbNum_,
            bool OutputWorkspace_, ElementCOMPX threshold_,
            GM_ADDR ptrBMean_, GM_ADDR ptrBMax_,
            GM_ADDR ptrThreZ_, LayoutCOMPX layoutThre_, ElementZ rounding_alpha_,
            float e_max_, float std_est_A_row_ratio_, float A_row_scale_ratio_, 
            const float (&std_est_ratios_)[2],
            const float (&kn_ratios_)[2],
            const float (&kn_scale_ratios_)[2],
            const float (&kn_sqrt_ratios_)[2],
            const float (&k_sqrt_n_ratios_)[2],
            uint32_t SplitNnum_, 
            uint32_t SplitBReduceM_, uint32_t SplitBReduceN_,
            uint32_t SplitCEReduceM_, uint32_t SplitCEReduceN_,
            bool outputThre_, bool outputABE_, uint32_t k_FT_tile_real_,
            uint32_t ft_k_interval_real_, uint32_t kFTRoundCountReal_,
            uint32_t slice_k_num_limit_
        ) : problemGemmShape(problemGemmShape_), problemShape(problemShape_),
            problemShapeCol(problemShapeCol_), problemCompShape(problemCompShape_),
            problemSliceShape(problemSliceShape_),
            ptrA(ptrA_), layoutA(layoutA_), layoutACol(layoutACol_),
            ptrB(ptrB_), layoutB(layoutB_), layoutBCol(layoutBCol_),
            ptrC(ptrC_), ptrCout(ptrCout_),
            layoutC(layoutC_), layoutCCol(layoutCCol_),
            ptrX(ptrX_), layoutX(layoutX_), ptrXV(ptrXV_), layoutXV(layoutXV_),
            ptrWorkspace(ptrWorkspace_),
            enc_type(enc_type_), ptrZRow(ptrZRow_), ptrZCol(ptrZCol_),
            ptrZRow2(ptrZRow2_), ptrZCol2(ptrZCol2_), ptrCOMPZRow(ptrCOMPZRow_),
            layoutCOMPZRow(layoutCOMPZRow_), ptrCOMPZCol(ptrCOMPZCol_),
            layoutCOMPZCol(layoutCOMPZCol_), layoutCOMPX(layoutCOMPX_),
            layoutCOMPY(layoutCOMPY_), 
            UbNum(UbNum_), OutputWorkspace(OutputWorkspace_), threshold(threshold_), 
            ptrBMean(ptrBMean_), ptrBMax(ptrBMax_),
            ptrThreZ(ptrThreZ_), layoutThre(layoutThre_), 
            rounding_alpha(rounding_alpha_), e_max(e_max_), 
            std_est_A_row_ratio(std_est_A_row_ratio_), A_row_scale_ratio(A_row_scale_ratio_),
            SplitNnum(SplitNnum_), 
            SplitBReduceM(SplitBReduceM_), SplitBReduceN(SplitBReduceN_),
            SplitCEReduceM(SplitCEReduceM_), SplitCEReduceN(SplitCEReduceN_),
            outputThre(outputThre_), outputABE(outputABE_), 
            k_FT_tile_real(k_FT_tile_real_), ft_k_interval_real(ft_k_interval_real_),
            kFTRoundCountReal(kFTRoundCountReal_), slice_k_num_limit(slice_k_num_limit_){
                for (int i = 0; i < 2; ++i) {
                    this->std_est_ratios[i] = std_est_ratios_[i];
                    this->kn_ratios[i] = kn_ratios_[i];
                    this->kn_scale_ratios[i] = kn_scale_ratios_[i];
                    this->kn_sqrt_ratios[i] = kn_sqrt_ratios_[i];
                    this->k_sqrt_n_ratios[i] = k_sqrt_n_ratios_[i];
                }
            } 
    };

    struct Arguments {
        Catlass::GemmCoord problemGemmShape;
        Catlass::GemvCoord problemShape;
        size_t elementSize;
        GM_ADDR ptrX;
        GM_ADDR ptrXV;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
        GM_ADDR ptrCout;
        GM_ADDR ptrZRow;
        GM_ADDR ptrZCol;
        GM_ADDR ptrZRow2;
        GM_ADDR ptrZCol2;
        GM_ADDR ptrCOMPZRow;
        GM_ADDR ptrCOMPZCol;
        GM_ADDR ptrBMean;
        GM_ADDR ptrBMax;
        GM_ADDR ptrThreZ;
        FT_ENC_TYPE enc_type;
        uint32_t UbNum;
        bool OutputWorkspace;
        ElementCOMPX threshold;
        float rounding_exponent;
        float size_beta;
        float e_max_raw;
        uint32_t reduce_cores;
        FT_RCE_THRE_TYPE rce_thre_type;
        bool outputThre;
        bool outputABE;
        uint32_t slice_k_num_limit;
    };

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        // args.problemGemmShape.m() + 
        // return args.elementSize * args.problemGemmShape.m() * args.problemGemmShape.n();
        uint32_t splitNnum = ((args.problemGemmShape.n() + L1TileShape::N - 1) / L1TileShape::N);
        uint kFTRoundCount = ((args.problemGemmShape.k() + L0TileShapeforFT::K - 1) / L0TileShapeforFT::K);
        /*
        uint32_t k_FT_tile_real;
        uint32_t ft_k_interval_real;
        uint32_t kFTRoundCountReal;
        */
        uint32_t k_FT_tile_real = L0TileShapeforFT::K;
        uint32_t ft_k_interval_real = L0TileShapeforFT::K / L1TileShape::K;
        if(kFTRoundCount > args.slice_k_num_limit){
            k_FT_tile_real = (args.problemGemmShape.k() + args.slice_k_num_limit - 1) / args.slice_k_num_limit;

            uint32_t k_FT_tile_l1num = (k_FT_tile_real + L1TileShape::K - 1) / L1TileShape::K;
            k_FT_tile_real = L1TileShape::K * k_FT_tile_l1num;
            ft_k_interval_real = k_FT_tile_real / L1TileShape::K;
        }else{
            ft_k_interval_real = L0TileShapeforFT::K / L1TileShape::K;
        }

        uint32_t kFTRoundCountReal = (args.problemGemmShape.k() + k_FT_tile_real -1) / k_FT_tile_real;
        if(k_FT_tile_real > args.problemGemmShape.k()){
            k_FT_tile_real = args.problemGemmShape.k();
            ft_k_interval_real = ((args.problemGemmShape.k() + L1TileShape::K - 1) / L1TileShape::K);
            kFTRoundCountReal = 1;
        }
        // + (splitNnum * args.problemGemmShape.m()) + (splitNnum * args.problemGemmShape.m()))
        return sizeof(ElementY) * (splitNnum * args.problemGemmShape.m() * kFTRoundCountReal + args.problemGemmShape.m() * args.problemGemmShape.n() * kFTRoundCountReal);
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

        Catlass::GemvCoord problemShapeCol{n, m};
        problemShapeCol.m() = n;
        problemShapeCol.n() = m;

        LayoutA layoutA{m2, k2};
        LayoutACol layoutACol{k2, m2};

        LayoutB layoutB{k2, n2};
        LayoutBCol layoutBCol{n2, k2};

        LayoutC layoutC{m, n};
        LayoutCCol layoutCCol{n, m};
        LayoutX layoutX{m+n};

        LayoutZ layoutZRow{m};
        LayoutZ layoutZCol{n};

        uint32_t xlen = (m2 > n2) ? m2 : n2;
        LayoutX layoutXV{xlen};

        LayoutCOMPX layoutThre{m};

        // float input_exponent = (args.rounding_exponent < 0.0f) ? args.rounding_exponent : (0.0 - args.rounding_exponent);

        // float rounding_error = std::pow(2.0f,input_exponent);

        // float row_sqrt = 1.0f;

        float slice_N = L1TileShape::N;

        // if(args.size_beta < 1.0f){
        //     row_sqrt = std::sqrt(slice_N*1.0f);
        // }else{
        //     row_sqrt = args.size_beta;
        // }

        ElementZ rounding_alpha = (ElementZ)1.0f;
        // static_cast<ElementZ>(row_sqrt * rounding_error);

        float e_max = (args.e_max_raw * 1.0f);

        uint32_t SplitBReduceM = (SplitNnum + args.reduce_cores - 1) / args.reduce_cores;

        SplitBReduceM = (SplitBReduceM >= UBTileShapeBReduce::M) ? UBTileShapeBReduce::M : SplitBReduceM;

        uint32_t SplitCEReduceM = (SplitNnum + args.reduce_cores - 1) / args.reduce_cores;

        SplitCEReduceM = (SplitCEReduceM >= UBTileShapeCEReduce::M) ? UBTileShapeCEReduce::M : SplitCEReduceM;

        SplitCEReduceM = (SplitCEReduceM < 2) ? 2 : SplitCEReduceM;

        uint32_t SplitBReduceN_num = (k2 + UBTileShapeBReduce::N - 1) / UBTileShapeBReduce::N;
        uint32_t SplitBReduceN = UBTileShapeBReduce::N;
        if(SplitBReduceN_num < 2){
            SplitBReduceN = (k2 + 2 - 1) / 2;
        }

        uint32_t SplitCEReduceN_num = (m2 + UBTileShapeCEReduce::N - 1) / UBTileShapeCEReduce::N;
        uint32_t SplitCEReduceN = UBTileShapeCEReduce::N;
        if(SplitCEReduceN_num < 2){
            SplitCEReduceN = (m2 + 2 - 1) / 2;
        }

        float std_est_ratios[2];
        float kn_ratios[2];
        float kn_scale_ratios[2];
        float kn_sqrt_ratios[2];
        float k_sqrt_n_ratios[2];

        uint32_t n_remain_split = (args.problemGemmShape.n() % L1TileShape::N);

        float common_size = k2 * L1TileShape::N * 1.0f;
        float remain_size = k2 * n_remain_split * 1.0f;

        float common_std_factor = std::sqrt(2.0f * logf(common_size));
        float remain_std_factor = common_std_factor;

        float std_est_ratio_common = (1.0f / common_std_factor);
        float std_est_ratio_remain = std_est_ratio_common;

        float common_kn_ratio = (1.0f * common_size);
        float remain_kn_ratio = common_kn_ratio;

        float common_kn_sqrt_ratio = (std::sqrt(common_kn_ratio));
        float remain_kn_sqrt_ratio = common_kn_sqrt_ratio;

        float common_k_sqrt_n_ratio = (std::sqrt(k2*1.0f) *(L1TileShape::N * 1.0f));
        float remain_k_sqrt_n_ratio = common_k_sqrt_n_ratio;

        float std_est_A_row_factor = std::sqrt(2.0f * logf((k2 * 1.0f)));
        float std_est_A_row_ratio = (1.0f / std_est_A_row_factor);
        float A_row_scale_ratio = 1.0f / (1.0f * k2);

        if(n_remain_split > 0){
            remain_std_factor = std::sqrt(2.0f * logf(remain_size));
            std_est_ratio_remain = (1.0f / remain_std_factor);

            remain_kn_ratio = (1.0f * remain_size);
            remain_kn_sqrt_ratio = (std::sqrt(n_remain_split));
            remain_k_sqrt_n_ratio = (std::sqrt(k2*1.0f) *(n_remain_split * 1.0f));
        }

        std_est_ratios[0] = std_est_ratio_common;
        std_est_ratios[1] = std_est_ratio_remain;

        kn_ratios[0] = common_kn_ratio;
        kn_ratios[1] = remain_kn_ratio;

        kn_scale_ratios[0] = (1.0f / common_kn_ratio);
        kn_scale_ratios[1] = (1.0f / remain_kn_ratio);

        kn_sqrt_ratios[0] = common_kn_sqrt_ratio;
        kn_sqrt_ratios[1] = remain_kn_sqrt_ratio;

        k_sqrt_n_ratios[0] = common_k_sqrt_n_ratio;
        k_sqrt_n_ratios[1] = remain_k_sqrt_n_ratio;

        uint kFTRoundCount = ((args.problemGemmShape.k() + L0TileShapeforFT::K - 1) / L0TileShapeforFT::K);
        /*
        uint32_t k_FT_tile_real;
        uint32_t ft_k_interval_real;
        uint32_t kFTRoundCountReal;
        */
        uint32_t k_FT_tile_real = L0TileShapeforFT::K;
        uint32_t ft_k_interval_real = L0TileShapeforFT::K / L1TileShape::K;
        if(kFTRoundCount > args.slice_k_num_limit){
            k_FT_tile_real = (args.problemGemmShape.k() + args.slice_k_num_limit - 1) / args.slice_k_num_limit;

            uint32_t k_FT_tile_l1num = (k_FT_tile_real + L1TileShape::K - 1) / L1TileShape::K;
            k_FT_tile_real = L1TileShape::K * k_FT_tile_l1num;
            ft_k_interval_real = k_FT_tile_real / L1TileShape::K;
        }else{
            ft_k_interval_real = L0TileShapeforFT::K / L1TileShape::K;
        }

        uint32_t kFTRoundCountReal = (args.problemGemmShape.k() + k_FT_tile_real -1) / k_FT_tile_real;
        if(k_FT_tile_real > args.problemGemmShape.k()){
            k_FT_tile_real = args.problemGemmShape.k();
            ft_k_interval_real = ((args.problemGemmShape.k() + L1TileShape::K - 1) / L1TileShape::K);
            kFTRoundCountReal = 1;
        }

        // printf("SplitBReduceM: %d\n", SplitBReduceM);
        // printf("SplitBReduceN: %d\n", SplitBReduceN);

        /*
        */
        
        Params params{
            problemGemmShape,
            problemShape,
            problemShapeCol,
            problemCompShape,
            problemSliceShape,
            args.ptrA, layoutA, layoutACol,
            args.ptrB, layoutB, layoutBCol,
            args.ptrC, args.ptrCout, layoutC, layoutCCol,
            args.ptrX, layoutX, args.ptrXV, layoutXV,
            workspace, args.enc_type, 
            args.ptrZRow, args.ptrZCol, args.ptrZRow2, args.ptrZCol2,
            args.ptrCOMPZRow, layoutCOMPZRow,
            args.ptrCOMPZCol, layoutCOMPZCol, 
            layoutCOMPX, layoutCOMPY, 
            args.UbNum, args.OutputWorkspace, args.threshold, 
            args.ptrBMean, args.ptrBMax,
            args.ptrThreZ, layoutThre, rounding_alpha, 
            e_max, std_est_A_row_ratio, A_row_scale_ratio,
            std_est_ratios, kn_ratios, kn_scale_ratios,
            kn_sqrt_ratios, k_sqrt_n_ratios,
            SplitNnum, SplitBReduceM, SplitBReduceN,
            SplitCEReduceM, SplitCEReduceN,
            args.outputThre, args.outputABE,k_FT_tile_real,ft_k_interval_real,kFTRoundCountReal,
            args.slice_k_num_limit
        };

        // printf("kn_scale_ratio: %f\n",params.kn_scale_ratios[0]);

        return params;
    }

    // Methods
    CATLASS_DEVICE
    MatmulFTSplitAICMixed() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    CATLASS_DEVICE
    void Matmul_op(Params const &params){
        BlockScheduler matmulBlockScheduler(params.problemGemmShape, Catlass::MakeCoord(L1TileShape::M,L1TileShape::N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);

        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);

        AscendC::GlobalTensor<ElementC> gmCSlice;
        // ptrWorkspace
        gmCSlice.SetGlobalBuffer((__gm__ ElementC *)params.ptrWorkspace);

        AscendC::GlobalTensor<ElementXforFT> gmX;
        gmX.SetGlobalBuffer((__gm__ ElementXforFT *)params.ptrX);

        AscendC::GlobalTensor<ElementSliceIn> gmY;
        gmY.SetGlobalBuffer((__gm__ ElementSliceIn *)params.ptrWorkspace);

        Catlass::layout::RowMajor layoutC(params.problemGemmShape.m(), params.problemGemmShape.n());
        LayoutYforFT layoutYforFT{params.SplitNnum, params.problemGemmShape.m()};

        LayoutX layoutX{params.problemGemmShape.n()};

        int64_t OffsetInCSliceWork = params.SplitNnum * params.problemGemmShape.m() * params.kFTRoundCountReal;
        int64_t gmOffsetCShape = params.problemGemmShape.m() * params.problemGemmShape.n();
        
        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            // Compute block location
            Catlass::GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            Catlass::GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
            Catlass::GemmCoord actualCoord = Catlass::GemmCoord({blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N, blockCoord.k() * L1TileShape::K});

            uint32_t splitNIdx = blockCoord.n() * L1TileShape::N / L1TileShape::N;
            // Compute initial location in logical coordinates
            Catlass::MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
            Catlass::MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
            Catlass::MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
            Catlass::MatrixCoord offsetYforFT{splitNIdx, blockCoord.m() * L1TileShape::M};
            
            int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
            int64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
            int64_t gmOffsetC = layoutC.GetOffset(offsetC);
            int64_t gmOffsetX = blockCoord.n() * L1TileShape::N;
            // params.problemGemmShape.m() +
            int64_t gmOffsetYforFT = layoutYforFT.GetOffset(offsetYforFT);
            int64_t gmOffsetCSlice = OffsetInCSliceWork + gmOffsetC; 

            /*
            CATLASS_DEVICE
            void operator()(
                AscendC::GlobalTensor<ElementA> const & gmA, LayoutA const &layoutA,
                AscendC::GlobalTensor<ElementB> const & gmB, LayoutB const &layoutB,
                AscendC::GlobalTensor<ElementC> const & gmC, LayoutC const &layoutC,
                AscendC::GlobalTensor<ElementX> const & gmBlockX, LayoutX const& layoutX,
                AscendC::GlobalTensor<ElementY> const & gmBlockY, LayoutY const& layoutY,
                Catlass::GemmCoord const &actualShape, 
                Catlass::GemmCoord const& actualCoord,
                Catlass::GemmCoord const &totalProblemShape,
                Catlass::GemvCoord const &totalSliceShape,
                uint32_t slice_k_num_limit, uint32_t kFTRoundCountReal, 
                uint32_t ft_k_interval_real,
                Catlass::Arch::CrossCoreFlagWithReverse<> & flagAicFinishStore
            )
            */
            
            blockMmad(
                gmA[gmOffsetA], params.layoutA,
                gmB[gmOffsetB], params.layoutB,
                gmCSlice[gmOffsetCSlice], layoutC,
                gmX[gmOffsetX], layoutX,
                gmY[gmOffsetYforFT], layoutYforFT,
                actualBlockShape, actualCoord,
                params.problemGemmShape,params.problemSliceShape,
                params.slice_k_num_limit, params.kFTRoundCountReal,
                params.ft_k_interval_real, flagAicFinishStore);

            // // 通知相应 AIV core，MMAD计算已经完成了，结果已经写入了GM 
            // Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
            
            // Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
        }
    }

    template<>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        Matmul_op(params);
        AscendC::PipeBarrier<PIPE_ALL>();

    }


    CATLASS_DEVICE
    void CE_reduce_for_thre_op(Params const &params)
    {
        AscendC::SetAtomicNone();
        // Arch::Resource<ArchTag> resource;

        // Represent the full gm

        uint32_t aivIndex = AscendC::GetBlockIdx();
        int32_t aivSubIndex = AscendC::GetSubBlockIdx();
        // aivIndex % AscendC::GetSubBlockNum();
        // AscendC::GetSubBlockIdx();
        uint32_t aicoreIndex = aivIndex / AscendC::GetSubBlockNum();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t aivNum = aicoreNum * AscendC::GetSubBlockNum();

        uint32_t half_aiv_num = aivNum / 2;
        uint32_t half_aivIndex = aivIndex;
        if(aivIndex >= half_aiv_num){
            half_aivIndex = aivIndex - half_aiv_num;
        }
        uint32_t aiv_part_num = 1 * AscendC::GetTaskRation();
        // AivCore aivCore = static_cast<AivCore>(AscendC::GetSubBlockIdx());
        uint32_t align = Catlass::BYTE_PER_C0 / sizeof(ElementY);
        // uint32_t aicoreIndex = aivIndex / AscendC::GetTaskRation();

        AscendC::GlobalTensor<ElementSliceOut> gmZRow;
        gmZRow.SetGlobalBuffer((__gm__ ElementSliceOut *)params.ptrZRow);
        AscendC::GlobalTensor<ElementY> gmSliceCE;
        gmSliceCE.SetGlobalBuffer((__gm__ ElementY *)params.ptrWorkspace);

        // Get aicore information

        uint32_t UBTileSplitM = params.SplitCEReduceM;
        // RoundUp(UBTileShapeBReduce::M, UBAlignHelper::ALIGN);
        uint32_t UBTileNRound = RoundUp(params.SplitCEReduceN, UBAlignHelper::ALIGN);
        //uint32_t UBTileKRound = 1;
        //uint32_t UBTileMRound = 1;

        uint32_t Reduce_M_size = params.SplitNnum;
        uint32_t loopsNum = CeilDiv(Reduce_M_size, UBTileSplitM);
        //uint32_t loopsNum = params.problemGemmShape.k();

        BlockSliceSum blockSliceSum(resource);

        int64_t OffsetInSliceInit = 0;
        // params.SplitNnum * params.problemGemmShape.k();

        LayoutYforFT layoutSliceOut{params.SplitNnum, params.problemGemmShape.m()};
        LayoutYforFT layoutSliceIn{params.SplitNnum, params.problemGemmShape.m()};

        for(uint32_t loopId = aivIndex; loopId < loopsNum; loopId += aivNum) {
                 
            uint32_t mActual = ((int32_t)loopId == (int32_t)(loopsNum - 1)) ?
                        (Reduce_M_size - loopId * UBTileSplitM) : UBTileSplitM;

            uint32_t nActual = params.problemGemmShape.m();
            int64_t gmOffsetInCESlice = OffsetInSliceInit + loopId * UBTileSplitM * params.problemGemmShape.m();
            int64_t gmOffsetOutCEMat = loopId * UBTileSplitM * params.problemGemmShape.m();

            Catlass::GemvCoord actualBlockShape = Catlass::GemvCoord{mActual, nActual};
            /*
            // AscendC::GlobalTensor<ElementY> const &gmZ,
            CATLASS_DEVICE
            void operator()(
            AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
            AscendC::GlobalTensor<ElementY> const &gmY, LayoutY const &layoutY,
            GemvCoord const &actualShape, uint32_t NRealRound, uint32_t KFTRoundCount)
            */
            blockSliceSum(gmSliceCE[gmOffsetInCESlice], layoutSliceIn,
                    gmZRow[gmOffsetOutCEMat], layoutSliceOut,
                    actualBlockShape,
                    UBTileNRound, params.kFTRoundCountReal);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_DEVICE
    void Matrix_Slice_Add(Params const &params)
    {
        AscendC::SetAtomicNone();
        // Arch::Resource<ArchTag> resource;

        // Represent the full gm
        // Get aicore information

        BlockMatrixAdd blockMatrixAdd(resource);

        BlockScheduler matmulBlockScheduler(params.problemGemmShape, Catlass::MakeCoord(L1TileShape::M,L1TileShape::N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        // uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetTaskRation();
        // AscendC::printf("%zu\n",AscendC::GetBlockNum());
        uint32_t aivIndex = AscendC::GetBlockIdx();
        uint32_t aicoreIndex = aivIndex / AscendC::GetSubBlockNum();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t aivNum = aicoreNum * AscendC::GetSubBlockNum();
        uint32_t aiv_part_num = 1 * AscendC::GetTaskRation();

        // uint32_t aicoreIndex = aivIndex / AscendC::GetTaskRation();

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        AscendC::GlobalTensor<ElementC> gmCout;
        gmCout.SetGlobalBuffer((__gm__ ElementC *)params.ptrCout);

        AscendC::GlobalTensor<ElementY> gmCSlice;
        gmCSlice.SetGlobalBuffer((__gm__ ElementY *)params.ptrWorkspace);

        Catlass::MatrixCoord loopsMN = matmulBlockScheduler.loopsMN;

        /*
        return loopsMN.row() * loopsMN.column();
        */

        uint32_t UBTileMRound = RoundUp(UBTileShapeMatAdd::M, UBAlignHelper::ALIGN);
        uint32_t UBTileNRound = RoundUp(UBTileShapeMatAdd::N, UBAlignHelper::ALIGN);

        uint32_t UBBlockMRound = RoundUp(UBBlockShapeMatAdd::M, UBAlignHelper::ALIGN);
        uint32_t UBBlockNRound = RoundUp(UBBlockShapeMatAdd::N, UBAlignHelper::ALIGN);

        Catlass::layout::RowMajor layoutC(params.problemGemmShape.m(), params.problemGemmShape.n());
        LayoutYforFT layoutYforFT{params.SplitNnum, params.problemGemmShape.m()};

        int64_t OffsetInCSliceWork = params.SplitNnum * params.problemGemmShape.m() * params.kFTRoundCountReal;
        int64_t gmOffsetCShape = params.problemGemmShape.m() * params.problemGemmShape.n();
        uint32_t kFTRoundCountReal = params.kFTRoundCountReal;
        for (uint32_t loopIdx = aicoreIndex; loopIdx < coreLoops; loopIdx += aicoreNum) {
            // Compute block location
            Catlass::GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            // Catlass::GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
            Catlass::GemmCoord actualCoord = Catlass::GemmCoord({blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N, blockCoord.k() * L1TileShape::K});

            uint32_t splitNIdx = blockCoord.n() * L1TileShape::N / L1TileShape::N;
            // Compute initial location in logical coordinates
            Catlass::MatrixCoord offsetCInit{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};

            uint32_t mActual = UBBlockMRound;

            if(blockCoord.m() == loopsMN.row() - 1) {
                mActual = params.problemGemmShape.m() - blockCoord.m() * L1TileShape::M;
            }

            uint32_t nActual = UBBlockNRound;

            if(blockCoord.n() == loopsMN.column() - 1) {
                nActual = params.problemGemmShape.n() - blockCoord.n() * L1TileShape::N;
            }
            
            int64_t gmOffsetCInit = params.layoutC.GetOffset(offsetCInit);

            int64_t gmOffsetCSliceInit = OffsetInCSliceWork + gmOffsetCInit; 
            
            Catlass::GemvCoord actualBlockShape = Catlass::GemvCoord{mActual, nActual};

            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);

            blockMatrixAdd(
                gmC[gmOffsetCInit],
                gmCSlice[gmOffsetCSliceInit], 
                layoutC,
                gmCout[gmOffsetCInit], layoutC,
                actualBlockShape, aiv_part_num);
            
            for(uint32_t kFTRoundIdx = 1; kFTRoundIdx < params.kFTRoundCountReal; kFTRoundIdx++){
                int64_t gmOffsetCSliceKRound = gmOffsetCSliceInit + kFTRoundIdx * gmOffsetCShape;
                Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);

                 /*
                CATLASS_DEVICE
                void operator()(
                    AscendC::GlobalTensor<ElementA> const &gmA1, 
                    AscendC::GlobalTensor<ElementA> const &gmA2,
                    LayoutA const &layoutA,
                    AscendC::GlobalTensor<ElementY> const &gmY, 
                    LayoutY const &layoutY,
                    GemvCoord const &actualShape, uint32_t aiv_part_num)
                */
                blockMatrixAdd(
                    gmCout[gmOffsetCInit],
                    gmCSlice[gmOffsetCSliceKRound], 
                    layoutC,
                    gmCout[gmOffsetCInit], layoutC,
                actualBlockShape, aiv_part_num);
            }    
            // Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // CATLASS_DEVICE
    // void SliceNSum_op(Params const &params)
    // {
    //     AscendC::SetAtomicNone();
    //     // Arch::Resource<ArchTag> resource;
    //     BlockSliceSum blockSliceSum(resource);
    //     uint32_t align = Catlass::BYTE_PER_C0 / sizeof(ElementSliceIn);
    //     uint32_t maxmPerBlock_round = RoundUp(SliceSumUBTileShape::M, align);
    //     uint32_t maxnPerBlock_round = RoundUp(SliceSumUBTileShape::N, align);
    //     uint32_t split = 1;

    //     LayoutYforFT layoutSliceIn{params.SplitNnum, params.problemGemmShape.m()};
    //     LayoutCOMPX layoutSliceOut{params.problemGemmShape.m()};

    //     // add split k
    //     uint32_t M_Split = params.SplitNnum;
    //     // RoundDown(, params.split) / params.split;
    //     uint32_t Nloopnum = CeilDiv(params.problemSliceShape.n(), maxnPerBlock_round);
    //     int32_t loopnum;

    //     loopnum = Nloopnum;

    //     uint32_t offset_matrix_CE;
    //     uint32_t offset_matrix_ABE;
    //     uint32_t offset_vector_out;
    //     uint32_t offset_vector_in = 0;
        
    //     // Represent the full gm
    //     AscendC::GlobalTensor<ElementSliceIn> gmIn;
    //     gmIn.SetGlobalBuffer((__gm__ ElementSliceIn *)params.ptrWorkspace);

    //     AscendC::GlobalTensor<ElementSliceOut> gmOutCE;
    //     gmOutCE.SetGlobalBuffer((__gm__ ElementSliceOut *)params.ptrZRow);

    //     AscendC::GlobalTensor<ElementSliceOut> gmOutABE;
    //     gmOutABE.SetGlobalBuffer((__gm__ ElementSliceOut *)params.ptrZRow2);


    //     uint32_t aiv_num = AscendC::GetBlockNum() * AscendC::GetTaskRation();
    //     for (uint32_t loop_id = 0; loop_id < loopnum; loop_id++) {
    //         uint32_t aiv_id = AscendC::GetBlockIdx();
    //         if (loop_id % aiv_num != aiv_id)
    //             continue;
    //         uint32_t n_actual = ((int32_t)loop_id > (int32_t)(loopnum - split - 1))
    //                                     ? params.problemSliceShape.n() - ((loop_id / split) * maxnPerBlock_round)
    //                                     : maxnPerBlock_round;
    //         uint32_t m_actual = params.problemSliceShape.m();

    //         if constexpr (std::is_same_v<LayoutYforFT, Catlass::layout::ColumnMajor>) {
    //             // params.problemGemmShape.m() +
    //             offset_matrix_CE = loop_id * maxnPerBlock_round * params.problemSliceShape.m() + params.SplitNnum * params.problemGemmShape.k() + params.SplitNnum * params.problemGemmShape.m();
    //             offset_matrix_ABE = loop_id * maxnPerBlock_round * params.problemSliceShape.m() + params.SplitNnum * params.problemGemmShape.k();
    //             offset_vector_out = (loop_id / split) * maxnPerBlock_round;

    //             m_actual = params.problemSliceShape.m();

    //         } else {
    //             // params.problemGemmShape.m() +
    //             offset_matrix_CE = loop_id * maxnPerBlock_round +  params.SplitNnum * params.problemGemmShape.k() + params.SplitNnum * params.problemGemmShape.m();
    //             offset_matrix_ABE = loop_id * maxnPerBlock_round +  params.SplitNnum * params.problemGemmShape.k();

    //             offset_vector_out = loop_id * maxnPerBlock_round;
    //         }
    //         Catlass::GemvCoord actualBlockShape = Catlass::GemvCoord{m_actual, n_actual};

    //         /*
    //         void operator()(AscendC::GlobalTensor<ElementA> const &gmA, 
    //         LayoutA const &layoutA,
    //         AscendC::GlobalTensor<ElementY> const &gmZ, 
    //         LayoutY const &layoutY, GemvCoord const &actualShape)
    //         */

    //         blockSliceSum(gmIn[offset_matrix_CE], layoutSliceIn,
    //             gmOutCE[offset_vector_out], layoutSliceOut,
    //             actualBlockShape);

    //         AscendC::PipeBarrier<PIPE_V>();

    //         blockSliceSum(gmIn[offset_matrix_ABE], layoutSliceIn,
    //             gmOutABE[offset_vector_out], layoutSliceOut,
    //             actualBlockShape);
            
    //     }

    //     AscendC::PipeBarrier<PIPE_ALL>();
    // }

    // CATLASS_DEVICE
    // void Threshold_op(Params const &params)
    // {
    //     AscendC::SetAtomicNone();
    //     // Arch::Resource<ArchTag> resource;
    //     BlockThreCalc blockThreCalc(resource);
    //     uint32_t align = Catlass::BYTE_PER_C0 / sizeof(ElementC);
    //     uint32_t maxmPerBlock_round = RoundUp(ThreCalcUBTileShape::M, align);
    //     uint32_t maxnPerBlock_round = RoundUp(ThreCalcUBTileShape::N, align);
    //     uint32_t split = 1;

    //     // add split k
    //     uint32_t N_Split = params.problemShape.n();
    //     // RoundDown(, params.split) / params.split;
    //     uint32_t Mloopnum = CeilDiv(params.problemShape.m(), maxmPerBlock_round);
    //     int32_t loopnum;
    //     // float Realbeta = params.alpha;
    //     if constexpr (std::is_same_v<LayoutC, Catlass::layout::ColumnMajor>) {
    //         loopnum = Mloopnum * split;
    //         // Realbeta = params.alpha;
    //     } else {
    //         loopnum = Mloopnum;
    //     }

    //     uint32_t offset_matrix;
    //     uint32_t offset_vector_out;
    //     uint32_t offset_vector_in = 0;

    //     // uint32_t total_workspace_offset = (params.problemShape.m() + params.problemGemmShape.k())*sizeof(ElementY) / sizeof(ElementZ);
    //     // Represent the full gm
    //     AscendC::GlobalTensor<ElementC> gmC;
    //     gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

    //     AscendC::GlobalTensor<ElementZ> gmT;
    //     gmT.SetGlobalBuffer((__gm__ ElementZ *)params.ptrThreZ);
    //     uint32_t aiv_num = AscendC::GetBlockNum() * AscendC::GetTaskRation();
    //     for (uint32_t loop_id = 0; loop_id < loopnum; loop_id++) {
    //         uint32_t aiv_id = AscendC::GetBlockIdx();
    //         if (loop_id % aiv_num != aiv_id)
    //             continue;
    //         uint32_t m_actual = ((int32_t)loop_id > (int32_t)(loopnum - split - 1))
    //                                     ? params.problemShape.m() - ((loop_id / split) * maxmPerBlock_round)
    //                                     : maxmPerBlock_round;
    //         uint32_t n_actual = params.problemShape.n();

    //         if constexpr (std::is_same_v<LayoutC, Catlass::layout::ColumnMajor>) {
    //             offset_matrix = (loop_id % split) * N_Split * params.problemShape.m() +
    //                             (loop_id / split) * maxmPerBlock_round;
    //             offset_vector_out = (loop_id / split) * maxmPerBlock_round;
    //             offset_vector_in = (loop_id % split) * N_Split;

    //             if ((loop_id % split) == split - 1) {
    //                 n_actual = params.problemShape.n() - N_Split * (split - 1);
    //             } else {
    //                 n_actual = N_Split;
    //             }
    //         } else {
    //             offset_matrix = loop_id * maxmPerBlock_round * params.problemShape.n();
    //             offset_vector_out = loop_id * maxmPerBlock_round;
    //         }
    //         Catlass::GemvCoord actualBlockShape = Catlass::GemvCoord{m_actual, n_actual};

    //         // float realbeta = (loop_id % split == 0) ? Realbeta : 0.0f;
    //         // +total_workspace_offset
    //         // params.rounding_alpha

    //         blockThreCalc(gmC[offset_matrix], params.layoutC,
    //             gmT[offset_vector_out], params.layoutThre,
    //             actualBlockShape, params.rounding_alpha);
    //     }

    //     AscendC::PipeBarrier<PIPE_ALL>();
    // }

    // CATLASS_DEVICE
    // void COMP_op_with_thre_vector(Params const &params, 
    //     uint32_t element_num, GM_ADDR ptrInputX, GM_ADDR ptrInputY, 
    //     GM_ADDR ptrOutputCOMP)
    // {
    //     AscendC::SetAtomicNone();
    //     // Arch::Resource<ArchTag> resource;

    //     BlockCompare blockCompare(resource);

    //     uint32_t align = Catlass::BYTE_PER_BLK / sizeof(ElementCOMPX);
    //     uint32_t total_block_elements = COMPUBTileShape::M * COMPUBTileShape::N * params.UbNum;
    //     uint32_t maxPerBlock_round = RoundUp(total_block_elements, align);

    //     // uint32_t total_workspace_offset = (params.problemShape.m() + params.problemGemmShape.k())*sizeof(ElementY) / sizeof(ElementZ);

    //     // uint32_t maxPerBlock_work = maxPerBlock_round * sizeof(ElementCOMPX) / sizeof(ElementWork);
    //     uint32_t maxPerBlock_out = maxPerBlock_round / 8;

    //     uint32_t total_input_elements = element_num;
        
    //     uint32_t total_input_bytes = total_input_elements * sizeof(ElementCOMPX);
    //     uint32_t total_output_elements = (total_input_elements + 8-1)/8;
    //     uint32_t total_workspace_bytes = RoundUp(total_input_bytes, static_cast<uint32_t>(sizeof(ElementWork)));

    //     uint32_t total_workspace_elements = total_workspace_bytes / sizeof(ElementWork);

    //     // add split k
    //     uint32_t loopnum = CeilDiv(total_input_elements, maxPerBlock_round);

    //     uint32_t offset_vector_in_x = 0;
    //     uint32_t offset_vector_in_y = 0;
    //     uint32_t offset_vector_out = 0;
    //     uint32_t offset_vector_workspace = 0;

    //     // Represent the full gm
    //     AscendC::GlobalTensor<ElementCOMPX> gmX;
    //     gmX.SetGlobalBuffer((__gm__ ElementCOMPX *)ptrInputX);
        
    //     AscendC::GlobalTensor<ElementCOMPY> gmY;
    //     gmY.SetGlobalBuffer((__gm__ ElementCOMPY *)ptrInputY);

    //     AscendC::GlobalTensor<ElementZ> gmW;
    //     gmW.SetGlobalBuffer((__gm__ ElementZ *)params.ptrThreZ);

    //     // AscendC::GlobalTensor<ElementWork> gmWork;
    //     // gmWork.SetGlobalBuffer((__gm__ ElementWork *)params.ptrWorkspace);
    //     AscendC::GlobalTensor<ElementCOMPZ> gmZ;
    //     gmZ.SetGlobalBuffer((__gm__ ElementCOMPZ *)ptrOutputCOMP);

    //     bool isFirstBlock = true;
    //     bool hasNextBlock = false;
    //     uint32_t aiv_num = AscendC::GetBlockNum() * AscendC::GetTaskRation();
    //     uint32_t aiv_id = AscendC::GetBlockIdx();

    //     for (uint32_t loop_id = aiv_id; loop_id < loopnum; loop_id+=aiv_num) {
            
    //         if (loop_id % aiv_num != aiv_id)
    //             continue;

    //         uint32_t InputGmBlockIdx = loop_id;
    //         uint32_t input_element_actual = (InputGmBlockIdx == loopnum - 1) ? (total_input_elements - InputGmBlockIdx * maxPerBlock_round) : maxPerBlock_round;

    //         int64_t gmOffsetX = InputGmBlockIdx * maxPerBlock_round;
    //         int64_t gmOffsetY = InputGmBlockIdx * maxPerBlock_round;
    //         int64_t gmOffsetW = InputGmBlockIdx * maxPerBlock_round;

    //         // int64_t gmOffsetWork = InputGmBlockIdx * maxPerBlock_work;
    //         int64_t gmOffsetZ = InputGmBlockIdx * maxPerBlock_out;

    //         Catlass::GemvCoord actualBlockShape = Catlass::GemvCoord{1, input_element_actual};

    //         uint32_t InputNextGmBlockIdx;
    //         int64_t gmOffsetNextX;
    //         int64_t gmOffsetNextY;
    //         int64_t gmOffsetNextW;

    //         // int64_t gmOffsetNextWork;
    //         int64_t gmOffsetNextZ;
    //         Catlass::GemvCoord nextActualBlockShape;

    //         if((loop_id + aiv_num) < loopnum){
    //             uint32_t nextLoopIdx = loop_id + aiv_num;
    //             InputNextGmBlockIdx = nextLoopIdx;

    //             uint32_t input_element_actual_next = 
    //                 (InputNextGmBlockIdx == loopnum - 1) ? (total_input_elements - InputNextGmBlockIdx * maxPerBlock_round) : maxPerBlock_round;
                
    //             nextActualBlockShape = Catlass::GemvCoord{1, input_element_actual_next};
                
    //             gmOffsetNextX = InputNextGmBlockIdx * maxPerBlock_round;
    //             gmOffsetNextY = InputNextGmBlockIdx * maxPerBlock_round;
    //             gmOffsetNextW = InputNextGmBlockIdx * maxPerBlock_round;

    //             // gmOffsetNextWork = InputNextGmBlockIdx * maxPerBlock_work;
    //             gmOffsetNextZ = InputNextGmBlockIdx * maxPerBlock_out;
    //         }

    //         LayoutCOMPX layoutInputX{element_num};
    //         LayoutCOMPY layoutInputY{element_num};
    //         LayoutCOMPX layoutInputW{element_num};
            
    //         LayoutCOMPZ layoutOutputZ{total_output_elements};

    //         // gmWork[gmOffsetWork], params.layoutWorkspace,
    //         blockCompare(gmX[gmOffsetX], layoutInputX,
    //                      gmY[gmOffsetY], layoutInputY,
    //                      gmW[gmOffsetW], layoutInputW,
    //                      gmX[gmOffsetNextX], gmY[gmOffsetNextY],
    //                      gmW[gmOffsetNextW],
    //                      gmZ[gmOffsetZ], layoutOutputZ,
    //                      actualBlockShape, nextActualBlockShape, isFirstBlock, 
    //                      hasNextBlock, params.OutputWorkspace, params.threshold);
    //     }

    //     AscendC::PipeBarrier<PIPE_ALL>();
    // }

    


    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params){

        Matrix_Slice_Add(params);

        Catlass::Arch::CrossCoreBarrierAIV<0x0, PIPE_V>();
        Catlass::Arch::CrossCoreBarrierAIV<0x0, PIPE_MTE3>();

        CE_reduce_for_thre_op(params);


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