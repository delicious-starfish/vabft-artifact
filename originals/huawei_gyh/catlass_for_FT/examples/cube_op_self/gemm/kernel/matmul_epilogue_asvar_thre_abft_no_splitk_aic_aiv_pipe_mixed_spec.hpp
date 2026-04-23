#ifndef CATLASS_GEMM_KERNEL_MATMUL_BE_ABE_ON_AIC_ASVAR_THRESHOLD_ABFT_NO_SPLITK_HPP_SPEC
#define CATLASS_GEMM_KERNEL_MATMUL_BE_ABE_ON_AIC_ASVAR_THRESHOLD_ABFT_NO_SPLITK_HPP_SPEC

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
namespace CubeSelf::Gemm::Kernel{
    // Template for matmul add kernel. Compute D = A * B + X
    // class BlockSumGemv_,
template <
    class BlockMmadABe_,
    class BlockMmad_,
    class BlockSchedulerFirst_,
    class BlockScheduler_,
    class BlockFTGemvAIC_,
    class BlockFTSum_,
    class BlockFTGemvAIV_,
    class BlockSliceRed_,
    class BlockSliceSum_
>
class MatmulAsVarABonAicNoSplitMixedSpec {
public:
    using BlockMmad = BlockMmad_;
    using BlockMmadABe = BlockMmadABe_;
    // using BlockGemv = BlockGemv_;
    using BlockSliceSum = BlockSliceSum_;
    // using BlockSumGemv = BlockSumGemv_;
    // using BlockThreCalc = BlockThreCalc_;

    using BlockFTSum = BlockFTSum_;
    using BlockFTGemvAIV = BlockFTGemvAIV_;
    using BlockSliceRed = BlockSliceRed_;

    using BlockFTGemvAIC = BlockFTGemvAIC_;

    // using BlockEpilogue = BlockEpilogue_;
    using FT_ENC_TYPE = Catlass::Gemv::helper::FT_ENC_TYPE;
    using FT_COMP_TYPE = Catlass::Gemv::helper::FT_COMP_TYPE;

    using FT_AIV_PIPE_FUSE_TYPE = Catlass::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE;
    using FT_THRESHOLD_ALGORITHM = Catlass::Gemv::helper::FT_THRESHOLD_ALGORITHM;

    using FT_RCE_THRE_TYPE = Catlass::Gemv::helper::FT_RCE_THRE_TYPE;
    
    static const FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE = BlockFTGemvAIV::FUSE_TYPE;
    static const FT_THRESHOLD_ALGORITHM ALGO_TYPE = FT_THRESHOLD_ALGORITHM::ASVAR;


    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using L0TileShape = typename BlockMmad::L0TileShape;

    using L1TileShapeforFT = typename BlockMmadABe::L1TileShapeforFT;
    using L0TileShapeforFT = typename BlockMmadABe::L0TileShapeforFT;

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
    
    using ElementXforFT = typename BlockMmadABe::ElementX;
    using LayoutXforFT = typename BlockMmadABe::LayoutX;

    using ElementYforFT = typename BlockMmadABe::ElementY;
    using LayoutYforFT = typename BlockMmadABe::LayoutY;

    using ElementYforB = typename BlockFTSum::ElementX;
    using LayoutYforB = typename BlockFTSum::LayoutX;

    using ElementYforA = typename BlockFTSum::ElementY;
    using LayoutYforA = typename BlockFTSum::LayoutY;

    using ElementX = typename BlockFTSum::ElementX;
    using LayoutX = typename BlockFTSum::LayoutX;

    using ElementY = typename BlockFTGemvAIC::ElementY;
    using LayoutY = typename BlockFTGemvAIV::LayoutX;
    

    using LayoutCCol = typename std::conditional<
        std::is_same<LayoutC, Catlass::layout::RowMajor>::value,
        Catlass::layout::ColumnMajor,
        Catlass::layout::RowMajor>::type;
    
    using CColType = Catlass::Gemm::GemmType<ElementC, LayoutCCol>;

    using ElementAccumulator = 
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementXforFT, ElementXforFT>::ElementAccumulator;

    using ElementZ = ElementYforFT;
    using ElementZInAiv = typename BlockFTGemvAIV::ElementY;
    using LayoutZ = typename BlockFTGemvAIV::LayoutY;

    using ElementZforBRed = ElementZ;

    using ElementCOMPX = ElementZ;
    using LayoutCOMPX = Catlass::layout::VectorLayout;

    using ElementCOMPY = ElementZ;
    using LayoutCOMPY = Catlass::layout::VectorLayout;

    using ElementSliceIn = ElementZ;
    using LayoutSliceIn = LayoutYforFT;

    using ElementSliceOut = ElementZ;
    using LayoutSliceOut = Catlass::layout::VectorLayout;

    // using UBTileShape = typename BlockSumGemv::UBTileShape;

    using UBTileShapeBMax = typename BlockFTSum::UBTileShapeforB;
    using UBBlockShapeBMax = typename BlockFTSum::UBBlockShapeforB;

    using UBTileShapeARed = typename BlockFTSum::UBTileShapeforA;

    using L1TileShapeBE = typename BlockFTGemvAIC::L1TileShape;
    using L0TileShapeBE = typename BlockFTGemvAIC::L0TileShape;
    using UBBlockShapeBE = typename BlockFTGemvAIC::UBBlockShape;

    using UBTileShapeCE = typename BlockFTGemvAIV::UBTileShape;
    using UBBlockShapeCE = typename BlockFTGemvAIV::UBBlockShape;

    using UBTileShapeBReduce = typename BlockSliceRed::UBTileShapeforB;
    using SliceSumUBTileShape = typename BlockSliceRed::UBTileShapeforA;

    using UBAlignHelper = Catlass::Gemv::helper::UBAlignHelper<ElementA>;

    using COMPUBTileShape = typename BlockFTGemvAIV::ThreCalcUBTileShapeTotal;

    using ThreCalcUBBlockShape = typename BlockFTGemvAIV::ThreCalcUBTileShapeTotal;
    using ThreCalcUBTileShape = typename BlockFTGemvAIV::ThreCalcUBTileShape;

    

    
    // using ElementCOMPZ = typename BlockThreCalc::ElementZ;
    using ElementCOMPZ = typename BlockFTGemvAIV::ElementZ;
    using LayoutCOMPZ = Catlass::layout::VectorLayout;

    using BlockScheduler = BlockScheduler_;
    using BlockSchedulerFirst = BlockSchedulerFirst_;

    static_assert(std::is_same_v<LayoutA, LayoutB>,
        "The LayoutA and LayoutB of Gemm should be consistent.");

    static_assert(std::is_same_v<ElementZ, ElementZInAiv>,
        "The LayoutA and LayoutB of Gemm should be consistent.");

    enum class AivCore {
        AIV0 = 0,
        AIV1
    };

    
    
    /// Parameters structure
    struct Params {
        // Data members
        Catlass::GemmCoord problemGemmShape; 
        Catlass::GemmCoord problemGemmShapeFirst;
        Catlass::GemmCoord problemGemmShapeRemain;
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
        GM_ADDR ptrBE;
        GM_ADDR ptrBMaxSlice;
        GM_ADDR ptrBMean;
        GM_ADDR ptrBMax;
        GM_ADDR ptrAMean;
        GM_ADDR ptrAMax;
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
        uint32_t SplitReduceM;
        uint32_t SplitReduceN;
        bool outputThre;
        bool outputCE;
        bool SplitKNum;
        // GM_ADDR ptrWorkspace;
        // EpilogueParams epilogueParams;
        // GM_ADDR ptrC;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {};

        CATLASS_HOST_DEVICE
        Params(
            Catlass::GemmCoord const &problemGemmShape_,
            Catlass::GemmCoord const &problemGemmShapeFirst_,
            Catlass::GemmCoord const &problemGemmShapeRemain_,
            Catlass::GemvCoord const &problemShape_,
            Catlass::GemvCoord const &problemShapeCol_,
            Catlass::GemvCoord const &problemCompShape_,
            Catlass::GemvCoord const &problemSliceShape_,
            GM_ADDR ptrA_, LayoutA layoutA_, LayoutACol layoutACol_,
            GM_ADDR ptrB_, LayoutB layoutB_, LayoutBCol layoutBCol_,
            GM_ADDR ptrC_, LayoutC layoutC_, LayoutCCol layoutCCol_,
            GM_ADDR ptrX_, LayoutX layoutX_, GM_ADDR ptrXV_, LayoutX layoutXV_,
            GM_ADDR ptrWorkspace_,
            FT_ENC_TYPE enc_type_, GM_ADDR ptrZRow_, GM_ADDR ptrZCol_,
            GM_ADDR ptrZRow2_, GM_ADDR ptrZCol2_, GM_ADDR ptrCOMPZRow_,
            LayoutCOMPZ layoutCOMPZRow_, GM_ADDR ptrCOMPZCol_,
            LayoutCOMPZ layoutCOMPZCol_, LayoutCOMPX layoutCOMPX_,
            LayoutCOMPY layoutCOMPY_, uint32_t UbNum_,
            bool OutputWorkspace_, ElementCOMPX threshold_,
            GM_ADDR ptrBE_,
            GM_ADDR ptrBMaxSlice_,
            GM_ADDR ptrBMean_, GM_ADDR ptrBMax_,
            GM_ADDR ptrAMean_, GM_ADDR ptrAMax_,
            GM_ADDR ptrThreZ_, LayoutCOMPX layoutThre_, ElementZ rounding_alpha_,
            float e_max_, float std_est_A_row_ratio_, float A_row_scale_ratio_, 
            const float (&std_est_ratios_)[2],
            const float (&kn_ratios_)[2],
            const float (&kn_scale_ratios_)[2],
            const float (&kn_sqrt_ratios_)[2],
            const float (&k_sqrt_n_ratios_)[2],
            uint32_t SplitNnum_, 
            uint32_t SplitReduceM_, uint32_t SplitReduceN_,
            bool outputThre_, bool outputCE_, uint32_t SplitKNum_
        ) : problemGemmShape(problemGemmShape_), 
            problemGemmShapeFirst(problemGemmShapeFirst_),
            problemGemmShapeRemain(problemGemmShapeRemain_),
            problemShape(problemShape_),
            problemShapeCol(problemShapeCol_), problemCompShape(problemCompShape_),
            problemSliceShape(problemSliceShape_),
            ptrA(ptrA_), layoutA(layoutA_), layoutACol(layoutACol_),
            ptrB(ptrB_), layoutB(layoutB_), layoutBCol(layoutBCol_),
            ptrC(ptrC_), layoutC(layoutC_), layoutCCol(layoutCCol_),
            ptrX(ptrX_), layoutX(layoutX_), ptrXV(ptrXV_), layoutXV(layoutXV_),
            ptrWorkspace(ptrWorkspace_),
            enc_type(enc_type_), ptrZRow(ptrZRow_), ptrZCol(ptrZCol_),
            ptrZRow2(ptrZRow2_), ptrZCol2(ptrZCol2_), ptrCOMPZRow(ptrCOMPZRow_),
            layoutCOMPZRow(layoutCOMPZRow_), ptrCOMPZCol(ptrCOMPZCol_),
            layoutCOMPZCol(layoutCOMPZCol_), layoutCOMPX(layoutCOMPX_),
            layoutCOMPY(layoutCOMPY_), 
            UbNum(UbNum_), OutputWorkspace(OutputWorkspace_), threshold(threshold_), 
            ptrBE(ptrBE_),
            ptrBMaxSlice(ptrBMaxSlice_),
            ptrBMean(ptrBMean_), ptrBMax(ptrBMax_),
            ptrAMean(ptrAMean_), ptrAMax(ptrAMax_),
            ptrThreZ(ptrThreZ_), layoutThre(layoutThre_), 
            rounding_alpha(rounding_alpha_), e_max(e_max_), 
            std_est_A_row_ratio(std_est_A_row_ratio_), A_row_scale_ratio(A_row_scale_ratio_),
            SplitNnum(SplitNnum_), 
            SplitReduceM(SplitReduceM_), SplitReduceN(SplitReduceN_),
            outputThre(outputThre_), outputCE(outputCE_), SplitKNum(SplitKNum_){
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
        GM_ADDR ptrZRow;
        GM_ADDR ptrZCol;
        GM_ADDR ptrZRow2;
        GM_ADDR ptrZCol2;
        GM_ADDR ptrCOMPZRow;
        GM_ADDR ptrCOMPZCol;
        GM_ADDR ptrBE;
        GM_ADDR ptrBMaxSlice;
        GM_ADDR ptrBMean;
        GM_ADDR ptrBMax;
        GM_ADDR ptrAMean;
        GM_ADDR ptrAMax;
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
        bool outputCE;
        uint32_t SplitKNum;
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
        // + (splitNnum * args.problemGemmShape.m()) + (splitNnum * args.problemGemmShape.m()))
        return sizeof(ElementYforFT) * (splitNnum * args.problemGemmShape.m() * args.SplitKNum);
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

        uint32_t FirstBlockN = SplitNnum;

        uint32_t RemainBlockN = n2;

        uint32_t SplitKNum = args.SplitKNum;
        if (SplitKNum < 1){
            SplitKNum = 1;
        }

        // GemmCoord problemGemmShape{128, 128, 128};
        Catlass::GemmCoord problemGemmShapeFirst{m2, FirstBlockN, k2};
        Catlass::GemmCoord problemGemmShapeRemain{m2, RemainBlockN, k2};

        problemGemmShapeFirst.m() = m2;
        problemGemmShapeFirst.n() = FirstBlockN;
        problemGemmShapeFirst.k() = k2;

        problemGemmShapeRemain.m() = m2;
        problemGemmShapeRemain.n() = RemainBlockN;
        problemGemmShapeRemain.k() = k2;

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

        uint32_t SplitReduceM = (SplitNnum + args.reduce_cores - 1) / args.reduce_cores;

        SplitReduceM = (SplitReduceM >= UBTileShapeBReduce::M) ? UBTileShapeBReduce::M : SplitReduceM;

        SplitReduceM = (SplitReduceM < 1) ? 1 : SplitReduceM;

        uint32_t SplitReduceN_num = (k2 + UBTileShapeBReduce::N - 1) / UBTileShapeBReduce::N;
        uint32_t SplitReduceN = UBTileShapeBReduce::N;
        if(SplitReduceN_num < 2){
            SplitReduceN = (k2 + 2 - 1) / 2;
        }

        // if(args.rce_thre_type == FT_RCE_THRE_TYPE::ROUND_WITH_ACC){
        //     float acc_rounding_error = std::pow(2.0f, -23.0f);
        //     float acc_scaling_factor = 1.0f * slice_N*(slice_N+1)*(2*slice_N+1) / 48.0f;
        //     acc_scaling_factor = std::sqrt(acc_scaling_factor);
        //     rounding_alpha = static_cast<ElementZ>(row_sqrt * rounding_error + acc_rounding_error * acc_scaling_factor); 
        // }

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

        // printf("SplitReduceM: %d\n", SplitReduceM);
        // printf("SplitReduceN: %d\n", SplitReduceN);

        /*
        Catlass::GemmCoord const &problemGemmShape_,
        Catlass::GemmCoord const &problemGemmShapeFirst_,
        Catlass::GemmCoord const &problemGemmShapeRemain_,
        Catlass::GemvCoord const &problemShape_,
        Catlass::GemvCoord const &problemShapeCol_,
        Catlass::GemvCoord const &problemCompShape_,
        Catlass::GemvCoord const &problemSliceShape_,
        */
        
        Params params{
            problemGemmShape,
            problemGemmShapeFirst,
            problemGemmShapeRemain,
            problemShape,
            problemShapeCol,
            problemCompShape,
            problemSliceShape,
            args.ptrA, layoutA, layoutACol,
            args.ptrB, layoutB, layoutBCol,
            args.ptrC, layoutC, layoutCCol,
            args.ptrX, layoutX, args.ptrXV, layoutXV,
            workspace, args.enc_type, 
            args.ptrZRow, args.ptrZCol, args.ptrZRow2, args.ptrZCol2,
            args.ptrCOMPZRow, layoutCOMPZRow,
            args.ptrCOMPZCol, layoutCOMPZCol, 
            layoutCOMPX, layoutCOMPY, 
            args.UbNum, args.OutputWorkspace, args.threshold, 
            args.ptrBE, args.ptrBMaxSlice,
            args.ptrBMean, args.ptrBMax,
            args.ptrAMean, args.ptrAMax,
            args.ptrThreZ, layoutThre, rounding_alpha, 
            e_max, std_est_A_row_ratio, A_row_scale_ratio,
            std_est_ratios, kn_ratios, kn_scale_ratios,
            kn_sqrt_ratios, k_sqrt_n_ratios,
            SplitNnum, SplitReduceM, SplitReduceN,
            args.outputThre, args.outputCE, SplitKNum
        };

        // printf("kn_scale_ratio: %f\n",params.kn_scale_ratios[0]);

        return params;
    }

    // Methods
    CATLASS_DEVICE
    MatmulAsVarABonAicNoSplitMixedSpec() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    CATLASS_DEVICE
    void BE_split_op_on_AIC(Params const &params)
    {
        // Arch::Resource<ArchTag> resource;

        // Represent the full gm
        // Get aicore information

        BlockFTGemvAIC blockFTGemvAIC(resource);
        
        uint32_t aicoreNum = AscendC::GetBlockNum();
        // BlockScheduler matmulBlockScheduler(params.problemGemmShape, Catlass::MakeCoord(L1TileShape::M,L1TileShape::N));
        // uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();
        // uint32_t aivNum = aicoreNum * AscendC::GetSubBlockNum();
        // AscendC::printf("%zu\n",AscendC::GetBlockNum());
        // uint32_t aicoreIndex = aivIndex / AscendC::GetTaskRation();

        AscendC::GlobalTensor<ElementX> gmXV;
        gmXV.SetGlobalBuffer((__gm__ ElementX *)params.ptrXV);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementY> gmY;
        gmY.SetGlobalBuffer((__gm__ ElementY *)params.ptrBE);

        LayoutYforFT layoutYforBE{params.SplitNnum, params.problemGemmShape.k()};
        LayoutX layoutXV{params.problemGemmShape.n()};
        // AscendC::GlobalTensor<ElementY> gmBMax;
        // gmBMax.SetGlobalBuffer((__gm__ ElementY *)params.ptrBMax);

        uint32_t TileKRound = L1TileShapeBE::M;
        uint32_t TileNRound = L1TileShape::N;

        uint32_t BlockKRound = UBBlockShapeBE::M;
        uint32_t BlockNRound = UBBlockShapeBE::N;

        uint32_t loopsNumN = CeilDiv(params.problemGemmShape.n(), BlockNRound);
        uint32_t loopsNumK = CeilDiv(params.problemGemmShape.k(), BlockKRound);

        uint32_t loopsNum = loopsNumK * loopsNumN;
        // CeilDiv(params.problemGemmShape.k(), UBTileKRound);
        //uint32_t loopsNum = params.problemGemmShape.k();

        float alpha{1.0};
        float beta{0.0};
        
        for(uint32_t loopId = AscendC::GetBlockIdx(); loopId < loopsNum; loopId += AscendC::GetBlockNum()) {
            
            uint32_t nLoopId = loopId / loopsNumK;
            uint32_t kLoopId = loopId % loopsNumK;

            int64_t gmOffsetX;
            int64_t gmOffsetB;
            int64_t gmOffsetY;
            int64_t gmOffsetNextX;
            int64_t gmOffsetNextB;
            int64_t gmOffsetNextY;
            // uint32_t aivId = AscendC::GetBlockIdx();
            // if (loopId % aivNum != aivId) continue;

            uint32_t nActual = ((int32_t)nLoopId == (int32_t)(loopsNumN - 1)) ?
                params.problemGemmShape.n() - nLoopId * BlockNRound : BlockNRound;

            uint32_t kActual = ((int32_t)kLoopId == (int32_t)(loopsNumK - 1)) ?
                params.problemGemmShape.k() - kLoopId * BlockKRound : BlockKRound;

             
            // params.SplitNnum

            int64_t gmOffsetBRow = kLoopId * BlockKRound;
            int64_t gmOffsetBCol = nLoopId * BlockNRound;
            uint32_t splitNIdx = nLoopId * BlockNRound / TileNRound;
            gmOffsetB = gmOffsetBRow * params.problemGemmShape.n() + gmOffsetBCol;
            gmOffsetY = splitNIdx * params.problemGemmShape.k() + kLoopId * BlockKRound;
            gmOffsetX = nLoopId * BlockNRound;

            Catlass::GemvCoord actualBlockShape = Catlass::GemvCoord{kActual, nActual};

            bool isFirstBlock = (loopId == AscendC::GetBlockIdx());
            // 
            bool hasNextBlock = false;
            uint32_t nLoopIdNext;
            uint32_t kLoopIdNext;
            Catlass::GemvCoord nextActualBlockShape;
            if (loopId + AscendC::GetBlockNum() < loopsNum) {
                hasNextBlock = true;
                uint32_t loopIdNext = loopId + AscendC::GetBlockNum();

                uint32_t nLoopIdNext = loopIdNext / loopsNumK;
                uint32_t kLoopIdNext = loopIdNext % loopsNumK;
                // uint32_t MNextGmActual =
                //     (MNextGmBlockIdx == MLoops - 1) ? (M - MNextGmBlockIdx * maxMPerBlock) : maxMPerBlock;

                uint32_t nActualNext = ((int32_t)nLoopIdNext == (int32_t)(loopsNumN - 1)) ?
                params.problemGemmShape.n() - nLoopIdNext * BlockNRound : BlockNRound;

                uint32_t kActualNext = ((int32_t)kLoopIdNext == (int32_t)(loopsNumK - 1)) ?
                params.problemGemmShape.k() - kLoopIdNext * BlockKRound : BlockKRound;

                nextActualBlockShape = Catlass::GemvCoord{kActualNext, nActualNext};

                int64_t gmOffsetBRowNext = kLoopIdNext * BlockKRound;
                int64_t gmOffsetBColNext = nLoopIdNext * BlockNRound;
                uint32_t splitNIdxNext = nLoopIdNext * BlockNRound / TileNRound;
                gmOffsetNextB = gmOffsetBRowNext * params.problemGemmShape.n() + gmOffsetBColNext;
                gmOffsetNextY = splitNIdxNext * params.problemGemmShape.k() + kLoopIdNext * BlockKRound;
                gmOffsetNextX = nLoopIdNext * BlockNRound;
            }

            LayoutYforFT layoutBlockBE{1,kActual};
            Catlass::layout::VectorLayout layoutE{nActual};

            /*
            void operator()(
                AscendC::GlobalTensor<ElementX> const& gmBlockX, LayoutX const& layoutX,
                AscendC::GlobalTensor<ElementA> const& gmBlockA, LayoutA const& layoutA,
                AscendC::GlobalTensor<ElementY> const& gmBlockY, LayoutY const& layoutY,
                AscendC::GlobalTensor<ElementX> const& gmNextBlockX,
                AscendC::GlobalTensor<ElementA> const& gmNextBlockA,
                GemvCoord const& actualShape, GemvCoord const& actualShapeNext,
                bool isFirstBlock, bool hasNextBlock)
            */
            
            blockFTGemvAIC(
                gmXV[gmOffsetX], layoutXV,
                gmB[gmOffsetB], params.layoutB,
                gmY[gmOffsetY], layoutYforBE,
                gmXV[gmOffsetNextX],
                gmB[gmOffsetNextB],
                actualBlockShape,
                nextActualBlockShape,
                isFirstBlock,hasNextBlock);
        }

        AscendC::PipeBarrier<PIPE_ALL>();

        // AscendC::SyncAll<true>();
    }

    CATLASS_DEVICE
    void ABe_spec_on_AIC(Params const &params){
        BlockSchedulerFirst matmulBlockSchedulerFirst(params.problemGemmShapeFirst, Catlass::MakeCoord(L1TileShapeforFT::M, L1TileShapeforFT::N));
        uint32_t coreLoops = matmulBlockSchedulerFirst.GetCoreLoops();

        uint32_t coreLoopsTotal = coreLoops * params.SplitKNum;

        uint32_t KBlockSize = (params.problemGemmShapeFirst.k() + params.SplitKNum - 1) / params.SplitKNum;

        BlockMmadABe blockMmadABe(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);

        AscendC::GlobalTensor<ElementXforFT> gmX;
        gmX.SetGlobalBuffer((__gm__ ElementXforFT *)params.ptrBE);

        AscendC::GlobalTensor<ElementYforFT> gmY;
        
        if(params.SplitKNum > 1){
            gmY.SetGlobalBuffer((__gm__ ElementYforFT *)params.ptrWorkspace);
        }else{
            gmY.SetGlobalBuffer((__gm__ ElementYforFT *)params.ptrZRow2);
        }
        

        LayoutXforFT layoutXforFTBlock{params.SplitNnum,params.problemGemmShape.k()};
        // , uint32_t(1)
        LayoutYforFT layoutXforFT{params.SplitNnum, params.problemGemmShape.k()};

        LayoutYforFT layoutYforFTBlock{params.SplitNnum, params.problemGemmShape.m()};
        LayoutYforFT layoutYforFT{params.SplitNnum, params.problemGemmShape.m()};


        // LayoutX layoutX{params.problemGemmShape.n()};
        // Catlass::MatrixCoord loopsMN = matmulBlockSchedulerFirst.loopsMN;

        int64_t base_slice_offset = params.SplitNnum * params.problemGemmShape.m();
        
        for (uint32_t loopIdxTotal = AscendC::GetBlockIdx(); loopIdxTotal < coreLoopsTotal; loopIdxTotal += AscendC::GetBlockNum()) {
            // Compute block location
            uint32_t KIdx = loopIdxTotal / coreLoops;
            uint32_t loopIdx = loopIdxTotal % coreLoops;
            uint32_t KStartCoord = KIdx * KBlockSize;
            uint32_t KBlockSizeActual = (KIdx == (params.SplitKNum - 1)) ? (params.problemGemmShapeFirst.k() - KStartCoord) : KBlockSize;
            Catlass::GemmCoord blockCoord = matmulBlockSchedulerFirst.GetBlockCoord(loopIdx);
            Catlass::GemmCoord actualBlockShapeInBlock = matmulBlockSchedulerFirst.GetActualBlockShape(blockCoord);

            Catlass::GemmCoord actualBlockShape = Catlass::GemmCoord({actualBlockShapeInBlock.m(), actualBlockShapeInBlock.n(), KBlockSizeActual});
            Catlass::GemmCoord actualCoord = Catlass::GemmCoord({blockCoord.m() * L1TileShapeforFT::M, blockCoord.n() * L1TileShapeforFT::N, KStartCoord});

            // Compute initial location in logical coordinates
            Catlass::MatrixCoord offsetA{blockCoord.m() * L1TileShapeforFT::M, KStartCoord};
            Catlass::MatrixCoord offsetXforFT{blockCoord.n() * L1TileShapeforFT::N, KStartCoord};
            Catlass::MatrixCoord offsetYforFT{blockCoord.n() * L1TileShapeforFT::N, blockCoord.m() * L1TileShapeforFT::M};
            
            
            int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
            int64_t gmOffsetXforFT = layoutXforFT.GetOffset(offsetXforFT);
            int64_t gmOffsetYforFTinBlock = layoutYforFT.GetOffset(offsetYforFT);

            int64_t gmOffsetYforFT = gmOffsetYforFTinBlock + KIdx * base_slice_offset;

            // Compute block-scoped matrix multiply-add
            /*
            /// Perform a block-scoped matrix multiply-accumulate
            CATLASS_DEVICE
            void operator()(
                AscendC::GlobalTensor<ElementA> const & gmA, LayoutA const &layoutA,
                AscendC::GlobalTensor<ElementX> const & gmX, LayoutX const &layoutX,
                AscendC::GlobalTensor<ElementY> const & gmY, LayoutY const &layoutY,
                Catlass::GemmCoord const &actualShape
            )
            */

            blockMmadABe(
                gmA[gmOffsetA], params.layoutA,
                gmX[gmOffsetXforFT], layoutXforFTBlock,
                gmY[gmOffsetYforFT], layoutYforFTBlock,
                actualBlockShape);
        }
    }

    CATLASS_DEVICE
    void Matmul_op(Params const &params)
    {
        
        BlockScheduler matmulBlockScheduler(params.problemGemmShapeRemain, Catlass::MakeCoord(L1TileShape::M,L1TileShape::N));
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

        Catlass::layout::RowMajor layoutC(params.problemGemmShape.m(), params.problemGemmShape.n());
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

            // Compute block-scoped matrix multiply-add
            blockMmad(
                gmA[gmOffsetA], params.layoutA,
                gmB[gmOffsetB], params.layoutB,
                gmC[gmOffsetC], layoutC,
                actualBlockShape);
            
            // 通知相应 AIV core，MMAD计算已经完成了，结果已经写入了GM 
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
        }

        // AscendC::PipeBarrier<PIPE_ALL>();
    }

    template<>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        BE_split_op_on_AIC(params);
        // AscendC::SyncAll<true>();
        Catlass::Arch::CrossCoreBarrierAIC<0x0, PIPE_FIX>();
        // Catlass::Arch::CrossCoreBarrierAIC<0x0, PIPE_M>();
        // Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
        // Matmul_op_ABe_fused(params);
        ABe_spec_on_AIC(params);
        Catlass::Arch::CrossCoreBarrierAIC<0x0, PIPE_FIX>();
        // Catlass::Arch::CrossCoreBarrierAIC<0x0, PIPE_M>();
        // AscendC::SyncAll<true>();
        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
        if(params.problemGemmShapeRemain.n() > 0){
            Matmul_op(params);   
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }


    CATLASS_DEVICE
    void AB_red_split_op(Params const &params)
    {
        AscendC::SetAtomicNone();
        // Arch::Resource<ArchTag> resource;

        // Represent the full gm
        // Get aicore information
        
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t aivNum = aicoreNum * AscendC::GetTaskRation();

        // BlockScheduler matmulBlockScheduler(params.problemGemmShape, Catlass::MakeCoord(L1TileShape::M,L1TileShape::N));
        // uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();
        // uint32_t aivNum = aicoreNum * AscendC::GetSubBlockNum();
        // AscendC::printf("%zu\n",AscendC::GetBlockNum());
        uint32_t aivIndex = AscendC::GetBlockIdx();
        uint32_t aicoreIndex = aivIndex / AscendC::GetSubBlockNum();
        // uint32_t aicoreIndex = aivIndex / AscendC::GetTaskRation();

        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);

        AscendC::GlobalTensor<ElementYforB> gmYforB;
        gmYforB.SetGlobalBuffer((__gm__ ElementYforB *)params.ptrBMaxSlice);

        AscendC::GlobalTensor<ElementYforA> gmAMean;
        gmAMean.SetGlobalBuffer((__gm__ ElementYforA *)params.ptrAMean);

        AscendC::GlobalTensor<ElementA> gmAMax;
        gmAMax.SetGlobalBuffer((__gm__ ElementA *)params.ptrAMax);

        uint32_t UBTileKRoundforB = RoundUp(UBTileShapeBMax::M, UBAlignHelper::ALIGN);
        uint32_t UBTileNRoundforB = RoundUp(UBTileShapeBMax::N, UBAlignHelper::ALIGN);

        uint32_t UBBlockKRoundforB = RoundUp(UBBlockShapeBMax::M, UBAlignHelper::ALIGN);
        uint32_t UBBlockNRoundforB = RoundUp(UBBlockShapeBMax::N, UBAlignHelper::ALIGN);

        uint32_t UBTileMRoundforA = RoundUp(UBTileShapeARed::M, UBAlignHelper::ALIGN);
        uint32_t UBTileKRoundforA = RoundUp(UBTileShapeARed::N, UBAlignHelper::ALIGN);

        //uint32_t UBTileKRound = 1;
        //uint32_t UBTileMRound = 1;

        uint32_t loopsNumNforB = CeilDiv(params.problemGemmShape.n(), UBBlockNRoundforB);
        uint32_t loopsNumKforB = CeilDiv(params.problemGemmShape.k(), UBBlockKRoundforB);

        uint32_t loopsNumforB = loopsNumKforB * loopsNumNforB;

        uint32_t loopsNumMforA = CeilDiv(params.problemGemmShape.m(), UBTileMRoundforA);
        
        uint32_t loopsNumforA = loopsNumMforA;

        uint32_t loopsNum = loopsNumforB + loopsNumforA;

        BlockFTSum blockFTSum(resource);

        float alpha{1.0};
        float beta{0.0};

        for(uint32_t loopId = aivIndex; loopId < loopsNum; loopId += aivNum) {
            
            if(loopId < loopsNumforB){
                uint32_t nLoopId = loopId % loopsNumNforB;
                uint32_t kLoopId = loopId / loopsNumNforB;

                uint32_t nActual = ((int32_t)nLoopId == (int32_t)(loopsNumNforB - 1)) ?
                    params.problemGemmShape.n() - nLoopId * UBBlockNRoundforB : UBBlockNRoundforB;

                uint32_t kActual = ((int32_t)kLoopId == (int32_t)(loopsNumKforB - 1)) ?
                    params.problemGemmShape.k() - kLoopId * UBBlockKRoundforB : UBBlockKRoundforB;

                int64_t gmOffsetBRow = kLoopId * UBBlockKRoundforB;
                int64_t gmOffsetBCol = nLoopId * UBBlockNRoundforB;
                uint32_t splitNIdx = nLoopId * UBBlockNRoundforB / UBTileNRoundforB;
                int64_t gmOffsetB = gmOffsetBRow * params.problemGemmShape.n() + gmOffsetBCol;
                int64_t gmOffsetBEMax = splitNIdx * params.problemGemmShape.k() + kLoopId * UBBlockKRoundforB;

                Catlass::GemvCoord actualBlockShapeforB = Catlass::GemvCoord{kActual, nActual};
                LayoutYforB layoutBE{kActual};
                LayoutYforB layoutE{nActual};

                /*
                template<>
                CATLASS_DEVICE
                void operator()<FT_REDUCE_TYPE::MAX>(
                    AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
                    AscendC::GlobalTensor<ElementY> const &gmZMax,
                    LayoutX const &layoutX,
                    GemvCoord const &actualShape)
                */
            
                blockFTSum.MaxRed(
                    gmB[gmOffsetB], params.layoutB,
                    gmYforB[gmOffsetBEMax], layoutBE, actualBlockShapeforB);
            }
            else{
                uint32_t loopIdlocal = loopId - loopsNumforB;
                uint32_t mLoopId = loopIdlocal % loopsNumMforA;
                uint32_t kLoopId = 0;

                uint32_t mActual = ((int32_t)mLoopId == (int32_t)(loopsNumMforA - 1)) ?
                    params.problemGemmShape.m() - mLoopId * UBTileMRoundforA : UBTileMRoundforA;

                uint32_t kActual = params.problemGemmShape.k();

                Catlass::MatrixCoord offsetA{mLoopId * UBTileMRoundforA, 0};
            
                int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);

                int64_t gmOffsetAMean = mLoopId * UBTileMRoundforA;
                int64_t gmOffsetAMax = mLoopId * UBTileMRoundforA;

                Catlass::GemvCoord actualBlockShapeforA = Catlass::GemvCoord{mActual, kActual};
                LayoutYforA layoutAred{mActual};

                /*
                template<>
                CATLASS_DEVICE
                void operator()<FT_REDUCE_TYPE::SUM_MAX_MIXED>(
                    AscendC::GlobalTensor<ElementA> const &gmA, 
                    LayoutA const &layoutA,
                    AscendC::GlobalTensor<ElementY> const &gmZSum, 
                    AscendC::GlobalTensor<ElementA> const &gmZMax,
                    LayoutY const &layoutY,
                    GemvCoord const &actualShape)
                */
            
                blockFTSum(
                    gmA[gmOffsetA], params.layoutA,
                    gmAMean[gmOffsetAMean],
                    gmAMax[gmOffsetAMax],
                    layoutAred, actualBlockShapeforA);
            }
            
        }
    }

    CATLASS_DEVICE
    void ABE_B_reduce_fused_op(Params const &params)
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

        AscendC::GlobalTensor<ElementZforBRed> gmBMean;
        gmBMean.SetGlobalBuffer((__gm__ ElementZforBRed *)params.ptrBMean);
        AscendC::GlobalTensor<ElementZforBRed> gmBMax;
        gmBMax.SetGlobalBuffer((__gm__ ElementZforBRed *)params.ptrBMax);
        AscendC::GlobalTensor<ElementYforB> gmBSumSlice;
        gmBSumSlice.SetGlobalBuffer((__gm__ ElementYforB *)params.ptrBE);
        AscendC::GlobalTensor<ElementYforB> gmBMaxSlice;
        gmBMaxSlice.SetGlobalBuffer((__gm__ ElementYforB *)params.ptrBMaxSlice);

        AscendC::GlobalTensor<ElementYforFT> gmZRow;
        gmZRow.SetGlobalBuffer((__gm__ ElementYforFT *)params.ptrZRow2);
        AscendC::GlobalTensor<ElementYforFT> gmSliceABE;
        gmSliceABE.SetGlobalBuffer((__gm__ ElementYforFT *)params.ptrWorkspace);

        // Get aicore information

        uint32_t UBTileSplitM = params.SplitReduceM;
        // RoundUp(UBTileShapeBReduce::M, UBAlignHelper::ALIGN);
        uint32_t UBTileNRound = RoundUp(params.SplitReduceN, UBAlignHelper::ALIGN);

        uint32_t Reduce_M_size = params.SplitNnum;
        uint32_t loopsNumforB = CeilDiv((Reduce_M_size - 1), UBTileSplitM);

        uint32_t UBTileMRoundforABE = RoundUp(SliceSumUBTileShape::M, UBAlignHelper::ALIGN);
        uint32_t UBTileNRoundforABE = RoundUp(SliceSumUBTileShape::N, UBAlignHelper::ALIGN);

        uint32_t ABE_M_size = params.SplitNnum;
        uint32_t ABE_N_size = params.problemGemmShape.m();
        uint32_t loopsNumforABE = 0;
        uint32_t loopsNumMforABE = 0;
        uint32_t loopsNumNforABE = 0;
        uint32_t ABE_N_coord_base = 0;
        uint32_t ABE_N_size_part = ABE_N_size / 2;

        if(params.SplitKNum > 1){
            loopsNumMforABE = CeilDiv(ABE_M_size, UBTileMRoundforABE);
            if(aivSubIndex < 1){
                ABE_N_size_part = ABE_N_size / 2;
                ABE_N_coord_base = 0;
                loopsNumNforABE = CeilDiv(ABE_N_size_part, UBTileNRoundforABE);
            }else{
                ABE_N_size_part = ABE_N_size - (ABE_N_size / 2);
                ABE_N_coord_base = ABE_N_size / 2;
                loopsNumNforABE = CeilDiv(ABE_N_size_part, UBTileNRoundforABE);
            }

            loopsNumforABE = loopsNumNforABE * loopsNumMforABE;
        }else{
            loopsNumforABE = 0;
        }

        int64_t OffsetInSliceInit = 0;
        // params.SplitNnum * params.problemGemmShape.k();

        LayoutYforFT layoutSliceOut{params.SplitNnum, params.problemGemmShape.m()};
        LayoutYforFT layoutSliceIn{params.SplitNnum, params.problemGemmShape.m()};
        
        BlockSliceRed blockSliceRed(resource);

        int64_t OffsetInMaxInit = 0;
        // params.SplitNnum * params.problemGemmShape.k();

        Catlass::layout::VectorLayout layoutOut{params.SplitNnum};
        LayoutYforFT layoutWorkforRed{params.SplitNnum, params.problemGemmShape.k()};

        if(aivSubIndex < 1){
            uint32_t loopsNumTotal = loopsNumforB + 1 + loopsNumforABE;
            for(uint32_t loopId = aicoreIndex; loopId < loopsNumTotal; loopId += aicoreNum) {
                
                if(loopId < (loopsNumforB + 1)){
                    uint32_t mActual = ((int32_t)loopId == (int32_t)(loopsNumforB - 1)) ?
                        (Reduce_M_size - 1 - loopId * UBTileSplitM) : UBTileSplitM;

                    uint32_t nActual = params.problemGemmShape.k();
                    int64_t gmOffsetInBSum = loopId * UBTileSplitM * params.problemGemmShape.k();
                    int64_t gmOffsetOutBSum = loopId * UBTileSplitM;
                    int64_t gmOffsetInBMax = OffsetInMaxInit + loopId * UBTileSplitM * params.problemGemmShape.k();
                    int64_t gmOffsetOutBMax = loopId * UBTileSplitM;

                    float kn_scale_ratio = (params.kn_scale_ratios[0]);
                    if((int32_t)loopId > (int32_t)(loopsNumforB - 1)){
                        mActual = 1;
                        kn_scale_ratio = (params.kn_scale_ratios[1]);
                        gmOffsetInBSum = (Reduce_M_size - 1) * params.problemGemmShape.k();
                        gmOffsetOutBSum = (Reduce_M_size - 1);
                        gmOffsetInBMax = OffsetInMaxInit + (Reduce_M_size - 1) * params.problemGemmShape.k();
                    }

                    Catlass::GemvCoord actualBlockShape = Catlass::GemvCoord{mActual, nActual};
                
                    /*
                    // AscendC::GlobalTensor<ElementY> const &gmZ,
                    CATLASS_DEVICE
                    void RowMean(
                        AscendC::GlobalTensor<ElementB> const &gmB, 
                        LayoutB const &layoutB,
                        AscendC::GlobalTensor<ElementY> const &gmZ, 
                        LayoutY const &layoutY,
                        GemvCoord const &actualShape,
                        uint32_t NRealRound,
                        float kn_scale_ratio)
                    */
                    blockSliceRed.RowMean(gmBSumSlice[gmOffsetInBSum], 
                        layoutWorkforRed,
                        gmBMean[gmOffsetOutBSum], layoutOut,
                        actualBlockShape,
                        UBTileNRound, kn_scale_ratio);
                }
                else{
                    uint32_t loopIdlocal = loopId - (loopsNumforB + 1);
                    
                    uint32_t loopIdM = loopIdlocal / loopsNumNforABE;
                    uint32_t loopIdN = loopIdlocal % loopsNumNforABE;
                    uint32_t mActual = ((int32_t)loopIdM == (int32_t)(loopsNumMforABE - 1)) ?
                            (ABE_M_size - loopIdM * UBTileMRoundforABE) : UBTileMRoundforABE;

                    uint32_t nActual = ((int32_t)loopIdN == (int32_t)(loopsNumNforABE - 1)) ?
                            (ABE_N_size_part - loopIdN * UBTileNRoundforABE) : UBTileNRoundforABE;

                    Catlass::MatrixCoord offsetSliceIn{loopIdM * UBTileMRoundforABE, ABE_N_coord_base + loopIdN * UBTileNRoundforABE};
                    Catlass::MatrixCoord offsetABEOut{loopIdM * UBTileMRoundforABE, ABE_N_coord_base + loopIdN * UBTileNRoundforABE};
            
                    int64_t gmOffsetInSlice = OffsetInSliceInit + layoutSliceIn.GetOffset(offsetSliceIn);
                    int64_t gmOffsetOutABE = layoutSliceOut.GetOffset(offsetABEOut); 

                    Catlass::GemvCoord actualBlockShape = Catlass::GemvCoord{mActual, nActual};
                    /*
                    // AscendC::GlobalTensor<ElementY> const &gmZ,
                    CATLASS_DEVICE
                    void operator()(
                        AscendC::GlobalTensor<ElementA> const &gmA, 
                        LayoutA const &layoutA,
                        AscendC::GlobalTensor<ElementY> const &gmY,
                        LayoutY const &layoutY,
                        GemvCoord const &actualShape, 
                        uint32_t NRealRound, uint32_t KFTRoundCount
                    )
                    */
                    blockSliceRed(gmSliceABE[gmOffsetInSlice], layoutSliceIn,
                        gmZRow[gmOffsetOutABE], layoutSliceOut,
                        actualBlockShape,
                        UBTileNRoundforABE, params.SplitKNum);
                }  
            }
        }else{
            uint32_t loopsNumTotal = loopsNumforB + 1 + loopsNumforABE;
            for(uint32_t loopId = aicoreIndex; loopId < loopsNumTotal; loopId += aicoreNum) {
                
                if(loopId < (loopsNumforB + 1)){
                    uint32_t mActual = ((int32_t)loopId == (int32_t)(loopsNumforB - 1)) ?
                        (Reduce_M_size - 1 - loopId * UBTileSplitM) : UBTileSplitM;

                    uint32_t nActual = params.problemGemmShape.k();
                    int64_t gmOffsetInBMax = OffsetInMaxInit + loopId * UBTileSplitM * params.problemGemmShape.k();
                    int64_t gmOffsetOutBMax = loopId * UBTileSplitM;
                    int64_t gmOffsetInBSum = loopId * UBTileSplitM * params.problemGemmShape.k();
                
                    float kn_scale_ratio = (params.kn_scale_ratios[0]);
                    if((int32_t)loopId > (int32_t)(loopsNumforB - 1)){
                        mActual = 1;
                        kn_scale_ratio = (params.kn_scale_ratios[1]);
                        gmOffsetInBMax = OffsetInMaxInit + (Reduce_M_size - 1) * params.problemGemmShape.k();
                        gmOffsetOutBMax = (Reduce_M_size - 1);
                        gmOffsetInBSum = (Reduce_M_size - 1) * params.problemGemmShape.k();
                    }

                    Catlass::GemvCoord actualBlockShape = Catlass::GemvCoord{mActual, nActual};
                    /*
                    CATLASS_DEVICE
                    void RowMax(
                        AscendC::GlobalTensor<ElementB> const &gmB, 
                        LayoutB const &layoutB,
                        AscendC::GlobalTensor<ElementY> const &gmZ, 
                        LayoutY const &layoutY,
                        GemvCoord const &actualShape,
                        uint32_t NRealRound,
                        float kn_scale_ratio)
                    */

                    /*
                    blockMeanMax.RowMax(gmBMaxSlice[gmOffsetInBMax], layoutWorkforRed,
                        gmBMax[gmOffsetOutBMax], layoutOut,
                        actualBlockShape,
                        UBTileNRound, kn_scale_ratio);
                    */
                    blockSliceRed.RowMax(gmBMaxSlice[gmOffsetInBMax], layoutWorkforRed,
                        gmBMax[gmOffsetOutBMax], layoutOut,
                        actualBlockShape,
                        UBTileNRound, kn_scale_ratio);
                }
                else{
                    uint32_t loopIdlocal = loopId - (loopsNumforB + 1);
                    
                    uint32_t loopIdM = loopIdlocal / loopsNumNforABE;
                    uint32_t loopIdN = loopIdlocal % loopsNumNforABE;
                    uint32_t mActual = ((int32_t)loopIdM == (int32_t)(loopsNumMforABE - 1)) ?
                            (ABE_M_size - loopIdM * UBTileMRoundforABE) : UBTileMRoundforABE;

                    uint32_t nActual = ((int32_t)loopIdN == (int32_t)(loopsNumNforABE - 1)) ?
                            (ABE_N_size_part - loopIdN * UBTileNRoundforABE) : UBTileNRoundforABE;

                    Catlass::MatrixCoord offsetSliceIn{loopIdM * UBTileMRoundforABE, ABE_N_coord_base + loopIdN * UBTileNRoundforABE};
                    Catlass::MatrixCoord offsetABEOut{loopIdM * UBTileMRoundforABE, ABE_N_coord_base + loopIdN * UBTileNRoundforABE};
            
                    int64_t gmOffsetInSlice = OffsetInSliceInit + layoutSliceIn.GetOffset(offsetSliceIn);
                    int64_t gmOffsetOutABE = layoutSliceOut.GetOffset(offsetABEOut); 

                    Catlass::GemvCoord actualBlockShape = Catlass::GemvCoord{mActual, nActual};
                    /*
                    // AscendC::GlobalTensor<ElementY> const &gmZ,
                    CATLASS_DEVICE
                    void operator()(
                        AscendC::GlobalTensor<ElementA> const &gmA, 
                        LayoutA const &layoutA,
                        AscendC::GlobalTensor<ElementY> const &gmY,
                        LayoutY const &layoutY,
                        GemvCoord const &actualShape, 
                        uint32_t NRealRound, uint32_t KFTRoundCount
                    )
                    */
                    blockSliceRed(gmSliceABE[gmOffsetInSlice], layoutSliceIn,
                        gmZRow[gmOffsetOutABE], layoutSliceOut,
                        actualBlockShape,
                        UBTileNRoundforABE, params.SplitKNum);
                }  
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }


    CATLASS_DEVICE
    void CE_split_op_fused(Params const &params, GM_ADDR ptrOutputCOMP)
    {
        

        AB_red_split_op(params);
        Catlass::Arch::CrossCoreBarrierAIV<0x0, PIPE_V>();
        Catlass::Arch::CrossCoreBarrierAIV<0x0, PIPE_MTE3>();
        
        Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);
        ABE_B_reduce_fused_op(params);
        // if(params.SplitKNum > 1){
        //     ABE_reduce_for_FT_on_AIV(params);
        // }
        

        // Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);        

        // Catlass::Arch::CrossCoreBarrierAIV<0x0, PIPE_V>();
        // Catlass::Arch::CrossCoreBarrierAIV<0x0, PIPE_MTE3>();
        

        // AscendC::SyncAll<true>(); 
        Catlass::Arch::CrossCoreBarrierAIV<0x0, PIPE_V>();
        Catlass::Arch::CrossCoreBarrierAIV<0x0, PIPE_MTE3>();

        // Represent the full gm
        // Get aicore information

        BlockFTGemvAIV blockFTGemvAIV(resource);

        // BlockThreCalc blockThreCalc(resource);

        BlockScheduler matmulBlockScheduler(params.problemGemmShapeRemain, Catlass::MakeCoord(L1TileShape::M,L1TileShape::N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        // uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetTaskRation();
        // AscendC::printf("%zu\n",AscendC::GetBlockNum());
        uint32_t aivIndex = AscendC::GetBlockIdx();
        uint32_t aicoreIndex = aivIndex / AscendC::GetSubBlockNum();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t aivNum = aicoreNum * AscendC::GetSubBlockNum();
        uint32_t aiv_part_num = 1 * AscendC::GetTaskRation();
        uint32_t align = Catlass::BYTE_PER_C0 / sizeof(ElementC);

        // uint32_t aicoreIndex = aivIndex / AscendC::GetTaskRation();

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        AscendC::GlobalTensor<ElementYforA> gmAMean;
        gmAMean.SetGlobalBuffer((__gm__ ElementYforA *)params.ptrAMean);

        AscendC::GlobalTensor<ElementA> gmAMax;
        gmAMax.SetGlobalBuffer((__gm__ ElementA *)params.ptrAMax);

        AscendC::GlobalTensor<ElementCOMPX> gmCOMPX;
        gmCOMPX.SetGlobalBuffer((__gm__ ElementCOMPX *)params.ptrZRow2);

        // AscendC::GlobalTensor<ElementCOMPX> gmOutX;
        // gmOutX.SetGlobalBuffer((__gm__ ElementCOMPX *)params.ptrZCol2);
        
        AscendC::GlobalTensor<ElementCOMPY> gmCOMPY;
        gmCOMPY.SetGlobalBuffer((__gm__ ElementCOMPY *)params.ptrZRow);

        AscendC::GlobalTensor<ElementCOMPZ> gmCOMPZ;
        gmCOMPZ.SetGlobalBuffer((__gm__ ElementCOMPZ *)ptrOutputCOMP);

        AscendC::GlobalTensor<ElementZ> gmT;
        gmT.SetGlobalBuffer((__gm__ ElementZ *)params.ptrThreZ);

        AscendC::GlobalTensor<ElementZforBRed> gmBMean;
        gmBMean.SetGlobalBuffer((__gm__ ElementZforBRed *)params.ptrBMean);
        AscendC::GlobalTensor<ElementZforBRed> gmBMax;
        gmBMax.SetGlobalBuffer((__gm__ ElementZforBRed *)params.ptrBMax);

        Catlass::MatrixCoord loopsMN = matmulBlockScheduler.loopsMN;

        /*
        return loopsMN.row() * loopsMN.column();
        */

        uint32_t UBTileMRound = RoundUp(UBTileShapeCE::M, UBAlignHelper::ALIGN);
        uint32_t UBTileKRound = RoundUp(UBTileShapeCE::N, UBAlignHelper::ALIGN);

        uint32_t UBBlockMRound = RoundUp(UBBlockShapeCE::M, UBAlignHelper::ALIGN);
        uint32_t UBBlockKRound = RoundUp(UBBlockShapeCE::N, UBAlignHelper::ALIGN);

        uint32_t element_num = params.problemGemmShape.m();

        uint32_t ThreUBTileMRound = UBTileMRound;
        uint32_t ThreUBTileNRound = RoundUp(L0TileShape::N, UBAlignHelper::ALIGN);

        uint32_t ThreUBBlockMRound = UBBlockMRound;
        uint32_t ThreUBBlockNRound = RoundUp(L1TileShape::N, UBAlignHelper::ALIGN);

        if(FUSE_TYPE == FT_AIV_PIPE_FUSE_TYPE::ABE_FUSED_THRE){
            ThreUBTileMRound = UBTileMRound;
            ThreUBTileNRound = RoundUp(L0TileShape::N, UBAlignHelper::ALIGN);

            ThreUBBlockMRound = UBBlockMRound;
            ThreUBBlockNRound = RoundUp(L1TileShape::N, UBAlignHelper::ALIGN);
        }else{
            ThreUBTileMRound = RoundUp(ThreCalcUBTileShape::M, UBAlignHelper::ALIGN);
            ThreUBTileNRound = RoundUp(ThreCalcUBTileShape::N, UBAlignHelper::ALIGN);

            ThreUBBlockMRound = RoundUp(ThreCalcUBBlockShape::M, UBAlignHelper::ALIGN);
            ThreUBBlockNRound = RoundUp(ThreCalcUBBlockShape::N, UBAlignHelper::ALIGN);
        }
        
        uint32_t ThreUBBlockZRound = ThreUBBlockMRound / 8;
        uint32_t ThreUBTileZRound = ThreUBTileZRound / 8;

        uint32_t total_input_elements = element_num;
        
        uint32_t total_input_bytes = total_input_elements * sizeof(ElementCOMPX);
        uint32_t total_output_elements = (total_input_elements + 8 - 1) / 8;

        LayoutYforFT layoutYforFT{params.SplitNnum, params.problemGemmShape.m()};
        LayoutYforFT layoutABeforFT{params.SplitNnum, params.problemGemmShape.m()};

        LayoutYforFT layoutThreforFT{params.SplitNnum, params.problemGemmShape.m()};

        LayoutYforFT layoutCOMPXforFT{params.SplitNnum, params.problemGemmShape.m()};
        LayoutYforFT layoutCOMPYforFT{params.SplitNnum, params.problemGemmShape.m()};

        LayoutYforFT layoutCOMPZforFT{params.SplitNnum, total_output_elements};

        LayoutCOMPX layoutInputX{element_num};
        // LayoutCOMPY layoutInputY{element_num};
        LayoutCOMPX layoutAforFT{element_num};
        LayoutCOMPX layoutARed{element_num};
        LayoutCOMPX layoutCE{element_num};
        LayoutCOMPX layoutBforFT{params.SplitNnum};
            
        LayoutCOMPZ layoutOutputZ{total_output_elements};

        for (uint32_t loopIdx = aicoreIndex; loopIdx < coreLoops; loopIdx += aicoreNum) {
            // Compute block location
            Catlass::GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            
            Catlass::GemmCoord actualCoord = Catlass::GemmCoord({blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N, blockCoord.k() * L1TileShape::K});

            uint32_t splitNIdx = blockCoord.n() * L1TileShape::N / L1TileShape::N;
            // Compute initial location in logical coordinates
            Catlass::MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};

            Catlass::MatrixCoord offsetYforFT{splitNIdx, blockCoord.m() * L1TileShape::M};

            Catlass::MatrixCoord offsetCOMPYforFT{splitNIdx, blockCoord.m() * L1TileShape::M};

            Catlass::MatrixCoord offsetCOMPXforFT{splitNIdx, blockCoord.m() * L1TileShape::M};

            Catlass::MatrixCoord offsetThreforFT{splitNIdx, blockCoord.m() * L1TileShape::M};

            uint32_t COMPZRowOffset = blockCoord.m() * L1TileShape::M / 8;
            Catlass::MatrixCoord offsetCOMPZforFT{splitNIdx, COMPZRowOffset};

            uint32_t mActual = UBBlockMRound;

            if(blockCoord.m() == loopsMN.row() -1) {
                mActual = params.problemGemmShape.m() - blockCoord.m() * L1TileShape::M;
            }

            uint32_t nActual = ThreUBBlockNRound;

            if(blockCoord.n() == loopsMN.column() - 1){
                nActual = params.problemGemmShape.n() - blockCoord.n() * L1TileShape::N;
            }
            
            int64_t gmOffsetC = params.layoutC.GetOffset(offsetC);
            
            int64_t gmOffsetYforFT = layoutYforFT.GetOffset(offsetYforFT);

            int64_t gmOffsetCOMPXforFT = layoutCOMPYforFT.GetOffset(offsetCOMPXforFT);
            int64_t gmOffsetCOMPYforFT = layoutCOMPYforFT.GetOffset(offsetCOMPYforFT);

            int64_t gmOffsetThreforFT = layoutThreforFT.GetOffset(offsetThreforFT);
            int64_t gmOffsetCOMPZforFT = layoutCOMPZforFT.GetOffset(offsetCOMPZforFT);
            int64_t gmOffsetARed = blockCoord.m() * L1TileShape::M;

            Catlass::layout::VectorLayout layoutABE{mActual};
            
            Catlass::GemvCoord actualBlockShape = Catlass::GemvCoord{mActual, nActual};
 
            float std_est_B_ratio = params.std_est_ratios[0];
            float kn_ratio_factor = params.kn_ratios[0];
            float kn_sqrt_ratio_factor = params.kn_sqrt_ratios[0];
            float k_sqrt_n_ratio_factor = params.k_sqrt_n_ratios[0];
            if(splitNIdx>=(params.SplitNnum - 1)){
                std_est_B_ratio = params.std_est_ratios[1];
                kn_ratio_factor = params.kn_ratios[1];
                kn_sqrt_ratio_factor = params.kn_sqrt_ratios[1];
                k_sqrt_n_ratio_factor = params.k_sqrt_n_ratios[1];
            }

            /*
            CATLASS_DEVICE
            void operator()(
                AscendC::GlobalTensor<ElementA> const &gmC, 
                LayoutA const &layoutC,
                AscendC::GlobalTensor<ElementX> const &gmAMax,
                AscendC::GlobalTensor<ElementY> const &gmAMean, 
                LayoutX const &layoutAforFT,
                AscendC::GlobalTensor<ElementY> const &gmY, 
                AscendC::GlobalTensor<ElementY> gmABe, 
                LayoutY const &layoutY,
                AscendC::GlobalTensor<ElementY> const &gmBMean,
                AscendC::GlobalTensor<ElementY> const &gmBMax, 
                LayoutX const &layoutBforFT,
                AscendC::GlobalTensor<ElementY> const &gmThreZ, 
                LayoutY const &layoutThre,
                AscendC::GlobalTensor<ElementZ> const &gmCOMPZ, 
                LayoutZ const &layoutZ,
                GemvCoord const &actualShape, 
                float std_est_A_row_ratio, 
                float std_est_B_ratio,
                float kn_ratio_factor, 
                float kn_sqrt_ratio_factor, 
                float k_sqrt_n_ratio_factor,
                float e_max,
                bool outputThre, 
                bool outputCE,
                uint32_t aiv_part_num, 
                Catlass::Arch::CrossCoreFlagWithReverse<> & flagAicFinishStore)
            */
            blockFTGemvAIV(
                gmC[gmOffsetC], params.layoutC,
                gmAMax[gmOffsetARed], gmAMean[gmOffsetARed],
                layoutARed, gmCOMPY[gmOffsetCOMPYforFT], gmCOMPX[gmOffsetCOMPYforFT],
                layoutCE, gmBMean[splitNIdx], gmBMax[splitNIdx],
                layoutBforFT, gmT[gmOffsetThreforFT], params.layoutThre,
                gmCOMPZ[gmOffsetCOMPZforFT], layoutOutputZ,
                actualBlockShape, params.std_est_A_row_ratio,
                std_est_B_ratio, kn_ratio_factor,
                kn_sqrt_ratio_factor, k_sqrt_n_ratio_factor, 
                params.e_max, params.outputThre, params.outputCE,
                aiv_part_num,flagAicFinishStore);
            
            // gmOutX[gmOffsetCOMPYforFT]
            // Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }


    CATLASS_DEVICE
    void ABE_reduce_for_FT_on_AIV(Params const &params)
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

        uint32_t align = Catlass::BYTE_PER_C0 / sizeof(ElementYforFT);
        // uint32_t aicoreIndex = aivIndex / AscendC::GetTaskRation();

        AscendC::GlobalTensor<ElementYforFT> gmZRow;
        gmZRow.SetGlobalBuffer((__gm__ ElementYforFT *)params.ptrZRow2);
        AscendC::GlobalTensor<ElementYforFT> gmSliceABE;
        gmSliceABE.SetGlobalBuffer((__gm__ ElementYforFT *)params.ptrWorkspace);

        // Get aicore information

        uint32_t UBTileMRound = RoundUp(SliceSumUBTileShape::M, UBAlignHelper::ALIGN);
        uint32_t UBTileNRound = RoundUp(SliceSumUBTileShape::N, UBAlignHelper::ALIGN);

        uint32_t Reduce_M_size = params.SplitNnum;
        uint32_t Reduce_N_size = params.problemGemmShape.m();
        uint32_t loopsNumN = CeilDiv(Reduce_N_size, UBTileNRound);
        uint32_t loopsNumM = CeilDiv(Reduce_M_size, UBTileMRound);
        uint32_t loopsNum = loopsNumN * loopsNumM;
        //uint32_t loopsNum = params.problemGemmShape.k();

        BlockSliceSum blockSliceSum(resource);

        int64_t OffsetInSliceInit = 0;
        // params.SplitNnum * params.problemGemmShape.k();

        LayoutYforFT layoutSliceOut{params.SplitNnum, params.problemGemmShape.m()};
        LayoutYforFT layoutSliceIn{params.SplitNnum, params.problemGemmShape.m()};

        for(uint32_t loopId = aivIndex; loopId < loopsNum; loopId += aivNum) {
            
            uint32_t loopIdM = loopId / loopsNumN;
            uint32_t loopIdN = loopId % loopsNumN;
            uint32_t mActual = ((int32_t)loopIdM == (int32_t)(loopsNumM - 1)) ?
                        (Reduce_M_size - loopIdM * UBTileMRound) : UBTileMRound;

            uint32_t nActual = ((int32_t)loopIdN == (int32_t)(loopsNumN - 1)) ?
                        (Reduce_N_size - loopIdN * UBTileNRound) : UBTileNRound;

            Catlass::MatrixCoord offsetSliceIn{loopIdM * UBTileMRound, loopIdN * UBTileNRound};
            Catlass::MatrixCoord offsetABEOut{loopIdM * UBTileMRound, loopIdN * UBTileNRound};
            
            int64_t gmOffsetInSlice = OffsetInSliceInit + layoutSliceIn.GetOffset(offsetSliceIn);
            int64_t gmOffsetOutABE = layoutSliceOut.GetOffset(offsetABEOut); 
            // loopId * UBTileSplitM * params.problemGemmShape.m();

            Catlass::GemvCoord actualBlockShape = Catlass::GemvCoord{mActual, nActual};
            /*
            // AscendC::GlobalTensor<ElementY> const &gmZ,
            CATLASS_DEVICE
            void operator()(
                AscendC::GlobalTensor<ElementA> const &gmA, 
                LayoutA const &layoutA,
                AscendC::GlobalTensor<ElementY> const &gmY, 
                LayoutY const &layoutY,
                GemvCoord const &actualShape, 
                uint32_t NRealRound, uint32_t KFTRoundCount)
            */
            blockSliceSum(gmSliceABE[gmOffsetInSlice], layoutSliceIn,
                    gmZRow[gmOffsetOutABE], layoutSliceOut,
                    actualBlockShape,
                    UBTileNRound, params.SplitKNum);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }
    

    


    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params){
        
        CE_split_op_fused(params,params.ptrCOMPZRow);
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

#endif