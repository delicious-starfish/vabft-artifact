#ifndef CATLASS_GEMM_KERNEL_MATMUL_BR_ABR_ON_AIC_ASVAR_THRESHOLD_ABFT_NO_SPLITK_HPP_SPEC_ROBUST_CHUNK_HALF_PRE
#define CATLASS_GEMM_KERNEL_MATMUL_BR_ABR_ON_AIC_ASVAR_THRESHOLD_ABFT_NO_SPLITK_HPP_SPEC_ROBUST_CHUNK_HALF_PRE

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
    class FTChunkShape_,
    class BlockMmadABr_,
    class BlockMmad_,
    class BlockSchedulerFirst_,
    class BlockScheduler_,
    class BlockFTGemvBRAIC_,
    class BlockFTGemvAEAIC_,
    class BlockFTSum_,
    class BlockFTGemvAIV_,
    class BlockSliceRed_,
    class BlockChunkVerify_
>
class MatmulAsVarABonAicChunkSpecRobustPreloadF16 {
public:
    using BlockMmad = BlockMmad_;
    using BlockMmadABr = BlockMmadABr_;
    // using BlockGemv = BlockGemv_;
    using BlockChunkVerify = BlockChunkVerify_;
    using FTChunkShape = FTChunkShape_;
    // using BlockSumGemv = BlockSumGemv_;
    // using BlockThreCalc = BlockThreCalc_;

    using BlockFTSum = BlockFTSum_;
    using BlockFTGemvAIV = BlockFTGemvAIV_;
    using BlockSliceRed = BlockSliceRed_;

    using BlockFTGemvBRAIC = BlockFTGemvBRAIC_;
    using BlockFTGemvAEAIC = BlockFTGemvAEAIC_;

    // using BlockEpilogue = BlockEpilogue_;
    using FT_ENC_TYPE = Catlass::Gemv::helper::FT_ENC_TYPE;
    using FT_COMP_TYPE = Catlass::Gemv::helper::FT_COMP_TYPE;

    using FT_AIV_PIPE_FUSE_TYPE = Catlass::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE;
    using FT_THRESHOLD_ALGORITHM = Catlass::Gemv::helper::FT_THRESHOLD_ALGORITHM;

    using FT_RCE_THRE_TYPE = Catlass::Gemv::helper::FT_RCE_THRE_TYPE;
    using FT_AIC_BE_SCHEME = Catlass::Gemv::helper::FT_AIC_BE_SCHEME;
    
    static const FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE = BlockFTGemvAIV::FUSE_TYPE;
    static const FT_THRESHOLD_ALGORITHM ALGO_TYPE = FT_THRESHOLD_ALGORITHM::ASVAR;

    static const FT_AIC_BE_SCHEME BE_SCHEME = BlockFTGemvBRAIC::BE_SCHEME;


    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using L0TileShape = typename BlockMmad::L0TileShape;

    using L1TileShapeforFT = typename BlockMmadABr::L1TileShapeforFT;
    using L0TileShapeforFT = typename BlockMmadABr::L0TileShapeforFT;

    using L1TileShapeforAe = typename BlockFTGemvAEAIC::L1TileShapeforFT;
    using L0TileShapeforAe = typename BlockFTGemvAEAIC::L0TileShapeforFT;

    using ElementA = typename BlockMmad::ElementA;
    using ElementAforFTChunk = typename BlockMmadABr::ElementAforFT;

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
    
    using ElementXforFTChunk = typename BlockMmadABr::ElementXforFT;

    using ElementXforFT = typename BlockFTGemvBRAIC::ElementX;
    using LayoutXforFT = typename BlockMmadABr::LayoutX;

    using ElementYforFT = typename BlockMmadABr::ElementY;
    using LayoutYforFT = typename BlockMmadABr::LayoutY;

    using ElementYforB = typename BlockFTSum::ElementX;
    using LayoutYforB = typename BlockFTSum::LayoutX;

    using ElementYforA = typename BlockFTSum::ElementY;
    using LayoutYforA = typename BlockFTSum::LayoutY;

    using ElementX = typename BlockFTSum::ElementX;
    using LayoutVX = typename BlockChunkVerify::LayoutVX;

    using LayoutMX = typename BlockChunkVerify::LayoutMX;

    using ElementY = typename BlockChunkVerify::ElementY;
    using LayoutVY = typename BlockChunkVerify::LayoutVY;

    using LayoutMY = typename BlockChunkVerify::LayoutMY;
    
    using LayoutCCol = typename std::conditional<
        std::is_same<LayoutC, Catlass::layout::RowMajor>::value,
        Catlass::layout::ColumnMajor,
        Catlass::layout::RowMajor>::type;
    
    using CColType = Catlass::Gemm::GemmType<ElementC, LayoutCCol>;

    using ElementAccumulator = 
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementXforFT, ElementXforFT>::ElementAccumulator;

    using ElementZ = ElementYforFT;
    using ElementZInAiv = typename BlockFTGemvBRAIC::ElementY;
    using LayoutVZ = typename BlockChunkVerify::LayoutVZ;
    using LayoutMZ = typename BlockChunkVerify::LayoutMZ;

    using ElementZforBRed = ElementZ;

    using ElementYforBEAIV = ElementZforBRed;

    using ElementCOMPX = ElementZ;
    using LayoutCOMPVX = Catlass::layout::VectorLayout;
    using LayoutCOMPMX = typename BlockChunkVerify::LayoutMX;

    using ElementCOMPY = ElementZ;
    using LayoutCOMPVY = Catlass::layout::VectorLayout;
    using LayoutCOMPMY = typename BlockChunkVerify::LayoutMY;

    // using UBTileShape = typename BlockSumGemv::UBTileShape;

    using UBTileShapeBMax = typename BlockFTSum::UBTileShapeforB;
    using UBBlockShapeBMax = typename BlockFTSum::UBBlockShapeforB;

    using UBTileShapeARed = typename BlockFTSum::UBTileShapeforA;

    using L1TileShapeBR = typename BlockFTGemvBRAIC::L1TileShape;
    using L0TileShapeBR = typename BlockFTGemvBRAIC::L0TileShape;
    using UBBlockShapeBR = typename BlockFTGemvBRAIC::UBBlockShape;

    static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;

    static constexpr uint32_t L0C_TILE_SIZE = 16 * L0TileShapeBR::M * sizeof(ElementAccumulator);

    static constexpr uint32_t L0C_TILE_SCALE = 16 * L0TileShapeBR::M;

    static constexpr uint32_t L0C_TILE_NUM_RAW  = L0C_SIZE / L0C_TILE_SIZE;

    static constexpr uint32_t L0C_TILE_NUM = (L0C_TILE_NUM_RAW < 8) ? L0C_TILE_NUM_RAW : 8;


    using UBTileShapeCR = typename BlockFTGemvAIV::UBTileShapeforC;
    using UBTileShapeACast = typename BlockFTGemvAIV::UBTileShapeforA;
    using UBBlockShapeCR = typename BlockFTGemvAIV::UBBlockShape;

    using UBTileShapeBReduce = typename BlockSliceRed::UBTileShapeforB;
    using SliceSumUBTileShape = typename BlockSliceRed::UBTileShapeforA;

    using UBAlignHelper = Catlass::Gemv::helper::UBAlignHelper<ElementA>;
    using UBAlignHelperOut = Catlass::Gemv::helper::UBAlignHelper<ElementZ>;

    using ThreCalcUBBlockShape = typename BlockFTGemvAIV::UBBlockShape;
    using ThreCalcUBTileShape = typename BlockFTGemvAIV::UBTileShapeforC;

    using UBTileShapeVerify = typename BlockChunkVerify::UBTileShape;
    using UBBlockShapeVerify = typename BlockChunkVerify::UBBlockShape;
    
    using ElementCOMPZ = typename BlockChunkVerify::ElementZ;
    using LayoutCOMPVZ = Catlass::layout::VectorLayout;
    using LayoutCOMPMZ = typename BlockChunkVerify::LayoutMZ;

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
        Catlass::GemvCoord problemCompShape;
        GM_ADDR ptrA;
        GM_ADDR ptrACast;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrX;
        GM_ADDR ptrXTail;
        GM_ADDR ptrVXR2;
        GM_ADDR ptrVXR2Tail;
        LayoutVX layoutVX;
        LayoutMX layoutMX;
        LayoutVX layoutVXTail;
        LayoutMX layoutMXTail;
        GM_ADDR ptrWorkspace;
        FT_ENC_TYPE enc_type;
        GM_ADDR ptrZRowR1;
        GM_ADDR ptrZRowR2;
        GM_ADDR ptrZRow2R1;
        GM_ADDR ptrZRow2R2;
        GM_ADDR ptrCR1Slice;
        GM_ADDR ptrCR2Slice;
        GM_ADDR ptrCOMPZRow;
        LayoutCOMPVZ layoutCOMPVZRow;
        LayoutCOMPMZ layoutCOMPMZRow;
        LayoutCOMPVX layoutCOMPVX;
        LayoutCOMPMX layoutCOMPMX;
        LayoutCOMPVY layoutCOMPVY;
        LayoutCOMPMY layoutCOMPMY;
        ElementCOMPX threshold;
        GM_ADDR ptrBR1;
        GM_ADDR ptrBR2;
        GM_ADDR ptrBRforAIV;
        GM_ADDR ptrBMaxSlice;
        GM_ADDR ptrBMinSlice;
        GM_ADDR ptrBMeanAbs;
        GM_ADDR ptrBMeanSquare;
        GM_ADDR ptrBVar;
        GM_ADDR ptrXVR1forA;
        GM_ADDR ptrAMean;
        GM_ADDR ptrAMax;
        GM_ADDR ptrAMin;
        GM_ADDR ptrThreZ;
        LayoutCOMPMX layoutThre;
        ElementZ rounding_alpha;
        float e_max;
        float std_est_A_row_ratio;
        float A_row_scale_ratio;
        float n_scale_ratios[2];
        float n_ratios[2];
        float n_sqrt_ratios[2];
        float n_square_ratios[2];
        uint32_t SplitNnumChunk;
        uint32_t SplitNnum;
        uint32_t SplitReduceM;
        uint32_t SplitReduceN;
        bool outputThre;
        uint32_t SplitKNum;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {};
            


        CATLASS_HOST_DEVICE
        Params(
            Catlass::GemmCoord const &problemGemmShape_,
            Catlass::GemmCoord const &problemGemmShapeFirst_,
            Catlass::GemmCoord const &problemGemmShapeRemain_,
            Catlass::GemvCoord const &problemShape_,
            Catlass::GemvCoord const &problemCompShape_,
            GM_ADDR ptrA_, GM_ADDR ptrACast_,
            LayoutA layoutA_,
            GM_ADDR ptrB_, LayoutB layoutB_,
            GM_ADDR ptrC_, LayoutC layoutC_, 
            GM_ADDR ptrX_, GM_ADDR ptrXTail_, 
            GM_ADDR ptrVXR2_, GM_ADDR ptrVXR2Tail_,
            LayoutVX layoutVX_, LayoutMX layoutMX_,
            LayoutVX layoutVXTail_, LayoutMX layoutMXTail_,
            GM_ADDR ptrWorkspace_,
            FT_ENC_TYPE enc_type_, GM_ADDR ptrZRowR1_, GM_ADDR ptrZRowR2_,
            GM_ADDR ptrZRow2R1_, GM_ADDR ptrZRow2R2_, 
            GM_ADDR ptrCR1Slice_, GM_ADDR ptrCR2Slice_,
            GM_ADDR ptrCOMPZRow_,
            LayoutCOMPVZ layoutCOMPVZRow_, LayoutCOMPMZ layoutCOMPMZRow_,
            LayoutCOMPVX layoutCOMPVX_, LayoutCOMPMX layoutCOMPMX_,
            LayoutCOMPVY layoutCOMPVY_, LayoutCOMPMY layoutCOMPMY_,
            ElementCOMPX threshold_,
            GM_ADDR ptrBR1_, GM_ADDR ptrBR2_,
            GM_ADDR ptrBRforAIV_,
            GM_ADDR ptrBMaxSlice_,
            GM_ADDR ptrBMinSlice_,
            GM_ADDR ptrBMeanAbs_,
            GM_ADDR ptrBMeanSquare_,
            GM_ADDR ptrBVar_,
            GM_ADDR ptrXVR1forA_,
            GM_ADDR ptrAMean_, GM_ADDR ptrAMax_, 
            GM_ADDR ptrAMin_, 
            GM_ADDR ptrThreZ_, 
            LayoutCOMPMX layoutThre_, ElementZ rounding_alpha_,
            float e_max_, float std_est_A_row_ratio_, float A_row_scale_ratio_, 
            const float (&n_scale_ratios_)[2],
            const float (&n_ratios_)[2],
            const float (&n_sqrt_ratios_)[2],
            const float (&n_square_ratios_)[2],
            uint32_t SplitNnumChunk_, uint32_t SplitNnum_, 
            uint32_t SplitReduceM_, uint32_t SplitReduceN_,
            bool outputThre_, uint32_t SplitKNum_
        ) : problemGemmShape(problemGemmShape_), 
            problemGemmShapeFirst(problemGemmShapeFirst_),
            problemGemmShapeRemain(problemGemmShapeRemain_),
            problemShape(problemShape_),
            problemCompShape(problemCompShape_),
            ptrA(ptrA_), ptrACast(ptrACast_),
            layoutA(layoutA_), 
            ptrB(ptrB_), layoutB(layoutB_), 
            ptrC(ptrC_), layoutC(layoutC_), 
            ptrX(ptrX_), ptrXTail(ptrXTail_),
            ptrVXR2(ptrVXR2_), ptrVXR2Tail(ptrVXR2Tail_),
            layoutVX(layoutVX_), layoutMX(layoutMX_),
            layoutVXTail(layoutVXTail_), layoutMXTail(layoutMXTail_),
            ptrWorkspace(ptrWorkspace_),
            enc_type(enc_type_), ptrZRowR1(ptrZRowR1_), ptrZRowR2(ptrZRowR2_),
            ptrZRow2R1(ptrZRow2R1_), ptrZRow2R2(ptrZRow2R2_), 
            ptrCR1Slice(ptrCR1Slice_), ptrCR2Slice(ptrCR2Slice_), 
            ptrCOMPZRow(ptrCOMPZRow_),
            layoutCOMPVZRow(layoutCOMPVZRow_), layoutCOMPMZRow(layoutCOMPMZRow_),
            layoutCOMPVX(layoutCOMPVX_), layoutCOMPMX(layoutCOMPMX_),
            layoutCOMPVY(layoutCOMPVY_), layoutCOMPMY(layoutCOMPMY_),
            threshold(threshold_), 
            ptrBR1(ptrBR1_), ptrBR2(ptrBR2_),
            ptrBRforAIV(ptrBRforAIV_),
            ptrBMaxSlice(ptrBMaxSlice_),
            ptrBMinSlice(ptrBMinSlice_),
            ptrBMeanAbs(ptrBMeanAbs_),
            ptrBMeanSquare(ptrBMeanSquare_),
            ptrBVar(ptrBVar_),
            ptrXVR1forA(ptrXVR1forA_),
            ptrAMean(ptrAMean_), ptrAMax(ptrAMax_), ptrAMin(ptrAMin_),
            ptrThreZ(ptrThreZ_), layoutThre(layoutThre_), 
            rounding_alpha(rounding_alpha_), e_max(e_max_), 
            std_est_A_row_ratio(std_est_A_row_ratio_), A_row_scale_ratio(A_row_scale_ratio_), 
            SplitNnumChunk(SplitNnumChunk_),
            SplitNnum(SplitNnum_), 
            SplitReduceM(SplitReduceM_), SplitReduceN(SplitReduceN_),
            outputThre(outputThre_), SplitKNum(SplitKNum_)
            {
                for (int i = 0; i < 2; ++i) {         
                    this->n_scale_ratios[i] = n_scale_ratios_[i];
                    this->n_ratios[i] = n_ratios_[i];
                    this->n_sqrt_ratios[i] = n_sqrt_ratios_[i];
                    this->n_square_ratios[i] = n_square_ratios_[i];
                }
            } 
    };

    struct Arguments {
        Catlass::GemmCoord problemGemmShape;
        Catlass::GemvCoord problemShape;
        size_t elementSize;
        GM_ADDR ptrX;
        GM_ADDR ptrXTail;
        GM_ADDR ptrVXR2;
        GM_ADDR ptrVXR2Tail;
        GM_ADDR ptrA;
        GM_ADDR ptrACast;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
        GM_ADDR ptrZRowR1;
        GM_ADDR ptrZRowR2;
        GM_ADDR ptrZRow2R1;
        GM_ADDR ptrZRow2R2;
        GM_ADDR ptrCR1Slice;
        GM_ADDR ptrCR2Slice;
        GM_ADDR ptrCOMPZRow;
        GM_ADDR ptrBR1;
        GM_ADDR ptrBR2;
        GM_ADDR ptrBRforAIV;
        GM_ADDR ptrBMaxSlice;
        GM_ADDR ptrBMinSlice;
        GM_ADDR ptrBMeanAbs;
        GM_ADDR ptrBMeanSquare;
        GM_ADDR ptrBVar; 
        GM_ADDR ptrXVR1forA;
        GM_ADDR ptrAMean;
        GM_ADDR ptrAMax;
        GM_ADDR ptrAMin;
        GM_ADDR ptrThreZ;
        FT_ENC_TYPE enc_type;
        ElementCOMPX threshold;
        float rounding_exponent;
        float size_beta;
        float e_max_raw;
        uint32_t reduce_cores;
        FT_RCE_THRE_TYPE rce_thre_type;
        bool outputThre;
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
        uint32_t splitNnumChunk = ((args.problemGemmShape.n() + FTChunkShape::N - 1) / FTChunkShape::N);
        // + (splitNnum * args.problemGemmShape.m()) + (splitNnum * args.problemGemmShape.m()))
        // args.SplitKNum
        return sizeof(ElementYforFT) * ((splitNnumChunk * 2 + 1) * args.problemGemmShape.m() * 1);
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        Catlass::GemmCoord problemGemmShape = args.problemGemmShape;
        Catlass::GemvCoord problemShape = args.problemShape;

        uint32_t SplitNnum = ((args.problemGemmShape.n() + L1TileShape::N - 1) / L1TileShape::N);
        uint32_t SplitNnumChunk = ((args.problemGemmShape.n() + FTChunkShape::N - 1) / FTChunkShape::N);

        uint32_t m = problemShape.m();
        uint32_t n = problemShape.n();

        uint32_t m2 = problemGemmShape.m();
        uint32_t n2 = problemGemmShape.n();
        uint32_t k2 = problemGemmShape.k();

        uint32_t FirstBlockN = SplitNnumChunk * 2;

        uint32_t RemainBlockN = n2;

        uint32_t SplitKNum = 1;
        // args.SplitKNum;
        // if (SplitKNum < 1){
        //     SplitKNum = 1;
        // }

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

        uint32_t total_input_elements = m + n;
        
        uint32_t total_input_bytes = total_input_elements * sizeof(ElementCOMPX);
        uint32_t total_output_elements = (m + 8 - 1) / 8 + (n + 8 - 1) / 8;
        uint32_t row_output_elements = (m + 8 - 1) / 8;
        uint32_t col_output_elements = (n + 8 - 1) / 8;
        // printf("Total input bytes: %d",total_input_bytes);

        LayoutCOMPVX layoutCOMPVX{total_input_elements};
        LayoutCOMPVY layoutCOMPVY{total_input_elements};

        LayoutCOMPMX layoutCOMPMX{SplitNnumChunk, total_input_elements};
        LayoutCOMPMY layoutCOMPMY{SplitNnumChunk, total_input_elements};

        LayoutCOMPVZ layoutCOMPVZ{total_output_elements};

        LayoutCOMPVZ layoutCOMPVZRow{row_output_elements};
        LayoutCOMPVZ layoutCOMPVZCol{col_output_elements};

        LayoutCOMPMZ layoutCOMPMZ{SplitNnumChunk, total_output_elements};

        LayoutCOMPMZ layoutCOMPMZRow{SplitNnumChunk, row_output_elements};
        LayoutCOMPMZ layoutCOMPMZCol{SplitNnumChunk, col_output_elements};

        LayoutA layoutA{m2, k2};

        LayoutB layoutB{k2, n2};

        LayoutC layoutC{m, n};
        
        LayoutVX layoutVX{FTChunkShape::N};
        LayoutMX layoutMX{2, FTChunkShape::N};

        LayoutVZ layoutVZRow{m};
        LayoutVZ layoutVZCol{n};

        LayoutMZ layoutMZRow{SplitNnumChunk, m};
        LayoutMZ layoutMZCol{SplitNnumChunk, n};

        LayoutCOMPMX layoutThre{SplitNnumChunk, m};

        uint32_t tail_chunk_n_size = n2 % FTChunkShape::N;
        if(tail_chunk_n_size < 1){
            tail_chunk_n_size = FTChunkShape::N;
        }

        LayoutVX layoutVXTail{tail_chunk_n_size};
        LayoutMX layoutMXTail{2, tail_chunk_n_size};

        ElementZ rounding_alpha = (ElementZ)1.0f;
        // static_cast<ElementZ>(row_sqrt * rounding_error);

        float e_max = (args.e_max_raw * 1.0f);

        uint32_t SplitReduceM = (SplitNnumChunk + args.reduce_cores - 1) / args.reduce_cores;
        // printf("SplitReduceM Raw: %d\n", SplitReduceM);

        SplitReduceM = (SplitReduceM >= UBTileShapeBReduce::M) ? UBTileShapeBReduce::M : SplitReduceM;

        SplitReduceM = (SplitReduceM <= 1) ? 1 : SplitReduceM;

        uint32_t SplitReduceN_num = (k2 + UBTileShapeBReduce::N - 1) / UBTileShapeBReduce::N;
        uint32_t SplitReduceN = UBTileShapeBReduce::N;
        if(SplitReduceN_num < 2){
            SplitReduceN = (k2 + 2 - 1) / 2;
        }

       
        float n_scale_ratios[2];
        float n_ratios[2];
        float n_sqrt_ratios[2];
        float n_square_ratios[2];

        uint32_t n_remain_split = (args.problemGemmShape.n() % FTChunkShape::N);

        float std_est_A_row_factor = std::sqrt(2.0f * logf((k2 * 1.0f)));
        float std_est_A_row_ratio = (1.0f / std_est_A_row_factor);
        float A_row_scale_ratio = 1.0f / (1.0f * k2);

        n_scale_ratios[0] = (1.0f / (FTChunkShape::N * 1.0f));
        n_ratios[0] = (FTChunkShape::N * 1.0f);
        n_sqrt_ratios[0] = std::sqrt(FTChunkShape::N * 1.0f);
        n_square_ratios[0] = (FTChunkShape::N * FTChunkShape::N * 1.0f);
        
        if(n_remain_split > 0){
            n_scale_ratios[1] = (1.0f / (n_remain_split * 1.0f));
            n_ratios[1] = (n_remain_split * 1.0f);
            n_sqrt_ratios[1] = std::sqrt(n_remain_split * 1.0f);
            n_square_ratios[1] = (n_remain_split * n_remain_split * 1.0f);
        }else{
            n_scale_ratios[1] = (1.0f / (FTChunkShape::N * 1.0f));
            n_ratios[1] = (FTChunkShape::N * 1.0f);
            n_sqrt_ratios[1] = std::sqrt(FTChunkShape::N * 1.0f);
            n_square_ratios[1] = (FTChunkShape::N * FTChunkShape::N * 1.0f);
        }

        Params params{
            problemGemmShape,
            problemGemmShapeFirst,
            problemGemmShapeRemain,
            problemShape,
            problemCompShape,
            args.ptrA, args.ptrACast,
            layoutA, 
            args.ptrB, layoutB, 
            args.ptrC, layoutC, 
            args.ptrX, args.ptrXTail, args.ptrVXR2, args.ptrVXR2Tail,
            layoutVX, layoutMX,
            layoutVXTail, layoutMXTail,
            workspace, args.enc_type, 
            args.ptrZRowR1, args.ptrZRowR2, args.ptrZRow2R1, args.ptrZRow2R2,
            args.ptrCR1Slice, args.ptrCR2Slice,
            args.ptrCOMPZRow, layoutCOMPVZRow, layoutCOMPMZRow, 
            layoutCOMPVX, layoutCOMPMX,
            layoutCOMPVY, layoutCOMPMY,
            args.threshold, 
            args.ptrBR1, args.ptrBR2,
            args.ptrBRforAIV, 
            args.ptrBMaxSlice,
            args.ptrBMinSlice,
            args.ptrBMeanAbs,
            args.ptrBMeanSquare,
            args.ptrBVar,
            args.ptrXVR1forA,
            args.ptrAMean, args.ptrAMax, args.ptrAMin,
            args.ptrThreZ, layoutThre, rounding_alpha, 
            e_max, std_est_A_row_ratio, A_row_scale_ratio,
            n_scale_ratios, n_ratios, 
            n_sqrt_ratios, n_square_ratios,
            SplitNnumChunk,
            SplitNnum, SplitReduceM, SplitReduceN,
            args.outputThre, SplitKNum
        };

        uint32_t UBTileMRoundforABR = RoundUp(SliceSumUBTileShape::M, UBAlignHelperOut::ALIGN);
        uint32_t UBTileNRoundforABR = RoundUp(SliceSumUBTileShape::N, UBAlignHelperOut::ALIGN);

        uint32_t ABR_M_size = params.SplitNnumChunk;
        uint32_t ABR_N_size = params.problemGemmShape.m();
        uint32_t loopsNumforABR = 0;
        uint32_t loopsNumMforABR = 0;
        uint32_t loopsNumNforABR = 0;
        uint32_t ABR_N_coord_base = 0;
        uint32_t ABR_N_size_part = ABR_N_size / 2;

        int64_t OffsetAeSliceInit = 2 * params.SplitNnumChunk * params.problemGemmShape.m() * params.SplitKNum;

        if(params.SplitKNum > 1){
            loopsNumMforABR = CeilDiv(ABR_M_size, UBTileMRoundforABR);
            ABR_N_size_part = ABR_N_size / 2;
            ABR_N_coord_base = 0;
            loopsNumNforABR = CeilDiv(ABR_N_size_part, UBTileNRoundforABR);

            loopsNumforABR = loopsNumNforABR * loopsNumMforABR;
        }else{
            loopsNumforABR = 0;
        }

        uint32_t UBTileSplitM = params.SplitReduceM;
        uint32_t UBTileNRoundforBRed = RoundUp(params.SplitReduceN, UBAlignHelperOut::ALIGN);

        uint32_t Reduce_M_size = params.SplitNnumChunk;
        
        uint32_t loopsNumforB = CeilDiv((Reduce_M_size - 1), UBTileSplitM);

        if (Reduce_M_size < 2){
            loopsNumforB = 1;
        } 

        printf("L0C_TILE_NUM: %d\n", L0C_TILE_NUM);
        printf("L0C_TILE_NUM_RAW: %d\n", L0C_TILE_NUM_RAW);

        // printf("SplitReduceM: %d\n", params.SplitReduceM);
        // printf("SplitReduceN: %d\n", params.SplitReduceN);
        // printf("UBTileNRoundforBRed: %d\n", UBTileNRoundforBRed);
        // printf("loopsNumforB: %d\n", loopsNumforB);
        // printf("n_remain_split: %d\n", n_remain_split);

        // // printf("%d\n",102);
        // printf("loopsNumforABE: %d\n", loopsNumforABE);
        // printf("loopsNumNforABE: %d\n",loopsNumNforABE);
        // printf("loopsNumMforABE: %d\n",loopsNumMforABE);
        // printf("ABR_N_size_part: %d\n",ABR_N_size_part);
        // printf("ABR_N_size: %d\n",ABR_N_size);

        return params;
    }

    // Methods
    CATLASS_DEVICE
    MatmulAsVarABonAicChunkSpecRobustPreloadF16() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    CATLASS_DEVICE
    void BR_split_op_on_AIC(Params const &params)
    {
        // Arch::Resource<ArchTag> resource;

        // Represent the full gm
        // Get aicore information

        BlockFTGemvBRAIC blockFTGemvBRAIC(resource);
        
        uint32_t aicoreNum = AscendC::GetBlockNum();
        // BlockScheduler matmulBlockScheduler(params.problemGemmShape, Catlass::MakeCoord(L1TileShape::M,L1TileShape::N));
        // uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();
        // uint32_t aivNum = aicoreNum * AscendC::GetSubBlockNum();
        // AscendC::printf("%zu\n",AscendC::GetBlockNum());
        // uint32_t aicoreIndex = aivIndex / AscendC::GetTaskRation();

        AscendC::GlobalTensor<ElementXforFT> gmXV;
        gmXV.SetGlobalBuffer((__gm__ ElementXforFT *)params.ptrX);

        AscendC::GlobalTensor<ElementXforFT> gmXVTail;
        gmXVTail.SetGlobalBuffer((__gm__ ElementXforFT *)params.ptrXTail);

        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        
        AscendC::GlobalTensor<ElementZforBRed> gmYR1;
        gmYR1.SetGlobalBuffer((__gm__ ElementZforBRed *)params.ptrBR1);

        AscendC::GlobalTensor<ElementZforBRed> gmYR2;
        gmYR2.SetGlobalBuffer((__gm__ ElementZforBRed *)params.ptrBR2);

        LayoutYforFT layoutYforBR{params.SplitNnumChunk, params.problemGemmShape.k()};
        LayoutMX layoutMX{2, FTChunkShape::N};
        
        uint32_t Remain_Chunk_N = params.problemGemmShape.n() % FTChunkShape::N;
        if (Remain_Chunk_N < 1){
            Remain_Chunk_N = FTChunkShape::N;
        }
        
        LayoutMX layoutMXTail{2, Remain_Chunk_N};

        uint32_t TileKRound = L1TileShapeBR::M;
        uint32_t TileNRound = L1TileShapeBR::N;

        uint32_t BlockKRound = UBBlockShapeBR::M;
        uint32_t BlockNRound = UBBlockShapeBR::N;

        uint32_t loopsNumN = CeilDiv(params.problemGemmShape.n(), BlockNRound);
        uint32_t loopsNumK = CeilDiv(params.problemGemmShape.k(), TileKRound);

        uint32_t loopsNum = loopsNumK * loopsNumN;
        // CeilDiv(params.problemGemmShape.k(), UBTileKRound);
        //uint32_t loopsNum = params.problemGemmShape.k();

        float alpha{1.0};
        float beta{0.0};

        uint32_t singleIdx = 0;

        #pragma unroll
        for (uint32_t i = 0; i < L0C_TILE_NUM; i++) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>((event_t)i);
        }
        
        for(uint32_t loopId = AscendC::GetBlockIdx(); loopId < loopsNum; loopId += AscendC::GetBlockNum()) {
            
            uint32_t nLoopId = loopId / loopsNumK;
            uint32_t kLoopId = loopId % loopsNumK;

            int64_t gmOffsetX;
            int64_t gmOffsetB;
            int64_t gmOffsetYR1;
            int64_t gmOffsetYR2;
            int64_t gmOffsetNextX;
            int64_t gmOffsetNextB;
            int64_t gmOffsetNextYR1;
            int64_t gmOffsetNextYR2;
            // uint32_t aivId = AscendC::GetBlockIdx();
            // if (loopId % aivNum != aivId) continue;

            uint32_t nActual = ((int32_t)nLoopId == (int32_t)(loopsNumN - 1)) ?
                params.problemGemmShape.n() - nLoopId * BlockNRound : BlockNRound;

            uint32_t kActual = ((int32_t)kLoopId == (int32_t)(loopsNumK - 1)) ?
                params.problemGemmShape.k() - kLoopId * TileKRound : TileKRound;

            int64_t gmOffsetBRow = kLoopId * TileKRound;
            int64_t gmOffsetBCol = nLoopId * BlockNRound;
            uint32_t splitNIdx = nLoopId * BlockNRound / BlockNRound;

            gmOffsetB = gmOffsetBRow * params.problemGemmShape.n() + gmOffsetBCol;
            gmOffsetYR1 = splitNIdx * params.problemGemmShape.k() + kLoopId * TileKRound;
            gmOffsetYR2 = splitNIdx * params.problemGemmShape.k() + kLoopId * TileKRound;

            gmOffsetX = 0;
            // nLoopId * BlockNRound;

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
                params.problemGemmShape.k() - kLoopIdNext * TileKRound : TileKRound;

                nextActualBlockShape = Catlass::GemvCoord{kActualNext, nActualNext};

                int64_t gmOffsetBRowNext = kLoopIdNext * TileKRound;
                int64_t gmOffsetBColNext = nLoopIdNext * BlockNRound;
                uint32_t splitNIdxNext = nLoopIdNext * BlockNRound / BlockNRound;
                gmOffsetNextB = gmOffsetBRowNext * params.problemGemmShape.n() + gmOffsetBColNext;
                gmOffsetNextYR1 = splitNIdxNext * params.problemGemmShape.k() + kLoopIdNext * TileKRound;
                gmOffsetNextYR2 = splitNIdxNext * params.problemGemmShape.k() + kLoopIdNext * TileKRound;
                gmOffsetNextX = 0;
                // nLoopIdNext * BlockNRound
            }

            LayoutYforFT layoutBlockBR{1, kActual};
            // LayoutYforFT layoutBlockBR2{1,kActual}

            LayoutXforFT layoutVR{2, TileKRound};

            /*
            void operator()(
                AscendC::GlobalTensor<ElementX> const& gmBlockX, LayoutX const& layoutX,
                AscendC::GlobalTensor<ElementA> const& gmBlockA, LayoutA const& layoutA,
                AscendC::GlobalTensor<ElementYforAIV> const& gmBlockYR1,
                AscendC::GlobalTensor<ElementYforAIV> const& gmBlockYR2,
                LayoutY const& layoutY,
                AscendC::GlobalTensor<ElementX> const& gmNextBlockX,
                AscendC::GlobalTensor<ElementA> const& gmNextBlockA,
                GemvCoord const& actualShape, GemvCoord const& actualShapeNext,
                bool isFirstBlock, bool hasNextBlock)
            */
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((event_t)(singleIdx % L0C_TILE_NUM));

            if((int32_t)nLoopId == (int32_t)(loopsNumN - 1)) {
                blockFTGemvBRAIC(
                    gmXVTail[gmOffsetX], layoutMXTail,
                    gmB[gmOffsetB], params.layoutB,
                    gmYR1[gmOffsetYR1], 
                    gmYR2[gmOffsetYR2],
                    layoutYforBR,
                    gmXVTail[gmOffsetNextX],
                    gmB[gmOffsetNextB],
                    actualBlockShape,
                    nextActualBlockShape,
                    isFirstBlock, hasNextBlock, singleIdx);  
            }else{
                blockFTGemvBRAIC(
                    gmXV[gmOffsetX], layoutMX,
                    gmB[gmOffsetB], params.layoutB,
                    gmYR1[gmOffsetYR1], 
                    gmYR2[gmOffsetYR2],
                    layoutYforBR,
                    gmXV[gmOffsetNextX],
                    gmB[gmOffsetNextB],
                    actualBlockShape,
                    nextActualBlockShape,
                    isFirstBlock,hasNextBlock, singleIdx);   
            }

            AscendC::SetFlag<AscendC::HardEvent::FIX_M>((event_t)(singleIdx % L0C_TILE_NUM));

            singleIdx++;
             
        }

        #pragma unroll
        for (uint32_t i = 0; i < L0C_TILE_NUM; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((event_t)i);
        }

        AscendC::PipeBarrier<PIPE_ALL>();

        // AscendC::SyncAll<true>();
    }

    CATLASS_DEVICE
    void Ae_spec_on_AIC(Params const &params){
        Catlass::GemmCoord problemGemmShapeAe{params.problemGemmShapeFirst.m(), L1TileShapeforAe::N, params.problemGemmShapeFirst.k()};
        BlockSchedulerFirst matmulBlockSchedulerAe(problemGemmShapeAe, Catlass::MakeCoord(L1TileShapeforAe::M, L1TileShapeforAe::N));
        uint32_t coreLoops = matmulBlockSchedulerAe.GetCoreLoops();

        uint32_t coreLoopsTotal = coreLoops;

        // * params.SplitKNum

        uint32_t KBlockSize = problemGemmShapeAe.k();
        // (params.problemGemmShapeFirst.k() + params.SplitKNum - 1) / params.SplitKNum;

        BlockFTGemvAEAIC blockFTGemvAEAIC(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);


        AscendC::GlobalTensor<ElementXforFT> gmVX;
        gmVX.SetGlobalBuffer((__gm__ ElementXforFT *)params.ptrXVR1forA);

        AscendC::GlobalTensor<ElementYforFT> gmVY;

        int64_t OffsetAeSliceInit = 0;
        gmVY.SetGlobalBuffer((__gm__ ElementYforFT *)params.ptrAMean);

        LayoutVX layoutVXforFTBlock{params.problemGemmShape.k()};
        // , uint32_t(1)
        LayoutVX layoutVXforFT{params.problemGemmShape.k()};

        LayoutYforFT layoutVYforFTBlock{1, params.problemGemmShape.m()};
        LayoutYforFT layoutVYforFT{1, params.problemGemmShape.m()};

        int64_t base_slice_offset_for_ae = params.problemGemmShape.m();
        
        for (uint32_t loopIdxTotal = AscendC::GetBlockIdx(); loopIdxTotal < coreLoopsTotal; loopIdxTotal += AscendC::GetBlockNum()) {
            // Compute block location
            uint32_t KIdx = 0;
            uint32_t loopIdx = loopIdxTotal % coreLoops;
            uint32_t KStartCoord = KIdx * KBlockSize;
            uint32_t KBlockSizeActual = KBlockSize;
            // (KIdx == (params.SplitKNum - 1)) ? (params.problemGemmShapeFirst.k() - KStartCoord) :
            Catlass::GemmCoord blockCoord = matmulBlockSchedulerAe.GetBlockCoord(loopIdx);
            Catlass::GemmCoord actualBlockShapeInBlock = matmulBlockSchedulerAe.GetActualBlockShape(blockCoord);

            Catlass::GemmCoord actualBlockShape = Catlass::GemmCoord({actualBlockShapeInBlock.m(), 1, KBlockSizeActual});
            Catlass::GemmCoord actualCoord = Catlass::GemmCoord({blockCoord.m() * L1TileShapeforAe::M, blockCoord.n() * L1TileShapeforAe::N, KStartCoord});

            // Compute initial location in logical coordinates
            Catlass::MatrixCoord offsetA{blockCoord.m() * L1TileShapeforAe::M, KStartCoord};
            Catlass::MatrixCoord offsetXforFT{blockCoord.n() * L1TileShapeforAe::N, KStartCoord};
            Catlass::MatrixCoord offsetYforFT{blockCoord.n() * L1TileShapeforAe::N, blockCoord.m() * L1TileShapeforAe::M};
            
            
            int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
            
            int64_t gmOffsetVXforFT = 0;
            int64_t gmOffsetVYforFTinBlock = blockCoord.m() * L1TileShapeforAe::M;
            int64_t gmOffsetVYforFT = OffsetAeSliceInit + gmOffsetVYforFTinBlock + KIdx * base_slice_offset_for_ae;

            // Compute block-scoped matrix multiply-add
            /*
            CATLASS_DEVICE
            void operator()(
                AscendC::GlobalTensor<ElementA> const & gmA, LayoutA const &layoutA,
                AscendC::GlobalTensor<ElementX> const & gmVX, LayoutVX const &layoutVX,
                AscendC::GlobalTensor<ElementY> const & gmVY, LayoutY const &layoutVY,
                Catlass::GemmCoord const &actualShape)
            */
            blockFTGemvAEAIC(
                gmA[gmOffsetA], params.layoutA,
                gmVX, layoutVXforFTBlock,
                gmVY[gmOffsetVYforFT], layoutVYforFTBlock,
                actualBlockShape);
            
            
        }
    }

    CATLASS_DEVICE
    void ABr_spec_on_AIC(Params const &params){
        BlockSchedulerFirst matmulBlockSchedulerFirst(params.problemGemmShapeFirst, Catlass::MakeCoord(L1TileShapeforFT::M, L1TileShapeforFT::N));
        uint32_t coreLoops = matmulBlockSchedulerFirst.GetCoreLoops();

        uint32_t coreLoopsTotal = coreLoops;
        // * params.SplitKNum

        uint32_t KBlockSize = params.problemGemmShapeFirst.k();
        // ( + params.SplitKNum - 1) / params.SplitKNum;

        BlockMmadABr blockMmadABr(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementYforFT> gmACast;
        gmACast.SetGlobalBuffer((__gm__ ElementYforFT *)params.ptrACast);

        AscendC::GlobalTensor<ElementYforFT> gmXR1;
        gmXR1.SetGlobalBuffer((__gm__ ElementYforFT *)params.ptrBR1);

        AscendC::GlobalTensor<ElementYforFT> gmXR2;
        gmXR2.SetGlobalBuffer((__gm__ ElementYforFT *)params.ptrBR2);

        AscendC::GlobalTensor<ElementYforFT> gmYR1;
        gmYR1.SetGlobalBuffer((__gm__ ElementYforFT *)params.ptrZRow2R1);

        AscendC::GlobalTensor<ElementYforFT> gmYR2;
        gmYR2.SetGlobalBuffer((__gm__ ElementYforFT *)params.ptrZRow2R2);
        
        // if(params.SplitKNum > 1){
        //     gmY.SetGlobalBuffer((__gm__ ElementYforFT *)params.ptrWorkspace);
        // }else{
            
        // }


        LayoutXforFT layoutXforFTBlock{params.SplitNnumChunk, params.problemGemmShape.k()};
        // , uint32_t(1)
        LayoutYforFT layoutXforFT{params.SplitNnumChunk, params.problemGemmShape.k()};

        LayoutYforFT layoutYforFTBlock{params.SplitNnumChunk, params.problemGemmShape.m()};
        LayoutYforFT layoutYforFT{params.SplitNnumChunk, params.problemGemmShape.m()};


        // LayoutX layoutX{params.problemGemmShape.n()};
        // Catlass::MatrixCoord loopsMN = matmulBlockSchedulerFirst.loopsMN;

        int64_t base_slice_offset = 2 * params.SplitNnumChunk * params.problemGemmShape.m();

        Catlass::MatrixCoord loopsMN = matmulBlockSchedulerFirst.loopsMN;

        uint32_t N_R1_size_common = L1TileShapeforFT::N / 2;
        uint32_t N_R2_size_common = L1TileShapeforFT::N - (L1TileShapeforFT::N / 2);
        
        for (uint32_t loopIdxTotal = AscendC::GetBlockIdx(); loopIdxTotal < coreLoopsTotal; loopIdxTotal += AscendC::GetBlockNum()) {
            // Compute block location
            uint32_t KIdx = 0;
            // loopIdxTotal / coreLoops;
            uint32_t loopIdx = loopIdxTotal % coreLoops;
            uint32_t KStartCoord = KIdx * KBlockSize;
            uint32_t KBlockSizeActual = KBlockSize;
            // (KIdx == (params.SplitKNum - 1)) ? (params.problemGemmShapeFirst.k() - KStartCoord) : KBlockSize;
            Catlass::GemmCoord blockCoord = matmulBlockSchedulerFirst.GetBlockCoord(loopIdx);
            Catlass::GemmCoord actualBlockShapeInBlock = matmulBlockSchedulerFirst.GetActualBlockShape(blockCoord);

            uint32_t actual_N_R1_size = N_R1_size_common;
            uint32_t actual_N_R2_size = N_R2_size_common;

            if(blockCoord.n() == loopsMN.column() - 1){
                actual_N_R1_size = params.SplitNnumChunk - blockCoord.n() * N_R1_size_common;
                actual_N_R2_size = params.SplitNnumChunk - blockCoord.n() * N_R2_size_common;
            }

            uint32_t actual_N_size = actual_N_R1_size + actual_N_R2_size;

            Catlass::GemmCoord actualBlockShape = Catlass::GemmCoord({actualBlockShapeInBlock.m(), actual_N_size, KBlockSizeActual});
            Catlass::GemmCoord actualCoord = Catlass::GemmCoord({blockCoord.m() * L1TileShapeforFT::M, blockCoord.n() * L1TileShapeforFT::N, KStartCoord});

            // uint32_t total_N_coord = blockCoord.n() * L1TileShapeforFT::N;
            uint32_t R1_N_coord = blockCoord.n() * N_R1_size_common;
            uint32_t R2_N_coord = blockCoord.n() * N_R2_size_common;
            // Compute initial location in logical coordinates
            Catlass::MatrixCoord offsetA{blockCoord.m() * L1TileShapeforFT::M, KStartCoord};
            Catlass::MatrixCoord offsetXR1forFT{R1_N_coord, KStartCoord};
            Catlass::MatrixCoord offsetXR2forFT{R2_N_coord, KStartCoord};

            Catlass::MatrixCoord offsetYR1forFT{R1_N_coord, blockCoord.m() * L1TileShapeforFT::M};
            Catlass::MatrixCoord offsetYR2forFT{R2_N_coord, blockCoord.m() * L1TileShapeforFT::M};
            
            int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);

            int64_t gmOffsetXR1forFT = layoutXforFT.GetOffset(offsetXR1forFT);
            int64_t gmOffsetXR2forFT = layoutXforFT.GetOffset(offsetXR2forFT);

            int64_t gmOffsetYR1forFTinBlock = layoutYforFT.GetOffset(offsetYR1forFT);
            int64_t gmOffsetYR2forFTinBlock = layoutYforFT.GetOffset(offsetYR2forFT);

            int64_t gmOffsetYR1forFT = gmOffsetYR1forFTinBlock + KIdx * base_slice_offset;
            int64_t gmOffsetYR2forFT = gmOffsetYR2forFTinBlock + KIdx * base_slice_offset;

            // Compute block-scoped matrix multiply-add
            /*
            CATLASS_DEVICE
            void operator()(
                AscendC::GlobalTensor<ElementAforFT> const & gmA, LayoutA const &layoutA,
                AscendC::GlobalTensor<ElementXforFT> const & gmXR1,
                AscendC::GlobalTensor<ElementXforFT> const & gmXR2, 
                LayoutX const &layoutX,
                AscendC::GlobalTensor<ElementY> const & gmYR1, 
                AscendC::GlobalTensor<ElementY> const & gmYR2,
                LayoutY const &layoutY,
                Catlass::GemmCoord const &actualShape, 
                uint32_t actual_N_R1_size)
            */

            blockMmadABr(
                gmACast[gmOffsetA], params.layoutA,
                gmXR1[gmOffsetXR1forFT],
                gmXR2[gmOffsetXR2forFT],
                layoutXforFTBlock,
                gmYR1[gmOffsetYR1forFT], 
                gmYR2[gmOffsetYR2forFT], 
                layoutYforFTBlock,
                actualBlockShape, 
                actual_N_R1_size);

            // 通知相应 AIV core，MMAD计算已经完成了，结果已经写入了GM 
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
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

            // blockMmad(
            //     gmA[gmOffsetA], params.layoutA,
            //     gmB[gmOffsetB], params.layoutB,
            //     gmC[gmOffsetC], layoutC,
            //     actualBlockShape);
            
            // 通知相应 AIV core，MMAD计算已经完成了，结果已经写入了GM 
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
        }

        // AscendC::PipeBarrier<PIPE_ALL>();
    }

    template<>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        
            
        
        BR_split_op_on_AIC(params);
        
        // AscendC::SyncAll<true>();
        Catlass::Arch::CrossCoreBarrierAIC<0x0, PIPE_FIX>();
        // Catlass::Arch::CrossCoreBarrierAIC<0x0, PIPE_M>();
        // Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
        // Matmul_op_ABe_fused(params);
        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
        
        Ae_spec_on_AIC(params);
        Catlass::Arch::CrossCoreBarrierAIC<0x0, PIPE_FIX>();

        // Catlass::Arch::CrossCoreBarrierAIC<0x0, PIPE_M>();
        // AscendC::SyncAll<true>();
        
        if(params.problemGemmShapeRemain.n() > 0){
            Matmul_op(params);   
        }
        Catlass::Arch::CrossCoreBarrierAIC<0x0, PIPE_FIX>();
        Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStoreAIV);
        
        ABr_spec_on_AIC(params);

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

        AscendC::GlobalTensor<ElementYforB> gmBMaxSlice;
        gmBMaxSlice.SetGlobalBuffer((__gm__ ElementYforB *)params.ptrBMaxSlice);

        AscendC::GlobalTensor<ElementYforB> gmBMinSlice;
        gmBMinSlice.SetGlobalBuffer((__gm__ ElementYforB *)params.ptrBMinSlice);

        AscendC::GlobalTensor<ElementYforA> gmAMin;
        gmAMin.SetGlobalBuffer((__gm__ ElementYforA *)params.ptrAMin);

        AscendC::GlobalTensor<ElementYforA> gmAMax;
        gmAMax.SetGlobalBuffer((__gm__ ElementYforA *)params.ptrAMax);

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
                uint32_t splitNIdx = nLoopId * UBBlockNRoundforB / UBBlockNRoundforB;
                int64_t gmOffsetB = gmOffsetBRow * params.problemGemmShape.n() + gmOffsetBCol;
                int64_t gmOffsetBEMax = splitNIdx * params.problemGemmShape.k() + kLoopId * UBBlockKRoundforB;

                Catlass::GemvCoord actualBlockShapeforB = Catlass::GemvCoord{kActual, nActual};
                LayoutYforB layoutBE{kActual};
                LayoutYforB layoutE{nActual};

                /*
                void BlockRed(
                    AscendC::GlobalTensor<ElementA> const &gmA, 
                    LayoutA const &layoutA,
                    AscendC::GlobalTensor<ElementX> const &gmZMin, 
                    AscendC::GlobalTensor<ElementX> const &gmZMax,
                    LayoutX const &layoutX, GemvCoord const &actualShape)
                */
            
                blockFTSum.BlockRed(
                    gmB[gmOffsetB], params.layoutB,
                    gmBMinSlice[gmOffsetBEMax], 
                    gmBMaxSlice[gmOffsetBEMax],
                    layoutBE, actualBlockShapeforB);
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
                CATLASS_DEVICE
                void operator()(
                    AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
                    AscendC::GlobalTensor<ElementY> const &gmZMin, 
                    AscendC::GlobalTensor<ElementY> const &gmZMax,
                    LayoutY const &layoutY,
                    GemvCoord const &actualShape)
                */
                blockFTSum(
                    gmA[gmOffsetA], params.layoutA,
                    gmAMin[gmOffsetAMax],
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

        AscendC::GlobalTensor<ElementZforBRed> gmBMeanAbs;
        gmBMeanAbs.SetGlobalBuffer((__gm__ ElementZforBRed *)params.ptrBMeanAbs);
        AscendC::GlobalTensor<ElementZforBRed> gmBMeanSquare;
        gmBMeanSquare.SetGlobalBuffer((__gm__ ElementZforBRed *)params.ptrBMeanSquare);
        AscendC::GlobalTensor<ElementZforBRed> gmBVar;
        gmBVar.SetGlobalBuffer((__gm__ ElementZforBRed *)params.ptrBVar);
        AscendC::GlobalTensor<ElementZforBRed> gmBSumSlice;
        gmBSumSlice.SetGlobalBuffer((__gm__ ElementZforBRed *)params.ptrBR1);
        AscendC::GlobalTensor<ElementYforB> gmBMaxSlice;
        gmBMaxSlice.SetGlobalBuffer((__gm__ ElementYforB *)params.ptrBMaxSlice);
        AscendC::GlobalTensor<ElementYforB> gmBMinSlice;
        gmBMinSlice.SetGlobalBuffer((__gm__ ElementYforB *)params.ptrBMinSlice);

        // Get aicore information

        uint32_t UBTileSplitM = params.SplitReduceM;
        // RoundUp(UBTileShapeBReduce::M, UBAlignHelper::ALIGN);
        uint32_t UBTileNRound = RoundUp(params.SplitReduceN, UBAlignHelperOut::ALIGN);

        uint32_t Reduce_M_size = params.SplitNnumChunk;
        uint32_t loopsNumforB = CeilDiv((Reduce_M_size - 1), UBTileSplitM);

        if(Reduce_M_size < 2){
            loopsNumforB = 0;
        }

        uint32_t ABR_M_size = params.SplitNnumChunk;
        uint32_t ABR_N_size = params.problemGemmShape.m();
        uint32_t loopsNumforABE = 0;
        uint32_t loopsNumMforABE = 0;
        uint32_t loopsNumNforABE = 0;
        uint32_t ABR_N_coord_base = 0;
        uint32_t ABR_N_size_part = ABR_N_size / 2;

        int64_t OffsetAeSliceInit = 2 * params.SplitNnumChunk * params.problemGemmShape.m() * params.SplitKNum;

        loopsNumforABE = 0;

        int64_t OffsetInSliceInit = 0;

        LayoutYforFT layoutSliceOut{params.SplitNnumChunk, params.problemGemmShape.m()};
        LayoutYforFT layoutSliceIn{params.SplitNnumChunk, params.problemGemmShape.m()};

        LayoutVX layoutSliceAE{params.problemGemmShape.m()};
        
        BlockSliceRed blockSliceRed(resource);

        int64_t OffsetInMaxInit = 0;

        Catlass::layout::VectorLayout layoutOut{params.SplitNnumChunk};
        LayoutYforFT layoutWorkforRed{params.SplitNnumChunk, params.problemGemmShape.k()};

        if(aivSubIndex < 1){
            uint32_t loopsNumTotal = loopsNumforB + 1 + loopsNumforABE;
            for(uint32_t loopId = aicoreIndex; loopId < loopsNumTotal; loopId += aicoreNum) {
                
                if(loopId < (loopsNumforB + 1)){
                    uint32_t mActual = ((int32_t)loopId == (int32_t)(loopsNumforB - 1)) ?
                        (Reduce_M_size - 1 - loopId * UBTileSplitM) : UBTileSplitM;

                    uint32_t nActual = params.problemGemmShape.k();
                    int64_t gmOffsetInBSum = loopId * UBTileSplitM * params.problemGemmShape.k();
                    int64_t gmOffsetOutBMeanAbs = loopId * UBTileSplitM;
                    int64_t gmOffsetOutBMeanSquare = loopId * UBTileSplitM;

                    int64_t gmOffsetInBMax = OffsetInMaxInit + loopId * UBTileSplitM * params.problemGemmShape.k();
                    int64_t gmOffsetInBMin = OffsetInMaxInit + loopId * UBTileSplitM * params.problemGemmShape.k();
                    int64_t gmOffsetOutBVar = loopId * UBTileSplitM;

                    float n_scale_ratio = (params.n_scale_ratios[0]);

                    if((int32_t)loopId > (int32_t)(loopsNumforB - 1)){
                        mActual = 1;
                        n_scale_ratio = (params.n_scale_ratios[1]);
                        gmOffsetInBSum = (Reduce_M_size - 1) * params.problemGemmShape.k();
                        gmOffsetOutBMeanAbs = (Reduce_M_size - 1);
                        gmOffsetOutBMeanSquare = (Reduce_M_size - 1);

                        gmOffsetInBMax = OffsetInMaxInit + (Reduce_M_size - 1) * params.problemGemmShape.k();
                        gmOffsetInBMin = OffsetInMaxInit + (Reduce_M_size - 1) * params.problemGemmShape.k();
                        gmOffsetOutBVar = (Reduce_M_size - 1);
                    }

                    Catlass::GemvCoord actualBlockShape = Catlass::GemvCoord{mActual, nActual};
                
                    /*
                    void RowMeanAbsSquare(
                    AscendC::GlobalTensor<ElementB> const &gmB, LayoutB const &layoutB,
                    AscendC::GlobalTensor<ElementY> const &gmZMeanAbs,
                    AscendC::GlobalTensor<ElementY> const &gmZMeanSquare, 
                    LayoutX const &layoutX,
                    GemvCoord const &actualShape,
                    uint32_t NRealRound,
                    float n_scale_ratio)
                    */
                    blockSliceRed.RowMeanAbsSquare(gmBSumSlice[gmOffsetInBSum], 
                        layoutWorkforRed,
                        gmBMeanAbs[gmOffsetOutBMeanAbs], 
                        gmBMeanSquare[gmOffsetOutBMeanSquare],
                        layoutOut,
                        actualBlockShape,
                        UBTileNRound, n_scale_ratio);
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
                    int64_t gmOffsetInBMin = OffsetInMaxInit + loopId * UBTileSplitM * params.problemGemmShape.k();

                    int64_t gmOffsetOutBVar = loopId * UBTileSplitM;
                    int64_t gmOffsetInBSum = loopId * UBTileSplitM * params.problemGemmShape.k();
                
                    float n_scale_ratio = (params.n_scale_ratios[0]);
                    if((int32_t)loopId > (int32_t)(loopsNumforB - 1)){
                        mActual = 1;
                        n_scale_ratio = (params.n_scale_ratios[1]);
                        gmOffsetInBMax = OffsetInMaxInit + (Reduce_M_size - 1) * params.problemGemmShape.k();
                        gmOffsetInBMin = OffsetInMaxInit + (Reduce_M_size - 1) * params.problemGemmShape.k();
                        gmOffsetOutBVar = (Reduce_M_size - 1);
                        gmOffsetInBSum = (Reduce_M_size - 1) * params.problemGemmShape.k();
                    }

                    Catlass::GemvCoord actualBlockShape = Catlass::GemvCoord{mActual, nActual};
                    /*
                    CATLASS_DEVICE
                    void RowVariance(
                        AscendC::GlobalTensor<ElementB> const &gmBMean,
                        AscendC::GlobalTensor<ElementB> const &gmBMax,
                        AscendC::GlobalTensor<ElementB> const &gmBMin, 
                        LayoutB const &layoutB,
                        AscendC::GlobalTensor<ElementY> const &gmZVar, 
                        LayoutX const &layoutX,
                        GemvCoord const &actualShape,
                        uint32_t NRealRound,
                        float n_scale_ratio
                    )
                    */

                    blockSliceRed.RowVariance(gmBSumSlice[gmOffsetInBSum], 
                        gmBMaxSlice[gmOffsetInBMax],
                        gmBMinSlice[gmOffsetInBMin],
                        layoutWorkforRed,
                        gmBVar[gmOffsetOutBVar], layoutOut,
                        actualBlockShape,
                        UBTileNRound, n_scale_ratio);
                }

            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_DEVICE
    void CR_split_op(Params const &params){
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

        AscendC::GlobalTensor<ElementYforFT> gmVXR2;
        gmVXR2.SetGlobalBuffer((__gm__ ElementYforFT *)params.ptrVXR2);

        AscendC::GlobalTensor<ElementYforFT> gmVXR2Tail;
        gmVXR2Tail.SetGlobalBuffer((__gm__ ElementYforFT *)params.ptrVXR2Tail);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);

        AscendC::GlobalTensor<ElementYforFT> gmACast;
        gmACast.SetGlobalBuffer((__gm__ ElementYforFT *)params.ptrACast);

        AscendC::GlobalTensor<ElementZforBRed> gmCR1Slice;
        gmCR1Slice.SetGlobalBuffer((__gm__ ElementZforBRed *)params.ptrCR1Slice);

        AscendC::GlobalTensor<ElementZforBRed> gmCR2Slice;
        gmCR2Slice.SetGlobalBuffer((__gm__ ElementZforBRed *)params.ptrCR2Slice);

        Catlass::MatrixCoord loopsMN = matmulBlockScheduler.loopsMN;

        uint32_t UBTileMRoundCR = RoundUp(UBTileShapeCR::M, UBAlignHelperOut::ALIGN);
        uint32_t UBTileNRoundCR = RoundUp(UBTileShapeCR::N, UBAlignHelperOut::ALIGN);

        uint32_t UBTileMRoundACast = RoundUp(UBTileShapeACast::M, UBAlignHelperOut::ALIGN);
        uint32_t UBTileNRoundACast = RoundUp(UBTileShapeACast::N, UBAlignHelperOut::ALIGN);

        uint32_t UBBlockMRound = RoundUp(UBBlockShapeCR::M, UBAlignHelperOut::ALIGN);
        uint32_t UBBlockNRound = RoundUp(UBBlockShapeCR::N, UBAlignHelperOut::ALIGN);

        uint32_t element_num = params.problemGemmShape.m();

        uint32_t total_input_elements = element_num;

        LayoutYforFT layoutCRforFT{params.SplitNnumChunk, params.problemGemmShape.m()};
        LayoutYforFT layoutABrforFT{params.SplitNnumChunk, params.problemGemmShape.m()};

        uint32_t Remain_N_size = params.problemGemmShape.n() % FTChunkShape::N;
        if(Remain_N_size < 1){
            Remain_N_size = FTChunkShape::N;
        }

        LayoutCOMPVX layoutInputVX{FTChunkShape::N};
        LayoutCOMPVX layoutInputVXTail{Remain_N_size};

        // LayoutCOMPY layoutInputY{element_num};
        LayoutCOMPVX layoutVAforFT{element_num};
        LayoutCOMPMX layoutMAforFT{params.SplitNnumChunk,element_num};

        LayoutCOMPVX layoutVARed{element_num};

        LayoutCOMPVX layoutVCR{element_num};
        LayoutCOMPMX layoutMCR{params.SplitNnumChunk,element_num};
        LayoutCOMPVX layoutVBforFT{params.SplitNnumChunk};

        uint32_t base_cr_slice_offset = params.SplitNnumChunk * element_num;

        uint32_t Nloop = (params.problemGemmShape.n() + L1TileShape::N - 1) / L1TileShape::N;
        uint32_t ACastKUnit = params.problemGemmShape.k() / Nloop;

        for (uint32_t loopIdx = aicoreIndex; loopIdx < coreLoops; loopIdx += aicoreNum) {
            // Compute block location
            Catlass::GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            
            Catlass::GemmCoord actualCoord = Catlass::GemmCoord({blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N, blockCoord.k() * L1TileShape::K});

            uint32_t BlockNStart = blockCoord.n() * L1TileShape::N;

            uint32_t splitNIdx = BlockNStart / FTChunkShape::N;
            uint32_t XNStart = BlockNStart % FTChunkShape::N;
            uint32_t sliceIdx = XNStart / L1TileShape::N;

            uint32_t blockNIdx = BlockNStart / L1TileShape::N;

            // Compute initial location in logical coordinates
            Catlass::MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
            Catlass::MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockNIdx * ACastKUnit};
            Catlass::MatrixCoord offsetACast{blockCoord.m() * L1TileShape::M, blockNIdx * ACastKUnit};

            Catlass::MatrixCoord offsetCRforFT{splitNIdx, blockCoord.m() * L1TileShape::M};

            uint32_t mActual = UBBlockMRound;

            if(blockCoord.m() == loopsMN.row() -1) {
                mActual = params.problemGemmShape.m() - blockCoord.m() * L1TileShape::M;
            }

            uint32_t nActual = UBBlockNRound;

            if(blockCoord.n() == loopsMN.column() - 1){
                nActual = params.problemGemmShape.n() - blockCoord.n() * L1TileShape::N;
            }

            uint32_t kActualforACast = ACastKUnit;

            if(blockNIdx == (Nloop - 1)){
                kActualforACast = params.problemGemmShape.k() - blockNIdx * ACastKUnit;
            }
            
            int64_t gmOffsetC = params.layoutC.GetOffset(offsetC);
            int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
            int64_t gmOffsetAOut = params.layoutA.GetOffset(offsetACast);
            
            int64_t gmOffsetCRforFT = sliceIdx * base_cr_slice_offset + layoutCRforFT.GetOffset(offsetCRforFT);

            int64_t gmOffsetARed = blockCoord.m() * L1TileShape::M;

            int64_t gmOffsetX = XNStart;
            
            Catlass::GemvCoord actualBlockShapeforC = Catlass::GemvCoord{mActual, nActual};
            Catlass::GemvCoord actualBlockShapeforA = Catlass::GemvCoord{mActual, kActualforACast};

            /*
            CATLASS_DEVICE
            void operator()(
                AscendC::GlobalTensor<ElementC> const &gmC, LayoutC const &layoutC,
                AscendC::GlobalTensor<ElementA> const &gmA, 
                AscendC::GlobalTensor<ElementY> const &gmAOut,
                LayoutA const &layoutA,
                AscendC::GlobalTensor<ElementX> const &gmXR2, 
                LayoutVX const &layoutXforFT,
                AscendC::GlobalTensor<ElementY> const &gmYR1, 
                AscendC::GlobalTensor<ElementY> const &gmYR2, 
                LayoutVY const &layoutY,
                GemvCoord const &actualShapeforC, 
                GemvCoord const &actualShapeforA, 
                uint32_t aiv_part_num,
                Catlass::Arch::CrossCoreFlagWithReverse<> & flagAicFinishStore)
            */

            if(splitNIdx < (params.SplitNnumChunk - 1)){
                blockFTGemvAIV(
                    gmC[gmOffsetC], params.layoutC,
                    gmA[gmOffsetA], gmACast[gmOffsetAOut],
                    params.layoutA,
                    gmVXR2[gmOffsetX], layoutInputVX,
                    gmCR1Slice[gmOffsetCRforFT], gmCR2Slice[gmOffsetCRforFT], layoutVCR, 
                    actualBlockShapeforC, actualBlockShapeforA,
                    aiv_part_num, flagAicFinishStore
                );
            }else{
                blockFTGemvAIV(
                    gmC[gmOffsetC], params.layoutC,
                    gmA[gmOffsetA], gmACast[gmOffsetAOut],
                    params.layoutA,
                    gmVXR2Tail[gmOffsetX], layoutInputVXTail,
                    gmCR1Slice[gmOffsetCRforFT], gmCR2Slice[gmOffsetCRforFT], layoutVCR, 
                    actualBlockShapeforC, actualBlockShapeforA,
                    aiv_part_num, flagAicFinishStore
                );
            }
            
            // Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_DEVICE
    void CR_verify_op_fused(Params const &params, GM_ADDR ptrOutputCOMP)
    {
        
        AB_red_split_op(params);
        Catlass::Arch::CrossCoreBarrierAIV<0x0, PIPE_V>();
        Catlass::Arch::CrossCoreBarrierAIV<0x0, PIPE_MTE3>();
        
        Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);
        ABE_B_reduce_fused_op(params);
        
        // AscendC::SyncAll<true>(); 
        Catlass::Arch::CrossCoreBarrierAIV<0x0, PIPE_V>();
        Catlass::Arch::CrossCoreBarrierAIV<0x0, PIPE_MTE3>();

        CR_split_op(params);

        Catlass::Arch::CrossCoreBarrierAIV<0x0, PIPE_V>();
        Catlass::Arch::CrossCoreBarrierAIV<0x0, PIPE_MTE3>();

        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStoreAIV);

        BlockChunkVerify blockChunkVerify(resource);


        // 完成最后的比较即可
        BlockSchedulerFirst matmulBlockSchedulerFirst(params.problemGemmShapeFirst, Catlass::MakeCoord(L1TileShapeforFT::M, L1TileShapeforFT::N));

        uint32_t coreLoops = matmulBlockSchedulerFirst.GetCoreLoops();

        uint32_t coreLoopsTotal = coreLoops;
        uint32_t aivIndex = AscendC::GetBlockIdx();
        uint32_t aicoreIndex = aivIndex / AscendC::GetSubBlockNum();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t aivNum = aicoreNum * AscendC::GetSubBlockNum();
        uint32_t aiv_part_num = 1 * AscendC::GetTaskRation();
        uint32_t align = Catlass::BYTE_PER_C0 / sizeof(ElementZforBRed);

        uint32_t KBlockSize = params.problemGemmShapeFirst.k();

        AscendC::GlobalTensor<ElementYforA> gmAMean;
        gmAMean.SetGlobalBuffer((__gm__ ElementYforA *)params.ptrAMean);

        AscendC::GlobalTensor<ElementYforA> gmAMax;
        gmAMax.SetGlobalBuffer((__gm__ ElementYforA *)params.ptrAMax);

        AscendC::GlobalTensor<ElementYforA> gmAMin;
        gmAMin.SetGlobalBuffer((__gm__ ElementYforA *)params.ptrAMin);

        AscendC::GlobalTensor<ElementCOMPX> gmCOMPXR1;
        gmCOMPXR1.SetGlobalBuffer((__gm__ ElementCOMPX *)params.ptrZRow2R1);

        AscendC::GlobalTensor<ElementCOMPX> gmCOMPXR2;
        gmCOMPXR2.SetGlobalBuffer((__gm__ ElementCOMPX *)params.ptrZRow2R2);

        AscendC::GlobalTensor<ElementCOMPY> gmCOMPYR1;
        gmCOMPYR1.SetGlobalBuffer((__gm__ ElementCOMPY *)params.ptrZRowR1);

        AscendC::GlobalTensor<ElementCOMPY> gmCOMPYR2;
        gmCOMPYR2.SetGlobalBuffer((__gm__ ElementCOMPY *)params.ptrZRowR2);

        AscendC::GlobalTensor<ElementCOMPY> gmCR1Slice;
        gmCR1Slice.SetGlobalBuffer((__gm__ ElementCOMPY *)params.ptrCR1Slice);

        AscendC::GlobalTensor<ElementCOMPY> gmCR2Slice;
        gmCR2Slice.SetGlobalBuffer((__gm__ ElementCOMPY *)params.ptrCR2Slice);

        AscendC::GlobalTensor<ElementCOMPZ> gmCOMPZ;
        gmCOMPZ.SetGlobalBuffer((__gm__ ElementCOMPZ *)ptrOutputCOMP);

        AscendC::GlobalTensor<ElementZ> gmT;
        gmT.SetGlobalBuffer((__gm__ ElementZ *)params.ptrThreZ);

        AscendC::GlobalTensor<ElementZforBRed> gmBMeanAbs;
        gmBMeanAbs.SetGlobalBuffer((__gm__ ElementZforBRed *)params.ptrBMeanAbs);

        AscendC::GlobalTensor<ElementZforBRed> gmBMeanSquare;
        gmBMeanSquare.SetGlobalBuffer((__gm__ ElementZforBRed *)params.ptrBMeanSquare);
        
        AscendC::GlobalTensor<ElementZforBRed> gmBVar;
        gmBVar.SetGlobalBuffer((__gm__ ElementZforBRed *)params.ptrBVar);

        Catlass::MatrixCoord loopsMN = matmulBlockSchedulerFirst.loopsMN;

        uint32_t UBTileMRound = RoundUp(UBTileShapeVerify::M, UBAlignHelperOut::ALIGN);
        uint32_t UBTileNRound = RoundUp(UBTileShapeVerify::N, UBAlignHelperOut::ALIGN);

        uint32_t UBBlockMRound = RoundUp(UBBlockShapeVerify::M, UBAlignHelperOut::ALIGN);
        uint32_t UBBlockNRound = RoundUp(UBBlockShapeVerify::N, UBAlignHelperOut::ALIGN);

        uint32_t element_num = params.problemGemmShape.m();

        uint32_t ThreUBTileNRound = RoundUp(L0TileShapeforFT::M / 2, UBAlignHelperOut::ALIGN);
        uint32_t ThreUBTileMRound = RoundUp(L0TileShapeforFT::N / 2, UBAlignHelperOut::ALIGN);

        uint32_t ThreUBBlockNRound = RoundUp(L1TileShapeforFT::M, UBAlignHelperOut::ALIGN);
        uint32_t ThreUBBlockMRound = RoundUp(L1TileShapeforFT::N / 2, UBAlignHelperOut::ALIGN);

        if(FUSE_TYPE == FT_AIV_PIPE_FUSE_TYPE::ABE_FUSED_THRE){
            ThreUBTileMRound = UBTileMRound;
            ThreUBTileNRound = RoundUp(L0TileShapeforFT::M / 2, UBAlignHelperOut::ALIGN);

            ThreUBBlockMRound = UBBlockMRound;
            ThreUBBlockNRound = RoundUp(L1TileShapeforFT::M, UBAlignHelperOut::ALIGN);
        }else{
            ThreUBTileNRound = RoundUp(L0TileShapeforFT::M / 2, UBAlignHelperOut::ALIGN);
            ThreUBTileMRound = RoundUp(L0TileShapeforFT::N / 2, UBAlignHelperOut::ALIGN);

            ThreUBBlockNRound = RoundUp(L1TileShapeforFT::M, UBAlignHelperOut::ALIGN);
            ThreUBBlockMRound = RoundUp(L1TileShapeforFT::N / 2, UBAlignHelperOut::ALIGN);
        }
        // Represent the full gm
        // Get aicore information

        uint32_t total_input_elements = element_num;

        uint32_t NBlockNumAlign = element_num / UBBlockNRound;
        uint32_t NBlockRemain = element_num % UBBlockNRound;

        uint32_t UBBlockPartNRound = (UBBlockNRound / aiv_part_num);
        uint32_t out_z_part_blk_align = (UBBlockPartNRound + 8 - 1) / 8;
        uint32_t out_z_blk_align = out_z_part_blk_align * aiv_part_num;
        uint32_t out_z_total_align = out_z_blk_align * NBlockNumAlign;

        uint32_t NBlockRemainPart1 = NBlockRemain / aiv_part_num;
        uint32_t NBlockRemainPart2 = NBlockRemain - NBlockRemainPart1;

        uint32_t out_z_part_blk_remain1 = (NBlockRemainPart1 < 1) ? 0 : ((NBlockRemainPart1 + 8 - 1) / 8);
        uint32_t out_z_part_blk_remain2 = (NBlockRemainPart2 < 1) ? 0 : ((NBlockRemainPart2 + 8 - 1) / 8);

        uint32_t total_output_elements = out_z_total_align + out_z_part_blk_remain1 + out_z_part_blk_remain2;

        uint32_t total_input_bytes = total_input_elements * sizeof(ElementCOMPX);

        LayoutYforFT layoutCRforFT{params.SplitNnumChunk, params.problemGemmShape.m()};
        LayoutYforFT layoutABRforFT{params.SplitNnumChunk, params.problemGemmShape.m()};

        LayoutYforFT layoutThreforFT{params.SplitNnumChunk, params.problemGemmShape.m()};

        LayoutYforFT layoutCOMPXforFT{params.SplitNnumChunk, params.problemGemmShape.m()};
        LayoutYforFT layoutCOMPYforFT{params.SplitNnumChunk, params.problemGemmShape.m()};

        LayoutYforFT layoutCOMPZforFT{params.SplitNnumChunk, total_output_elements};

        LayoutCOMPVX layoutInputX{element_num};
        // LayoutCOMPY layoutInputY{element_num};
        LayoutCOMPVX layoutAforFT{element_num};
        LayoutCOMPVX layoutARed{element_num};
        LayoutCOMPVX layoutCR{element_num};
        LayoutCOMPVX layoutBforFT{params.SplitNnumChunk};
            
        LayoutCOMPVZ layoutOutputVZ{total_output_elements};

        uint32_t M_R1_size_common = L1TileShapeforFT::N / 2;
        uint32_t M_R2_size_common = L1TileShapeforFT::N - (L1TileShapeforFT::N / 2);

        uint32_t CRSliceNum = (FTChunkShape::N + L1TileShape::N - 1) / L1TileShape::N;

        for (uint32_t loopIdx = aicoreIndex; loopIdx < coreLoops; loopIdx += aicoreNum) {
            // Compute block location
            uint32_t KIdx = 0;
            // loopIdxTotal / coreLoops;
            uint32_t KStartCoord = KIdx * KBlockSize;
            uint32_t KBlockSizeActual = KBlockSize;
            // (KIdx == (params.SplitKNum - 1)) ? (params.problemGemmShapeFirst.k() - KStartCoord) : KBlockSize;
            Catlass::GemmCoord blockCoord = matmulBlockSchedulerFirst.GetBlockCoord(loopIdx);
            Catlass::GemmCoord actualBlockShapeInBlock = matmulBlockSchedulerFirst.GetActualBlockShape(blockCoord);

            uint32_t actual_M_R1_size = M_R1_size_common;

            if(blockCoord.n() == loopsMN.column() - 1){
                actual_M_R1_size = params.SplitNnumChunk - blockCoord.n() * M_R1_size_common;
            }


            Catlass::GemvCoord actualBlockShape = Catlass::GemvCoord({actual_M_R1_size, actualBlockShapeInBlock.m()});

            uint32_t m_actual = actual_M_R1_size;
            uint32_t n_actual = actualBlockShapeInBlock.m();

            Catlass::GemmCoord actualCoord = Catlass::GemmCoord({blockCoord.m() * L1TileShapeforFT::M, blockCoord.n() * L1TileShapeforFT::N, KStartCoord});

            // uint32_t total_N_coord = blockCoord.n() * L1TileShapeforFT::N;
            uint32_t R1_M_coord = blockCoord.n() * M_R1_size_common;
            uint32_t R2_M_coord = blockCoord.n() * M_R2_size_common;

            uint32_t Blk_N_coord = blockCoord.m() * L1TileShapeforFT::M;

            Catlass::MatrixCoord offsetCR1{R1_M_coord, Blk_N_coord};
            Catlass::MatrixCoord offsetCR2{R2_M_coord, Blk_N_coord};

            Catlass::MatrixCoord offsetABR1forFT{R1_M_coord, Blk_N_coord};
            Catlass::MatrixCoord offsetABR2forFT{R2_M_coord, Blk_N_coord};

            Catlass::MatrixCoord offsetCOMPYR1forFT{R1_M_coord, Blk_N_coord};
            Catlass::MatrixCoord offsetCOMPYR2forFT{R2_M_coord, Blk_N_coord};

            Catlass::MatrixCoord offsetCOMPXR1forFT{R1_M_coord, Blk_N_coord};
            Catlass::MatrixCoord offsetCOMPXR2forFT{R2_M_coord, Blk_N_coord};

            Catlass::MatrixCoord offsetThreforFT{R1_M_coord, Blk_N_coord};

            uint32_t COMPZRowOffset = blockCoord.m() * L1TileShapeforFT::M / L1TileShapeforFT::M;
            COMPZRowOffset = COMPZRowOffset * out_z_blk_align;

            Catlass::MatrixCoord offsetCOMPZforFT{R1_M_coord, COMPZRowOffset};

            int64_t gmOffsetCR1 = layoutCRforFT.GetOffset(offsetCR1);
            int64_t gmOffsetCR2 = layoutCRforFT.GetOffset(offsetCR2);

            int64_t gmOffsetCOMPABR1forFT = layoutABRforFT.GetOffset(offsetABR1forFT);
            int64_t gmOffsetCOMPABR2forFT = layoutABRforFT.GetOffset(offsetABR2forFT);

            int64_t gmOffsetCOMPCR1forFT = layoutCOMPYforFT.GetOffset(offsetCOMPYR1forFT);
            int64_t gmOffsetCOMPCR2forFT = layoutCOMPYforFT.GetOffset(offsetCOMPYR2forFT);

            int64_t gmOffsetThreforFT = layoutThreforFT.GetOffset(offsetThreforFT);
            int64_t gmOffsetCOMPZforFT = layoutCOMPZforFT.GetOffset(offsetCOMPZforFT);
            int64_t gmOffsetARed = Blk_N_coord;

            int64_t gmOffsetBRed = R1_M_coord;

            Catlass::layout::VectorLayout layoutVABR{n_actual};

            float n_ratio_factor = params.n_ratios[0];
            float n_sqrt_ratio_factor = params.n_sqrt_ratios[0];
            float n_square_ratio_factor = params.n_square_ratios[0];

            float n_ratio_factor_tail = params.n_ratios[0];
            float n_sqrt_ratio_factor_tail = params.n_sqrt_ratios[0];
            float n_square_ratio_factor_tail = params.n_square_ratios[0];

            if(R1_M_coord >=(params.SplitNnumChunk - 1)){
                n_ratio_factor = params.n_ratios[1];
                n_sqrt_ratio_factor = params.n_sqrt_ratios[1];
                n_square_ratio_factor = params.n_square_ratios[1];
            }

            if(blockCoord.n() == loopsMN.column() - 1){
                n_ratio_factor_tail = params.n_ratios[1];
                n_sqrt_ratio_factor_tail = params.n_sqrt_ratios[1];
                n_square_ratio_factor_tail = params.n_square_ratios[1];
            }

            /*
            CATLASS_DEVICE
            void operator()(
            AscendC::GlobalTensor<ElementA> const &gmCR1Tmp, 
            AscendC::GlobalTensor<ElementY> const &gmCR2Tmp,
            LayoutA const &layoutCRIn,
            AscendC::GlobalTensor<ElementY> const &gmAMax, 
            AscendC::GlobalTensor<ElementY> const &gmAMean,
            AscendC::GlobalTensor<ElementY> const &gmAMin,  
            LayoutVX const &layoutAforFT,
            AscendC::GlobalTensor<ElementY> const &gmCR1Out,
            AscnedC::GlobalTensor<ElementY> const &gmCR2Out, 
            AscendC::GlobalTensor<ElementY> const &gmABR1, 
            AscendC::GlobalTensor<ElementY> const &gmABR2, 
            LayoutMY const &layoutABRIn,
            AscendC::GlobalTensor<ElementY> const &gmBMeanAbs,
            AscendC::GlobalTensor<ElementY> const &gmBMeanSquare, 
            AscendC::GlobalTensor<ElementY> const &gmBVar,
            LayoutVX const &layoutBforFT,
            AscendC::GlobalTensor<ElementY> const &gmThreZ, 
            LayoutMY const &layoutThre,
            AscendC::GlobalTensor<ElementZ> const &gmCOMPZ, 
            LayoutMZ const &layoutZ,
            GemvCoord const &actualShape, 
            float n_ratio_factor, 
            float n_sqrt_ratio_factor, 
            float n_square_ratio_factor,
            float n_ratio_factor_tail, 
            float n_sqrt_ratio_factor_tail, 
            float n_square_ratio_factor_tail,
            float e_max,
            bool outputThre, uint32_t aiv_part_num, uint32_t CRSliceNum,
            Catlass::Arch::CrossCoreFlagWithReverse<> & flagAicFinishStore)
            */

            blockChunkVerify(
                gmCR1Slice[gmOffsetCR1],gmCR2Slice[gmOffsetCR2],
                layoutCRforFT,
                gmAMax[gmOffsetARed], gmAMean[gmOffsetARed], gmAMin[gmOffsetARed],
                layoutAforFT,
                gmCOMPYR1[gmOffsetCOMPCR1forFT],
                gmCOMPYR2[gmOffsetCOMPCR2forFT],
                gmCOMPXR1[gmOffsetCOMPABR1forFT],
                gmCOMPXR2[gmOffsetCOMPABR2forFT],
                layoutABRforFT,
                gmBMeanAbs[gmOffsetBRed],
                gmBMeanSquare[gmOffsetBRed],
                gmBVar[gmOffsetBRed],
                layoutBforFT,
                gmT[gmOffsetThreforFT],
                layoutThreforFT,
                gmCOMPZ[gmOffsetCOMPZforFT],
                layoutCOMPZforFT,
                actualBlockShape,
                n_ratio_factor, n_sqrt_ratio_factor, n_square_ratio_factor,
                n_ratio_factor_tail, n_sqrt_ratio_factor_tail, n_square_ratio_factor_tail,
                params.e_max, params.outputThre, aiv_part_num, CRSliceNum, flagAicFinishStore,
                params.ptrC  // pass C pointer to enable online error correction
            );
        }

        AscendC::PipeBarrier<PIPE_ALL>();

        // Online correction: check error flag from the last iteration.
        // Placed outside the loop to avoid affecting hot-path compilation.
        if (blockChunkVerify.error_detected_ && params.ptrC != nullptr) {
            constexpr float corr_lambda = 2.821f / FTChunkShape::N;
            constexpr float corr_half_N = FTChunkShape::N / 2.0f;
            constexpr int corr_chunk_N = FTChunkShape::N;
            uint32_t last_blk_n = (coreLoops > aicoreNum) ?
                ((coreLoops - 1 - aicoreIndex) / aicoreNum) * aicoreNum + aicoreIndex : aicoreIndex;
            Catlass::GemmCoord lastBlockCoord = matmulBlockSchedulerFirst.GetBlockCoord(last_blk_n);
            uint32_t last_Blk_N_coord = lastBlockCoord.m() * L1TileShapeforFT::M;
            blockChunkVerify.correctErrors(
                blockChunkVerify.last_stage_id_,
                params.ptrC,
                params.problemGemmShape.n(),
                lastBlockCoord.n() * M_R1_size_common,
                last_Blk_N_coord,
                corr_lambda, corr_half_N, corr_chunk_N);
        }
    }
    

    


    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params){
        
        CR_verify_op_fused(params,params.ptrCOMPZRow);
        Catlass::Arch::CrossCoreBarrierAIV<0x0, PIPE_V>();
        Catlass::Arch::CrossCoreBarrierAIV<0x0, PIPE_MTE3>();
        
        
    }


private:
    // ID used for inter-core synchronization
    static constexpr Catlass::Arch::FlagID FLAG_AIC_FINISH_STORE = 0;
    static constexpr Catlass::Arch::FlagID RV_FLAG_AIC_FINISH_STORE = 1;
    static constexpr Catlass::Arch::FlagID FLAG_AIV_FINISH_STORE = 2;
    static constexpr Catlass::Arch::FlagID RV_FLAG_AIV_FINISH_STORE = 3;
    Catlass::Arch::CrossCoreFlagWithReverse<> flagAicFinishStore{FLAG_AIC_FINISH_STORE,RV_FLAG_AIC_FINISH_STORE};
    Catlass::Arch::CrossCoreFlagWithReverse<> flagAicFinishStoreAIV{FLAG_AIV_FINISH_STORE,RV_FLAG_AIV_FINISH_STORE};
    Catlass::Arch::Resource<ArchTag> resource;
};

} // namespace CubeSelf::Gemm::Kernel

#endif