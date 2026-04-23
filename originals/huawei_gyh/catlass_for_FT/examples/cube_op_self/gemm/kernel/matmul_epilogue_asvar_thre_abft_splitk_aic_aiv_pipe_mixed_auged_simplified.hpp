#ifndef CATLASS_GEMM_KERNEL_MATMUL_BE_ABE_ON_AIC_ASVAR_THRESHOLD_ABFT_SPLITK_HPP_AUGED_SIMPLIFIED
#define CATLASS_GEMM_KERNEL_MATMUL_BE_ABE_ON_AIC_ASVAR_THRESHOLD_ABFT_SPLITK_HPP_AUGED_SIMPLIFIED

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

template<
    class ArchTag_,
    class ElementAccumulator_,
    class ElementOut_,
    uint32_t COMPUTE_LENGTH
>
struct ReduceAdd {
    using ArchTag = ArchTag_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementOut = ElementOut_;

    CATLASS_DEVICE
    ReduceAdd(Catlass::Arch::Resource<ArchTag> &resource)
    {
        int64_t bufferOffset = 0;
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            inputBuffer[i] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(bufferOffset);
            bufferOffset += COMPUTE_LENGTH * sizeof(ElementAccumulator);
            accumulatorBuffer[i] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(bufferOffset);
            bufferOffset += COMPUTE_LENGTH * sizeof(ElementAccumulator);
            outputBuffer[i] = resource.ubBuf.template GetBufferByByte<ElementOut>(bufferOffset);
            bufferOffset += COMPUTE_LENGTH * sizeof(ElementOut);
        }
    }

    CATLASS_DEVICE
    void Gm2Ub(AscendC::LocalTensor<ElementAccumulator> const &dst,
        AscendC::GlobalTensor<ElementAccumulator> const &src,
        uint32_t dataNum)
    {
        AscendC::DataCopyExtParams dataCopyParams(1, dataNum * sizeof(ElementAccumulator), 0, 0, 0);
        AscendC::DataCopyPadExtParams<ElementAccumulator> padParams(false, 0, 0, 0);
        AscendC::DataCopyPad(dst, src, dataCopyParams, padParams);
    }

    CATLASS_DEVICE
    void Ub2Gm(AscendC::GlobalTensor<ElementOut> const &dst,
        AscendC::LocalTensor<ElementOut> const &src,
        uint32_t dataNum)
    {
        AscendC::DataCopyExtParams dataCopyParams(1, dataNum * sizeof(ElementOut), 0, 0, 0);
        AscendC::DataCopyPad(dst, src, dataCopyParams);
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementOut> const &dst,
        AscendC::GlobalTensor<ElementAccumulator> const &src,
        uint64_t elementCount, uint32_t splitkFactor)
    {
        // The vec mte processes 256 bytes of data at a time.
        constexpr uint32_t ELE_PER_VECOTR_BLOCK = 256 / sizeof(ElementAccumulator);
        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint32_t aivId = AscendC::GetBlockIdx();
        uint64_t taskPerAiv =
            (elementCount / aivNum + ELE_PER_VECOTR_BLOCK - 1) / ELE_PER_VECOTR_BLOCK * ELE_PER_VECOTR_BLOCK;
        if (taskPerAiv == 0) taskPerAiv = ELE_PER_VECOTR_BLOCK;
        uint32_t tileLen;
        if (taskPerAiv > COMPUTE_LENGTH) {
            tileLen = COMPUTE_LENGTH;
        } else {
            tileLen = taskPerAiv;
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputEventIds[0]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputEventIds[1]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[0]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[1]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(accumulatorEventIds[0]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(accumulatorEventIds[1]);

        uint32_t loops = (elementCount + tileLen - 1) / tileLen;
        for (uint32_t loopIdx = aivId; loopIdx < loops; loopIdx += aivNum) {
            uint32_t actualTileLen = tileLen;
            if (loopIdx == loops - 1) {
                actualTileLen = elementCount - loopIdx * tileLen;
            }

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(accumulatorEventIds[bufferIndex]);
            Gm2Ub(accumulatorBuffer[bufferIndex], src[loopIdx * tileLen], actualTileLen);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(accumulatorEventIds[bufferIndex]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(accumulatorEventIds[bufferIndex]);

            for (uint32_t sliceIdx = 1; sliceIdx < splitkFactor; ++sliceIdx) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputEventIds[bufferIndex]);
                Gm2Ub(inputBuffer[bufferIndex],
                    src[sliceIdx * elementCount + loopIdx * tileLen], actualTileLen);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputEventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputEventIds[bufferIndex]);

                AscendC::Add(accumulatorBuffer[bufferIndex],
                    accumulatorBuffer[bufferIndex], inputBuffer[bufferIndex], actualTileLen);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputEventIds[bufferIndex]);
            }
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[bufferIndex]);
            if constexpr (!std::is_same_v<ElementAccumulator, ElementOut>) {
                if constexpr (std::is_same_v<ElementOut, half>) {
                    AscendC::Cast(outputBuffer[bufferIndex],
                        accumulatorBuffer[bufferIndex], AscendC::RoundMode::CAST_NONE, actualTileLen);
                } else {
                    AscendC::Cast(outputBuffer[bufferIndex],
                        accumulatorBuffer[bufferIndex], AscendC::RoundMode::CAST_RINT, actualTileLen);
                }
            } else {
                AscendC::DataCopy(outputBuffer[bufferIndex], accumulatorBuffer[bufferIndex], tileLen);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(accumulatorEventIds[bufferIndex]);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputEventIds[bufferIndex]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputEventIds[bufferIndex]);
            Ub2Gm(dst[loopIdx * tileLen], outputBuffer[bufferIndex], actualTileLen);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[bufferIndex]);

            bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
        }

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputEventIds[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputEventIds[1]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[1]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(accumulatorEventIds[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(accumulatorEventIds[1]);
    }

private:
    static const uint32_t BUFFER_NUM = 2;
    AscendC::LocalTensor<ElementAccumulator> inputBuffer[BUFFER_NUM];
    AscendC::LocalTensor<ElementAccumulator> accumulatorBuffer[BUFFER_NUM];
    AscendC::LocalTensor<ElementOut> outputBuffer[BUFFER_NUM];
    AscendC::TEventID inputEventIds[BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};
    AscendC::TEventID accumulatorEventIds[BUFFER_NUM] = {EVENT_ID2, EVENT_ID3};
    AscendC::TEventID outputEventIds[BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};
    uint32_t bufferIndex{ 0 };
    static_assert(BUFFER_NUM * COMPUTE_LENGTH * sizeof(ElementAccumulator) * 2
        +  BUFFER_NUM * COMPUTE_LENGTH * sizeof(ElementOut) <= ArchTag::UB_SIZE, "Excedding the UB space!");
};

template <
    class BlockMmadFirst_,
    class BlockMmad_,
    class BlockSchedulerFirst_,
    class BlockScheduler_,
    class BlockFTGemvAIC_,
    class BlockFTSum_,
    class BlockFTGemvAIV_,
    class BlockSliceRed_,
    class BlockSliceSum_,
    class BlockTranspose_,
    class ReduceAdd_
>
class MatmulAsVarABonAicSplitKAugedSimplified {
public:
    using BlockMmad = BlockMmad_;
    using BlockMmadFirst = BlockMmadFirst_;
    // using BlockGemv = BlockGemv_;
    using BlockSliceSum = BlockSliceSum_;
    // using BlockSumGemv = BlockSumGemv_;
    // using BlockThreCalc = BlockThreCalc_;

    using BlockFTSum = BlockFTSum_;
    using BlockFTGemvAIV = BlockFTGemvAIV_;
    using BlockSliceRed = BlockSliceRed_;

    using BlockFTGemvAIC = BlockFTGemvAIC_;

    using BlockTranspose = BlockTranspose_;

    using ReduceAdd = ReduceAdd_;

    // using BlockEpilogue = BlockEpilogue_;
    using FT_ENC_TYPE = Catlass::Gemv::helper::FT_ENC_TYPE;
    using FT_COMP_TYPE = Catlass::Gemv::helper::FT_COMP_TYPE;

    using FT_AIV_PIPE_FUSE_TYPE = Catlass::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE;
    using FT_THRESHOLD_ALGORITHM = Catlass::Gemv::helper::FT_THRESHOLD_ALGORITHM;

    using FT_RCE_THRE_TYPE = Catlass::Gemv::helper::FT_RCE_THRE_TYPE;
    using FT_AIC_BE_SCHEME = Catlass::Gemv::helper::FT_AIC_BE_SCHEME;
    
    static const FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE = BlockFTGemvAIV::FUSE_TYPE;
    static const FT_THRESHOLD_ALGORITHM ALGO_TYPE = FT_THRESHOLD_ALGORITHM::ASVAR;

    static const FT_AIC_BE_SCHEME BE_SCHEME = BlockFTGemvAIC::BE_SCHEME;

    static const uint32_t REMAIN_ALIGNED = 8;

    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using L0TileShape = typename BlockMmad::L0TileShape;

    using L1TileShapeFirst = typename BlockMmadFirst::L1TileShape;
    using L0TileShapeFirst = typename BlockMmadFirst::L0TileShape;

    using L1TileShapeforFT = typename BlockMmadFirst::L1TileShapeforFT;
    using L0TileShapeforFT = typename BlockMmadFirst::L0TileShapeforFT;

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
    
    using ElementXforFT = typename BlockMmadFirst::ElementX;
    using LayoutXforFT = typename BlockMmadFirst::LayoutX;
    using LayoutXforFTCol = typename BlockMmadFirst::LayoutXCol;

    // using LayoutVXforFT = typename BlockMmadFirst::LayoutVX;
    using LayoutVXforFT = Catlass::layout::VectorLayout;
    // using LayoutVXforFTCol = typename BlockMmadFirst::LayoutVXCol;
    using LayoutVXforFTCol = Catlass::layout::ColumnMajor;

    using ElementYforFT = typename BlockMmadFirst::ElementY;
    using LayoutYforFT = typename BlockMmadFirst::LayoutY;

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

    using ElementYforBEAIV = typename std::conditional<
        BE_SCHEME == FT_AIC_BE_SCHEME::ROWCOMPLETE_BF,
        ElementYforB,
        ElementXforFT>::type;

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

    using UBTileShapeFTTrans = typename BlockTranspose::UBTileShape;
    using UBBlockShapeFTTrans = typename BlockTranspose::UBBlockShape;

    using UBTileTailShapeFTTrans = typename BlockTranspose::UBTileTailShape;
    using UBBlockTailShapeFTTrans = typename BlockTranspose::UBBlockTailShape;

    using UBTileShapeARed = typename BlockFTSum::UBTileShapeforA;

    using L1TileShapeBE = typename BlockFTGemvAIC::L1TileShape;
    using L0TileShapeBE = typename BlockFTGemvAIC::L0TileShape;
    using UBBlockShapeBE = typename BlockFTGemvAIC::UBBlockShape;

    using UBTileShapeCE = typename BlockFTGemvAIV::UBTileShape;
    using UBBlockShapeCE = typename BlockFTGemvAIV::UBBlockShape;

    using UBTileShapeBReduce = typename BlockSliceRed::UBTileShape;
    using SliceSumUBTileShape = typename BlockSliceSum::UBTileShape;

    using UBAlignHelper = Catlass::Gemv::helper::UBAlignHelper<ElementA>;
    using UBTransposeAlignHelper = Catlass::Gemv::helper::UBTransposeAlignHelper<ElementYforFT>;
    using L1XAlignHelper = Catlass::Gemv::helper::L1AlignHelper<ElementXforFT, LayoutVXforFT>;

    // using COMPUBTileShape = typename BlockThreCalc::UBTileShapeTotal;

    // using ThreCalcUBBlockShape = typename BlockThreCalc::UBTileShapeTotal;
    // using ThreCalcUBTileShape = typename BlockThreCalc::UBTileShape;

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
        GM_ADDR ptrBEforAIV;
        GM_ADDR ptrFTTmpSpace;
        GM_ADDR ptrBMaxSlice;
        GM_ADDR ptrBMinSlice;
        GM_ADDR ptrBMeanAbs;
        GM_ADDR ptrBMeanSquare;
        GM_ADDR ptrBVar;
        GM_ADDR ptrVXforA;
        GM_ADDR ptrAMean;
        GM_ADDR ptrAMax;
        GM_ADDR ptrAMin;
        GM_ADDR ptrThreZ;
        LayoutCOMPX layoutThre;
        ElementZ rounding_alpha;
        float e_max;
        float n_scale_ratios[2];
        float n_ratios[2];
        float n_sqrt_ratios[2];
        float n_square_ratios[2];
        float simple_std_est_ratios[2];
        float simple_std_est_A_row_ratios[3];
        uint32_t SplitNnum;
        uint32_t SplitReduceM;
        uint32_t SplitReduceN;
        bool outputThre;
        bool outputCE;
        uint32_t remainMSize;
        uint32_t remainNum;
        uint32_t firstNnum;
        uint32_t SplitKNum = 1;
        uint32_t SplitKNumLimit = 8;
        uint32_t HeadKSliceNum = 0;
        uint32_t actualKSliceSize[3];
        // GM_ADDR ptrWorkspace;
        // EpilogueParams epilogueParams;
        // GM_ADDR ptrC;
        /*
        float std_est_A_row_ratio;
        float A_row_scale_ratio;
        float std_est_ratios[2];
        float kn_ratios[2];
        float kn_scale_ratios[2];
        float kn_sqrt_ratios[2];
        float k_sqrt_n_ratios[2];
        */

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
            GM_ADDR ptrBEforAIV_,
            GM_ADDR ptrFTTmpSpace_,
            GM_ADDR ptrBMaxSlice_,
            GM_ADDR ptrBMinSlice_,
            GM_ADDR ptrBMeanAbs_,
            GM_ADDR ptrBMeanSquare_,
            GM_ADDR ptrBVar_,
            GM_ADDR ptrVXforA_,
            GM_ADDR ptrAMean_, GM_ADDR ptrAMax_, GM_ADDR ptrAMin_, 
            GM_ADDR ptrThreZ_, LayoutCOMPX layoutThre_, ElementZ rounding_alpha_,
            float e_max_,
            const float (&n_scale_ratios_)[2],
            const float (&n_ratios_)[2],
            const float (&n_sqrt_ratios_)[2],
            const float (&n_square_ratios_)[2],
            const float (&simple_std_est_ratios_)[2],
            const float (&simple_std_est_A_row_ratios_)[3],
            uint32_t SplitNnum_, 
            uint32_t SplitReduceM_, uint32_t SplitReduceN_,
            bool outputThre_, bool outputCE_, 
            uint32_t remainMSize_, uint32_t remainNum_, uint32_t firstNnum_,
            uint32_t SplitKNum_,
            uint32_t SplitKNumLimit_, uint32_t HeadKSliceNum_,
            const uint32_t (&actualKSliceSize_)[3]
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
            ptrBE(ptrBE_), ptrBEforAIV(ptrBEforAIV_), ptrFTTmpSpace(ptrFTTmpSpace_),
            ptrBMaxSlice(ptrBMaxSlice_), ptrBMinSlice(ptrBMinSlice_), 
            ptrBMeanAbs(ptrBMeanAbs_), ptrBMeanSquare(ptrBMeanSquare_), ptrBVar(ptrBVar_),
            ptrVXforA(ptrVXforA_),
            ptrAMean(ptrAMean_), ptrAMax(ptrAMax_), ptrAMin(ptrAMin_),
            ptrThreZ(ptrThreZ_), layoutThre(layoutThre_), 
            rounding_alpha(rounding_alpha_), 
            e_max(e_max_), 
            SplitNnum(SplitNnum_), 
            SplitReduceM(SplitReduceM_), SplitReduceN(SplitReduceN_),
            outputThre(outputThre_), outputCE(outputCE_), 
            remainMSize(remainMSize_), remainNum(remainNum_), firstNnum(firstNnum_),
            SplitKNum(SplitKNum_), SplitKNumLimit(SplitKNumLimit_), HeadKSliceNum(HeadKSliceNum_)
            {
                for (int i = 0; i < 2; ++i) {
                    /*
                    const float (&kn_ratios_)[2],
                    const float (&kn_scale_ratios_)[2],
                    const float (&kn_sqrt_ratios_)[2],
                    const float (&k_sqrt_n_ratios_)[2],
                    const float (&std_est_ratios_)[2],
                    */
                    this->n_scale_ratios[i] = n_scale_ratios_[i];
                    this->n_ratios[i] = n_ratios_[i];
                    this->n_sqrt_ratios[i] = n_sqrt_ratios_[i];
                    this->n_square_ratios[i] = n_square_ratios_[i];
                    this->simple_std_est_ratios[i] = simple_std_est_ratios_[i];
                }

                for (int i = 0; i < 3; ++i) {
                    this->actualKSliceSize[i] = actualKSliceSize_[i];
                    this->simple_std_est_A_row_ratios[i] = simple_std_est_A_row_ratios_[i];
                }
            } 
    };
    /*
    simple_std_est_A_row_ratio(simple_std_est_A_row_ratio_),
    float std_est_A_row_ratio_, float A_row_scale_ratio_, 
    */

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
        GM_ADDR ptrBEforAIV;
        GM_ADDR ptrFTTmpSpace;
        GM_ADDR ptrBMaxSlice;
        GM_ADDR ptrBMinSlice;
        GM_ADDR ptrBMeanAbs;
        GM_ADDR ptrBMeanSquare;
        GM_ADDR ptrBVar;
        GM_ADDR ptrVXforA;
        GM_ADDR ptrAMean;
        GM_ADDR ptrAMax;
        GM_ADDR ptrAMin;
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
        uint32_t SliceKUnit;
        uint32_t SplitKNumLimit;
        int32_t useLogRatio;
    };

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
        return sizeof(ElementC) * args.problemGemmShape.m() * args.problemGemmShape.n() *
            GetSplitkFactorForABFT(args.problemGemmShape.k(), args.SliceKUnit, args.SplitKNumLimit);
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        Catlass::GemmCoord problemGemmShape = args.problemGemmShape;
        Catlass::GemvCoord problemShape = args.problemShape;

        uint32_t SplitNnum = ((args.problemGemmShape.n() + L1TileShape::N - 1) / L1TileShape::N);

        uint32_t FirstBlockNum = ((SplitNnum + L1TileShapeforFT::N - 1) / L1TileShapeforFT::N);
        uint32_t FirstBlockN = FirstBlockNum * L1TileShapeFirst::N;
        uint32_t FirstXStep = L1TileShapeFirst::N;

        uint32_t m = problemShape.m();
        uint32_t n = problemShape.n();

        uint32_t m2 = problemGemmShape.m();
        uint32_t n2 = problemGemmShape.n();
        uint32_t k2 = problemGemmShape.k();

        if(FirstBlockN > n2){
            FirstBlockN = n2;
        }

        uint32_t RemainBlockN = n2 - FirstBlockN;
        uint32_t RemainBlockNum = ((RemainBlockN + L1TileShape::N - 1) / L1TileShape::N);
        uint32_t RemainCENumRaw = RemainBlockNum / FirstBlockNum;
        uint32_t firstNnum = FirstBlockNum;
        SplitNnum = RemainBlockNum + FirstBlockNum;
        if(RemainCENumRaw < 1){
            RemainCENumRaw = 1;
        }

        uint32_t RemainMSize = (UBTileShapeCE::M + RemainCENumRaw - 1) / RemainCENumRaw;
        RemainMSize = ((RemainMSize + REMAIN_ALIGNED - 1) / REMAIN_ALIGNED) * REMAIN_ALIGNED;
        uint32_t RemainCENum = (UBTileShapeCE::M + RemainMSize - 1) / RemainMSize;

        uint32_t SplitKNum = GetSplitkFactorForABFT(args.problemGemmShape.k(), args.SliceKUnit, args.SplitKNumLimit);
        uint32_t SplitKNumLimit = args.SplitKNumLimit;

        uint32_t actualKSliceSize[3];

        uint32_t TileKNum = (args.problemGemmShape.k() + L1TileShape::K - 1) / L1TileShape::K;
        uint32_t BasicSliceTileKNum = (TileKNum / SplitKNum);

        if(TileKNum % SplitKNum == 0){
            // (TileKNum / SplitKNum)
            actualKSliceSize[0] =  BasicSliceTileKNum * L1TileShape::K;
            // (TileKNum / SplitKNum)
            actualKSliceSize[1] = BasicSliceTileKNum * L1TileShape::K;
        }else{
            // (TileKNum / SplitKNum)
            actualKSliceSize[0] = (BasicSliceTileKNum + 1) * L1TileShape::K;
            // (TileKNum / SplitKNum)
            actualKSliceSize[1] = BasicSliceTileKNum * L1TileShape::K;
        }
        // (TileKNum / SplitKNum)
        actualKSliceSize[2] = args.problemGemmShape.k() - (((TileKNum % SplitKNum) + BasicSliceTileKNum * (SplitKNum - 1)) * L1TileShape::K);
        
        uint32_t HeadKSliceNum = (TileKNum % SplitKNum);

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

        uint32_t longest_k_slice_size = actualKSliceSize[0];

        if(SplitKNum < 2){
            longest_k_slice_size = k2;
        }else if(TileKNum < 2){
            longest_k_slice_size = k2;
        }

        uint32_t SplitReduceN_num = (longest_k_slice_size + UBTileShapeBReduce::N - 1) / UBTileShapeBReduce::N;
        uint32_t SplitReduceN = UBTileShapeBReduce::N;
        if(SplitReduceN_num < 2){
            SplitReduceN = (longest_k_slice_size + 2 - 1) / 2;
        }
        // if(args.rce_thre_type == FT_RCE_THRE_TYPE::ROUND_WITH_ACC){
        //     float acc_rounding_error = std::pow(2.0f, -23.0f);
        //     float acc_scaling_factor = 1.0f * slice_N*(slice_N+1)*(2*slice_N+1) / 48.0f;
        //     acc_scaling_factor = std::sqrt(acc_scaling_factor);
        //     rounding_alpha = static_cast<ElementZ>(row_sqrt * rounding_error + acc_rounding_error * acc_scaling_factor); 
        // }

        // float std_est_ratios[2];
        // float kn_ratios[2];
        // float kn_scale_ratios[2];
        // float kn_sqrt_ratios[2];
        // float k_sqrt_n_ratios[2];
        float n_scale_ratios[2];
        float n_ratios[2];
        float n_sqrt_ratios[2];
        float n_square_ratios[2];
        float simple_std_est_ratios[2];
        float simple_std_est_A_row_ratios[3];

        uint32_t n_remain_split = (args.problemGemmShape.n() % L1TileShape::N);

        float simple_common_std_factor = std::sqrt(2.0f * logf(L1TileShape::N * 1.0f));
        float simple_remain_std_factor = simple_common_std_factor;

        if(args.useLogRatio < 1){
            simple_common_std_factor = 1.0f;
            simple_remain_std_factor = 1.0f;
        }


        float simple_std_est_ratio_common = (1.0f / simple_common_std_factor);
        float simple_std_est_ratio_remain = simple_std_est_ratio_common;

        if(args.useLogRatio < 1){
            float simple_std_est_A_row_factor = 1.0f;
            float simple_std_est_A_row_ratio = 1.0f / simple_std_est_A_row_factor;

            for(int i = 0; i < 3; ++i){
                simple_std_est_A_row_ratios[i] = simple_std_est_A_row_ratio;
            }

        }else{
            for(int i = 0; i < 3; ++i){
                uint32_t k_slice_scale = actualKSliceSize[i];

                float k_slice_scale_factor = k_slice_scale * 1.0f;

                if(k_slice_scale < 1){
                    k_slice_scale_factor = (k2 * 1.0f);
                }
                // (k2 * 1.0f)
                simple_std_est_A_row_ratios[i] = (1.0f / std::sqrt(2.0f * logf(k_slice_scale_factor)));
            }
        }

        if(n_remain_split > 0){
            
            if(args.useLogRatio < 1){
                simple_remain_std_factor = 1.0f;
            }else{
                simple_remain_std_factor = std::sqrt(2.0f * logf((n_remain_split*1.0f)));
            }
             
            simple_std_est_ratio_remain = (1.0f / simple_remain_std_factor);
        }

        simple_std_est_ratios[0] = simple_std_est_ratio_common;
        simple_std_est_ratios[1] = simple_std_est_ratio_remain;

        n_scale_ratios[0] = (1.0f / (L1TileShape::N * 1.0f));
        n_ratios[0] = (L1TileShape::N * 1.0f);
        n_sqrt_ratios[0] = std::sqrt((L1TileShape::N * 1.0f));
        n_square_ratios[0] = (L1TileShape::N * L1TileShape::N * 1.0f);
        
        if(n_remain_split > 0){
            n_scale_ratios[1] = (1.0f / (n_remain_split * 1.0f));
            n_ratios[1] = (n_remain_split * 1.0f);
            n_sqrt_ratios[1] = std::sqrt((n_remain_split * 1.0f));
            n_square_ratios[1] = (n_remain_split * n_remain_split * 1.0f);
        }else{
            n_scale_ratios[1] = (1.0f / (L1TileShape::N * 1.0f));
            n_ratios[1] = (L1TileShape::N * 1.0f);
            n_sqrt_ratios[1] = std::sqrt(L1TileShape::N * 1.0f);
            n_square_ratios[1] = (L1TileShape::N * L1TileShape::N * 1.0f);
        }

        // printf("SplitReduceM: %d\n", SplitReduceM);
        // printf("SplitReduceN: %d\n", SplitReduceN);
        
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
            args.ptrBE,
            args.ptrBEforAIV, 
            args.ptrFTTmpSpace,
            args.ptrBMaxSlice,
            args.ptrBMinSlice,
            args.ptrBMeanAbs,
            args.ptrBMeanSquare,
            args.ptrBVar,
            args.ptrVXforA,
            args.ptrAMean, args.ptrAMax, args.ptrAMin,
            args.ptrThreZ, layoutThre, rounding_alpha, 
            e_max,
            n_scale_ratios, n_ratios, n_sqrt_ratios, n_square_ratios, 
            simple_std_est_ratios,
            simple_std_est_A_row_ratios,
            SplitNnum, SplitReduceM, SplitReduceN,
            args.outputThre, args.outputCE, 
            RemainMSize, RemainCENum, firstNnum, SplitKNum, 
            SplitKNumLimit, HeadKSliceNum, actualKSliceSize
        };

        /*
        float e_max_,
        const float (&n_scale_ratios_)[2],
        const float (&n_ratios_)[2],
        const float (&n_sqrt_ratios_)[2],
        const float (&n_square_ratios_)[2],
        const float (&simple_std_est_ratios_)[2],
        const float (&simple_std_est_A_row_ratios_)[3],
        uint32_t SplitNnum_, 
        uint32_t SplitReduceM_, uint32_t SplitReduceN_,
        bool outputThre_, bool outputCE_, 
        uint32_t remainMSize_, uint32_t remainNum_, uint32_t firstNnum_,
        uint32_t SplitKNum_,
        uint32_t SplitKNumLimit_, uint32_t HeadKSliceNum_,
        const uint32_t (&actualKSliceSize_)[3]
        */

        /*
        std_est_A_row_ratio, A_row_scale_ratio,
        std_est_ratios, kn_ratios, kn_scale_ratios,
        kn_sqrt_ratios, k_sqrt_n_ratios, 
        */
        
        /*
        uint32_t SplitNnum_, 
            uint32_t SplitReduceM_, uint32_t SplitReduceN_,
            bool outputThre_, bool outputCE_, uint32_t remainMSize_, uint32_t remainNum_
        */

        // printf("kn_scale_ratio: %f\n",params.kn_scale_ratios[0]);

        return params;
    }

    // Methods
    CATLASS_DEVICE
    MatmulAsVarABonAicSplitKAugedSimplified() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    CATLASS_DEVICE
    void BE_split_op_on_AIC_BF(Params const &params)
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

        AscendC::GlobalTensor<ElementXforFT> gmXV;
        gmXV.SetGlobalBuffer((__gm__ ElementXforFT *)params.ptrXV);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementXforFT> gmY;
        gmY.SetGlobalBuffer((__gm__ ElementXforFT *)params.ptrBE);
        AscendC::GlobalTensor<ElementZforBRed> gmYforAIV;
        gmYforAIV.SetGlobalBuffer((__gm__ ElementZforBRed *)params.ptrBEforAIV);

        LayoutYforFT layoutYforBE{params.SplitNnum, params.problemGemmShape.k()};
        LayoutX layoutXV{params.problemGemmShape.n()};

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

            blockFTGemvAIC.op_with_addition_copy(
                gmXV[gmOffsetX], layoutXV,
                gmB[gmOffsetB], params.layoutB,
                gmY[gmOffsetY], gmYforAIV[gmOffsetY],
                layoutYforBE,
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

        AscendC::GlobalTensor<ElementXforFT> gmXV;
        gmXV.SetGlobalBuffer((__gm__ ElementXforFT *)params.ptrXV);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementXforFT> gmY;
        gmY.SetGlobalBuffer((__gm__ ElementXforFT *)params.ptrBE);

        LayoutYforFT layoutYforBE{params.SplitNnum, params.problemGemmShape.k()};
        LayoutX layoutXV{params.problemGemmShape.n()};

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
    void Matmul_op_ABe_fused(Params const &params){

        BlockSchedulerFirst matmulBlockSchedulerFirst(params.problemGemmShapeFirst,
            Catlass::GemmCoord(L1TileShapeFirst::M, L1TileShapeFirst::N, L1TileShapeFirst::K), 
            params.SplitKNum);

        uint32_t coreLoopsTotal = matmulBlockSchedulerFirst.GetCoreLoops();

        BlockMmadFirst blockMmadFirst(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);

        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);

        AscendC::GlobalTensor<ElementC> gmC;
        if(params.SplitKNum > 1){
            gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrWorkspace);
        }else{
            gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);
        }

        AscendC::GlobalTensor<ElementXforFT> gmX;
        gmX.SetGlobalBuffer((__gm__ ElementXforFT *)params.ptrBE);

        AscendC::GlobalTensor<ElementYforFT> gmFTTmpSpace;
        gmFTTmpSpace.SetGlobalBuffer((__gm__ ElementYforFT *)params.ptrFTTmpSpace);

        // AscendC::GlobalTensor<ElementYforFT> gmY;
        // gmY.SetGlobalBuffer((__gm__ ElementYforFT *)params.ptrZRow2);

        int64_t OffsetAeSliceInit = 0;

        uint32_t VX_ALIGNED = L1XAlignHelper::M_ALIGNED;
        uint32_t X_ALIGNED = L1XAlignHelper::M_ALIGNED;

        Catlass::layout::RowMajor layoutC(params.problemGemmShape.m(), params.problemGemmShape.n());

        LayoutXforFT layoutXforFTBlock{params.SplitNnum, params.problemGemmShape.k()};
       
        LayoutYforFT layoutXforFT{params.SplitNnum, params.problemGemmShape.k()};

        LayoutXforFTCol layoutXforFTBlockCol{params.problemGemmShape.k(), params.SplitNnum};
        LayoutXforFTCol layoutXforFTTotalCol{params.problemGemmShape.k(), params.SplitNnum};

        LayoutX layoutVXforFTBlock{params.problemGemmShape.k()};
        LayoutX layoutVXforFT{params.problemGemmShape.k()};

        LayoutYforFT layoutYforFTBlock{params.SplitNnum, params.problemGemmShape.m()};
        LayoutYforFT layoutYforFTTotal{params.SplitNnum, params.problemGemmShape.m()};

        LayoutYforFT layoutYforFTBlockCol{params.problemGemmShape.m(), params.SplitNnum};
        LayoutYforFT layoutYforFTTotalCol{params.problemGemmShape.m(), params.SplitNnum};

        // LayoutYforFT layoutYforFTBlockCol{params.problemGemmShape.m(), params.SplitNnum + VX_ALIGNED};
        // LayoutYforFT layoutYforFTTotalCol{params.problemGemmShape.m(), params.SplitNnum + VX_ALIGNED};

        LayoutYforFT layoutVYforFTBlock{1, params.problemGemmShape.m()};
        LayoutYforFT layoutVYforFT{1, params.problemGemmShape.m()};

        uint64_t MatrixElementSize = static_cast<uint64_t>(params.problemGemmShape.m()) * static_cast<uint64_t>(params.problemGemmShape.n());
        uint64_t base_slice_offset = static_cast<uint64_t>(params.problemGemmShape.m()) * static_cast<uint64_t>(params.SplitNnum);

        // LayoutX layoutX{params.problemGemmShape.n()};
        Catlass::GemmCoord loopsMNK = matmulBlockSchedulerFirst.loopsMNK;
        
        for (uint32_t loopIdxTotal = AscendC::GetBlockIdx(); loopIdxTotal < coreLoopsTotal; loopIdxTotal += AscendC::GetBlockNum()) {
            // Compute block location
            uint32_t KSliceIdx = matmulBlockSchedulerFirst.GetSplitkSliceIdx(loopIdxTotal);
            
            // uint32_t KBlockSizeActual = params.actualKSliceSize[0];
            // if(KSliceIdx >= params.HeadKSliceNum){
            //     KBlockSizeActual = (KSliceIdx == params.SplitKNum -1) ? params.actualKSliceSize[2] : params.actualKSliceSize[1];
            // }
            Catlass::GemmCoord blockCoord = matmulBlockSchedulerFirst.GetBlockCoord(loopIdxTotal);
            Catlass::GemmCoord actualBlockShape = matmulBlockSchedulerFirst.GetActualBlockShape(
                blockCoord, KSliceIdx);

            uint32_t splitNIdx = blockCoord.n() * L1TileShapeFirst::N / L1TileShapeFirst::N;
            // Compute initial location in logical coordinates
            Catlass::MatrixCoord offsetA{blockCoord.m() * L1TileShapeFirst::M, blockCoord.k() * L1TileShapeFirst::K};
            Catlass::MatrixCoord offsetB{blockCoord.k() * L1TileShapeFirst::K, blockCoord.n() * L1TileShapeFirst::N};
            Catlass::MatrixCoord offsetC{blockCoord.m() * L1TileShapeFirst::M, blockCoord.n() * L1TileShapeFirst::N};

            Catlass::MatrixCoord offsetXforFT{splitNIdx * L1TileShapeforFT::N, blockCoord.k() * L1TileShapeFirst::K};
            Catlass::MatrixCoord offsetYforFT{splitNIdx * L1TileShapeforFT::N, blockCoord.m() * L1TileShapeFirst::M};
            
            Catlass::MatrixCoord offsetXforFTCol{blockCoord.k() * L1TileShapeFirst::K, splitNIdx * L1TileShapeforFT::N};
            Catlass::MatrixCoord offsetYforFTCol{blockCoord.m() * L1TileShapeFirst::M, splitNIdx * L1TileShapeforFT::N};
            
            uint64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
            uint64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
            uint64_t gmOffsetC = layoutC.GetOffset(offsetC)
                + MatrixElementSize * static_cast<uint64_t>(KSliceIdx);

            uint64_t gmOffsetXforFT = layoutXforFT.GetOffset(offsetXforFT);
            uint64_t gmOffsetXforFTCol = layoutXforFTTotalCol.GetOffset(offsetXforFTCol);
            // blockCoord.n() * L1TileShape::N;
            // params.problemGemmShape.m() +
            uint64_t gmOffsetYforFTinBlock = layoutYforFTTotal.GetOffset(offsetYforFT);
            uint64_t gmOffsetYforFT = gmOffsetYforFTinBlock + static_cast<uint64_t>(KSliceIdx) * base_slice_offset;

            uint64_t gmOffsetYforFTColinBlock = layoutYforFTTotalCol.GetOffset(offsetYforFTCol);
            uint64_t gmOffsetYforFTCol = gmOffsetYforFTColinBlock + static_cast<uint64_t>(KSliceIdx) * base_slice_offset;


            uint64_t gmOffsetVXforFT = 0;
            uint64_t gmOffsetVYforFTinBlock = blockCoord.m() * L1TileShapeFirst::M;
            uint64_t gmOffsetVYforFT = OffsetAeSliceInit + gmOffsetVYforFTinBlock;

            // Compute block-scoped matrix multiply-add
            /*
            /// Perform a block-scoped matrix multiply-accumulate
            CATLASS_DEVICE
            void operator()(
                AscendC::GlobalTensor<ElementA> const & gmA, LayoutA const &layoutA,
                AscendC::GlobalTensor<ElementB> const & gmB, LayoutB const &layoutB,
                AscendC::GlobalTensor<ElementC> const & gmC, LayoutC const &layoutC,
                AscendC::GlobalTensor<ElementX> const & gmX, LayoutX const &layoutX,
                AscendC::GlobalTensor<ElementY> const & gmY, LayoutY const &layoutY,
                Catlass::GemmCoord const &actualShape, Catlass::GemvCoord &actualShapeforX)
            */
            uint32_t kActualforX = actualBlockShape.k();

            // uint32_t KBlockSizeActual = params.actualKSliceSize[0];
            // if(KSliceIdx >= params.HeadKSliceNum){
            //     KBlockSizeActual = (KSliceIdx == params.SplitKNum -1) ? params.actualKSliceSize[2] : params.actualKSliceSize[1];
            // }

            uint32_t nActualforX = (blockCoord.n() == (loopsMNK.n() - 1)) ? (params.SplitNnum - splitNIdx * L1TileShapeforFT::N) : L1TileShapeforFT::N;
            
            Catlass::GemvCoord actualBlockShapeforX = Catlass::GemvCoord{nActualforX, kActualforX};
            Catlass::GemvCoord actualBlockShapeforXCol = Catlass::GemvCoord{kActualforX, nActualforX};

            /*
            CATLASS_DEVICE
            void operator()(
                AscendC::GlobalTensor<ElementA> const & gmA, LayoutA const &layoutA,
                AscendC::GlobalTensor<ElementB> const & gmB, LayoutB const &layoutB,
                AscendC::GlobalTensor<ElementC> const & gmC, LayoutC const &layoutC,
                AscendC::GlobalTensor<ElementX> const & gmX, LayoutXCol const &layoutXCol,
                AscendC::GlobalTensor<ElementY> const & gmY, LayoutY const &layoutY,
                Catlass::GemmCoord const &actualShape, 
                Catlass::GemvCoord &actualShapeforX)
            */
            blockMmadFirst(
                gmA[gmOffsetA], params.layoutA,
                gmB[gmOffsetB], params.layoutB,
                gmC[gmOffsetC], layoutC,
                gmX[gmOffsetXforFTCol], layoutXforFTBlockCol,
                gmFTTmpSpace[gmOffsetYforFTCol], layoutYforFTBlockCol,
                actualBlockShape, actualBlockShapeforXCol);
        }
    }

    CATLASS_DEVICE
    void Matmul_op(Params const &params)
    {
        // BlockSchedulerFirst matmulBlockSchedulerFirst(params.problemGemmShapeFirst, Catlass::MakeCoord(L1TileShapeFirst::M,L1TileShapeFirst::N));

        BlockScheduler matmulBlockScheduler(params.problemGemmShapeRemain, 
            Catlass::GemmCoord(L1TileShape::M, L1TileShape::N, L1TileShape::K), 
            params.SplitKNum);

        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);

        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);

        AscendC::GlobalTensor<ElementC> gmC;
        if(params.SplitKNum > 1){
            gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrWorkspace);
        }else{
            gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);
        }

        Catlass::layout::RowMajor layoutC(params.problemGemmShape.m(), params.problemGemmShape.n());

        uint64_t MatrixElementSize = static_cast<uint64_t>(params.problemGemmShape.m()) * static_cast<uint64_t>(params.problemGemmShape.n());
        // 共24个核，以其核的编号作为起始loop循环的位置，每次处理的循环编号为间隔核的数量
        // 此处,对于AIV 而言，GetBlockIdx()获取的是其 AIV core 的 Block ID，对于AIC而言，获取的是AIC core 的BLOCK ID
        // GetBlockNum(): 获取的是AI Core的数量，或者说是AIC 与 AIV 组合的数量，一个AIC 对应多个AIV， 往往获得的值等于使用的AIC的数量
        
        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            // Compute block location

            uint32_t KSliceIdx = matmulBlockScheduler.GetSplitkSliceIdx(loopIdx);
            Catlass::GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            Catlass::GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(
                blockCoord, KSliceIdx);

            // Compute initial location in logical coordinates
            Catlass::MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
            Catlass::MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, params.problemGemmShapeFirst.n() + blockCoord.n() * L1TileShape::N};
            Catlass::MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, params.problemGemmShapeFirst.n() + blockCoord.n() * L1TileShape::N};

            uint64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
            uint64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
            uint64_t gmOffsetC = layoutC.GetOffset(offsetC)
                + MatrixElementSize * static_cast<uint64_t>(KSliceIdx);

            bool isFirstBlock = (loopIdx == AscendC::GetBlockIdx());
            // 
            bool hasNextBlock = false;
            uint64_t gmOffsetNextA = gmOffsetA;
            uint64_t gmOffsetNextB = gmOffsetB;
            uint64_t gmOffsetNextC = gmOffsetC;
            Catlass::GemmCoord nextActualBlockShape = Catlass::GemmCoord{128, 256, L1TileShape::K};
            uint32_t loopIdxNext = loopIdx + AscendC::GetBlockNum();

            if(loopIdxNext < coreLoops){
                hasNextBlock = true;
                uint32_t KSliceIdxNext = matmulBlockScheduler.GetSplitkSliceIdx(loopIdxNext);
                Catlass::GemmCoord blockCoordNext = matmulBlockScheduler.GetBlockCoord(loopIdxNext);
                Catlass::GemmCoord nextActualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoordNext, KSliceIdxNext);

                // Compute initial location in logical coordinates
                Catlass::MatrixCoord offsetNextA{blockCoordNext.m() * L1TileShape::M, blockCoordNext.k() * L1TileShape::K};
                Catlass::MatrixCoord offsetNextB{blockCoordNext.k() * L1TileShape::K, params.problemGemmShapeFirst.n() + blockCoordNext.n() * L1TileShape::N};
                Catlass::MatrixCoord OffsetNextC{blockCoordNext.m() * L1TileShape::M, params.problemGemmShapeFirst.n() + blockCoordNext.n() * L1TileShape::N};

                gmOffsetNextA = params.layoutA.GetOffset(offsetNextA);
                gmOffsetNextB = params.layoutB.GetOffset(offsetNextB);
                gmOffsetNextC = layoutC.GetOffset(OffsetNextC) + MatrixElementSize * static_cast<uint64_t>(KSliceIdxNext);
            }

            // Compute block-scoped matrix multiply-add
            // blockMmad(
            //     gmA[gmOffsetA], params.layoutA,
            //     gmB[gmOffsetB], params.layoutB,
            //     gmC[gmOffsetC], layoutC,
            //     actualBlockShape);

            blockMmad(gmA[gmOffsetA], gmA[gmOffsetNextA], params.layoutA,
                gmB[gmOffsetB], gmB[gmOffsetNextB], params.layoutB,
                gmC[gmOffsetC], layoutC, 
                actualBlockShape, nextActualBlockShape, isFirstBlock, hasNextBlock);
            
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
        Matmul_op_ABe_fused(params);
        Catlass::Arch::CrossCoreBarrierAIC<0x0, PIPE_FIX>();
        // Catlass::Arch::CrossCoreBarrierAIC<0x0, PIPE_M>();
        // AscendC::SyncAll<true>();
        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
        if(params.problemGemmShapeRemain.n() > 0){
            Matmul_op(params);   
        }

        if(params.SplitKNum > 1){
            Catlass::Arch::CrossCoreBarrierAIC<0x0, PIPE_FIX>();
        }

        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
        
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_DEVICE
    void ABFT_Transpose(Params const &params)
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
        // bool isFirstBlock = (loopId == AscendC::GetBlockIdx());
        uint32_t aivIndex = AscendC::GetBlockIdx();
        uint32_t aicoreIndex = aivIndex / AscendC::GetSubBlockNum();
        // uint32_t aicoreIndex = aivIndex / AscendC::GetTaskRation();

        AscendC::GlobalTensor<ElementYforFT> gmFTTmpSpace;
        gmFTTmpSpace.SetGlobalBuffer((__gm__ ElementYforFT *)params.ptrFTTmpSpace);

        AscendC::GlobalTensor<ElementYforFT> gmY;
        gmY.SetGlobalBuffer((__gm__ ElementYforFT *)params.ptrZRow2);

        uint32_t UBTileMRound = RoundUp(UBTileShapeFTTrans::M, UBTransposeAlignHelper::BLK_ALIGN);
        UBTileMRound = RoundUp(UBTileMRound, UBTransposeAlignHelper::H_ALIGN);
        uint32_t UBTileNRound = RoundUp(UBTileShapeFTTrans::N, UBTransposeAlignHelper::C_ALIGN);

        uint32_t UBTileTailMRound = RoundUp(UBTileTailShapeFTTrans::M, UBTransposeAlignHelper::BLK_ALIGN);
        UBTileTailMRound = RoundUp(UBTileTailMRound, UBTransposeAlignHelper::H_ALIGN);
        uint32_t UBTileTailNRound = RoundUp(UBTileTailShapeFTTrans::N, UBTransposeAlignHelper::C_ALIGN);

        uint32_t UBBlockMRound = RoundUp(UBBlockShapeFTTrans::M, UBTransposeAlignHelper::BLK_ALIGN);
        UBBlockMRound = RoundUp(UBBlockMRound, UBTransposeAlignHelper::H_ALIGN);
        uint32_t UBBlockNRound = RoundUp(UBBlockShapeFTTrans::N, UBTransposeAlignHelper::C_ALIGN);

        uint32_t UBBlockTailMRound = RoundUp(UBBlockTailShapeFTTrans::M, UBTransposeAlignHelper::BLK_ALIGN);
        UBBlockTailMRound = RoundUp(UBBlockTailMRound, UBTransposeAlignHelper::H_ALIGN);
        uint32_t UBBlockTailNRound = RoundUp(UBBlockTailShapeFTTrans::N, UBTransposeAlignHelper::C_ALIGN);
        

        //uint32_t UBTileKRound = 1;
        //uint32_t UBTileMRound = 1;

        uint32_t loopsNumM = CeilDiv(params.problemGemmShape.m(), UBBlockMRound);
        uint32_t loopsNumN = CeilDiv(params.SplitNnum, UBBlockNRound);
        uint32_t loopsNumK = params.SplitKNum;

        uint32_t loopsNum = loopsNumM * loopsNumN * loopsNumK;
        uint32_t loopsNumMN = loopsNumM * loopsNumN;

        BlockTranspose blockTranspose(resource);

        float alpha{1.0};
        float beta{0.0};

        LayoutYforFT layoutYInforFTBlockCol{params.problemGemmShape.m(), params.SplitNnum};
        LayoutYforFT layoutYInforFTTotalCol{params.problemGemmShape.m(), params.SplitNnum};

        LayoutYforFT layoutYOutforFTBlock{params.SplitNnum, params.problemGemmShape.m()};
        LayoutYforFT layoutYOutforFTTotal{params.SplitNnum, params.problemGemmShape.m()};

        uint64_t MatrixElementSize = static_cast<uint64_t>(params.problemGemmShape.m()) * static_cast<uint64_t>(params.problemGemmShape.n());
        uint64_t base_slice_offset = static_cast<uint64_t>(params.problemGemmShape.m()) * static_cast<uint64_t>(params.SplitNnum);

        for(uint32_t loopId = aivIndex; loopId < loopsNum; loopId += aivNum) {

            uint32_t loopIdLocal = loopId % loopsNumMN;
            uint32_t kLoopId = loopId / loopsNumMN;

            uint32_t nLoopId = loopIdLocal % loopsNumN;
            uint32_t mLoopId = loopIdLocal / loopsNumN;

            uint32_t nActual = ((int32_t)nLoopId == (int32_t)(loopsNumN - 1)) ?
                params.SplitNnum - nLoopId * UBBlockNRound : UBBlockNRound;

            uint32_t mActual = ((int32_t)mLoopId == (int32_t)(loopsNumM - 1)) ?
                params.problemGemmShape.m() - mLoopId * UBBlockMRound : UBBlockMRound;

            Catlass::MatrixCoord offsetYInforFTCol{mLoopId * UBBlockMRound, nLoopId * UBBlockNRound};
            Catlass::MatrixCoord offsetYOutforFT{nLoopId * UBBlockNRound, mLoopId * UBBlockMRound};
             
            uint64_t gmOffsetYInforFTColinBlock = layoutYInforFTTotalCol.GetOffset(offsetYInforFTCol);
            uint64_t gmOffsetYInforFTCol = gmOffsetYInforFTColinBlock + static_cast<uint64_t>(kLoopId) * base_slice_offset;

            uint64_t gmOffsetYOutforFTinBlock = layoutYOutforFTTotal.GetOffset(offsetYOutforFT);
            uint64_t gmOffsetYOutforFT = gmOffsetYOutforFTinBlock + static_cast<uint64_t>(kLoopId) * base_slice_offset;
            
            Catlass::GemvCoord actualBlockShape = Catlass::GemvCoord{mActual, nActual};

            // bool isFirstBlock = (loopId == AscendC::GetBlockIdx());
            bool isFirstBlock = (loopId == aivIndex);
            bool hasNextBlock = false;
            uint32_t nLoopIdNext;
            uint32_t mLoopIdNext;
            uint32_t kLoopIdNext;
            uint64_t gmOffsetYInforFTColNext;
            uint64_t gmOffsetYOutforFTNext;  
            Catlass::GemvCoord nextActualBlockShape;

            if ((loopId + aivNum) < loopsNum){
                hasNextBlock = true;
                uint32_t loopIdNext = loopId + aivNum;

                uint32_t loopIdLocalNext = loopIdNext % loopsNumMN;
                kLoopIdNext = loopIdNext / loopsNumMN;

                nLoopIdNext = loopIdLocalNext % loopsNumN;
                mLoopIdNext = loopIdLocalNext / loopsNumN;

                uint32_t nActualNext = ((int32_t)nLoopIdNext == (int32_t)(loopsNumN - 1)) ?
                    params.SplitNnum - nLoopIdNext * UBBlockNRound : UBBlockNRound;

                uint32_t mActualNext = ((int32_t)mLoopIdNext == (int32_t)(loopsNumM - 1)) ?
                    params.problemGemmShape.m() - mLoopIdNext * UBBlockMRound : UBBlockMRound;

                nextActualBlockShape = Catlass::GemvCoord{mActualNext, nActualNext};

                Catlass::MatrixCoord offsetYInforFTColNext{mLoopIdNext * UBBlockMRound, nLoopIdNext * UBBlockNRound};
                Catlass::MatrixCoord offsetYOutforFTNext{nLoopIdNext * UBBlockNRound, mLoopIdNext * UBBlockMRound};
             
                uint64_t gmOffsetYInforFTColNextinBlock = layoutYInforFTTotalCol.GetOffset(offsetYInforFTColNext);
                gmOffsetYInforFTColNext = gmOffsetYInforFTColNextinBlock + static_cast<uint64_t>(kLoopIdNext) * base_slice_offset;
                
                uint64_t gmOffsetYOutforFTNextinBlock = layoutYOutforFTTotal.GetOffset(offsetYOutforFTNext);  
                gmOffsetYOutforFTNext = gmOffsetYOutforFTNextinBlock + static_cast<uint64_t>(kLoopIdNext) * base_slice_offset;
            }

            /*
            CATLASS_DEVICE
            void operator()(
                AscendC::GlobalTensor<ElementA> const &gmA,
                AscendC::GlobalTensor<ElementA> const &gmNextBlockA,
                LayoutA const &layoutA,
                AscendC::GlobalTensor<ElementY> const &gmY, LayoutY const &layoutY,
                GemvCoord const &actualShape, GemvCoord const &actualShapeNext,
                bool isFirstBlock, bool hasNextBlock)
            */
                
            blockTranspose(
                gmFTTmpSpace[gmOffsetYInforFTCol], 
                gmFTTmpSpace[gmOffsetYInforFTColNext],
                layoutYInforFTTotalCol, 
                gmY[gmOffsetYOutforFT], layoutYOutforFTTotal,
                actualBlockShape, nextActualBlockShape,
                isFirstBlock, hasNextBlock);
        }

        // AscendC::SyncAll<true>();
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
        uint32_t loopsNumKforA = params.SplitKNum; // 划分K维度的数量
        
        uint32_t loopsNumforA = loopsNumMforA * loopsNumKforA;

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
                CATLASS_DEVICE
                void BlockRed(
                    AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
                    AscendC::GlobalTensor<ElementX> const &gmZMax,
                    LayoutX const &layoutX,
                    GemvCoord const &actualShape)
                */
                
                blockFTSum.BlockRed(
                    gmB[gmOffsetB], params.layoutB,
                    gmBMaxSlice[gmOffsetBEMax],
                    layoutBE, actualBlockShapeforB);
            }
            else{
                uint32_t loopIdlocal = loopId - loopsNumforB;
                uint32_t mLoopId = loopIdlocal % loopsNumMforA;
                uint32_t kLoopId = loopIdlocal / loopsNumMforA;

                uint32_t mActual = ((int32_t)mLoopId == (int32_t)(loopsNumMforA - 1)) ?
                    params.problemGemmShape.m() - mLoopId * UBTileMRoundforA : UBTileMRoundforA;

                uint32_t kActual = params.actualKSliceSize[0];
                uint32_t AOffsetonK = 0;

                if(kLoopId >= params.HeadKSliceNum){
                    kActual = (kLoopId == (params.SplitKNum -1)) ? params.actualKSliceSize[2] : params.actualKSliceSize[1];

                    uint32_t RemainKSliceNum = kLoopId - params.HeadKSliceNum;

                    AOffsetonK = params.HeadKSliceNum * params.actualKSliceSize[0] + RemainKSliceNum * params.actualKSliceSize[1];
                }else{
                    AOffsetonK = kLoopId * params.actualKSliceSize[0];
                }

                Catlass::MatrixCoord offsetA{mLoopId * UBTileMRoundforA, AOffsetonK};
            
                int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);

                int64_t gmOffsetAMean = mLoopId * UBTileMRoundforA + kLoopId * params.problemGemmShape.m();
                int64_t gmOffsetAMax = mLoopId * UBTileMRoundforA + kLoopId * params.problemGemmShape.m();

                Catlass::GemvCoord actualBlockShapeforA = Catlass::GemvCoord{mActual, kActual};
                LayoutYforA layoutAred{mActual};

                /*
                CATLASS_DEVICE
                void operator()(
                    AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
                    AscendC::GlobalTensor<ElementY> const &gmZMax,
                    LayoutY const &layoutY,
                    GemvCoord const &actualShape)
                */

                blockFTSum(
                    gmA[gmOffsetA], params.layoutA,
                    gmAMax[gmOffsetAMax],
                    layoutAred, actualBlockShapeforA);
            }
            
        }

        // AscendC::SyncAll<true>();
    }

    CATLASS_DEVICE
    void B_reduce_for_thre_op(Params const &params)
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
        uint32_t aivNum = aicoreNum * AscendC::GetTaskRation();

        uint32_t half_aiv_num = aivNum / 2;
        uint32_t half_aivIndex = aivIndex;
        if(aivIndex >= half_aiv_num){
            half_aivIndex = aivIndex - half_aiv_num;
        }
        uint32_t aiv_part_num = 1 * AscendC::GetTaskRation();
        // AivCore aivCore = static_cast<AivCore>(AscendC::GetSubBlockIdx());
        uint32_t align = Catlass::BYTE_PER_C0 / sizeof(ElementZforBRed);
        // uint32_t aicoreIndex = aivIndex / AscendC::GetTaskRation();

        AscendC::GlobalTensor<ElementZforBRed> gmBVar;
        gmBVar.SetGlobalBuffer((__gm__ ElementZforBRed *)params.ptrBVar);
        // ElementYforBEAIV
        
        AscendC::GlobalTensor<ElementYforB> gmBMaxSlice;
        gmBMaxSlice.SetGlobalBuffer((__gm__ ElementYforB *)params.ptrBMaxSlice);


       // Get aicore information
        uint32_t UBTileSplitM = params.SplitReduceM;
        // RoundUp(UBTileShapeBReduce::M, UBAlignHelper::ALIGN);
        uint32_t UBTileNRound = RoundUp(params.SplitReduceN, UBAlignHelper::ALIGN);
        //uint32_t UBTileKRound = 1;
        //uint32_t UBTileMRound = 1;

        uint32_t Reduce_M_size = params.SplitNnum;
        uint32_t loopsNumM = CeilDiv((Reduce_M_size - 1), UBTileSplitM);
        uint32_t loopsNumK = params.SplitKNum;
        uint32_t loopsNum = (loopsNumM + 1) * loopsNumK;

        uint32_t split_block_num = (Reduce_M_size + align - 1) / align;

        uint32_t SplitNnumAligned = split_block_num * align + align;
        //uint32_t loopsNum = params.problemGemmShape.k();

        BlockSliceRed blockSliceRed(resource);

        int64_t OffsetInMaxInit = 0;
        // params.SplitNnum * params.problemGemmShape.k();

        Catlass::layout::VectorLayout layoutOut{params.SplitNnum};
        LayoutYforFT layoutWorkforRed{params.SplitNnum, params.problemGemmShape.k()};

        // for(uint32_t loopId = aivIndex; loopId < loopsNum; loopId += aivNum)
        for(uint32_t loopId = aivIndex; loopId < loopsNum; loopId += aivNum) {
            // (loopsNum+1)
                 
            uint32_t mLoopId = loopId % (loopsNumM + 1);
            uint32_t kLoopId = loopId / (loopsNumM + 1);

            uint32_t mActual = ((int32_t)mLoopId == (int32_t)(loopsNumM - 1)) ?
                    (Reduce_M_size - 1 - mLoopId * UBTileSplitM) : UBTileSplitM;
            
            uint32_t kActual = params.actualKSliceSize[0];
            uint32_t BSliceOffsetonK = 0;

            if(kLoopId >= params.HeadKSliceNum){
                kActual = (kLoopId == (params.SplitKNum -1)) ? params.actualKSliceSize[2] : params.actualKSliceSize[1];

                uint32_t RemainKSliceNum = kLoopId - params.HeadKSliceNum;

                BSliceOffsetonK = params.HeadKSliceNum * params.actualKSliceSize[0] + RemainKSliceNum * params.actualKSliceSize[1];
            }else{
                BSliceOffsetonK = kLoopId * params.actualKSliceSize[0];
            }

            uint64_t gmOffsetInBMaxRow = mLoopId * UBTileSplitM * params.problemGemmShape.k();
            uint64_t gmOffsetInBMaxCol = BSliceOffsetonK;
            uint64_t gmOffsetInBMax = OffsetInMaxInit + gmOffsetInBMaxRow + gmOffsetInBMaxCol;

            uint64_t gmOffsetInBMinRow = mLoopId * UBTileSplitM * params.problemGemmShape.k();
            uint64_t gmOffsetInBMinCol = BSliceOffsetonK;
            uint64_t gmOffsetInBMin = OffsetInMaxInit + gmOffsetInBMinRow + gmOffsetInBMinCol;

            uint64_t gmOffsetInBSumRow = mLoopId * UBTileSplitM * params.problemGemmShape.k();
            uint64_t gmOffsetInBSumCol = BSliceOffsetonK;
            uint64_t gmOffsetInBSum = gmOffsetInBSumRow + gmOffsetInBSumCol;

            uint64_t gmOffsetOutBVar = mLoopId * UBTileSplitM + kLoopId * SplitNnumAligned;
                
            float n_scale_ratio = (params.n_scale_ratios[0]);

            if((int32_t)mLoopId > (int32_t)(loopsNumM - 1)){
                mActual = 1;
                n_scale_ratio = (params.n_scale_ratios[1]);

                gmOffsetInBMaxRow = (Reduce_M_size - 1) * params.problemGemmShape.k();
                gmOffsetInBMaxCol = BSliceOffsetonK;
                gmOffsetInBMax = OffsetInMaxInit + gmOffsetInBMaxRow + gmOffsetInBMaxCol;

                gmOffsetInBMinRow = (Reduce_M_size - 1) * params.problemGemmShape.k();
                gmOffsetInBMinCol = BSliceOffsetonK;
                gmOffsetInBMin = OffsetInMaxInit + gmOffsetInBMinRow + gmOffsetInBMinCol;

                gmOffsetInBSumRow = (Reduce_M_size - 1) * params.problemGemmShape.k();
                gmOffsetInBSumCol = BSliceOffsetonK;
                gmOffsetInBSum = gmOffsetInBSumRow + gmOffsetInBSumCol;

                gmOffsetOutBVar = (Reduce_M_size - 1) + kLoopId * SplitNnumAligned;  
            }

            Catlass::GemvCoord actualBlockShape = Catlass::GemvCoord{mActual, kActual};
                
            /*
            // AscendC::GlobalTensor<ElementY> const &gmZ,
            CATLASS_DEVICE
            void RowVariance(
                AscendC::GlobalTensor<ElementA> const &gmBMax,
                LayoutA const &layoutA,
                AscendC::GlobalTensor<ElementY> const &gmZVar, 
                LayoutX const &layoutX,
                GemvCoord const &actualShape,
                uint32_t NRealRound)
            */
            
            blockSliceRed.RowVariance( 
                gmBMaxSlice[gmOffsetInBMax],
                layoutWorkforRed,
                gmBVar[gmOffsetOutBVar], layoutOut,
                actualBlockShape,
                UBTileNRound);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_DEVICE
    void CE_split_op_fused(Params const &params, GM_ADDR ptrOutputCOMP)
    {
        
        AB_red_split_op(params);
        Catlass::Arch::CrossCoreBarrierAIV<0x0, PIPE_V>();
        Catlass::Arch::CrossCoreBarrierAIV<0x0, PIPE_MTE3>();
        
        // Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);
        B_reduce_for_thre_op(params);

        Catlass::Arch::CrossCoreBarrierAIV<0x0, PIPE_V>();
        Catlass::Arch::CrossCoreBarrierAIV<0x0, PIPE_MTE3>();

        Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);  
        
        ABFT_Transpose(params);

        Catlass::Arch::CrossCoreBarrierAIV<0x0, PIPE_V>();
        Catlass::Arch::CrossCoreBarrierAIV<0x0, PIPE_MTE3>();

        // Represent the full gm
        // Get aicore information
        /*
        /// Construct
        CATLASS_DEVICE
        BlockFTGemvCENoSplitK(Arch::ResourceAIV<ArchTag> &resource, float n_sqrt_ratio_factor_remain,
            uint32_t UBufAddrStart = 0, uint32_t remainNum=4, uint32_t remainMSize=16)
        */

        float n_sqrt_ratio_factor_first = params.n_sqrt_ratios[0];

        BlockFTGemvAIV blockFTGemvAIV(resource, n_sqrt_ratio_factor_first, 
            (uint32_t)0, params.remainNum, params.remainMSize);

        // BlockThreCalc blockThreCalc(resource);

        BlockScheduler matmulBlockScheduler(params.problemGemmShapeRemain, 
            Catlass::GemmCoord(L1TileShape::M, L1TileShape::N, L1TileShape::K), 
            params.SplitKNum);
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        BlockSchedulerFirst matmulBlockSchedulerFirst(params.problemGemmShapeFirst,
            Catlass::GemmCoord(L1TileShapeFirst::M, L1TileShapeFirst::N, L1TileShapeFirst::K), 
            params.SplitKNum);

        Catlass::GemmCoord loopsMNK = matmulBlockScheduler.loopsMNK;
        
        if(params.problemGemmShapeRemain.n() < 1){
            coreLoops = matmulBlockSchedulerFirst.GetCoreLoops();
            loopsMNK = matmulBlockSchedulerFirst.loopsMNK;
        }

        // uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetTaskRation();
        // AscendC::printf("%zu\n",AscendC::GetBlockNum());
        uint32_t aivIndex = AscendC::GetBlockIdx();
        uint32_t aicoreIndex = aivIndex / AscendC::GetSubBlockNum();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t aivNum = aicoreNum * AscendC::GetSubBlockNum();
        uint32_t aiv_part_num = 1 * AscendC::GetTaskRation();
        uint32_t align = Catlass::BYTE_PER_C0 / sizeof(ElementZforBRed);

        // uint32_t aicoreIndex = aivIndex / AscendC::GetTaskRation();

        AscendC::GlobalTensor<ElementC> gmC;
        if(params.SplitKNum > 1){
            gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrWorkspace);
        }else{
            gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);
        }

        AscendC::GlobalTensor<ElementYforA> gmAMax;
        gmAMax.SetGlobalBuffer((__gm__ ElementYforA *)params.ptrAMax);

        // AscendC::GlobalTensor<ElementYforA> gmStdA;
        // gmStdA.SetGlobalBuffer((__gm__ ElementYforA *)params.ptrAMin);

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
        
        AscendC::GlobalTensor<ElementZforBRed> gmBVar;
        gmBVar.SetGlobalBuffer((__gm__ ElementZforBRed *)params.ptrBVar);

        

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

        uint64_t MatrixElementSize = static_cast<uint64_t>(params.problemGemmShape.m()) * static_cast<uint64_t>(params.problemGemmShape.n());
        uint64_t base_slice_offset = static_cast<uint64_t>(params.SplitNnum) * static_cast<uint64_t>(params.problemGemmShape.m());
        uint64_t base_slice_offset_forZ = static_cast<uint64_t>(params.SplitNnum) * static_cast<uint64_t>(total_output_elements);

        uint64_t base_M_offset = static_cast<uint64_t>(params.problemGemmShape.m());

        uint32_t split_block_num = (params.SplitNnum + align - 1) / align;

        uint32_t SplitNnumAligned = split_block_num * align + align;

        if(params.problemGemmShapeRemain.n() > 0){
            for (uint32_t loopIdx = aicoreIndex; loopIdx < coreLoops; loopIdx += aicoreNum) {
                // Compute block location

                uint32_t KSliceIdx = matmulBlockScheduler.GetSplitkSliceIdx(loopIdx);
                Catlass::GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);

                uint32_t splitNIdx = blockCoord.n() * L1TileShape::N / L1TileShape::N;

                uint32_t splitNIdxTotal = splitNIdx + params.firstNnum;
                uint32_t splitNIdxFirst = splitNIdx / params.remainNum;
                uint32_t remainIdxFirst = splitNIdx % params.remainNum;
                
                // Compute initial location in logical coordinates
                Catlass::MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, params.problemGemmShapeFirst.n() + blockCoord.n() * L1TileShape::N};
                Catlass::MatrixCoord offsetYforFT{splitNIdxTotal, blockCoord.m() * L1TileShape::M};

                Catlass::MatrixCoord offsetCOMPYforFT{splitNIdxTotal, blockCoord.m() * L1TileShape::M};
                Catlass::MatrixCoord offsetCOMPXforFT{splitNIdxTotal, blockCoord.m() * L1TileShape::M};
                Catlass::MatrixCoord offsetThreforFT{splitNIdxTotal, blockCoord.m() * L1TileShape::M};

                Catlass::MatrixCoord offsetCFirst{blockCoord.m() * L1TileShape::M, splitNIdxFirst * L1TileShapeFirst::N};
                Catlass::MatrixCoord offsetYforFTFirst{splitNIdxFirst, blockCoord.m() * L1TileShape::M};

                Catlass::MatrixCoord offsetCOMPYforFTFirst{splitNIdxFirst, blockCoord.m() * L1TileShape::M};
                Catlass::MatrixCoord offsetCOMPXforFTFirst{splitNIdxFirst, blockCoord.m() * L1TileShape::M};
                Catlass::MatrixCoord offsetThreforFTFirst{splitNIdxFirst, blockCoord.m() * L1TileShape::M};

                uint32_t COMPZRowOffset = blockCoord.m() * L1TileShape::M / 8;
                Catlass::MatrixCoord offsetCOMPZforFT{splitNIdxTotal, COMPZRowOffset};
                Catlass::MatrixCoord offsetCOMPZforFTFirst{splitNIdxFirst, COMPZRowOffset};

                uint32_t mActual = UBBlockMRound;

                if(blockCoord.m() == loopsMNK.m() -1) {
                    mActual = params.problemGemmShapeRemain.m() - blockCoord.m() * L1TileShape::M;
                }

                uint32_t nActual = ThreUBBlockNRound;
                uint32_t nActualFirst = L1TileShapeFirst::N;

                if(blockCoord.n() == loopsMNK.n() - 1){
                    nActual = params.problemGemmShapeRemain.n() - blockCoord.n() * L1TileShape::N;
                }

                if(splitNIdxFirst == (params.firstNnum - 1)){
                    nActualFirst = params.problemGemmShapeFirst.n() - splitNIdxFirst * L1TileShapeFirst::N;
                }

                uint64_t gmOffsetCSlice = params.layoutC.GetOffset(offsetC);
                uint64_t gmOffsetC = gmOffsetCSlice + MatrixElementSize * static_cast<uint64_t>(KSliceIdx);

                uint64_t gmOffsetYforFTSlice = layoutYforFT.GetOffset(offsetYforFT);
                uint64_t gmOffsetYforFT = gmOffsetYforFTSlice + base_slice_offset * static_cast<uint64_t>(KSliceIdx);

                uint64_t gmOffsetCOMPXforFTSlice = layoutCOMPYforFT.GetOffset(offsetCOMPXforFT);
                uint64_t gmOffsetCOMPXforFT = gmOffsetCOMPXforFTSlice + base_slice_offset * static_cast<uint64_t>(KSliceIdx);

                uint64_t gmOffsetCOMPYforFTSlice = layoutCOMPYforFT.GetOffset(offsetCOMPYforFT);
                uint64_t gmOffsetCOMPYforFT = gmOffsetCOMPYforFTSlice + base_slice_offset * static_cast<uint64_t>(KSliceIdx);

                uint64_t gmOffsetThreforFTSlice = layoutThreforFT.GetOffset(offsetThreforFT);
                uint64_t gmOffsetThreforFT = gmOffsetThreforFTSlice + base_slice_offset * static_cast<uint64_t>(KSliceIdx);
                
                uint64_t gmOffsetCOMPZforFTSlice = layoutCOMPZforFT.GetOffset(offsetCOMPZforFT);
                uint64_t gmOffsetCOMPZforFT = gmOffsetCOMPZforFTSlice + base_slice_offset_forZ * static_cast<uint64_t>(KSliceIdx);

                uint64_t gmOffsetCFirstSlice = params.layoutC.GetOffset(offsetCFirst);
                uint64_t gmOffsetCFirst = gmOffsetCFirstSlice + MatrixElementSize * static_cast<uint64_t>(KSliceIdx);

                uint64_t gmOffsetYforFTFirstSlice = layoutYforFT.GetOffset(offsetYforFTFirst);
                uint64_t gmOffsetYforFTFirst = gmOffsetYforFTFirstSlice + base_slice_offset * static_cast<uint64_t>(KSliceIdx);

                uint64_t gmOffsetCOMPXforFTFirstSlice = layoutCOMPYforFT.GetOffset(offsetCOMPXforFTFirst);
                uint64_t gmOffsetCOMPXforFTFirst = gmOffsetCOMPXforFTFirstSlice + base_slice_offset * static_cast<uint64_t>(KSliceIdx);

                uint64_t gmOffsetCOMPYforFTFirstSlice = layoutCOMPYforFT.GetOffset(offsetCOMPYforFTFirst);
                uint64_t gmOffsetCOMPYforFTFirst = gmOffsetCOMPYforFTFirstSlice + base_slice_offset * static_cast<uint64_t>(KSliceIdx);

                uint64_t gmOffsetThreforFTFirstSlice = layoutThreforFT.GetOffset(offsetThreforFTFirst);
                uint64_t gmOffsetThreforFTFirst = gmOffsetThreforFTFirstSlice + base_slice_offset * static_cast<uint64_t>(KSliceIdx);

                uint64_t gmOffsetCOMPZforFTFirstSlice = layoutCOMPZforFT.GetOffset(offsetCOMPZforFTFirst);
                uint64_t gmOffsetCOMPZforFTFirst = gmOffsetCOMPZforFTFirstSlice + base_slice_offset_forZ * static_cast<uint64_t>(KSliceIdx);

                uint64_t gmOffsetARedSlice = blockCoord.m() * L1TileShape::M;
                uint64_t gmOffsetARed = gmOffsetARedSlice + base_M_offset * static_cast<uint64_t>(KSliceIdx);

                uint32_t gmOffsetBRedSlice = splitNIdxTotal;
                uint32_t gmOffsetBRed = gmOffsetBRedSlice + KSliceIdx * SplitNnumAligned;

                uint32_t gmOffsetBRedFirstSlice = splitNIdxFirst;
                uint32_t gmOffsetBRedFirst = gmOffsetBRedFirstSlice + KSliceIdx * SplitNnumAligned;

                Catlass::layout::VectorLayout layoutABE{mActual};
            
                Catlass::GemvCoord actualBlockShape = Catlass::GemvCoord{mActual, nActual};
 
                float std_est_B_ratio = params.simple_std_est_ratios[0];

                float n_ratio_factor = params.n_ratios[0];
                float n_sqrt_ratio_factor = params.n_sqrt_ratios[0];
                float n_square_ratio_factor = params.n_square_ratios[0];
                float simple_std_est_A_row_ratio = params.simple_std_est_A_row_ratios[0];

                if(splitNIdx >= (params.SplitNnum - 1)){
                    std_est_B_ratio = params.simple_std_est_ratios[1];
                    n_ratio_factor = params.n_ratios[1];
                    n_sqrt_ratio_factor = params.n_sqrt_ratios[1];
                    n_square_ratio_factor = params.n_square_ratios[1];
                }

                if(KSliceIdx >= params.HeadKSliceNum){
                    simple_std_est_A_row_ratio = (KSliceIdx == params.SplitKNum -1) ? params.simple_std_est_A_row_ratios[2] : params.simple_std_est_A_row_ratios[1];
                }

                bool hasRemain = false;
                if(splitNIdxFirst < params.firstNnum){
                    hasRemain = true;
                }
                /*
                CATLASS_DEVICE
                void operator()(
                    AscendC::GlobalTensor<ElementA> const &gmC,
                    AscendC::GlobalTensor<ElementA> const &gmCRemain, LayoutA const &layoutC,
                    AscendC::GlobalTensor<ElementY> const &gmAMax, LayoutX const &layoutAforFT,
                    AscendC::GlobalTensor<ElementY> const &gmY, 
                    AscendC::GlobalTensor<ElementY> const &gmYRemain,
                    AscendC::GlobalTensor<ElementY> const &gmABe,
                    AscendC::GlobalTensor<ElementY> const &gmABeRemain, LayoutY const &layoutY,
                    AscendC::GlobalTensor<ElementY> const &gmBVar,
                    AscendC::GlobalTensor<ElementY> const &gmBVarRemain, 
                    LayoutX const &layoutBforFT,
                    AscendC::GlobalTensor<ElementY> const &gmThreZ,
                    AscendC::GlobalTensor<ElementY> const &gmThreZRemain, LayoutY const &layoutThre,
                    AscendC::GlobalTensor<ElementZ> const &gmCOMPZ,
                    AscendC::GlobalTensor<ElementZ> const &gmCOMPZRemain, LayoutZ const &layoutZ,
                    GemvCoord const &actualShape,
                    float std_est_A_row_ratio, 
                    float std_est_B_ratio, 
                    float n_sqrt_ratio_factor, 
                    float e_max,
                    bool outputThre, bool outputCE, 
                    uint32_t aiv_part_num, bool hasRemain, uint32_t remainIdx, uint32_t remainNSize,
                    Catlass::Arch::CrossCoreFlagWithReverse<> & flagAicFinishStore)
                */

                blockFTGemvAIV(
                    gmC[gmOffsetC], gmC[gmOffsetCFirst], params.layoutC,
                    gmAMax[gmOffsetARed], layoutARed, 
                    gmCOMPY[gmOffsetCOMPYforFT], gmCOMPY[gmOffsetCOMPYforFTFirst],
                    gmCOMPX[gmOffsetCOMPYforFT], gmCOMPX[gmOffsetCOMPYforFTFirst], layoutCE, 
                    gmBVar[gmOffsetBRed], gmBVar[gmOffsetBRedFirst], layoutBforFT, 
                    gmT[gmOffsetThreforFT], gmT[gmOffsetThreforFTFirst], params.layoutThre,
                    gmCOMPZ[gmOffsetCOMPZforFT], gmCOMPZ[gmOffsetCOMPZforFTFirst], layoutOutputZ,
                    actualBlockShape, simple_std_est_A_row_ratio, std_est_B_ratio,
                    n_sqrt_ratio_factor, params.e_max, params.outputThre, params.outputCE,
                    aiv_part_num, hasRemain, remainIdxFirst, nActualFirst, flagAicFinishStore);

                // Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);   
            }
        }else{
            for (uint32_t loopIdx = aicoreIndex; loopIdx < coreLoops; loopIdx += aicoreNum) {
                // Compute block location
                uint32_t KSliceIdx = matmulBlockSchedulerFirst.GetSplitkSliceIdx(loopIdx);
                Catlass::GemmCoord blockCoord = matmulBlockSchedulerFirst.GetBlockCoord(loopIdx);

                uint32_t splitNIdx = blockCoord.n() * L1TileShapeFirst::N / L1TileShapeFirst::N;
                uint32_t splitNIdxTotal = splitNIdx;
                uint32_t splitNIdxFirst = 0;
                uint32_t remainIdxFirst = 0;

                bool hasRemain = false;
                // Compute initial location in logical coordinates
                Catlass::MatrixCoord offsetC{blockCoord.m() * L1TileShapeFirst::M, blockCoord.n() * L1TileShapeFirst::N};
                Catlass::MatrixCoord offsetYforFT{splitNIdxTotal, blockCoord.m() * L1TileShapeFirst::M};

                Catlass::MatrixCoord offsetCOMPYforFT{splitNIdxTotal, blockCoord.m() * L1TileShapeFirst::M};
                Catlass::MatrixCoord offsetCOMPXforFT{splitNIdxTotal, blockCoord.m() * L1TileShapeFirst::M};
                Catlass::MatrixCoord offsetThreforFT{splitNIdxTotal, blockCoord.m() * L1TileShapeFirst::M};

                Catlass::MatrixCoord offsetCFirst{blockCoord.m() * L1TileShapeFirst::M, splitNIdxFirst * L1TileShapeFirst::N};
                Catlass::MatrixCoord offsetYforFTFirst{splitNIdxFirst, blockCoord.m() * L1TileShapeFirst::M};

                Catlass::MatrixCoord offsetCOMPYforFTFirst{splitNIdxFirst, blockCoord.m() * L1TileShapeFirst::M};
                Catlass::MatrixCoord offsetCOMPXforFTFirst{splitNIdxFirst, blockCoord.m() * L1TileShapeFirst::M};
                Catlass::MatrixCoord offsetThreforFTFirst{splitNIdxFirst, blockCoord.m() * L1TileShapeFirst::M};


                uint32_t COMPZRowOffset = blockCoord.m() * L1TileShapeFirst::M / 8;
                Catlass::MatrixCoord offsetCOMPZforFT{splitNIdxTotal, COMPZRowOffset};
                Catlass::MatrixCoord offsetCOMPZforFTFirst{splitNIdxFirst, COMPZRowOffset};

                uint32_t mActual = L1TileShapeFirst::M;
                if(blockCoord.m() == loopsMNK.m() -1) {
                    mActual = params.problemGemmShapeFirst.m() - blockCoord.m() * L1TileShapeFirst::M;
                }

                uint32_t nActual = L1TileShapeFirst::N;
                uint32_t nActualFirst = L1TileShapeFirst::N;

                if(blockCoord.n() == loopsMNK.n() - 1){
                    nActual = params.problemGemmShapeFirst.n() - blockCoord.n() * L1TileShapeFirst::N;
                }

                if(splitNIdxFirst == (params.firstNnum - 1)){
                    nActualFirst = params.problemGemmShapeFirst.n() - splitNIdxFirst * L1TileShapeFirst::N;
                }

                uint64_t gmOffsetCSlice = params.layoutC.GetOffset(offsetC);
                uint64_t gmOffsetC = gmOffsetCSlice + MatrixElementSize * static_cast<uint64_t>(KSliceIdx);

                uint64_t gmOffsetYforFTSlice = layoutYforFT.GetOffset(offsetYforFT);
                uint64_t gmOffsetYforFT = gmOffsetYforFTSlice + base_slice_offset * static_cast<uint64_t>(KSliceIdx);

                uint64_t gmOffsetCOMPXforFTSlice = layoutCOMPYforFT.GetOffset(offsetCOMPXforFT);
                uint64_t gmOffsetCOMPXforFT = gmOffsetCOMPXforFTSlice + base_slice_offset * static_cast<uint64_t>(KSliceIdx);

                uint64_t gmOffsetCOMPYforFTSlice = layoutCOMPYforFT.GetOffset(offsetCOMPYforFT);
                uint64_t gmOffsetCOMPYforFT = gmOffsetCOMPYforFTSlice + base_slice_offset * static_cast<uint64_t>(KSliceIdx);

                uint64_t gmOffsetThreforFTSlice = layoutThreforFT.GetOffset(offsetThreforFT);
                uint64_t gmOffsetThreforFT = gmOffsetThreforFTSlice + base_slice_offset * static_cast<uint64_t>(KSliceIdx);
                
                uint64_t gmOffsetCOMPZforFTSlice = layoutCOMPZforFT.GetOffset(offsetCOMPZforFT);
                uint64_t gmOffsetCOMPZforFT = gmOffsetCOMPZforFTSlice + base_slice_offset_forZ * static_cast<uint64_t>(KSliceIdx);

                uint64_t gmOffsetCFirstSlice = params.layoutC.GetOffset(offsetCFirst);
                uint64_t gmOffsetCFirst = gmOffsetCFirstSlice + MatrixElementSize * static_cast<uint64_t>(KSliceIdx);

                uint64_t gmOffsetYforFTFirstSlice = layoutYforFT.GetOffset(offsetYforFTFirst);
                uint64_t gmOffsetYforFTFirst = gmOffsetYforFTFirstSlice + base_slice_offset * static_cast<uint64_t>(KSliceIdx);

                uint64_t gmOffsetCOMPXforFTFirstSlice = layoutCOMPYforFT.GetOffset(offsetCOMPXforFTFirst);
                uint64_t gmOffsetCOMPXforFTFirst = gmOffsetCOMPXforFTFirstSlice + base_slice_offset * static_cast<uint64_t>(KSliceIdx);

                uint64_t gmOffsetCOMPYforFTFirstSlice = layoutCOMPYforFT.GetOffset(offsetCOMPYforFTFirst);
                uint64_t gmOffsetCOMPYforFTFirst = gmOffsetCOMPYforFTFirstSlice + base_slice_offset * static_cast<uint64_t>(KSliceIdx);

                uint64_t gmOffsetThreforFTFirstSlice = layoutThreforFT.GetOffset(offsetThreforFTFirst);
                uint64_t gmOffsetThreforFTFirst = gmOffsetThreforFTFirstSlice + base_slice_offset * static_cast<uint64_t>(KSliceIdx);

                uint64_t gmOffsetCOMPZforFTFirstSlice = layoutCOMPZforFT.GetOffset(offsetCOMPZforFTFirst);
                uint64_t gmOffsetCOMPZforFTFirst = gmOffsetCOMPZforFTFirstSlice + base_slice_offset_forZ * static_cast<uint64_t>(KSliceIdx);

                uint64_t gmOffsetARedSlice = blockCoord.m() * L1TileShape::M;
                uint64_t gmOffsetARed = gmOffsetARedSlice + base_M_offset * static_cast<uint64_t>(KSliceIdx);

                uint32_t gmOffsetBRedSlice = splitNIdxTotal;
                uint32_t gmOffsetBRed = gmOffsetBRedSlice + KSliceIdx * SplitNnumAligned;

                uint32_t gmOffsetBRedFirstSlice = splitNIdxFirst;
                uint32_t gmOffsetBRedFirst = gmOffsetBRedFirstSlice + KSliceIdx * SplitNnumAligned;

                Catlass::layout::VectorLayout layoutABE{mActual};
            
                Catlass::GemvCoord actualBlockShape = Catlass::GemvCoord{mActual, nActual};
 
                float std_est_B_ratio = params.simple_std_est_ratios[0];

                float n_ratio_factor = params.n_ratios[0];
                float n_sqrt_ratio_factor = params.n_sqrt_ratios[0];
                float n_square_ratio_factor = params.n_square_ratios[0];
                float simple_std_est_A_row_ratio = params.simple_std_est_A_row_ratios[0];

                if(splitNIdx >= (params.SplitNnum - 1)){
                    std_est_B_ratio = params.simple_std_est_ratios[1];

                    n_ratio_factor = params.n_ratios[1];
                    n_sqrt_ratio_factor = params.n_sqrt_ratios[1];
                    n_square_ratio_factor = params.n_square_ratios[1];
                }

                if(KSliceIdx >= params.HeadKSliceNum){
                    simple_std_est_A_row_ratio = (KSliceIdx == params.SplitKNum -1) ? params.simple_std_est_A_row_ratios[2] : params.simple_std_est_A_row_ratios[1];
                }

                /*
                CATLASS_DEVICE
                void op_without_flag(
                    AscendC::GlobalTensor<ElementA> const &gmC,
                    AscendC::GlobalTensor<ElementA> const &gmCRemain, 
                    LayoutA const &layoutC,
                    AscendC::GlobalTensor<ElementY> const &gmAMax, LayoutX const &layoutAforFT,
                    AscendC::GlobalTensor<ElementY> const &gmY, 
                    AscendC::GlobalTensor<ElementY> const &gmYRemain,
                    AscendC::GlobalTensor<ElementY> const &gmABe,
                    AscendC::GlobalTensor<ElementY> const &gmABeRemain, LayoutY const &layoutY,
                    AscendC::GlobalTensor<ElementY> const &gmBVar,
                    AscendC::GlobalTensor<ElementY> const &gmBVarRemain, 
                    LayoutX const &layoutBforFT,
                    AscendC::GlobalTensor<ElementY> const &gmThreZ,
                    AscendC::GlobalTensor<ElementY> const &gmThreZRemain, LayoutY const &layoutThre,
                    AscendC::GlobalTensor<ElementZ> const &gmCOMPZ,
                    AscendC::GlobalTensor<ElementZ> const &gmCOMPZRemain, LayoutZ const &layoutZ,
                    GemvCoord const &actualShape,
                    float std_est_A_row_ratio, 
                    float std_est_B_ratio, 
                    float n_sqrt_ratio_factor, 
                    float e_max,
                    bool outputThre, bool outputCE, 
                    uint32_t aiv_part_num, bool hasRemain, uint32_t remainIdx, uint32_t remainNSize)
                */

                blockFTGemvAIV.op_without_flag(
                    gmC[gmOffsetC], gmC[gmOffsetCFirst], params.layoutC,
                    gmAMax[gmOffsetARed], layoutARed, 
                    gmCOMPY[gmOffsetCOMPYforFT], gmCOMPY[gmOffsetCOMPYforFTFirst],
                    gmCOMPX[gmOffsetCOMPYforFT], gmCOMPX[gmOffsetCOMPYforFTFirst], layoutCE, 
                    gmBVar[gmOffsetBRed], gmBVar[gmOffsetBRedFirst], layoutBforFT, 
                    gmT[gmOffsetThreforFT], gmT[gmOffsetThreforFTFirst], params.layoutThre,
                    gmCOMPZ[gmOffsetCOMPZforFT], gmCOMPZ[gmOffsetCOMPZforFTFirst], layoutOutputZ,
                    actualBlockShape, simple_std_est_A_row_ratio, std_est_B_ratio,
                    n_sqrt_ratio_factor, params.e_max, params.outputThre, params.outputCE,
                    aiv_part_num, hasRemain, remainIdxFirst, nActualFirst);
            }
        }
        
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params){

        // using ElementOut = typename ReduceAdd::ElementOut;
        // using ElementAccumulator = typename ReduceAdd::ElementAccumulator;
        
        CE_split_op_fused(params,params.ptrCOMPZRow);
        
        // AscendC::SyncAll<true>(); 
        Catlass::Arch::CrossCoreBarrierAIV<0x0, PIPE_V>();
        Catlass::Arch::CrossCoreBarrierAIV<0x0, PIPE_MTE3>();

        Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);

        if(params.SplitKNum > 1){
            AscendC::GlobalTensor<ElementC> gmC;
            AscendC::GlobalTensor<ElementC> gmWorkspace;
            gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC*>(params.ptrC));
            gmWorkspace.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC*>(params.ptrWorkspace));
            ReduceAdd reduceAdd(resource);
            reduceAdd(gmC, gmWorkspace,
                static_cast<uint64_t>(params.problemGemmShape.m()) * static_cast<uint64_t>(params.problemGemmShape.n()),
                params.SplitKNum);
        }   


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