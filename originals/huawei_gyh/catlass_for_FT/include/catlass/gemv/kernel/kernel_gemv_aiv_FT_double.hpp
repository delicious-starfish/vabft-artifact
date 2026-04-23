#ifndef CATLASS_GEMV_KERNEL_AIV_FT_HPP
#define CATLASS_GEMV_KERNEL_AIV_FT_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemv_coord.hpp"
#include "catlass/gemv/helper.hpp"

namespace Catlass::Gemm::Kernel {

template <
    //class BlockMmad_,
    //class BlockScheduler_,
    //class BlockEncodeScheduler_,
    class BlockSumGemv_
>
class GemvKernelFTAiv {
public:
    //using BlockMmad = BlockMmad_;
    using BlockSumGemv = BlockSumGemv_;
    using ArchTag = typename BlockSumGemv::ArchTag;
    //using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockSumGemv::ElementA;
    using LayoutA = typename BlockSumGemv::LayoutA;
    using ElementB = typename BlockSumGemv::ElementA;
    using LayoutB = typename BlockSumGemv::LayoutA;
    using ElementX = typename BlockSumGemv::ElementX;
    using LayoutX = typename BlockSumGemv::LayoutX;

    using LayoutACol = typename std::conditional<
        std::is_same<LayoutA, Catlass::layout::RowMajor>::value,
        Catlass::layout::ColumnMajor,
        Catlass::layout::RowMajor>::type;
    using LayoutBCol = typename std::conditional<
        std::is_same<LayoutB, Catlass::layout::RowMajor>::value,
        Catlass::layout::ColumnMajor,
        Catlass::layout::RowMajor>::type;

    using ElementY = typename BlockSumGemv::ElementY;
    using LayoutY = typename BlockSumGemv::LayoutY;
    using UBTileShape = typename BlockSumGemv::UBTileShape;
    
    using UBAlignHelper = Gemv::helper::UBAlignHelper<ElementA>;

    using ElementZ = ElementX;

    using FT_ENC_TYPE = Gemv::helper::FT_ENC_TYPE;

    static_assert(std::is_same_v<LayoutA, LayoutB>,
        "The LayoutA and LayoutB of Gemm should be consistent.");

    enum class AivCore {
        AIV0 = 0,
        AIV1
    };

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemGemmShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        LayoutACol layoutACol;
        GM_ADDR ptrB;
        LayoutB layoutB;
        LayoutBCol layoutBCol;
        GM_ADDR ptrXV;
        LayoutX layoutXV;
        GM_ADDR ptrZRow2; // 最终 列和/行和 checksum 与 eTC; Ce sum 结果的存放位置
        GM_ADDR ptrZCol2;
        GM_ADDR ptrWorkspace; // 中间暂存MMAD 输出结果的GM，这些结果要通过Ce/eTC 求行和/列和以便和ABe/eTAB 作比较，以检测校验soft silent errors.
        FT_ENC_TYPE enc_type;


        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const &problemGemmShape_,
            GM_ADDR ptrA_, LayoutA const &layoutA_, LayoutACol layoutACol_,
            GM_ADDR ptrB_, LayoutB const &layoutB_, LayoutBCol layoutBCol_,
            GM_ADDR ptrXV_, LayoutX const &layoutXV_,
            GM_ADDR ptrZRow2_, GM_ADDR ptrZCol2_,
            GM_ADDR ptrWorkspace_,
            FT_ENC_TYPE enc_type_
        ) : problemGemmShape(problemGemmShape_), ptrA(ptrA_), layoutA(layoutA_), layoutACol(layoutACol_),
            ptrB(ptrB_), layoutB(layoutB_), layoutBCol(layoutBCol_),
            ptrXV(ptrXV_), layoutXV(layoutXV_),
            ptrZRow2(ptrZRow2_), ptrZCol2(ptrZCol2_),
            ptrWorkspace(ptrWorkspace_),
            enc_type(enc_type_) {}
    };

    struct Arguments {
        GemmCoord problemGemmShape;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrXV;
        GM_ADDR ptrZRow2; // 最终 列和/行和 checksum 与 eTC; Ce sum 结果的存放位置
        GM_ADDR ptrZCol2;
        uint32_t blockNum;
        FT_ENC_TYPE enc_type;
    };

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        // 每个AI Core 上 Cube Core 的输出暂存在16个 GM 空间上
        // 每个空间大小为一个L1 Tile, 这是因为AIC-AIV 同步时最多连续setFlag 16 次
        //return args.blockNum * (Catlass::Arch::MAX_REVERSE_DEPTH + 1) * L1TileShape::M * L1TileShape::N * sizeof(ElementC);
        return sizeof(ElementX) * (args.problemGemmShape.m() + args.problemGemmShape.k());
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
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
        Params params{
            problemGemmShape, args.ptrA, layoutA, layoutACol,
            args.ptrB, layoutB, layoutBCol,
            args.ptrXV, layoutXV, args.ptrZRow2, args.ptrZCol2, workspace, args.enc_type};
        return params;
    }

    // Methods
    CATLASS_DEVICE
    GemvKernelFTAiv() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params){}

    CATLASS_DEVICE
    void ABE_op(Params const &params)
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


    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params){
        if(params.enc_type == FT_ENC_TYPE::NO){
            //EpilogueOp(params);
        }else if(params.enc_type == FT_ENC_TYPE::CE) {
            ABE_op(params);
        }else if(params.enc_type == FT_ENC_TYPE::ETC) {
            ETAB_op(params);
        }else if(params.enc_type == FT_ENC_TYPE::BOTHC) {
            ABE_op(params);
            // device 级别的同步
            AscendC::SyncAll<true>();
            ETAB_op(params);
        }
    }

private:
    // ID used for inter-core synchronization
    static constexpr Arch::FlagID FLAG_AIC_FINISH_STORE = 0;
    static constexpr Arch::FlagID RV_FLAG_AIC_FINISH_STORE = 1;
    Arch::CrossCoreFlagWithReverse<> flagAicFinishStore{FLAG_AIC_FINISH_STORE, RV_FLAG_AIC_FINISH_STORE};

    Arch::Resource<ArchTag> resource;
};

}

#endif // CATLASS_GEMV_KERNEL_AIV_FT_HPP