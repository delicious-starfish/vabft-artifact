
#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_HPP_FT_ABE_NO_SPLITK_SELF_ROBUST
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_HPP_FT_ABE_NO_SPLITK_SELF_ROBUST

# include "catlass/catlass.hpp"
# include "catlass/arch/resource.hpp"
# include "catlass/arch/cross_core_sync.hpp"
# include "catlass/coord.hpp"
# include "catlass/gemm_coord.hpp"
# include "gemm/dispatch_policy.hpp"
# include "catlass/gemm/helper.hpp"
# include "catlass/gemv/helper.hpp"
# include "catlass/gemm/tile/tile_copy.hpp"
# include "catlass/gemm/tile/tile_mmad.hpp"

namespace CubeSelf::Gemm::Block {
template<
    bool ENABLE_UNIT_FLAG_,
    class L1TileShape_,
    class L1TileShapeforFT_,
    class L0TileShape_,
    class L0TileShapeforFT_,
    class AType_,
    class BType_,
    class CType_,
    class XType_,
    class YType_,
    class BiasType_,
    class TileCopyFTABonAic_,
    class TileMmad_
>
struct BlockMmadFTABeNoSplitKRobust<
    CubeSelf::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L1TileShapeforFT_,
    L0TileShape_,
    L0TileShapeforFT_,
    AType_,
    BType_,
    CType_,
    XType_,
    YType_,
    BiasType_,
    TileCopyFTABonAic_,
    TileMmad_
>{
public:

    // Type Aliases
    using DispatchPolicy = MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using L1TileShape = L1TileShape_;
    using L1TileShapeforFT = L1TileShapeforFT_;
    using L0TileShape = L0TileShape_;
    using L0TileShapeforFT = L0TileShapeforFT_;

    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;

    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;

    using ElementX = typename XType_::Element;
    using LayoutX = typename XType_::Layout;

    using LayoutVX = Catlass::layout::VectorLayout;
    using VXType = Catlass::Gemm::GemmType<ElementX, LayoutVX>;

    using ElementY = typename YType_::Element;
    using LayoutY = typename YType_::Layout;

    using TileMmad = TileMmad_;

    using CopyGmToL1A = typename TileCopyFTABonAic_::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopyFTABonAic_::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopyFTABonAic_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopyFTABonAic_::CopyL1ToL0B;
    using CopyL0CToGm = typename TileCopyFTABonAic_::CopyL0CToGm;

    using CopyGmToL1X = typename TileCopyFTABonAic_::CopyGmToL1X;
    using CopyL1ToL0X = typename TileCopyFTABonAic_::CopyL1ToL0X;
    using CopyGmToL1VX = typename TileCopyFTABonAic_::CopyGmToL1VX;
    using CopyL1ToL0AforFT = typename TileCopyFTABonAic_::CopyL1ToL0AforFT;
    using CopyL0CToGmforABE = typename TileCopyFTABonAic_::CopyL0CToGmforABE;

    using ElementAccumulator = 
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA,ElementB>::ElementAccumulator;

    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;
    using LayoutXInL1 = typename CopyL1ToL0X::LayoutSrc;

    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;

    using LayoutAInL0forFT = typename CopyL1ToL0AforFT::LayoutDst;
    using LayoutXInL0 = typename CopyL1ToL0X::LayoutDst;

    using LayoutCInL0 = Catlass::layout::zN;
    using LayoutYInL0 = Catlass::layout::zN;

    using L1AAlignHelper = Catlass::Gemm::helper::L1AlignHelper<ElementA, LayoutA>;
    using L1BAlignHelper = Catlass::Gemm::helper::L1AlignHelper<ElementB, LayoutB>;
    using L1XAlignHelper = Catlass::Gemv::helper::L1AlignHelper<ElementX, LayoutVX>;

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;

    static constexpr uint32_t L1A_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L1B_SIZE = L1TileShape::K * L1TileShape::N * sizeof(ElementB);
    static constexpr uint32_t L1X_SIZE = L1TileShapeforFT::N * L1TileShapeforFT::K * sizeof(ElementX);
    static constexpr uint32_t L1VX_SIZE = 16 * L1TileShapeforFT::K * sizeof(ElementX);

    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
    static constexpr uint32_t L0A_SIZE_FOR_A = L0TileShape::M * L0TileShape::K * sizeof(ElementA) * STAGES;
    static constexpr uint32_t L0A_SIZE_FOR_X = L0TileShapeforFT::N * L0TileShapeforFT::K * sizeof(ElementX) * STAGES;
    static constexpr uint32_t L0A_SIZE_FOR_VX = 16 * L0TileShapeforFT::K * sizeof(ElementX) * STAGES;

    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
    static constexpr uint32_t L0B_SIZE_FOR_B = L0TileShape::K * L0TileShape::N * sizeof(ElementB) * STAGES;
    static constexpr uint32_t L0B_SIZE_FOR_A = L0TileShape::M * L0TileShape::K * sizeof(ElementA) * STAGES;
    
    static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;

    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE_FOR_A = L0A_SIZE_FOR_A / STAGES;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE_FOR_X = (L0A_SIZE_FOR_X + L0A_SIZE_FOR_VX) / STAGES;

    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE_FOR_B = L0B_SIZE_FOR_B / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE_FOR_A = L0B_SIZE_FOR_A / STAGES;
    

    // Check LayoutC
    static_assert(std::is_same_v<LayoutC, Catlass::layout::RowMajor>, "LayoutC only support RowMajor yet!");
    static_assert(std::is_same_v<LayoutY, Catlass::layout::RowMajor>, "LayoutY only support RowMajor yet!");

    // Check L1TileShape: 统一通过 A1 存储传输，所以要相加小于整体size
    static_assert((L1A_SIZE * STAGES + L1B_SIZE * STAGES + L1X_SIZE * STAGES + L1VX_SIZE * STAGES) <= ArchTag::L1_SIZE, 
        "L1TileShape exceeding the L1 space!");

    // Check L0TileShape
    static constexpr uint32_t L0A_TILE_SIZE = L0TileShape::M * L0TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L0X_TILE_SIZE = L0TileShapeforFT::N * L0TileShapeforFT::K * sizeof(ElementX);
    static constexpr uint32_t L0VX_TILE_SIZE = L1XAlignHelper::M_ALIGNED * L0TileShapeforFT::K * sizeof(ElementX);

    static constexpr uint32_t L0B_TILE_SIZE = L0TileShape::K * L0TileShape::N * sizeof(ElementB);

    static constexpr uint32_t L0C_SIZE_for_C = L0TileShape::M * L0TileShape::N * sizeof(ElementAccumulator);
    static constexpr uint32_t L0C_SIZE_for_ABE = L0TileShapeforFT::N * L0TileShape::M * sizeof(ElementAccumulator);
    static constexpr uint32_t L0C_SIZE_for_AE = L1XAlignHelper::M_ALIGNED * L0TileShape::M * sizeof(ElementAccumulator);

    static_assert((L0A_TILE_SIZE * STAGES) <= L0A_SIZE_FOR_A, "L0TileShape exceeding the space of L0A for A!");
    static_assert((L0X_TILE_SIZE * STAGES) <= L0A_SIZE_FOR_X, "L0TileShape exceeding the space of L0A for X of ABE!");
    static_assert((L0VX_TILE_SIZE * STAGES) <= L0A_SIZE_FOR_VX, "L0TileShape exceeding the space of L0A for X of AE!");

    static_assert((L0A_TILE_SIZE * STAGES + L0X_TILE_SIZE * STAGES + L0VX_TILE_SIZE * STAGES) <= L0A_SIZE, "L0TileShape exceeding the space of L0A!");
    static_assert((L0A_SIZE_FOR_A + L0A_SIZE_FOR_X + L0A_SIZE_FOR_VX) <= L0A_SIZE, "space for A, X for ABE and AE exceeding the L0A space!");
    

    static_assert((L0B_TILE_SIZE * STAGES) <= L0B_SIZE_FOR_B, "L0TileShape exceeding the space OF l0B for B!");
    static_assert((L0A_TILE_SIZE * STAGES) <= L0B_SIZE_FOR_A, "L0TileShape exceeding the space OF l0B for A!");
    static_assert((L0A_TILE_SIZE * STAGES + L0B_TILE_SIZE * STAGES) <= L0B_SIZE, "L0TileShape exceeding the space of L0B!");
    static_assert((L0B_SIZE_FOR_A + L0B_SIZE_FOR_B) <= L0B_SIZE, "space for B and A exceeding the L0B space!");

    static_assert((L0C_SIZE_for_C + L0C_SIZE_for_ABE + L0C_SIZE_for_AE) <= L0C_SIZE, "L0TileShape for C and ABe exceeding the global L0C space");


    static_assert(L1TileShape::M == L0TileShape::M && L1TileShape::N == L0TileShape::N && L1TileShapeforFT::N == L0TileShapeforFT::N && L1TileShapeforFT::M == L0TileShapeforFT::M,
        "The situation where the basic blocks of L1 and L0 differ on the m and n axes is not supported yet");

    static_assert(L1TileShape::K == L1TileShapeforFT::K,
        "The situation where the basic blocks of L1 for A/B and X differ on the K is not supported yet");

    static_assert(L1TileShape::M == L1TileShapeforFT::M,
        "The situation where the basic blocks of L1 for A and X differ on the M is not supported yet");

    static_assert(L0TileShape::K == L0TileShapeforFT::K,
        "The situation where the basic blocks of L0 for A/B and X differ on the K is not supported yet");

    static_assert(L0TileShapeforFT::N >= L1XAlignHelper::M_ALIGNED,
        "The situation where the L0TileShapeforFT::N < 16 is not supported yet");
    
    static_assert((L0TileShapeforFT::N % L1XAlignHelper::M_ALIGNED == 0),
        "The situation where the L0TileShapeforFT::N % 16 != 0 is not supported yet");

    // uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
    // uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShape.n());
    static_assert((L0TileShape::M % L1AAlignHelper::M_ALIGNED == 0),
        "The situation where the L0TileShape::M % 16 != 0 is not supported yet");
    
    static_assert((L0TileShape::N % L1BAlignHelper::N_ALIGNED == 0),
        "The situation where the L0TileShape::N % (ELE_NUM_PER_BLK) != 0 is not supported yet");
    
    /// Construct
    CATLASS_DEVICE
    BlockMmadFTABeNoSplitKRobust(Catlass::Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0)
    {   
        // block 上 L1 Cache 中 A1 和 B1 的起始地址偏移
        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + L1A_SIZE * STAGES;
        uint32_t l1XOffset = l1BufAddrStart + L1A_SIZE * STAGES + L1B_SIZE * STAGES;
        uint32_t l1VXOffset = l1BufAddrStart + L1A_SIZE * STAGES + L1B_SIZE * STAGES + L1X_SIZE * STAGES;

        uint32_t l0AOffsetforA = 0;
        uint32_t l0AOffsetforX = l0AOffsetforA + L0A_PINGPONG_BUF_SIZE_FOR_A * STAGES;
        
        uint32_t l0BOffsetforB = 0;
        uint32_t l0BOffsetforA = l0BOffsetforB + L0B_PINGPONG_BUF_SIZE_FOR_B * STAGES;

        uint32_t l0COffsetforC = 0;
        uint32_t l0COffsetforABe = l0COffsetforC + L0C_SIZE_for_C;

        // Init buffers
        for(uint32_t i=0; i < STAGES; i++) {
            // Catlass::Arch::Resource 封装了不同level Cache 上的 LocalTensorBuffer
            // Assign L1/L0A/L0B space for each stages
            /*
            LocalTensorBuffer<ArchTag, AscendC::TPosition::A1> l1Buf;
            */
            /*
            这里 L1 的存储共用 A1 位置，所以要注意起始地址要分隔开来，顺序访问
            */
            l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + L1A_SIZE * i);
            l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + L1B_SIZE * i);
            l1XTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementX>(l1XOffset + L1X_SIZE * i);
            l1VXTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementX>(l1VXOffset + L1VX_SIZE * i);

            /*
            L0 Cache 上面，分别使用 A2, B2 两个position，所以两个tensor的L0 Cache 均从0开始
            */
            l0ATensorListforA[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(l0AOffsetforA + L0A_PINGPONG_BUF_SIZE_FOR_A * i);
            l0ATensorListforX[i] = resource.l0ABuf.template GetBufferByByte<ElementX>(l0AOffsetforX + L0A_PINGPONG_BUF_SIZE_FOR_X * i);

            l0BTensorListforB[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(l0BOffsetforB + L0B_PINGPONG_BUF_SIZE_FOR_B * i);
            l0BTensorListforA[i] = resource.l0BBuf.template GetBufferByByte<ElementA>(l0BOffsetforA + L0B_PINGPONG_BUF_SIZE_FOR_A * i);

            // Assign event ID for each stages
            l1AEventList[i] = i;
            l1BEventList[i] = i + STAGES; // 保证同步时间编号不会冲突
            l1XEventList[i] = i + STAGES * 2;
            l1VXEventList[i] = i + STAGES * 3;

            l0AEventListforA[i] = i;
            l0AEventListforX[i] = i + STAGES * 2;
            l0BEventListforB[i] = i + STAGES;
            l0BEventListforA[i] = i + STAGES * 3;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1XEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1VXEventList[i]);

            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventListforA[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventListforB[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventListforX[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventListforA[i]);
        }
        l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(l0COffsetforC);
        l0CTensorforFT = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(l0COffsetforABe); 
        
        MMAD_EVENT_ID0 = 0;
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(MMAD_EVENT_ID0);

        MMAD_ABE_EVENT_ID1 = 1;
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(MMAD_ABE_EVENT_ID1);
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockMmadFTABeNoSplitKRobust()
    {
        for(uint32_t i=0; i < STAGES; i++){
            // 等待相关内存事件完成后再结束运行
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1XEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1VXEventList[i]);

            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventListforA[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventListforB[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventListforX[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventListforA[i]);
        }
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(MMAD_EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(MMAD_ABE_EVENT_ID1);
    }

    /// Perform a block-scoped matrix multiply-accumulate
    CATLASS_DEVICE
    void add_ae_op(
        AscendC::GlobalTensor<ElementA> const & gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementB> const & gmB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementC> const & gmC, LayoutC const &layoutC,
        AscendC::GlobalTensor<ElementX> const & gmX, LayoutX const &layoutX,
        AscendC::GlobalTensor<ElementX> const & gmVX, LayoutVX const &layoutVX,
        AscendC::GlobalTensor<ElementY> const & gmY, LayoutY const &layoutY,
        AscendC::GlobalTensor<ElementY> const & gmVY, LayoutY const &layoutVY,
        Catlass::GemmCoord const &actualShape, Catlass::GemvCoord &actualShapeforX)
    {   
        
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
        uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShape.n());

        uint32_t nRoundforFT = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShapeforX.m());

        auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
        auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);
        auto layoutXInL1 = LayoutXInL1::template MakeLayout<ElementX>(L1TileShapeforFT::N, L1TileShapeforFT::K);
        auto layoutVXInL1 = LayoutXInL1::template MakeLayout<ElementX>(L1XAlignHelper::M_ALIGNED, L1TileShapeforFT::K);
        
        auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(Catlass::MakeCoord(mRound, nRound));
        
        auto layoutInL0CTotalforFT = LayoutCInL0::MakeLayoutInL0C(Catlass::MakeCoord((nRoundforFT + L1XAlignHelper::M_ALIGNED), mRound));

        auto layoutInL0CforABe = layoutInL0CTotalforFT.GetTileLayout(Catlass::MatrixCoord(nRoundforFT, mRound));
        auto layoutInL0CforAe = layoutInL0CTotalforFT.GetTileLayout(Catlass::MatrixCoord(L1XAlignHelper::M_ALIGNED, mRound));

        uint32_t kActual = min(actualShape.k(), L1TileShape::K);

        // load first matrix A tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
        // 设定Tile 在global memory 中的layout, 即将一个Tile作为一个矩阵中的一部分，
        // 将其中的元素重新组织为与layoutA相同类型的的layout，其中shape为Tile规模
        // 但是每个元素/分形 行和列之间的 stride 还是按照原来整体layout的 shape/stride 来进行组织
        // 因为这里每个block是只输入并处理一个L1 Tile
        auto layoutTileA = layoutA.GetTileLayout(Catlass::MakeCoord(actualShape.m(),kActual));
        copyGmToL1A(l1ATensorList[l1ListId], gmA, layoutAInL1, layoutTileA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);

        // load first matrix B tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
        auto layoutTileB = layoutB.GetTileLayout(Catlass::MakeCoord(kActual,actualShape.n()));
        copyGmToL1B(l1BTensorList[l1ListId], gmB, layoutBInL1, layoutTileB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);

        // load first vector x tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1XEventList[l1ListId]);
        auto layoutTileX = layoutX.GetTileLayout(Catlass::MakeCoord(actualShapeforX.m(),kActual));
        copyGmToL1X(l1XTensorList[l1ListId], gmX, layoutXInL1, layoutTileX);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1XEventList[l1ListId]);

        // load first vector x tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1VXEventList[l1VXListId]);
        auto layoutTileVX = layoutVX.GetTileLayout(Catlass::MakeCoord(kActual));
        copyGmToL1VX(l1VXTensorList[l1VXListId], gmVX, layoutVXInL1, layoutTileVX);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1VXEventList[l1VXListId]);

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(MMAD_EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(MMAD_ABE_EVENT_ID1);
        }

        uint32_t mPartLoop = CeilDiv<L0TileShape::M>(mRound);
        uint32_t nPartLoop = CeilDiv<L0TileShape::N>(nRound);

        // main loop
        uint kTileCount = CeilDiv<L1TileShape::K>(actualShape.k());
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1VXEventList[l1VXListId]);
        for(uint32_t kLoopIdx=0; kLoopIdx < kTileCount; kLoopIdx++)
        {
            // 下一阶段执行的stage id
            uint32_t l1ListIdNext = (l1ListId + 1 < STAGES)? (l1ListId + 1) : 0;
            uint32_t kActualNext{0}; 

            // 流水线，提前将下一阶段的数据从 GM 加载到 L1 中与计算overlap
            // preload next tile from GM to L1
            if (kLoopIdx < kTileCount - 1){
                uint32_t kLoopIdxNext = kLoopIdx + 1;
                // 下一阶段 若非最后一个 loop，那么执行一个L1TileShape::K,否则执行剩余的数据
                kActualNext = (kLoopIdxNext < kTileCount - 1) ?
                    L1TileShape::K : (actualShape.k() - kLoopIdxNext * L1TileShape::K);
                
                // Get L1 Tensor for next stage
                auto l1ATensor = l1ATensorList[l1ListIdNext];
                auto l1BTensor = l1BTensorList[l1ListIdNext];
                auto l1XTensor = l1XTensorList[l1ListIdNext];
                // auto l1VXTensor = l1VXTensorList[l1ListIdNext];

                // Get GM tile for next stage
                Catlass::MatrixCoord gmTileAOffset{0, kLoopIdxNext * L1TileShape::K};
                Catlass::MatrixCoord gmTileBOffset{kLoopIdxNext * L1TileShape::K, 0};
                Catlass::MatrixCoord gmTileXOffset{0, kLoopIdxNext * L1TileShape::K};
                uint32_t gmTileVXOffset{kLoopIdxNext * L1TileShapeforFT::K};

                auto gmTileA = gmA[layoutA.GetOffset(gmTileAOffset)];
                auto gmTileB = gmB[layoutB.GetOffset(gmTileBOffset)];
                auto gmTileX = gmX[layoutX.GetOffset(gmTileXOffset)];

                // load next matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                layoutTileA = layoutA.GetTileLayout(Catlass::MakeCoord(actualShape.m(), kActualNext));
                copyGmToL1A(l1ATensor, gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                // load next matrix B tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                layoutTileB = layoutB.GetTileLayout(Catlass::MakeCoord(kActualNext, actualShape.n()));
                copyGmToL1B(l1BTensor, gmTileB, layoutBInL1, layoutTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);

                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1XEventList[l1ListIdNext]);
                auto layoutTileX = layoutX.GetTileLayout(Catlass::MakeCoord(actualShapeforX.m(),kActualNext));
                copyGmToL1X(l1XTensor, gmTileX, layoutXInL1, layoutTileX);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1XEventList[l1ListIdNext]);
            }

            // Get L1 Tensor for current usage
            auto l1ATensor = l1ATensorList[l1ListId];
            auto l1BTensor = l1BTensorList[l1ListId];
            auto l1XTensor = l1XTensorList[l1ListId];
            auto l1VXTensor = l1VXTensorList[l1VXListId];

            // Get the loop nums on L0
            uint32_t kPartLoop = CeilDiv<L0TileShape::K>(kActual);

            for(int mPartIdx=0; mPartIdx < mPartLoop; mPartIdx++){
                uint32_t mPartActual = (mPartIdx < mPartLoop - 1) ?
                    L0TileShape::M : (mRound - mPartIdx * L0TileShape::M);
                uint32_t mPartActualforX = nRoundforFT;

                for(int kPartIdx=0; kPartIdx < kPartLoop; kPartIdx++){
                    uint32_t kPartActual = (kPartIdx < kPartLoop - 1) ?
                        L0TileShape::K : (kActual - kPartIdx * L0TileShape::K);
                    
                    // Locate the current tile on L0A
                    auto l0ATileforA = l0ATensorListforA[l0AListIdforA];
                    auto l0BTileforA = l0BTensorListforA[l0BListIdforA];

                    LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mPartActual,kPartActual);
                    LayoutAInL0forFT layoutAInL0forFT = LayoutAInL0forFT::template MakeLayout<ElementA>(mPartActual,kPartActual);

                    // Locate the current tile of matrix A on L1
                    Catlass::MatrixCoord l1AOffset{mPartIdx * L0TileShape::M, kPartIdx * L0TileShape::K};

                    auto l1ATile = l1ATensor[layoutAInL1.GetOffset(l1AOffset)];
                    
                    if((mPartIdx == 0) && (kPartIdx == 0)){
                        // 若为当前stage第一次迭代，需要等待到第一批数据，
                        // 即在当前stage涉及的迭代前已经preload 完的A数据成功preload 到 L1 上才可进行A数据向 L0 上写
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
                    }
                    // 等待现在的L0阶段中之前在L0上的数据消费已经完成，即相应 MMAD 计算完成
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventListforA[l0AListIdforA]);
                    // Load current tile from L1 to L0A
                    copyL1ToL0A(l0ATileforA, l1ATile, layoutAInL0, layoutAInL1);

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventListforA[l0BListIdforA]);
                    copyL1ToL0AforFT(l0BTileforA, l1ATile, layoutAInL0forFT, layoutAInL1);

                    if ((mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1))
                    {
                        // 若这是当前stage 最后一次的迭代，则需要设置Flag，
                        // 即当前stage涉及到现有 A1的数据的操作已经完成
                        // 之后再进行PING-PONG preload时，可以向该阶段对应的L1 A1 数据中写入新的数据了
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                    }

                    for(int nPartIdx = 0; nPartIdx < nPartLoop; nPartIdx++){
                        uint32_t nPartActual = (nPartIdx < nPartLoop - 1) ?
                            L0TileShape::N : (nRound - nPartIdx * L0TileShape::N);
                        
                        // Locate the current tile on L0B
                        auto l0BTileforB = l0BTensorListforB[l0BListIdforB];
                        auto l0ATensorforX = l0ATensorListforX[l0AListIdforX];

                        LayoutBInL0 layoutBInL0 = 
                            LayoutBInL0::template MakeLayout<ElementB>(kPartActual, nPartActual); 

                        LayoutXInL0 layoutXInL0Total =
                            LayoutXInL0::template MakeLayout<ElementX>((mPartActualforX + L1XAlignHelper::M_ALIGNED), kPartActual); 

                        LayoutXInL0 layoutXInL0forABe = 
                            LayoutXInL0::template MakeLayout<ElementX>(mPartActualforX, kPartActual); 
                    
                        LayoutXInL0 layoutXInL0forAe = 
                            LayoutXInL0::template MakeLayout<ElementX>(L1XAlignHelper::M_ALIGNED, kPartActual);

                        // Locate the current tile of matrix X for ABe on L0
                        Catlass::MatrixCoord l0XOffsetforABe{nPartIdx * (mPartActualforX + L1XAlignHelper::M_ALIGNED), 0};

                        auto l0XTileforABe = l0ATensorforX[uint32_t(0)];

                        // Locate the current tile of matrix X for Ae on L0
                        Catlass::MatrixCoord l0XOffsetforAe{ 
                            nPartIdx * (mPartActualforX + L1XAlignHelper::M_ALIGNED) + mPartActualforX, 0};
                        
                        auto l0XTileforAe = l0ATensorforX[uint32_t(mPartActualforX) * uint32_t(kPartActual)];
                        // layoutXInL0Total.GetOffset(l0XOffsetforAe)
                        
                        // Locate the current tile of matrix B on L1
                        Catlass::MatrixCoord l1BOffset{kPartIdx * L0TileShape::K, 
                            nPartIdx * L0TileShape::N};
                        
                        auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BOffset)];

                        Catlass::MatrixCoord l1XOffsetforABe{0, kPartIdx * L0TileShape::K};
                        auto l1XTile = l1XTensor[layoutXInL1.GetOffset(l1XOffsetforABe)];

                        Catlass::MatrixCoord l1VXOffsetforAe{0, kPartIdx * L0TileShape::K};
                        auto l1VXTile = l1VXTensor[layoutVXInL1.GetOffset(l1VXOffsetforAe)];

                        // If the current tile is the first one on the k&n axis, wait for loading matrix B from GM to L1
                        if((nPartIdx == 0) && (kPartIdx == 0)){
                            // 若为当前stage第一次迭代，需要等待到第一批数据，
                            // 即在当前stage涉及的迭代前已经preload 完的B数据成功
                            // preload 到 L1 上才可进行B数据向 L0 上写
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1XEventList[l1ListId]);
                        }

                        // Wait for mmad finished
                        // 等待当前阶段L0 数据消费完成，即相应 MMAD 计算完成
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventListforB[l0BListIdforB]);
                        // Load current tile from L1 to L0B
                        copyL1ToL0B(l0BTileforB, l1BTile, layoutBInL0, layoutBInL1);

                        // Notify to do mmad
                        // 标记L0 Cache 中 A2 和 B2 的数据写入已经完成，可以进行 MMAD 运算了
                        // 同时也标记当前开始进行 L0 的 CO1 Cache 了。
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(MMAD_EVENT_ID0);

                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventListforX[l0AListIdforX]);
                        copyL1ToL0X(l0XTileforABe, l1XTile, layoutXInL0forABe, layoutXInL1);
                        copyL1ToL0X(l0XTileforAe, l1VXTile, layoutXInL0forAe, layoutVXInL1);

                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(MMAD_ABE_EVENT_ID1);

                        // If the current tile is the last one on the k&n axis, notify to load matrix B from GM to L1
                        if ((kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                            // 若这是当前stage 最后一次的迭代，则需要设置Flag，
                            // 即当前stage涉及到现有 L0 B的数据的操作已经完成
                            // 之后再进行PING-PONG preload时，可以向该阶段对应的L1 B 数据中写入新的数据了
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1XEventList[l1ListId]);
                        }

                        // Locate the current tile on L0C
                        Catlass::MatrixCoord l0COffset{mPartIdx * L0TileShape::M, nPartIdx * L0TileShape::N};
                        // 获取当前局部输出
                        auto l0CTile = l0CTensor[layoutInL0C.GetOffset(l0COffset)];

                        Catlass::MatrixCoord l0COffsetforFTTotal{nPartIdx * (mPartActualforX + L1XAlignHelper::M_ALIGNED), mPartIdx * L0TileShape::M};
                        auto l0CTileforFT = l0CTensorforFT[layoutInL0CTotalforFT.GetOffset(l0COffsetforFTTotal)];

                        // Compute the matrix multiplication on L0A and L0B and write the result to the accumulator
                        // Wait for loading L0B
                        // 等待 L0 上 A2 和 B2 的写入完成，当源数据写入完成后，即可进行矩阵运算了。
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(MMAD_EVENT_ID0);

                        // If the current tile is the first tile on the k axis, the accumulator needs to be reset to 0
                        // 当前 M,N Tile 的第一个 K 时，需要初始化输出的C矩阵为0

                        bool initC = ((kLoopIdx == 0) && (kPartIdx == 0));
                        // If the unit flag is enabled, the unit flag is set according to the calculation progress
                        uint8_t unitFlag = 0b00;
                        if constexpr (ENABLE_UNIT_FLAG) {
                            if ((kLoopIdx == kTileCount - 1) && (mPartIdx == mPartLoop - 1) &&
                                (kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                                unitFlag = 0b11;
                            } else {
                                unitFlag = 0b10;
                            }
                        }
                        // Perform calculation operations
                        tileMmad(l0CTile, l0ATileforA, l0BTileforB, mPartActual, 
                            nPartActual, kPartActual, initC, unitFlag);
                        
                        // Notify to move the next L0B tile
                        // 标记计算完成，即当前已经完成了一个l0 tile 的运算，可以加载下一个l0 tile了
                        // 这里最内层为 B 矩阵
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventListforB[l0BListIdforB]);
                        l0BListIdforB = (l0BListIdforB + 1 < STAGES) ? (l0BListIdforB + 1) : 0;

                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(MMAD_ABE_EVENT_ID1);

                        tileMmadforABe(l0CTileforFT, l0ATensorforX, l0BTileforA,
                            (mPartActualforX + L1XAlignHelper::M_ALIGNED), mPartActual, kPartActual, initC, unitFlag);

                        // Notify to move the next L0B tile
                        // 标记计算完成，即当前已经完成了一个l0 tile 的运算，可以加载下一个l0 tile了
                        // 这里最内层为 B 矩阵
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventListforX[l0AListIdforX]);
                        l0AListIdforX = (l0AListIdforX + 1 < STAGES) ? (l0AListIdforX + 1) : 0;
                    }
                    // 交替进行阶段，实现 L1 与 L0 之间数据传输与 MMAD 计算的PING-PANG
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventListforA[l0AListIdforA]);
                    l0AListIdforA = (l0AListIdforA + 1 < STAGES) ? (l0AListIdforA + 1) : 0;
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventListforA[l0BListIdforA]);
                    l0BListIdforA = (l0BListIdforA + 1 < STAGES) ? (l0BListIdforA + 1) : 0;
                }
            }
            // 交替进行阶段，实现L1 与 Global 之间数据传输 与 MMAD计算 的PING-PANG
            l1ListId = l1ListIdNext;
            kActual = kActualNext;
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1VXEventList[l1VXListId]);
        l1VXListId = (l1VXListId + 1 < STAGES)? (l1VXListId + 1) : 0;

        // copy block out
        // 将最终结果从 L0 的 CO1 输出到GM即可
        LayoutC layoutBlock = layoutC.GetTileLayout(actualShape.GetCoordMN());
        LayoutY layoutBlockforABe = layoutY.GetTileLayout(Catlass::MakeCoord(actualShapeforX.m(), actualShape.m()));
        LayoutY layoutBlockforAe = layoutVY.GetTileLayout(Catlass::MakeCoord(uint32_t(1), actualShape.m()));

        if constexpr (!ENABLE_UNIT_FLAG) {
            // 标记开始写入cGM 数据
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(MMAD_EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(MMAD_ABE_EVENT_ID1);
            // 等待允许写入开始
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(MMAD_EVENT_ID0);
            // 写入数据
            copyL0CToGm(gmC, l0CTensor, layoutBlock, layoutInL0C);

            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(MMAD_ABE_EVENT_ID1);

            Catlass::MatrixCoord l0COffsetforABe{uint32_t(0), uint32_t(0)};
            auto l0CTileforABe = l0CTensorforFT[layoutInL0CTotalforFT.GetOffset(l0COffsetforABe)];
            
            Catlass::MatrixCoord l0COffsetforAe{nRoundforFT, uint32_t(0)};
            // 获取当前局部输出
            auto l0CTileforAe = l0CTensorforFT[layoutInL0CTotalforFT.GetOffset(l0COffsetforAe)];

            copyL0CToGmforABE(gmY, l0CTileforABe, layoutBlockforABe, layoutInL0CforABe);
            copyL0CToGmforABE(gmVY, l0CTileforAe, layoutBlockforAe, layoutInL0CforAe);
            // 标记数据写入 cGM 已经完成
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(MMAD_EVENT_ID0);
            // 标记数据写入 cGM 已经完成
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(MMAD_ABE_EVENT_ID1);
        } else {

            Catlass::MatrixCoord l0COffsetforABe{uint32_t(0), uint32_t(0)};
            auto l0CTileforABe = l0CTensorforFT[layoutInL0CTotalforFT.GetOffset(l0COffsetforABe)];
            
            Catlass::MatrixCoord l0COffsetforAe{nRoundforFT, uint32_t(0)};
            // 获取当前局部输出
            auto l0CTileforAe = l0CTensorforFT[layoutInL0CTotalforFT.GetOffset(l0COffsetforAe)];

            copyL0CToGm(gmC, l0CTensor, layoutBlock, layoutInL0C, 0b11);
            
            copyL0CToGmforABE(gmY, l0CTileforABe, layoutBlockforABe, layoutInL0CforABe, 0b11);
            copyL0CToGmforABE(gmVY, l0CTileforAe, layoutBlockforAe, layoutInL0CforAe, 0b11);
        }
    }

    /// Perform a block-scoped matrix multiply-accumulate
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const & gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementB> const & gmB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementC> const & gmC, LayoutC const &layoutC,
        AscendC::GlobalTensor<ElementX> const & gmX, LayoutX const &layoutX,
        AscendC::GlobalTensor<ElementY> const & gmY, LayoutY const &layoutY,
        Catlass::GemmCoord const &actualShape, Catlass::GemvCoord &actualShapeforX)
    {   
        
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
        uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShape.n());

        uint32_t nRoundforFT = RoundUp<L1XAlignHelper::M_ALIGNED>(actualShapeforX.m());

        auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
        auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);
        auto layoutXInL1 = LayoutXInL1::template MakeLayout<ElementX>(L1TileShapeforFT::N, L1TileShapeforFT::K);

        auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(Catlass::MakeCoord(mRound, nRound));
        auto layoutInL0CforABe = LayoutYInL0::MakeLayoutInL0C(Catlass::MatrixCoord(nRoundforFT, mRound));

        uint32_t kActual = min(actualShape.k(), L1TileShape::K);

        // load first matrix A tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
        // 设定Tile 在global memory 中的layout, 即将一个Tile作为一个矩阵中的一部分，
        // 将其中的元素重新组织为与layoutA相同类型的的layout，其中shape为Tile规模
        // 但是每个元素/分形 行和列之间的 stride 还是按照原来整体layout的 shape/stride 来进行组织
        // 因为这里每个block是只输入并处理一个L1 Tile
        auto layoutTileA = layoutA.GetTileLayout(Catlass::MakeCoord(actualShape.m(),kActual));
        copyGmToL1A(l1ATensorList[l1ListId], gmA, layoutAInL1, layoutTileA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);

        // load first matrix B tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
        auto layoutTileB = layoutB.GetTileLayout(Catlass::MakeCoord(kActual,actualShape.n()));
        copyGmToL1B(l1BTensorList[l1ListId], gmB, layoutBInL1, layoutTileB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);

        // load first vector x tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1XEventList[l1ListId]);
        auto layoutTileX = layoutX.GetTileLayout(Catlass::MakeCoord(actualShapeforX.m(),kActual));
        copyGmToL1X(l1XTensorList[l1ListId], gmX, layoutXInL1, layoutTileX);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1XEventList[l1ListId]);

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(MMAD_EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(MMAD_ABE_EVENT_ID1);
        }

        uint32_t mPartLoop = CeilDiv<L0TileShape::M>(mRound);
        uint32_t nPartLoop = CeilDiv<L0TileShape::N>(nRound);

        // main loop
        uint kTileCount = CeilDiv<L1TileShape::K>(actualShape.k());
        for(uint32_t kLoopIdx=0; kLoopIdx < kTileCount; kLoopIdx++)
        {
            // 下一阶段执行的stage id
            uint32_t l1ListIdNext = (l1ListId + 1 < STAGES)? (l1ListId + 1) : 0;
            uint32_t kActualNext{0}; 

            // 流水线，提前将下一阶段的数据从 GM 加载到 L1 中与计算overlap
            // preload next tile from GM to L1
            if (kLoopIdx < kTileCount - 1){
                uint32_t kLoopIdxNext = kLoopIdx + 1;
                // 下一阶段 若非最后一个 loop，那么执行一个L1TileShape::K,否则执行剩余的数据
                kActualNext = (kLoopIdxNext < kTileCount - 1) ?
                    L1TileShape::K : (actualShape.k() - kLoopIdxNext * L1TileShape::K);
                
                // Get L1 Tensor for next stage
                auto l1ATensor = l1ATensorList[l1ListIdNext];
                auto l1BTensor = l1BTensorList[l1ListIdNext];
                auto l1XTensor = l1XTensorList[l1ListIdNext];

                // Get GM tile for next stage
                Catlass::MatrixCoord gmTileAOffset{0, kLoopIdxNext * L1TileShape::K};
                Catlass::MatrixCoord gmTileBOffset{kLoopIdxNext * L1TileShape::K, 0};
                Catlass::MatrixCoord gmTileXOffset{0, kLoopIdxNext * L1TileShape::K};
                // uint32_t gmTilexOffsetNext{kLoopIdxNext * L1TileShape::K};

                auto gmTileA = gmA[layoutA.GetOffset(gmTileAOffset)];
                auto gmTileB = gmB[layoutB.GetOffset(gmTileBOffset)];
                auto gmTileX = gmX[layoutX.GetOffset(gmTileXOffset)];

                // load next matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                layoutTileA = layoutA.GetTileLayout(Catlass::MakeCoord(actualShape.m(), kActualNext));
                copyGmToL1A(l1ATensor, gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                // load next matrix B tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                layoutTileB = layoutB.GetTileLayout(Catlass::MakeCoord(kActualNext, actualShape.n()));
                copyGmToL1B(l1BTensor, gmTileB, layoutBInL1, layoutTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);

                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1XEventList[l1ListIdNext]);
                auto layoutTileX = layoutX.GetTileLayout(Catlass::MakeCoord(actualShapeforX.m(),kActualNext));
                copyGmToL1X(l1XTensor, gmTileX, layoutXInL1, layoutTileX);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1XEventList[l1ListIdNext]);
            }

            // Get L1 Tensor for current usage
            auto l1ATensor = l1ATensorList[l1ListId];
            auto l1BTensor = l1BTensorList[l1ListId];
            auto l1XTensor = l1XTensorList[l1ListId];

            // Get the loop nums on L0
            uint32_t kPartLoop = CeilDiv<L0TileShape::K>(kActual);

            for(int mPartIdx=0; mPartIdx < mPartLoop; mPartIdx++){
                uint32_t mPartActual = (mPartIdx < mPartLoop - 1) ?
                    L0TileShape::M : (mRound - mPartIdx * L0TileShape::M);
                uint32_t mPartActualforX = nRoundforFT;

                for(int kPartIdx=0; kPartIdx < kPartLoop; kPartIdx++){
                    uint32_t kPartActual = (kPartIdx < kPartLoop - 1) ?
                        L0TileShape::K : (kActual - kPartIdx * L0TileShape::K);
                    
                    // Locate the current tile on L0A
                    auto l0ATileforA = l0ATensorListforA[l0AListIdforA];
                    auto l0BTileforA = l0BTensorListforA[l0BListIdforA];

                    LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mPartActual,kPartActual);
                    LayoutAInL0forFT layoutAInL0forFT = LayoutAInL0forFT::template MakeLayout<ElementA>(mPartActual,kPartActual);

                    // Locate the current tile of matrix A on L1
                    Catlass::MatrixCoord l1AOffset{mPartIdx * L0TileShape::M, kPartIdx * L0TileShape::K};

                    auto l1ATile = l1ATensor[layoutAInL1.GetOffset(l1AOffset)];
                    
                    if((mPartIdx == 0) && (kPartIdx == 0)){
                        // 若为当前stage第一次迭代，需要等待到第一批数据，
                        // 即在当前stage涉及的迭代前已经preload 完的A数据成功preload 到 L1 上才可进行A数据向 L0 上写
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
                    }
                    // 等待现在的L0阶段中之前在L0上的数据消费已经完成，即相应 MMAD 计算完成
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventListforA[l0AListIdforA]);

                    // Load current tile from L1 to L0A
                    copyL1ToL0A(l0ATileforA, l1ATile, layoutAInL0, layoutAInL1);

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventListforA[l0BListIdforA]);
                    copyL1ToL0AforFT(l0BTileforA, l1ATile, layoutAInL0forFT, layoutAInL1);

                    if ((mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1))
                    {
                        // 若这是当前stage 最后一次的迭代，则需要设置Flag，
                        // 即当前stage涉及到现有 A1的数据的操作已经完成
                        // 之后再进行PING-PONG preload时，可以向该阶段对应的L1 A1 数据中写入新的数据了
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                    }

                    for(int nPartIdx = 0; nPartIdx < nPartLoop; nPartIdx++){
                        uint32_t nPartActual = (nPartIdx < nPartLoop - 1) ?
                            L0TileShape::N : (nRound - nPartIdx * L0TileShape::N);
                        
                        // Locate the current tile on L0B
                        auto l0BTileforB = l0BTensorListforB[l0BListIdforB];
                        auto l0ATileforX = l0ATensorListforX[l0AListIdforX];

                        LayoutBInL0 layoutBInL0 = 
                            LayoutBInL0::template MakeLayout<ElementB>(kPartActual, nPartActual);
                        
                        LayoutXInL0 layoutXInL0 =
                        LayoutXInL0::template MakeLayout<ElementX>(mPartActualforX, kPartActual); 
                        
                        // Locate the current tile of matrix B on L1
                        Catlass::MatrixCoord l1BOffset{kPartIdx * L0TileShape::K, 
                            nPartIdx * L0TileShape::N};
                        
                        auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BOffset)];

                        Catlass::MatrixCoord l1XOffset{0, kPartIdx * L0TileShape::K};
                        auto l1XTile = l1XTensor[layoutXInL1.GetOffset(l1XOffset)];

                        
                        // If the current tile is the first one on the k&n axis, wait for loading matrix B from GM to L1
                        if((nPartIdx == 0) && (kPartIdx == 0)){
                            // 若为当前stage第一次迭代，需要等待到第一批数据，
                            // 即在当前stage涉及的迭代前已经preload 完的B数据成功
                            // preload 到 L1 上才可进行B数据向 L0 上写
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1XEventList[l1ListId]);
                        }

                        // Wait for mmad finished
                        // 等待当前阶段L0 数据消费完成，即相应 MMAD 计算完成
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventListforB[l0BListIdforB]);
                        // Load current tile from L1 to L0B
                        copyL1ToL0B(l0BTileforB, l1BTile, layoutBInL0, layoutBInL1);

                        // Notify to do mmad
                        // 标记L0 Cache 中 A2 和 B2 的数据写入已经完成，可以进行 MMAD 运算了
                        // 同时也标记当前开始进行 L0 的 CO1 Cache 了。
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(MMAD_EVENT_ID0);

                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventListforX[l0AListIdforX]);
                        copyL1ToL0X(l0ATileforX, l1XTile, layoutXInL0, layoutXInL1);

                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(MMAD_ABE_EVENT_ID1);

                        // If the current tile is the last one on the k&n axis, notify to load matrix B from GM to L1
                        if ((kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                            // 若这是当前stage 最后一次的迭代，则需要设置Flag，
                            // 即当前stage涉及到现有 L0 B的数据的操作已经完成
                            // 之后再进行PING-PONG preload时，可以向该阶段对应的L1 B 数据中写入新的数据了
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1XEventList[l1ListId]);
                        }

                        // Locate the current tile on L0C
                        Catlass::MatrixCoord l0COffset{mPartIdx * L0TileShape::M, nPartIdx * L0TileShape::N};
                        // 获取当前局部输出
                        auto l0CTile = l0CTensor[layoutInL0C.GetOffset(l0COffset)];

                        Catlass::MatrixCoord l0COffsetforABe{nPartIdx * mPartActualforX, mPartIdx * L0TileShape::M};
                        auto l0CTileforABe = l0CTensorforFT[layoutInL0CforABe.GetOffset(l0COffsetforABe)];

                        // Compute the matrix multiplication on L0A and L0B and write the result to the accumulator
                        // Wait for loading L0B
                        // 等待 L0 上 A2 和 B2 的写入完成，当源数据写入完成后，即可进行矩阵运算了。
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(MMAD_EVENT_ID0);

                        // If the current tile is the first tile on the k axis, the accumulator needs to be reset to 0
                        // 当前 M,N Tile 的第一个 K 时，需要初始化输出的C矩阵为0

                        bool initC = ((kLoopIdx == 0) && (kPartIdx == 0));
                        // If the unit flag is enabled, the unit flag is set according to the calculation progress
                        uint8_t unitFlag = 0b00;
                        if constexpr (ENABLE_UNIT_FLAG) {
                            if ((kLoopIdx == kTileCount - 1) && (mPartIdx == mPartLoop - 1) &&
                                (kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                                unitFlag = 0b11;
                            } else {
                                unitFlag = 0b10;
                            }
                        }
                        // Perform calculation operations
                        tileMmad(l0CTile, l0ATileforA, l0BTileforB, mPartActual, 
                            nPartActual, kPartActual, initC, unitFlag);
                        
                        // Notify to move the next L0B tile
                        // 标记计算完成，即当前已经完成了一个l0 tile 的运算，可以加载下一个l0 tile了
                        // 这里最内层为 B 矩阵
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventListforB[l0BListIdforB]);
                        l0BListIdforB = (l0BListIdforB + 1 < STAGES) ? (l0BListIdforB + 1) : 0;

                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(MMAD_ABE_EVENT_ID1);
                        // tileMmad(l0CTile, l0ATile, l0BTile, L1XAlignHelper::M_ALIGNED, m_actual_now, nPartActual, initC);
                        tileMmadforABe(l0CTileforABe, l0ATileforX, l0BTileforA,
                            mPartActualforX, mPartActual, kPartActual,initC,unitFlag);
                        // Notify to move the next L0B tile
                        // 标记计算完成，即当前已经完成了一个l0 tile 的运算，可以加载下一个l0 tile了
                        // 这里最内层为 B 矩阵
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventListforX[l0AListIdforX]);
                        l0AListIdforX = (l0AListIdforX + 1 < STAGES) ? (l0AListIdforX + 1) : 0;
                    }
                    // 交替进行阶段，实现 L1 与 L0 之间数据传输与 MMAD 计算的PING-PANG
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventListforA[l0AListIdforA]);
                    l0AListIdforA = (l0AListIdforA + 1 < STAGES) ? (l0AListIdforA + 1) : 0;
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventListforA[l0BListIdforA]);
                    l0BListIdforA = (l0BListIdforA + 1 < STAGES) ? (l0BListIdforA + 1) : 0;
                }
            }
            // 交替进行阶段，实现L1 与 Global 之间数据传输 与 MMAD计算 的PING-PANG
            l1ListId = l1ListIdNext;
            kActual = kActualNext;
        }

        // copy block out
        // 将最终结果从 L0 的 CO1 输出到GM即可
        LayoutC layoutBlock = layoutC.GetTileLayout(actualShape.GetCoordMN());
        LayoutY layoutBlockforABe = layoutY.GetTileLayout(Catlass::MakeCoord(actualShapeforX.m(), actualShape.m()));

        if constexpr (!ENABLE_UNIT_FLAG) {
            // 标记开始写入cGM 数据
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(MMAD_EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(MMAD_ABE_EVENT_ID1);
            // 等待允许写入开始
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(MMAD_EVENT_ID0);
            // 写入数据
            copyL0CToGm(gmC, l0CTensor, layoutBlock, layoutInL0C);

            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(MMAD_ABE_EVENT_ID1);
            copyL0CToGmforABE(gmY, l0CTensorforFT, layoutBlockforABe, layoutInL0CforABe);
            // 标记数据写入 cGM 已经完成
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(MMAD_EVENT_ID0);
            // 标记数据写入 cGM 已经完成
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(MMAD_ABE_EVENT_ID1);
        } else {
            copyL0CToGm(gmC, l0CTensor, layoutBlock, layoutInL0C, 0b11);
            copyL0CToGmforABE(gmY, l0CTensorforFT, layoutBlockforABe, layoutInL0CforABe, 0b11);
        }
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> l1ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList[STAGES];
    AscendC::LocalTensor<ElementX> l1XTensorList[STAGES];
    AscendC::LocalTensor<ElementX> l1VXTensorList[STAGES];

    AscendC::LocalTensor<ElementA> l0ATensorListforA[STAGES];
    AscendC::LocalTensor<ElementX> l0ATensorListforX[STAGES];

    AscendC::LocalTensor<ElementB> l0BTensorListforB[STAGES];
    AscendC::LocalTensor<ElementA> l0BTensorListforA[STAGES];

    AscendC::LocalTensor<ElementAccumulator> l0CTensor;
    AscendC::LocalTensor<ElementAccumulator> l0CTensorforFT;

    // Multi-stage event id list
    int32_t l1AEventList[STAGES];
    int32_t l1BEventList[STAGES];
    int32_t l1XEventList[STAGES];
    int32_t l1VXEventList[STAGES];

    int32_t l0AEventListforA[STAGES];
    int32_t l0AEventListforX[STAGES];

    int32_t l0BEventListforB[STAGES];
    int32_t l0BEventListforA[STAGES];

    // The id of current stage
    // 指示当前所处的pipeline 中的阶段（双阶段PING-PONG）
    uint32_t l1ListId{0};
    uint32_t l1VXListId{0};

    uint32_t l0AListIdforA{0};
    uint32_t l0AListIdforX{0};

    uint32_t l0BListIdforB{0};
    uint32_t l0BListIdforA{0};

    int32_t MMAD_EVENT_ID0;
    int32_t MMAD_ABE_EVENT_ID1;

    TileMmad tileMmad;
    TileMmad tileMmadforABe;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;

    CopyGmToL1VX copyGmToL1VX;
    CopyGmToL1X copyGmToL1X;
    CopyL1ToL0X copyL1ToL0X;
    CopyL1ToL0AforFT copyL1ToL0AforFT;
    CopyL0CToGmforABE copyL0CToGmforABE;
};
}

#endif // CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_HPP