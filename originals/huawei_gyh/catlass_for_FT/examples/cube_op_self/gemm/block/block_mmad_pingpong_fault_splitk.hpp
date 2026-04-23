
#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_HPP_FT_SPLITK_SELF
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_HPP_FT_SPLITK_SELF

# include "catlass/catlass.hpp"
# include "catlass/arch/resource.hpp"
# include "catlass/arch/cross_core_sync.hpp"
# include "catlass/coord.hpp"
# include "catlass/gemm_coord.hpp"
# include "gemm/dispatch_policy.hpp"
# include "catlass/gemm/helper.hpp"
# include "catlass/gemm/tile/tile_copy.hpp"
# include "catlass/gemm/tile/tile_mmad.hpp"



namespace CubeSelf::Gemm::Block {
template<
    bool ENABLE_UNIT_FLAG_,
    Catlass::Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    Catlass::Gemv::helper::FT_L02L1_TYPE COPY_TYPE_,
    class L1TileShape_,
    class L0TileShape_,
    class L0TileShapeforFT_,
    class AType_,
    class BType_,
    class CType_,
    class XType_,
    class YType_,
    class BiasType_,
    class TileCopyFT_,
    class TileMmad_
>
struct BlockMmadFTSpiltK<
    CubeSelf::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>,
    ENC_TYPE_,
    COPY_TYPE_,
    L1TileShape_,
    L0TileShape_,
    L0TileShapeforFT_,
    AType_,
    BType_,
    CType_,
    XType_,
    YType_,
    BiasType_,
    TileCopyFT_,
    TileMmad_
>{
public:
    /*
    template<
    class DispatchPolicy,
    class L1TileShape,
    class L0TileShape,
    class L0TileShapeforFT,
    class AType,
    class BType,
    class CType,
    class XType,
    class YType,
    class BiasType = void,
    class TileCopy = CubeSelf::Gemm::Tile::TileCopyFT<typename DispatchPolicy::ArchTag, AType, BType, CType, BiasType>,
    class TileMmad = CubeSelf::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>
    >
    struct BlockMmadFTSpiltK
    */
    // Type Aliases
    using DispatchPolicy = MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_;
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

    using ElementY = typename YType_::Element;
    using LayoutY = typename YType_::Layout;

    using TileMmad = TileMmad_;

    using ElementCInL1 = typename XType_::Element;
    using CTypeForL1 = Catlass::Gemm::GemmType<ElementCInL1,LayoutC>;
    
    using TileMmadforFT = Catlass::Gemm::Tile::TileMmad<ArchTag, XType_, CTypeForL1, BiasType_>;
    
    using CopyGmToL1A = typename TileCopyFT_::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopyFT_::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopyFT_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopyFT_::CopyL1ToL0B;

    using CopyL0CToGm = typename TileCopyFT_::CopyL0CToGm;

    using CopyGmToL1X = typename TileCopyFT_::CopyGmToL1X;
    using CopyL1ToL0X = typename TileCopyFT_::CopyL1ToL0X;

    using CopyL0CToL1forFT = typename TileCopyFT_::CopyL0CToL1forFT;
    using CopyL1ToL0CforFT = typename TileCopyFT_::CopyL1ToL0CforFT;
    using CopyL0CToGmforFT = typename TileCopyFT_::CopyL0CToGmforFT;


    using ElementAccumulator = 
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA,ElementB>::ElementAccumulator;

    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;
    using LayoutXInL1 = typename CopyL1ToL0X::LayoutSrc;
    using LayoutCInL1forFT = typename CopyL1ToL0CforFT::LayoutSrc;

    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
    using LayoutXInL0 = typename CopyL1ToL0X::LayoutDst;
    using LayoutCInL0forFT = typename CopyL1ToL0CforFT::LayoutDst;

    using LayoutCInL0 = Catlass::layout::zN;
    using LayoutYInL0 = Catlass::layout::zN;

    using L1AAlignHelper = Catlass::Gemm::helper::L1AlignHelper<ElementA, LayoutA>;
    using L1BAlignHelper = Catlass::Gemm::helper::L1AlignHelper<ElementB, LayoutB>;

    using L1BAlignHelperforFT = Catlass::Gemm::helper::L1AlignHelper<ElementY, LayoutYInL0>;

    using FT_ENC_TYPE = Catlass::Gemv::helper::FT_ENC_TYPE;

    using L1XAlignHelper = Catlass::Gemv::helper::L1AlignHelper<ElementX, LayoutX>;

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;

    static constexpr uint32_t L1A_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L1C_SIZE = L1TileShape::M * L1TileShape::N * sizeof(ElementX);

    static constexpr uint32_t L1X_SIZE = 16 * L1TileShape::N * sizeof(ElementX);
    static constexpr uint32_t ELEM_PER_CO_X_FT = Catlass::BYTE_PER_C0 / sizeof(ElementX);
    
    static constexpr uint32_t L1_SYNC_SIZE = 16 * ELEM_PER_CO_X_FT * sizeof(ElementX);
    static constexpr uint32_t L1FT_SIZE = L1C_SIZE;
    // ( > L1X_SIZE) ? L1C_SIZE : L1X_SIZE;

    static constexpr uint32_t L1B_SIZE = L1TileShape::K * L1TileShape::N * sizeof(ElementB);

    static constexpr uint32_t L0A_TOTAL_SIZE = ArchTag::L0A_SIZE;
    static constexpr uint32_t L0A_SIZE = L0TileShape::M * L0TileShape::K * sizeof(ElementA) * STAGES;
    static constexpr uint32_t L0X_SIZE = L1X_SIZE;
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;

    static constexpr uint32_t L0_M_NUM_FOR_FT = (L0TileShapeforFT::M + Catlass::C0_NUM_PER_FRACTAL- 1) / Catlass::C0_NUM_PER_FRACTAL;

    static constexpr uint32_t L0_M_ALIGNED_FT = (L0TileShapeforFT::M % Catlass::C0_NUM_PER_FRACTAL == 0)? L0TileShapeforFT::M : L0_M_NUM_FOR_FT * Catlass::C0_NUM_PER_FRACTAL;


    static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
    static constexpr uint32_t L0C_TILE_NUM_FT = (L1TileShape::M + L0_M_ALIGNED_FT - 1) / L0_M_ALIGNED_FT;
    static constexpr uint32_t L0C_TILE_STAGES = (L0C_TILE_NUM_FT <= STAGES) ? L0C_TILE_NUM_FT : STAGES;

    static constexpr uint32_t TOTAL_C_STAGES_FT = STAGES * L0C_TILE_STAGES;
    
    static constexpr uint32_t L0C_TILE_SIZE_FT = L1XAlignHelper::M_ALIGNED * L0_M_ALIGNED_FT; 
    static constexpr uint32_t L0C_TILE_SIZE_MMA = L1TileShape::M * L1TileShape::N; 

    static constexpr uint32_t L0C_TILE_BUFFER_SIZE_FT = L0C_TILE_SIZE_FT * sizeof(ElementAccumulator);
    static constexpr uint32_t L0C_TILE_BUFFER_SIZE_MMA = L0C_TILE_SIZE_MMA * sizeof(ElementAccumulator);
    

    //  * STAGES
    // Check LayoutC
    static_assert(std::is_same_v<LayoutC, Catlass::layout::RowMajor>, "LayoutC only support RowMajor yet!");

    // Check L1TileShape: 统一通过 A1 存储传输，所以要相加小于整体size
    static_assert((L1A_SIZE * STAGES + L1B_SIZE * STAGES + L1FT_SIZE * STAGES + L1_SYNC_SIZE * STAGES) <= ArchTag::L1_SIZE, 
        "L1TileShape exceeding the L1 space!");

    // Check L0TileShape
    static constexpr uint32_t L0A_TILE_SIZE = L0TileShape::M * L0TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0TileShape::K * L0TileShape::N * sizeof(ElementB);

    static_assert((L0A_TILE_SIZE * STAGES) <= L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert((L0A_TILE_SIZE * STAGES + L0X_SIZE)<=L0A_TOTAL_SIZE, "L0TileShape exceeding the L0A space because of storing X");
    static_assert((L0B_TILE_SIZE * STAGES) <= L0B_SIZE, "L0TileShape exceeding the L0B space!");

    static_assert(L1TileShape::M == L0TileShape::M && L1TileShape::N == L0TileShape::N,
        "The situation where the basic blocks of L1 and L0 differ on the m and n axes is not supported yet");
    
    static_assert(L0TileShapeforFT::N == L1TileShape::N,
        "The situation where the basic blocks of L0 for FT and L0 for MMA differ on the n axes is not supported yet");

    static_assert(L0TileShapeforFT::K % L1TileShape::K == 0,
        "L0 Tile Shape for FT on K axis should be multiple of L1 Tile Shape K!");

    static_assert((L0C_TILE_BUFFER_SIZE_MMA + (L0C_TILE_BUFFER_SIZE_FT * TOTAL_C_STAGES_FT)) <= L0C_SIZE,
        "L0C Tile Shape for FT exceeding the L0C space!");

    CATLASS_DEVICE
    BlockMmadFTSpiltK(Catlass::Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0)
    {   
        // block 上 L1 Cache 中 A1 和 B1 的起始地址偏移
        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + L1A_SIZE * STAGES;
        uint32_t l1FTOffset = l1BufAddrStart + L1A_SIZE * STAGES + L1B_SIZE * STAGES;
        uint32_t l1FTSyncOffset = l1BufAddrStart + L1A_SIZE * STAGES + L1B_SIZE * STAGES + L1FT_SIZE * STAGES;

        uint32_t l0COffsetMMA = 0;
        uint32_t l0COffsetFT = L0C_TILE_BUFFER_SIZE_MMA;

        uint32_t l0XOffset = L0A_SIZE;

        FT_K_INTERVAL = L0TileShapeforFT::K / L1TileShape::K;

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

            l1FTTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementX>(l1FTOffset + L1FT_SIZE * i);
            l1FTSyncTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementX>(l1FTSyncOffset + L1_SYNC_SIZE * i);

            /*
            L0 Cache 上面，分别使用 A2, B2 两个position，所以两个tensor的L0 Cache 均从0开始
            */
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0CTensorFTList[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(l0COffsetFT + L0C_TILE_BUFFER_SIZE_FT * L0C_TILE_STAGES * i);

            l0FTTensorList[i] = l0BTensorList[i].template ReinterpretCast<ElementX>();

            // Assign event ID for each stages
            l1AEventList[i] = i;
            l1BEventList[i] = i + STAGES; // 保证同步时间编号不会冲突
            l1FTEventList[i] = i + (STAGES * 2) + 1;
             
            l0AEventList[i] = i;
            l0BEventList[i] = i + STAGES;

            
            // l1FTEventList[i] = i + STAGES * 2;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);

            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1FTEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_FIX>(l1FTEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l1FTEventList[i]);
            
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }

        l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(l0COffsetMMA);
        
        MMAD_EVENT_ID0 = 0;
        l1FTEventforX = 4;
        FT_EVENT_ID1 = 7;
        
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l1FTEventforX);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(MMAD_EVENT_ID0);
        

        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(FT_EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(FT_EVENT_ID1);


        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FT_EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(FT_EVENT_ID1);

        l0XTensor = resource.l0ABuf.template GetBufferByByte<ElementX>(l0XOffset);
        // AscendC::SetFlag<AscendC::HardEvent::>

        for(uint32_t j = 0; j < L0C_TILE_STAGES; j++){
            l0CforFTEventList[j] = MMAD_EVENT_ID0 + (STAGES) + j;
            // AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CforFTEventList[j]);
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockMmadFTSpiltK()
    {
        for(uint32_t i=0; i < STAGES; i++){
            // 等待相关内存事件完成后再结束运行
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1FTEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_FIX>(l1FTEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l1FTEventList[i]);

            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l1FTEventforX);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(MMAD_EVENT_ID0);
        
        
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(FT_EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(FT_EVENT_ID1);

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FT_EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(FT_EVENT_ID1);
    }

    // Copy Vector to the L0X for the GEMV
    CATLASS_DEVICE
    void FillL0XWithVec(
        AscendC::GlobalTensor<ElementX> const& gmBlockX, LayoutX const& layoutX,
        Catlass::GemmCoord const& actualShape)
    {
        auto layoutXInL1 = LayoutXInL1::template MakeLayout<ElementX>(L1XAlignHelper::M_ALIGNED, L1TileShape::N);
        uint32_t shuffleKIdx = 0;
        uint32_t gmTilexOffset{shuffleKIdx * L1TileShape::N};
        uint32_t nActual = L1TileShape::N;
        uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(nActual);

        auto l1XTensorforFT = l1ATensorList[l1ListId].template ReinterpretCast<ElementX>();

        auto gmTilex = gmBlockX[gmTilexOffset];

         // load first vector x tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(FT_EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);

        auto layoutTilex = layoutX.GetTileLayout(Catlass::MakeCoord(nRound));
        copyGmToL1X(l1XTensorforFT, gmTilex, layoutXInL1, layoutTilex);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1FTEventforX);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1FTEventforX);

        uint32_t nPartLoop = 1;
        uint32_t nPartActual = L1TileShape::N;
        uint32_t nPartIdx = 0;

        /*
        MatrixCoord l1xOffset{0, nPartIdx * L0TileNsize};
        auto l1ATile = l1ATensor[layoutXInL1.GetOffset(l1xOffset)];
        */
        
        LayoutXInL0 layoutxInL0 =
            LayoutXInL0::template MakeLayout<ElementX>(L1XAlignHelper::M_ALIGNED, nPartActual);

        Catlass::MatrixCoord l1xOffset{0, nPartIdx * L1TileShape::N};
        // auto l0ATile = l0XTensor;

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l1FTEventforX);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
        // Load current tile from L1 to L0A
        copyL1ToL0X(l0XTensor, l1XTensorforFT, layoutxInL0, layoutXInL1);
        // AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l1FTEventforX);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);

        // AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(FT_EVENT_ID1);
        // l1FTListId = (l1FTListId + 1) % STAGES;
    }

    CATLASS_DEVICE
    void RowSum(
        AscendC::GlobalTensor<ElementY> const& gmBlockY,
        AscendC::GlobalTensor<ElementX> const& gmBlockXSync, 
        LayoutY const& layoutY, LayoutX const& layoutX,
        Catlass::GemmCoord const& actualShape, 
        Catlass::GemmCoord const& actualCoord,
        Catlass::GemvCoord const& totalSliceShape,
        uint32_t RowSumLastId,uint32_t CESliceLastId, bool isLastRowSum)
    {
        auto layoutXSyncInL1 = LayoutXInL1::template MakeLayout<ElementX>(L1XAlignHelper::M_ALIGNED, ELEM_PER_CO_X_FT);
        auto layoutCInL1forFT = LayoutCInL1forFT::template MakeLayout<ElementX>(L1TileShape::M, L1TileShape::N);
        auto layoutInL0C = LayoutYInL0::MakeLayoutInL0C(Catlass::MatrixCoord(L1XAlignHelper::M_ALIGNED, L0_M_ALIGNED_FT));

        uint32_t nTileCount = CeilDiv<L0TileShapeforFT::N>(actualShape.n());
        uint32_t CEOutputOffset = CESliceLastId * totalSliceShape.m() * totalSliceShape.n();

        // Optimize points：ShuffleK
        uint32_t startTileIdx = 0;
        
        uint32_t firstTileIdx = 0;

        uint32_t nActual = actualShape.n();
        uint32_t mActual = actualShape.m();

        uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShape.n());
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());

        uint32_t mPartLoop = CeilDiv<L0_M_ALIGNED_FT>(actualShape.m());

        uint32_t startMCoord = actualCoord.m();
        uint32_t MtotalCoord = startMCoord;

        // get L1 Tensor for current stage
        auto l1CTensorforFT = l1FTTensorList[RowSumLastId];
        uint32_t SyncNTile = actualShape.n() / STAGES;
        uint32_t gmTilexOffset = actualShape.n() / STAGES * (RowSumLastId);

        if(SyncNTile < 1){
            SyncNTile = 1;
            gmTilexOffset = 0;
        }
        
        uint32_t SyncNActual = (ELEM_PER_CO_X_FT < SyncNTile) ? ELEM_PER_CO_X_FT : SyncNTile;


        // main loop
        auto l1XTensorforSync = l1FTSyncTensorList[RowSumLastId];

        auto gmTilex = gmBlockXSync[gmTilexOffset];

        // load first vector x tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(l1FTEventList[RowSumLastId]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1FTEventList[RowSumLastId]);

        /*
        void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
        CEOutputOffset
        */
        auto layoutTilex = layoutX.GetTileLayout(Catlass::MakeCoord(SyncNActual));
        copyGmToL1X(l1XTensorforSync, gmTilex, layoutXSyncInL1, layoutTilex);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1FTEventList[RowSumLastId]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1FTEventList[RowSumLastId]);
        
        auto l0CTensorforFTOut = l0CTensorFTList[RowSumLastId];

        // AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l1FTEventforX);
        // AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(MMAD_EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l1FTEventList[RowSumLastId]);
        
        for(uint32_t mPartIdx=0; mPartIdx < mPartLoop; mPartIdx++){
            uint32_t shuffleKIdx = mPartIdx;
            uint32_t singleIdx = (mPartIdx % L0C_TILE_NUM_FT);
            uint32_t nowtmpL0ListId = (uint32_t)(singleIdx % L0C_TILE_STAGES);
            uint32_t L1CMOffset = mPartIdx * L0_M_ALIGNED_FT;

            uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(nActual);
            uint32_t nPartLoop = CeilDiv<L0TileShapeforFT::N>(nActual);

            uint32_t mPartActual =
                    (mPartIdx < mPartLoop - 1) ? L0_M_ALIGNED_FT : (mActual - mPartIdx * L0_M_ALIGNED_FT);

            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CforFTEventList[(uint32_t)(singleIdx % L0C_TILE_STAGES)]);
            
            for (uint32_t nPartIdx = 0; nPartIdx < nPartLoop; nPartIdx++) {
                uint32_t nPartActual =
                    (nPartIdx < nPartLoop - 1) ? L0TileShapeforFT::N : (nActual - nPartIdx * L0TileShapeforFT::N);
                        
                // Locate the current tile on L0A
                auto l0XTile = l0XTensor;
                LayoutXInL0 layoutxInL0 =
                    LayoutXInL0::template MakeLayout<ElementX>(L1XAlignHelper::M_ALIGNED, nPartActual);

                Catlass::MatrixCoord l1xOffset{0, nPartIdx * L0TileShape::N};

                // Locate the current tile on L0B
                auto l0CTileforFT = l0FTTensorList[l0BListId];
                LayoutCInL0forFT layoutCInL0forFT = LayoutCInL0forFT::template MakeLayout<ElementX>(mPartActual, nPartActual);

                Catlass::MatrixCoord l1COffsetforFT{L1CMOffset, nPartIdx * L0TileShape::N};
                auto l1CTileforFT = l1CTensorforFT[layoutCInL1forFT.GetOffset(l1COffsetforFT)];

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                // // // Load current tile from L1 to L0B
                copyL1ToL0CforFT(l0CTileforFT, l1CTileforFT, layoutCInL0forFT, layoutCInL1forFT);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0CforFTEventList[nowtmpL0ListId]);
                
                auto l0CTile = l0CTensorforFTOut[singleIdx * L0C_TILE_SIZE_FT];

                // If the current tile is the first tile on the k axis, the accumulator needs to be reset to 0
                bool initC = (nPartIdx == 0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0CforFTEventList[nowtmpL0ListId]);
                tileMmadforFT(l0CTile, l0XTile, l0CTileforFT, L1XAlignHelper::M_ALIGNED, mPartActual, nPartActual, initC);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                
                l0BListId = (l0BListId + 1 < STAGES) ? (l0BListId + 1) : 0;
            }

            auto l0CTile = l0CTensorforFTOut[singleIdx * L0C_TILE_SIZE_FT];

            // copy block out
            LayoutY layoutBlock = layoutY.GetTileLayout(Catlass::MakeCoord(uint32_t(1), mPartActual));

            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CforFTEventList[(uint32_t)(singleIdx % L0C_TILE_STAGES)]);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CforFTEventList[(uint32_t)(singleIdx % L0C_TILE_STAGES)]);
            // startMCoord + 
            copyL0CToGmforFT(gmBlockY[CEOutputOffset + singleIdx * L0_M_ALIGNED_FT], l0CTile, layoutBlock, layoutInL0C);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CforFTEventList[(uint32_t)(singleIdx % L0C_TILE_STAGES)]);

            // AscendC::PipeBarrier<PIPE_FIX>();
            // MtotalCoord = MtotalCoord + mPartActual;     
        }
        
        
        // if(isLastRowSum){
        //     AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l1FTEventforX);
        // }else{
        //     AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l1FTEventforX);
        // }
        
        // AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(FT_EVENT_ID1);
        // AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(FT_EVENT_ID1);
        // AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FT_EVENT_ID1);
        
        AscendC::SetFlag<AscendC::HardEvent::MTE2_FIX>(l1FTEventList[RowSumLastId]);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1FTEventList[RowSumLastId]);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l1FTEventList[RowSumLastId]);

        
    }



    /// Perform a block-scoped matrix multiply-accumulate
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
    {   
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
        uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShape.n());

        /*
        void FillL0XWithVec(
        AscendC::GlobalTensor<ElementX> const& gmBlockX, LayoutX const& layoutX,
        Catlass::GemmCoord const& actualShape)
        */

        auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
        auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);

        auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(Catlass::MakeCoord(mRound, nRound));

        uint32_t kActual = min(actualShape.k(), L1TileShape::K);


        // 在进行MMAD之前，先把一个常数向量/矩阵直接放到L0A中，用于容错计算，其不会对矩阵乘法产生影响

        FillL0XWithVec(gmBlockX,layoutX,actualShape);
        AscendC::PipeBarrier<PIPE_ALL>();

        // AscendC::PipeBarrier<PIPE_ALL>();

        
        

        uint32_t mPartLoop = CeilDiv<L0TileShape::M>(mRound);
        uint32_t nPartLoop = CeilDiv<L0TileShape::N>(nRound);

        // main loop
        uint kTileCount = CeilDiv<L1TileShape::K>(actualShape.k());
        uint kFTRoundCount = CeilDiv<L0TileShapeforFT::K>(actualShape.k());

        // uint32_t K_FT_TILE_REAL = L0TileShapeforFT::K;
        // if(kFTRoundCount > slice_k_num_limit){
        //     K_FT_TILE_REAL = (actualShape.k() + slice_k_num_limit - 1) / slice_k_num_limit;
        //     K_FT_TILE_REAL = RoundUp<L1TileShape::K>(K_FT_TILE_REAL);
        //     ft_k_interval_real = K_FT_TILE_REAL / L1TileShape::K;
        // }else{
        //     ft_k_interval_real = FT_K_INTERVAL;
        // }
        
        // uint32_t kFTRoundCountReal = (actualShape.k() + K_FT_TILE_REAL -1) / K_FT_TILE_REAL;

        // if(K_FT_TILE_REAL > actualShape.k()){
        //     K_FT_TILE_REAL = actualShape.k();
        //     ft_k_interval_real = kTileCount;
        //     kFTRoundCountReal = 1;
        // }

        // load first matrix A tile from GM to L1
        // AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(FT_EVENT_ID1);
        // kFTRoundIdx * ft_k_interval_real
        
        kActual = min(actualShape.k(), L1TileShape::K);

        for(uint32_t kFTRoundIdx = 0; kFTRoundIdx < kFTRoundCountReal; kFTRoundIdx++){
            uint32_t kLoopGlobalIdxInit = kFTRoundIdx * ft_k_interval_real;
            kActual = (kLoopGlobalIdxInit < (kTileCount - 1)) ?
                        L1TileShape::K : (actualShape.k() - kLoopGlobalIdxInit * L1TileShape::K);
            // Get GM tile for next stage
            Catlass::MatrixCoord gmTileAOffsetInit{0, kLoopGlobalIdxInit * L1TileShape::K};
            Catlass::MatrixCoord gmTileBOffsetInit{kLoopGlobalIdxInit * L1TileShape::K, 0};

            auto gmTileAInit = gmA[layoutA.GetOffset(gmTileAOffsetInit)];
            auto gmTileBInit = gmB[layoutB.GetOffset(gmTileBOffsetInit)];
            
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
            auto layoutTileA = layoutA.GetTileLayout(Catlass::MakeCoord(actualShape.m(),kActual));
            copyGmToL1A(l1ATensorList[l1ListId], gmTileAInit, layoutAInL1, layoutTileA);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);

            // load first matrix B tile from GM to L1
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
            auto layoutTileB = layoutB.GetTileLayout(Catlass::MakeCoord(kActual,actualShape.n()));
            copyGmToL1B(l1BTensorList[l1ListId], gmTileBInit, layoutBInL1, layoutTileB);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);

            // AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(FT_EVENT_ID1);
            // AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(FT_EVENT_ID1);
            // AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FT_EVENT_ID1);

            if constexpr (!ENABLE_UNIT_FLAG) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(MMAD_EVENT_ID0);
            }

            uint32_t kTileCountIntraFT = (kFTRoundIdx < kFTRoundCountReal - 1) ? ft_k_interval_real : kTileCount - kFTRoundIdx * ft_k_interval_real;

            for(uint32_t kLoopIdx=0; kLoopIdx < kTileCountIntraFT; kLoopIdx++){
                // 下一阶段执行的stage id
                uint32_t l1ListIdNext = (l1ListId + 1 < STAGES)? (l1ListId + 1) : 0;
                uint32_t kActualNext{0}; 

                // 流水线，提前将下一阶段的数据从 GM 加载到 L1 中与计算overlap
                // preload next tile from GM to L1
                if (kLoopIdx < kTileCountIntraFT - 1){
                    uint32_t kLoopIdxNext = kLoopIdx + 1;
                    uint32_t kLoopGlobalIdxNext = kFTRoundIdx * ft_k_interval_real + kLoopIdxNext;
                    // 下一阶段 若非最后一个 loop，那么执行一个L1TileShape::K,否则执行剩余的数据
                    kActualNext = (kLoopGlobalIdxNext < (kTileCount - 1)) ?
                        L1TileShape::K : (actualShape.k() - kLoopGlobalIdxNext * L1TileShape::K);
                
                    // Get L1 Tensor for next stage
                    auto l1ATensor = l1ATensorList[l1ListIdNext];
                    auto l1BTensor = l1BTensorList[l1ListIdNext];

                    // Get GM tile for next stage
                    Catlass::MatrixCoord gmTileAOffset{0, kLoopGlobalIdxNext * L1TileShape::K};
                    Catlass::MatrixCoord gmTileBOffset{kLoopGlobalIdxNext * L1TileShape::K, 0};

                    auto gmTileA = gmA[layoutA.GetOffset(gmTileAOffset)];
                    auto gmTileB = gmB[layoutB.GetOffset(gmTileBOffset)];

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
                }
                // else if (kFTRoundIdx < (kFTRoundCountReal - 1)){
                //     uint32_t kFTRoundIdxNext = kFTRoundIdx + 1;
                //     uint32_t kLoopGlobalIdxInitNext = kFTRoundIdxNext * ft_k_interval_real;
                //     kActualNext = (kLoopGlobalIdxInitNext < (kTileCount - 1)) ?
                //         L1TileShape::K : (actualShape.k() - kLoopGlobalIdxInitNext * L1TileShape::K);
            
                //     // Get GM tile for next stage
                //     Catlass::MatrixCoord gmTileAOffsetInitNext{0, kLoopGlobalIdxInitNext * L1TileShape::K};
                //     Catlass::MatrixCoord gmTileBOffsetInitNext{kLoopGlobalIdxInitNext * L1TileShape::K, 0};

                //     auto gmTileAInitNext = gmA[layoutA.GetOffset(gmTileAOffsetInitNext)];
                //     auto gmTileBInitNext = gmB[layoutB.GetOffset(gmTileBOffsetInitNext)];
            
                //     AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);

                //     auto layoutTileANext = layoutA.GetTileLayout(Catlass::MakeCoord(actualShape.m(),kActualNext));
                //     copyGmToL1A(l1ATensorList[l1ListIdNext], gmTileAInitNext, layoutAInL1, layoutTileANext);
                //     AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                //     // load first matrix B tile from GM to L1
                //     AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                //     auto layoutTileBNext = layoutB.GetTileLayout(Catlass::MakeCoord(kActualNext,actualShape.n()));
                //     copyGmToL1B(l1BTensorList[l1ListIdNext], gmTileBInitNext, layoutBInL1, layoutTileBNext);
                //     AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
                // }

                // Get L1 Tensor for current usage
                auto l1ATensor = l1ATensorList[l1ListId];
                auto l1BTensor = l1BTensorList[l1ListId];

                // Get the loop nums on L0
                uint32_t kPartLoop = CeilDiv<L0TileShape::K>(kActual);
            
                for(int mPartIdx=0; mPartIdx < mPartLoop; mPartIdx++){
                    uint32_t mPartActual = (mPartIdx < mPartLoop - 1) ?
                        L0TileShape::M : (mRound - mPartIdx * L0TileShape::M);

                    for(int kPartIdx=0; kPartIdx < kPartLoop; kPartIdx++){
                        uint32_t kPartActual = (kPartIdx < kPartLoop - 1) ?
                            L0TileShape::K : (kActual - kPartIdx * L0TileShape::K);
                    
                        // Locate the current tile on L0A
                        auto l0ATile = l0ATensorList[l0AListId];
                        LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mPartActual,kPartActual);

                        // Locate the current tile of matrix A on L1
                        Catlass::MatrixCoord l1AOffset{mPartIdx * L0TileShape::M, kPartIdx * L0TileShape::K};

                        auto l1ATile = l1ATensor[layoutAInL1.GetOffset(l1AOffset)];
                    
                        // 等待现在的L0阶段中之前在L0上的数据消费已经完成，即相应 MMAD 计算完成
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                        if((mPartIdx == 0) && (kPartIdx == 0)){
                            // 若为当前stage第一次迭代，需要等待到第一批数据，
                            // 即在当前stage涉及的迭代前已经preload 完的A数据成功preload 到 L1 上才可进行A数据向 L0 上写
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
                        }

                        // Load current tile from L1 to L0A
                        copyL1ToL0A(l0ATile, l1ATile, layoutAInL0, layoutAInL1);

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
                            auto l0BTile = l0BTensorList[l0BListId];
                            LayoutBInL0 layoutBInL0 = 
                                LayoutBInL0::template MakeLayout<ElementB>(kPartActual, nPartActual);
                        
                            // Locate the current tile of matrix B on L1
                            Catlass::MatrixCoord l1BOffset{kPartIdx * L0TileShape::K, 
                                nPartIdx * L0TileShape::N};
                        
                            auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BOffset)];

                            // Wait for mmad finished
                            // 等待当前阶段L0 数据消费完成，即相应 MMAD 计算完成
                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                            // If the current tile is the first one on the k&n axis, wait for loading matrix B from GM to L1
                            if((nPartIdx == 0) && (kPartIdx == 0)){
                                // 若为当前stage第一次迭代，需要等待到第一批数据，
                                // 即在当前stage涉及的迭代前已经preload 完的B数据成功
                                // preload 到 L1 上才可进行B数据向 L0 上写
                                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
                            }

                            // Load current tile from L1 to L0B
                            copyL1ToL0B(l0BTile, l1BTile, layoutBInL0, layoutBInL1);

                            // If the current tile is the last one on the k&n axis, notify to load matrix B from GM to L1
                            if ((kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                                // 若这是当前stage 最后一次的迭代，则需要设置Flag，
                                // 即当前stage涉及到现有 L0 B的数据的操作已经完成
                                // 之后再进行PING-PONG preload时，可以向该阶段对应的L1 B 数据中写入新的数据了
                                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                            }

                            // Notify to do mmad
                            // 标记L0 Cache 中 A2 和 B2 的数据写入已经完成，可以进行 MMAD 运算了
                            // 同时也标记当前开始进行 L0 的 CO1 Cache 了。
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(MMAD_EVENT_ID0);

                            // Locate the current tile on L0C
                            Catlass::MatrixCoord l0COffset{mPartIdx * L0TileShape::M, nPartIdx * L0TileShape::N};
                            // 获取当前局部输出
                            auto l0CTile = l0CTensor[layoutInL0C.GetOffset(l0COffset)];

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
                                if ((kLoopIdx  == kTileCountIntraFT - 1) && (mPartIdx == mPartLoop - 1) &&
                                    (kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                                    unitFlag = 0b11;
                                } else {
                                    unitFlag = 0b10;
                                }
                            }
                            // Perform calculation operations
                            tileMmad(l0CTile, l0ATile, l0BTile, mPartActual, 
                                nPartActual, kPartActual, initC, unitFlag);

                            // Notify to move the next L0B tile
                            // 标记计算完成，即当前已经完成了一个l0 tile 的运算，可以加载下一个l0 tile了
                            // 这里最内层为 B 矩阵
                            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                            l0BListId = (l0BListId + 1 < STAGES) ? (l0BListId + 1) : 0;
                        }
                        // 交替进行阶段，实现 L1 与 L0 之间数据传输与 MMAD 计算的PING-PANG
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                        l0AListId = (l0AListId + 1 < STAGES) ? (l0AListId + 1) : 0;
                    }
                }
                // 交替进行阶段，实现L1 与 Global 之间数据传输 与 MMAD计算 的PING-PANG
                l1ListId = l1ListIdNext;
                kActual = kActualNext;
            }

            // copy block out
            // 将局部累加结果从 L0 的 CO1 输出到GM即可
            LayoutC layoutBlock = layoutC.GetTileLayout(actualShape.GetCoordMN());
            auto layoutCInL1forFT = LayoutCInL1forFT::template MakeLayout<ElementX>(L1TileShape::M, L1TileShape::N);

            uint32_t CoutputOffset = kFTRoundIdx * totalProblemShape.m() * totalProblemShape.n();
            if constexpr (!ENABLE_UNIT_FLAG) {
                // 标记开始写入cGM 数据
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(MMAD_EVENT_ID0);
                // 等待允许写入开始
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(MMAD_EVENT_ID0);
                // 写入数据
                copyL0CToGm(gmC[CoutputOffset], l0CTensor, layoutBlock, layoutInL0C);

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_FIX>(l1FTEventList[l1FTListId]);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l1FTEventList[l1FTListId]);
                copyL0CToL1forFT(l1FTTensorList[l1FTListId], l0CTensor, layoutCInL1forFT, layoutInL0C);
                    
            } else {
                copyL0CToGm(gmC[CoutputOffset], l0CTensor, layoutBlock, layoutInL0C, 0b11);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_FIX>(l1FTEventList[l1FTListId]);
                copyL0CToL1forFT(l1FTTensorList[l1FTListId], l0CTensor, layoutCInL1forFT, layoutInL0C, 0b11, false);
            }

            AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(l1FTEventList[l1FTListId]);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l1FTEventList[l1FTListId]);

            // 若非第一次，则存在上一个LastId，可以开始计算上一个stage的FT数据了
            //  if(kFTRoundIdx < 1)
            if((kFTRoundIdx >= 1)){
                #pragma unroll
                for(uint32_t j = 0; j < L0C_TILE_STAGES; j++){
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CforFTEventList[j]);
                }

                // AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(MMAD_EVENT_ID0);

                /*
                void RowSum(
                    AscendC::GlobalTensor<ElementY> const& gmBlockY,
                    AscendC::GlobalTensor<ElementX> const& gmBlockXSync, 
                    LayoutY const& layoutY, LayoutX const& layoutX,
                    Catlass::GemmCoord const& actualShape, 
                    Catlass::GemmCoord const& actualCoord,
                    Catlass::GemvCoord const& totalSliceShape,
                    uint32_t RowSumLastId,
                    uint32_t CESliceLastId, bool isLastRowSum)
                */
            
                RowSum(gmBlockY,gmBlockX,
                    layoutY, layoutX,
                    actualShape,actualCoord,
                    totalSliceShape,
                    l1FTListLastId,
                    SliceKLastIdx, false);

                // AscendC::PipeBarrier<PIPE_FIX>();

                #pragma unroll
                for(uint32_t j = 0; j < L0C_TILE_STAGES; j++){
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CforFTEventList[j]);
                }

                AscendC::PipeBarrier<PIPE_FIX>();  
                AscendC::PipeBarrier<PIPE_MTE1>();  
                AscendC::PipeBarrier<PIPE_M>();
                
                // 通知相应 AIV core，MMAD计算已经完成了，结果已经写入了GM 
                Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
            }else{
                // AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(FT_EVENT_ID1);
                // AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(FT_EVENT_ID1);
                // AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FT_EVENT_ID1);
                AscendC::PipeBarrier<PIPE_FIX>();  
                AscendC::PipeBarrier<PIPE_MTE1>();  
                AscendC::PipeBarrier<PIPE_M>();
            }

            l1FTListLastId = l1FTListId;
            l1FTListId = (l1FTListId + 1 < STAGES) ? (l1FTListId + 1) : 0;
            SliceKLastIdx = kFTRoundIdx;

            if constexpr (!ENABLE_UNIT_FLAG) {
                // 标记数据写入 cGM 已经完成
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(MMAD_EVENT_ID0);
            }
        }
        
        AscendC::PipeBarrier<PIPE_ALL>();

        // 最后一步，计算剩余的AIC 上的CE校验即可
        if constexpr (!ENABLE_UNIT_FLAG) {
            // 标记数据写入 cGM 已经完成
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(MMAD_EVENT_ID0);
        }

        #pragma unroll
        for(uint32_t j = 0; j < L0C_TILE_STAGES; j++){
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CforFTEventList[j]);
        }

        // AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(MMAD_EVENT_ID0);

        /*
        void RowSum(
            AscendC::GlobalTensor<ElementY> const& gmBlockY,
            AscendC::GlobalTensor<ElementX> const& gmBlockXSync, 
            LayoutY const& layoutY, LayoutX const& layoutX,
            Catlass::GemmCoord const& actualShape, 
            Catlass::GemmCoord const& actualCoord,
            Catlass::GemvCoord const& totalSliceShape,
            uint32_t RowSumLastId,
            uint32_t CESliceLastId, bool isLastRowSum)
        */
        RowSum(gmBlockY,gmBlockX,
            layoutY, layoutX,
            actualShape, actualCoord,
            totalSliceShape,
            l1FTListLastId,
            SliceKLastIdx, true);

        // AscendC::PipeBarrier<PIPE_FIX>();

        #pragma unroll
        for(uint32_t j = 0; j < L0C_TILE_STAGES; j++){
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CforFTEventList[j]);
        }

        AscendC::PipeBarrier<PIPE_FIX>();
        
        if constexpr (!ENABLE_UNIT_FLAG) {
            // 标记数据写入 cGM 已经完成
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(MMAD_EVENT_ID0);
        }

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l1FTEventforX);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(FT_EVENT_ID1);

        // 通知相应 AIV core，MMAD计算已经完成了，结果已经写入了GM 
        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> l1ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList[STAGES];
    AscendC::LocalTensor<ElementX> l1FTTensorList[STAGES];
    AscendC::LocalTensor<ElementX> l1FTSyncTensorList[STAGES];

    AscendC::LocalTensor<ElementA> l0ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList[STAGES];
    AscendC::LocalTensor<ElementX> l0XTensorList[STAGES];
    AscendC::LocalTensor<ElementX> l0FTTensorList[STAGES];
    
    AscendC::LocalTensor<ElementAccumulator> l0CTensor;

    AscendC::LocalTensor<ElementAccumulator> l0CTensorFTList[STAGES];

    AscendC::LocalTensor<ElementX> l0XTensor;

    // Multi-stage event id list
    int32_t l1AEventList[STAGES];
    int32_t l1BEventList[STAGES];
    int32_t FT_EVENT_ID1;
    int32_t l1FTEventforX;
    int32_t l1FTEventList[STAGES];

    int32_t MMAD_EVENT_ID0;

    int32_t l0AEventList[STAGES];
    int32_t l0BEventList[STAGES];

    int32_t l0CforFTEventList[L0C_TILE_STAGES];

    // int32_t l1FTEventList[STAGES];

    // The id of current stage
    // 指示当前所处的pipeline 中的阶段（双阶段PING-PONG）
    uint32_t l1ListId{0};
    uint32_t l0AListId{0};
    uint32_t l0BListId{0};
    uint32_t l1FTListId{0};
    uint32_t l1FTListLastId{0};
    uint32_t SliceKLastIdx{0};
    uint32_t FT_K_INTERVAL{4};
    // uint32_t ft_k_interval_real{4};

    TileMmad tileMmad;
    TileMmadforFT tileMmadforFT;

    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyGmToL1X copyGmToL1X;

    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL1ToL0X copyL1ToL0X;
    CopyL1ToL0CforFT copyL1ToL0CforFT;

    CopyL0CToGm copyL0CToGm;
    CopyL0CToL1forFT copyL0CToL1forFT;
    CopyL0CToGmforFT copyL0CToGmforFT;
};
}

#endif // CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_HPP