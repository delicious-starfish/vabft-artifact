/// Perform a block-scoped vector-matrix multiply-accumulate
    CATLASS_DEVICE
    void RowSum2(
        AscendC::GlobalTensor<ElementY> const& gmBlockY, LayoutY const& layoutY,
        Catlass::GemmCoord const& actualShape, Catlass::GemmCoord const& actualCoord)
    {
        auto layoutXInL1 = LayoutXInL1::template MakeLayout<ElementX>(L1XAlignHelper::M_ALIGNED, L1TileShape::N);
        auto layoutCInL1forFT = LayoutCInL1forFT::template MakeLayout<ElementX>(L1TileShape::M, L1TileShape::N);
        auto layoutInL0C = LayoutYInL0::MakeLayoutInL0C(Catlass::MatrixCoord(L1XAlignHelper::M_ALIGNED, L1TileShape::M));

        uint32_t nTileCount = 1;
        // CeilDiv<L1TileShape::N>(L1TileShape::N);

        // Optimize points：ShuffleK
        uint32_t startTileIdx = 0;

        uint32_t firstTileIdx = startTileIdx % nTileCount;
        uint32_t lastTileIdx = (startTileIdx + nTileCount - 1) % nTileCount;

        uint32_t nActual =
            (actualShape.n() <  L1TileShape::N) ? actualShape.n() : L1TileShape::N;
        uint32_t mActual =  (actualShape.m() <  L1TileShape::M) ? actualShape.m() : L1TileShape::M;

        uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(nActual);
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(mActual);

        uint32_t singleIdx = l1ListId;

        // main loop
        
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((int32_t)(singleIdx % L0C_TILE_NUM));

        for (uint32_t nLoopIdx = 0; nLoopIdx < nTileCount; nLoopIdx++) {
            uint32_t shuffleKIdx = (startTileIdx + nLoopIdx) % nTileCount;

            // get L1 Tensor for current stage
            auto l1CTensorforFT = l1FTTensor;

            uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(nActual);
            uint32_t nPartLoop = CeilDiv<L0TileShapeforFT::N>(nActual);

            for (uint32_t nPartIdx = 0; nPartIdx < nPartLoop; nPartIdx++) {
                uint32_t nPartActual =
                    (nPartIdx < nPartLoop - 1) ? L0TileShapeforFT::N : (nActual - nPartIdx * L0TileShapeforFT::N);

                // Locate the current tile on L0A
                auto l0XTile = l0XTensor;
                LayoutXInL0 layoutxInL0 =
                    LayoutXInL0::template MakeLayout<ElementX>(L1XAlignHelper::M_ALIGNED, nPartActual);

                Catlass::MatrixCoord l1xOffset{0, nPartIdx * L0TileShapeforFT::N};

                // Locate the current tile on L0B
                auto l0CTileforFT = l0FTTensorList[l0FTListId];
                LayoutCInL0forFT layoutCInL0forFT = LayoutCInL0forFT::template MakeLayout<ElementX>(L0TileShapeforFT::M, nPartActual);

                Catlass::MatrixCoord l1COffsetforFT{0, nPartIdx * L0TileShapeforFT::N};
                auto l1CTileforFT = l1FTTensor[layoutCInL1forFT.GetOffset(l1COffsetforFT)];

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0FTEventList[l0FTListId]);
                // Load current tile from L1 to L0B
                copyL1ToL0CforFT(l0CTileforFT, l1CTileforFT, layoutCInL0forFT, layoutCInL1forFT);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0FTEventList[l0FTListId]);

                auto l0CTile = l0CTensor[(singleIdx % L0C_TILE_NUM) * L0C_TILE_SIZE];

                // If the current tile is the first tile on the k axis, the accumulator needs to be reset to 0
                bool initC = ((nLoopIdx == 0) && (nPartIdx == 0));

                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0FTEventList[l0FTListId]);
                // L0TileShapeforFT::M
                tileMmadforFT(l0CTile, l0XTile, l0CTileforFT, L1XAlignHelper::M_ALIGNED, L0TileShapeforFT::M, nPartActual, initC);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0FTEventList[l0FTListId]);

                l0FTListId = (l0FTListId + 1) % STAGES;
            }
        }

        auto l0CTile = l0CTensor[(singleIdx % L0C_TILE_NUM) * L0C_TILE_SIZE];

        // copy block out
        // actualShape.m()
        LayoutY layoutBlock = layoutY.GetTileLayout(MakeCoord(uint32_t(1), mActual));

        AscendC::SetFlag<AscendC::HardEvent::M_FIX>((int32_t)(singleIdx % L0C_TILE_NUM));
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>((int32_t)(singleIdx % L0C_TILE_NUM));
        copyL0CToGmforFT(gmBlockY, l0CTile, layoutBlock, layoutInL0C);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>((int32_t)(singleIdx % L0C_TILE_NUM));
    }