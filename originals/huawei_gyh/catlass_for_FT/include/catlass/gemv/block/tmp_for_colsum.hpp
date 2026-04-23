CATLASS_DEVICE
    void ColSum(
        AscendC::GlobalTensor<ElementX> const& gmBlockX, LayoutX const& layoutX,
        AscendC::GlobalTensor<ElementA> const& gmBlockA, LayoutACol const& layoutA,
        AscendC::GlobalTensor<ElementY> const& gmBlockY, LayoutY const& layoutY,
        AscendC::GlobalTensor<ElementX> const& gmNextBlockX,
        AscendC::GlobalTensor<ElementA> const& gmNextBlockA,
        GemvCoord const& actualShape, GemvCoord const& actualShapeNext,
        bool isFirstBlock, bool hasNextBlock, uint32_t singleIdx)
    {
        auto layoutXInL1 = LayoutXInL1::template MakeLayout<ElementX>(L1XAlignHelper::M_ALIGNED, L1TileShape::N);
        auto layoutAInL1 = LayoutAInL1Col::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::N);
        auto layoutInL0C = LayoutYInL0::MakeLayoutInL0C(MatrixCoord(L1XAlignHelper::M_ALIGNED, actualShape.m()));

        uint32_t nTileCount = CeilDiv<L1TileShape::N>(actualShape.n());
        uint32_t nTileCountNext = CeilDiv<L1TileShape::N>(actualShapeNext.n());

        // Optimize points：ShuffleK
        uint32_t startTileIdx = 0;
        if constexpr (ENABLE_SHUFFLE_K_) {
            startTileIdx = AscendC::GetBlockIdx();
        }
        uint32_t firstTileIdx = startTileIdx % nTileCount;
        uint32_t lastTileIdx = (startTileIdx + nTileCount - 1) % nTileCount;
        uint32_t firstTileIdxNext = startTileIdx % nTileCountNext;

        uint32_t nActual =
            (firstTileIdx < nTileCount - 1) ? L1TileShape::N : (actualShape.n() - firstTileIdx * L1TileShape::N);
        uint32_t nRound = RoundUp<L1AColAlignHelper::N_ALIGNED>(nActual);

        // main loop
        for (uint32_t nLoopIdx = 0; nLoopIdx < nTileCount; nLoopIdx++) {
            uint32_t shuffleKIdx = (startTileIdx + nLoopIdx) % nTileCount;
            if (shuffleKIdx == firstTileIdx && isFirstBlock) {
                MatrixCoord gmTileAOffset{0, shuffleKIdx * L1TileShape::N};
                uint32_t gmTilexOffset{shuffleKIdx * L1TileShape::N};

                auto gmTileA = gmBlockA[layoutA.GetOffset(gmTileAOffset)];
                auto gmTilex = gmBlockX[gmTilexOffset];

                // load first vector x tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                auto layoutTilex = layoutX.GetTileLayout(MakeCoord(nRound));
                copyGmToL1A(l1ATensorList[l1ListId], gmTilex, layoutXInL1, layoutTilex);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);

                // load first matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), nRound));
                copyGmToL1BCol(l1BTensorList[l1ListId], gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
            }

            uint32_t l1ListIdNext = (l1ListId + 1) % STAGES;
            uint32_t nActualNext{0};
            uint32_t nRoundNext{0};

            // preload next tile from GM to L1
            if (shuffleKIdx != lastTileIdx) {
                uint32_t shuffleKIdxNext = (startTileIdx + nLoopIdx + 1) % nTileCount;
                nActualNext = (shuffleKIdxNext < nTileCount - 1) ? L1TileShape::N
                                                                 : (actualShape.n() - shuffleKIdxNext * L1TileShape::N);
                nRoundNext = RoundUp<L1AColAlignHelper::N_ALIGNED>(nActualNext);

                // Get L1 tensor
                auto l1ATensor = l1ATensorList[l1ListIdNext];
                auto l1BTensor = l1BTensorList[l1ListIdNext];

                // Get GM tile
                MatrixCoord gmTileAOffset{0, shuffleKIdxNext * L1TileShape::N};
                uint32_t gmTilexOffset{shuffleKIdxNext * L1TileShape::N};

                auto gmTileA = gmBlockA[layoutA.GetOffset(gmTileAOffset)];
                auto gmTilex = gmBlockX[gmTilexOffset];

                // load vector x tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                auto layoutTilex = layoutX.GetTileLayout(MakeCoord(nRoundNext));

                copyGmToL1A(l1ATensor, gmTilex, layoutXInL1, layoutTilex);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                // load Matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), nRoundNext));

                copyGmToL1BCol(l1BTensor, gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
            }
            if (shuffleKIdx == lastTileIdx && hasNextBlock) {
                // Get L1 tensor
                auto l1ATensor = l1ATensorList[l1ListIdNext];
                auto l1BTensor = l1BTensorList[l1ListIdNext];

                // Get GM tensor for next stage
                nActualNext = (firstTileIdxNext < nTileCountNext - 1)
                    ? L1TileShape::N : (actualShapeNext.n() - firstTileIdxNext * L1TileShape::N);
                nRoundNext = RoundUp<L1AColAlignHelper::N_ALIGNED>(nActualNext);

                // Get GM tile
                MatrixCoord gmTileAOffset{0, firstTileIdxNext * L1TileShape::N};
                uint32_t gmTilexOffset{firstTileIdxNext * L1TileShape::N};

                auto gmTileA = gmNextBlockA[layoutA.GetOffset(gmTileAOffset)];
                auto gmTilex = gmNextBlockX[gmTilexOffset];

                // load vector x tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);

                auto layoutTilex = layoutX.GetTileLayout(MakeCoord(nRoundNext));

                copyGmToL1A(l1ATensor, gmTilex, layoutXInL1, layoutTilex);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                // load Matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShapeNext.m(), nRoundNext));

                copyGmToL1BCol(l1BTensor, gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
            }

            // get L1 Tensor for current stage
            auto l1ATensor = l1ATensorList[l1ListId];
            auto l1BTensor = l1BTensorList[l1ListId];

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);

            uint32_t nRound = RoundUp<L1AColAlignHelper::N_ALIGNED>(nActual);
            uint32_t nPartLoop = CeilDiv<L0TileShape::N>(nActual);

            for (uint32_t nPartIdx = 0; nPartIdx < nPartLoop; nPartIdx++) {
                uint32_t nPartActual =
                    (nPartIdx < nPartLoop - 1) ? L0TileShape::N : (nActual - nPartIdx * L0TileShape::N);

                // Locate the current tile on L0A
                auto l0ATile = l0ATensorList[l0AListId];
                LayoutXInL0 layoutxInL0 =
                    LayoutXInL0::template MakeLayout<ElementX>(L1XAlignHelper::M_ALIGNED, nPartActual);

                MatrixCoord l1xOffset{0, nPartIdx * L0TileShape::N};
                auto l1ATile = l1ATensor[layoutXInL1.GetOffset(l1xOffset)];

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                // Load current tile from L1 to L0A
                copyL1ToL0A(l0ATile, l1ATile, layoutxInL0, layoutXInL1);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0AListId]);

                // Locate the current tile on L0B
                auto l0BTile = l0BTensorList[l0BListId];
                LayoutAInL0Col layoutAInL0 = LayoutAInL0Col::template MakeLayout<ElementA>(L0TileShape::M, nPartActual);

                MatrixCoord l1AOffset{0, nPartIdx * L0TileShape::N};
                auto l1BTile = l1BTensor[layoutAInL1.GetOffset(l1AOffset)];

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                // Load current tile from L1 to L0B
                copyL1ToL0BCol(l0BTile, l1BTile, layoutAInL0, layoutAInL1);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0BListId]);

                auto l0CTile = l0CTensor[(singleIdx % L0C_TILE_NUM) * L0C_TILE_SIZE];

                // If the current tile is the first tile on the k axis, the accumulator needs to be reset to 0
                bool initC = ((nLoopIdx == 0) && (nPartIdx == 0));

                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0BListId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0AListId]);
                tileMmad(l0CTile, l0ATile, l0BTile, L1XAlignHelper::M_ALIGNED, L0TileShape::M, nPartActual, initC);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);

                l0AListId = (l0AListId + 1) % STAGES;
                l0BListId = (l0BListId + 1) % STAGES;
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);

            l1ListId = l1ListIdNext;

            nActual = nActualNext;
        }

        auto l0CTile = l0CTensor[(singleIdx % L0C_TILE_NUM) * L0C_TILE_SIZE];

        // copy block out
        LayoutY layoutBlock = layoutY.GetTileLayout(MakeCoord(uint32_t(1), actualShape.m()));

        AscendC::SetFlag<AscendC::HardEvent::M_FIX>((int32_t)(singleIdx % L0C_TILE_NUM));
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>((int32_t)(singleIdx % L0C_TILE_NUM));

        copyL0CToGm(gmBlockY, l0CTile, layoutBlock, layoutInL0C);
    }
