/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMV_BLOCK_BLOCK_GEMV_ASVAR_VERIFY_AIV_NO_SPLIT_HPP_SPEC_ROBUST_PRELOAD
#define CATLASS_GEMV_BLOCK_BLOCK_GEMV_ASVAR_VERIFY_AIV_NO_SPLIT_HPP_SPEC_ROBUST_PRELOAD

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemv_coord.hpp"
#include "catlass/gemv/helper.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/gemm/dispatch_policy.hpp"

#include "catlass/gemv/tile/tile_threshold.hpp"

#include "catlass/gemv/tile/tile_matmul_elem_add.hpp"
#include "catlass/gemv/tile/tile_vmuls.hpp"
#include "catlass/gemv/tile/tile_std_estimate.hpp"

namespace Catlass::Gemv::Block {


template <
    class UBTileShape_,
    class UBBlockShape_,
    class L1TileShape_,
    class AType_,
    class XType_,
    class YType_,
    class ZType_,
    class BiasType_,
    class TileCopy_,
    class TileMatrixAdd_,
    class TileThreCalc_,
    class TileStdEst_>
struct BlockFTVerifyNoSplitKPreload <
    Gemm::GemvAtlasA2,
    Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
    Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::THRE_FUSED,
    Gemv::helper::FT_ENC_TYPE::RCE,
    Gemv::helper::FT_COMP_TYPE::RSUB,
    Gemv::helper::FT_ABE_TYPE::CENTRAL_BLOCK,
    UBTileShape_,
    UBBlockShape_,
    L1TileShape_,
    AType_,
    XType_,
    YType_,
    ZType_,
    BiasType_,
    TileCopy_,
    TileMatrixAdd_,
    TileThreCalc_,
    TileStdEst_> 
{
public:
    // Type Aliases
    using DispatchPolicy = Gemm::GemvAtlasA2;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using FT_ENC_TYPE = Catlass::Gemv::helper::FT_ENC_TYPE;
    using FT_COMP_TYPE = Catlass::Gemv::helper::FT_COMP_TYPE;
    using FT_REDUCE_TYPE = Catlass::Gemv::helper::FT_REDUCE_TYPE;
    using FT_AIV_PIPE_FUSE_TYPE = Catlass::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE;
    using FT_THRESHOLD_ALGORITHM = Catlass::Gemv::helper::FT_THRESHOLD_ALGORITHM;
    using FT_ABE_TYPE = Catlass::Gemv::helper::FT_ABE_TYPE;

    using UBTileShape = UBTileShape_;
    using UBBlockShape = UBBlockShape_;
    using L1TileShape = L1TileShape_;

    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;

    using ElementX = typename XType_::Element;
    using LayoutX = typename XType_::Layout;

    using ElementY = typename YType_::Element;
    using LayoutY = typename YType_::Layout;

    using ElementZ = typename ZType_::Element;
    using LayoutZ = typename ZType_::Layout;


    using TileThreCalc = TileThreCalc_;
    using TileStdEst = TileStdEst_;
    using TileMatrixAdd = TileMatrixAdd_;

    using LayoutMX = layout::RowMajor;
    using LayoutMY = layout::RowMajor;
    using LayoutMZ = layout::RowMajor;

    using LayoutVX = layout::VectorLayout;
    using LayoutVY = layout::VectorLayout;
    using LayoutVZ = layout::VectorLayout;

    using MXType = Gemm::GemmType<ElementX, LayoutMX>;
    using MYType = Gemm::GemmType<ElementY, LayoutMY>;
    using MZType = Gemm::GemmType<ElementZ, LayoutMZ>;

    using VXType = Gemm::GemmType<ElementX, LayoutVX>;
    using VYType = Gemm::GemmType<ElementY, LayoutVY>;
    using VZType = Gemm::GemmType<ElementZ, LayoutVZ>;

    using TileMatrixAddforCRReduce = Gemv::Tile::TileMatmulAdd<
        ArchTag, MYType, MYType, void>;

    using TileStdEstMatrix = Gemv::Tile::TileStdEstRobust<
        ArchTag,
        MYType,
        MYType>;
    
    /*
    struct TileStdEstRobust<Arch::AtlasA2,
                Gemm::GemmType<float, layout::VectorLayout>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
    */
    
    using TileStdEstVector = Gemv::Tile::TileStdEstRobust<
        ArchTag,
        VYType,
        VYType>;

    /*
    struct TileThreCalc<Arch::AtlasA2,
                helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
                Gemm::GemmType<ElementA, layout::RowMajor>,
                Gemm::GemmType<float, layout::RowMajor>,
                Gemm::GemmType<float, layout::RowMajor>,
                void>
    */

    using TileThreCalcMatrix = Gemv::Tile::TileThreCalc<
        ArchTag, 
        FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
        AType_, MYType, MYType, void>;

    /*
        template <
        class ElementA
    >
    struct TileThreCalcChunk<Arch::AtlasA2,
                    helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
                    Gemm::GemmType<ElementA, layout::RowMajor>,
                    Gemm::GemmType<float, layout::VectorLayout>,
                    Gemm::GemmType<float, layout::VectorLayout>,
                    void>
    */

    using TileThreCalcChunk = Gemv::Tile::TileThreCalcChunk<
        ArchTag,
        FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
        AType_, VYType, VYType, void>;
    

    using VecCopyGmToUb = typename TileCopy_::VecCopyGmToUb;
    using VecCopyUbToGm = typename TileCopy_::VecCopyUbToGm;
    using MatrixCopyGmToUb = typename TileCopy_::MatrixCopyGmToUb;
    using MatrixCopyGmToUbforThre = typename TileCopy_::MatrixCopyGmToUbforThre;
    using VecCopyGmToUbInY = typename TileCopy_::VecCopyGmToUbInY;
    using VecCopyUbToGmZ = typename TileCopy_::VecCopyUbToGmZ;
    using VecCopyUbToGmforThre = typename TileCopy_::VecCopyUbToGmforThre;

    using TileCompare = Gemv::Tile::TileFaultVcompare<FT_COMP_TYPE::RSUB, ArchTag, 
                                        MZType, MYType, MYType>;

    using MatrixCopyGmToUbforThreAIn = typename TileCopy_::MatrixCopyGmToUbforThreAIn;
    using MatrixCopyGmToUbforABRIn = typename TileCopy_::MatrixCopyGmToUbforABRIn;
    using MatrixCopyGmToUbforCRIn = typename TileCopy_::MatrixCopyGmToUbforCRIn;

    using MatrixCopyUbToGmforZ = typename TileCopy_::MatrixCopyUbToGmforZ;
    using MatrixCopyUbToGmforCROut = typename TileCopy_::MatrixCopyUbToGmforCROut;

    using MatrixCopyUbToGmforThreOut = typename TileCopy_::MatrixCopyUbToGmforThreOut;

    
    
    static constexpr FT_COMP_TYPE COMP_TYPE = FT_COMP_TYPE::RSUB;
    static constexpr FT_ENC_TYPE ENC_TYPE = FT_ENC_TYPE::RCE;
    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = FT_THRESHOLD_ALGORITHM::ASVAR;
    static constexpr FT_ABE_TYPE ABE_TYPE = FT_ABE_TYPE::CENTRAL_BLOCK;

    static constexpr uint32_t ELE_NUM_PER_BLK_OUTZ = BYTE_PER_BLK / sizeof(ElementZ);

    static constexpr bool NEED_CAST_FOR_RED = std::is_same<ElementX, ElementY>::value;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    using UBAlignHelper = Gemv::helper::UBAlignHelper<ElementA>;
    using UBAlignHelperOut = Gemv::helper::UBAlignHelper<ElementY>;
    using TensorCoord = layout::VectorLayout::TensorCoord;

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE = Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::NO_FUSED;
    static constexpr uint32_t CR1buf_SIZE_ = 16 * 1024;
    static constexpr uint32_t CR2buf_SIZE_ = 16 * 1024;

    static constexpr uint32_t AMeanbuf_SIZE_ = 4 * 1024;
    static constexpr uint32_t AMaxbuf_SIZE_forY_ = 4 * 1024;
    static constexpr uint32_t AMinbuf_SIZE_forY_ = 4 * 1024;
    static constexpr uint32_t Threbuf_SIZE_ = 16 * 1024;

    static constexpr uint32_t workspace_forThre_SIZE_ = 28 * 1024;
    

    static constexpr uint32_t Zbuf_SIZE_ = 2 *1024;
    static constexpr uint32_t BMeanbuf_SIZE_ = 2 * 1024;
    static constexpr uint32_t BMeanSquarebuf_SIZE_ = 2 * 1024;
    static constexpr uint32_t BVarbuf_SIZE_ = 2 * 1024;


    static constexpr uint32_t workspace_forR1_SIZE_ = 32 * 1024;
    static constexpr uint32_t workspace_forR2_SIZE_ = 32 * 1024;
    static constexpr uint32_t ABR1buf_SIZE_ = 16 * 1024;
    static constexpr uint32_t ABR2buf_SIZE_ = 16 * 1024;
    

    static constexpr uint32_t ELE_NUM_PER_BLK_FOR_C = BYTE_PER_BLK / sizeof(ElementY);

    static_assert(L1TileShape::M == UBBlockShape::N,
        "The situation where the basic Tile of UB and L1 for MMA differ on the m axes is not supported yet");

    static_assert(L1TileShape::N == (UBBlockShape::M * 2),
        "The situation where the basic Tile of UB and L1 for MMA differ on the n axes is not supported yet");


    static_assert((UBBlockShape::N % UBTileShape::N) == 0,
        "The situation where the basic Tile of UB and L1 for MMA differ on the n axes is not supported yet");

    static_assert(UBBlockShape::N / UBTileShape::N <= 2,
        "The situation where the basic Tile of UB In Total AICores and L1 for MMA differ on the n axes is not supported yet");

    CATLASS_DEVICE
    BlockFTVerifyNoSplitKPreload() {}

    /// Construct
    CATLASS_DEVICE
    BlockFTVerifyNoSplitKPreload(Arch::Resource<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbCR1Offset = UBufAddrStart;
        uint32_t UbCR2Offset = UBufAddrStart + CR1buf_SIZE_;

        uint32_t UbAMeanOffset = UBufAddrStart + CR1buf_SIZE_ + CR2buf_SIZE_;
        uint32_t UbAMinOffset_forY = UBufAddrStart + CR1buf_SIZE_ + CR2buf_SIZE_ + AMeanbuf_SIZE_;
        uint32_t UbAMaxOffset_forY = UBufAddrStart + CR1buf_SIZE_ + CR2buf_SIZE_ + AMeanbuf_SIZE_ + AMinbuf_SIZE_forY_;
        uint32_t UbThreOffset = UBufAddrStart + CR1buf_SIZE_ + CR2buf_SIZE_ + AMeanbuf_SIZE_ + AMinbuf_SIZE_forY_ + AMaxbuf_SIZE_forY_;

        uint32_t UbZOffset = UBufAddrStart + CR1buf_SIZE_ + CR2buf_SIZE_ + AMeanbuf_SIZE_ + AMinbuf_SIZE_forY_ + AMaxbuf_SIZE_forY_ + Threbuf_SIZE_;
        uint32_t UbBMeanOffset = UBufAddrStart + CR1buf_SIZE_ + CR2buf_SIZE_ + AMeanbuf_SIZE_ + AMinbuf_SIZE_forY_ + AMaxbuf_SIZE_forY_ + Threbuf_SIZE_ + Zbuf_SIZE_;
        uint32_t UbBMeanSqaureOffset = UBufAddrStart + CR1buf_SIZE_ + CR2buf_SIZE_ + AMeanbuf_SIZE_ + AMinbuf_SIZE_forY_ + AMaxbuf_SIZE_forY_ + Threbuf_SIZE_ + Zbuf_SIZE_ + BMeanbuf_SIZE_;
        uint32_t UbBVarOffset = UBufAddrStart + CR1buf_SIZE_ + CR2buf_SIZE_ + AMeanbuf_SIZE_ + AMinbuf_SIZE_forY_ + AMaxbuf_SIZE_forY_ + Threbuf_SIZE_ + Zbuf_SIZE_ + BMeanbuf_SIZE_ + BMeanSquarebuf_SIZE_;
        uint32_t UbWOffset_thre_ = UBufAddrStart + CR1buf_SIZE_ + CR2buf_SIZE_ + AMeanbuf_SIZE_ + AMinbuf_SIZE_forY_ + AMaxbuf_SIZE_forY_ + Threbuf_SIZE_ + Zbuf_SIZE_ + BMeanbuf_SIZE_ + BMeanSquarebuf_SIZE_ + BVarbuf_SIZE_;
        uint32_t UbWOffset_R1_ = UBufAddrStart + CR1buf_SIZE_ + CR2buf_SIZE_ + AMeanbuf_SIZE_ + AMinbuf_SIZE_forY_ + AMaxbuf_SIZE_forY_ + Threbuf_SIZE_ + Zbuf_SIZE_ + BMeanbuf_SIZE_ + BMeanSquarebuf_SIZE_ + BVarbuf_SIZE_ + workspace_forThre_SIZE_;
        uint32_t UbWOffset_R2_ = UBufAddrStart + CR1buf_SIZE_ + CR2buf_SIZE_ + AMeanbuf_SIZE_ + AMinbuf_SIZE_forY_ + AMaxbuf_SIZE_forY_ + Threbuf_SIZE_ + Zbuf_SIZE_ + BMeanbuf_SIZE_ + BMeanSquarebuf_SIZE_ + BVarbuf_SIZE_ + workspace_forThre_SIZE_ + workspace_forR1_SIZE_;

        uint32_t UbABR1Offset = UBufAddrStart + CR1buf_SIZE_ + CR2buf_SIZE_ + AMeanbuf_SIZE_ + AMinbuf_SIZE_forY_ + AMaxbuf_SIZE_forY_ + Threbuf_SIZE_ + Zbuf_SIZE_ + BMeanbuf_SIZE_ + BMeanSquarebuf_SIZE_ + BVarbuf_SIZE_ + workspace_forThre_SIZE_ + workspace_forR1_SIZE_ + workspace_forR2_SIZE_;
        uint32_t UbABR2Offset = UBufAddrStart + CR1buf_SIZE_ + CR2buf_SIZE_ + AMeanbuf_SIZE_ + AMinbuf_SIZE_forY_ + AMaxbuf_SIZE_forY_ + Threbuf_SIZE_ + Zbuf_SIZE_ + BMeanbuf_SIZE_ + BMeanSquarebuf_SIZE_ + BVarbuf_SIZE_ + workspace_forThre_SIZE_ + workspace_forR1_SIZE_ + workspace_forR2_SIZE_ + ABR1buf_SIZE_;
        
        // Init buffers
        // UbInBRedEvent = 0;

        for (uint32_t i = 0; i < STAGES; i++) {

            UbCR1TensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbCR1Offset + i * (CR1buf_SIZE_ / 2)); // CR1结果矩阵
            UbCR2TensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbCR2Offset + i * (CR2buf_SIZE_ / 2)); // Cr2结果矩阵 

            UbThreTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbThreOffset + i * (Threbuf_SIZE_ / 2)); // Thre为阈值结果矩阵

            UbAMeanTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbAMeanOffset + i * (AMeanbuf_SIZE_ / 2)); // 存储 A 的行 mean 向量
            UbAMinTensorforYList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbAMinOffset_forY + i * (AMinbuf_SIZE_forY_ / 2)); //  存储 A 的行 min 向量（ElementY）
            UbAMaxTensorforYList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbAMaxOffset_forY + i * (AMaxbuf_SIZE_forY_ / 2)); //  存储 A 的行 max 向量 (ElementY)

            UbZTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementZ>(UbZOffset + i * (Zbuf_SIZE_ / 2)); // Z 为比较结果的向量

            /*
                B聚合结果输入向量矩阵
            */
            UbBMeanAbsTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbBMeanOffset + i * (BMeanbuf_SIZE_ / 2));
            UbBMeanSquareTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbBMeanSqaureOffset + i * (BMeanSquarebuf_SIZE_ / 2));
            UbBVarTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbBVarOffset + i * (BVarbuf_SIZE_ / 2));

            UbABR1TensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbABR1Offset + i * (ABR1buf_SIZE_ / 2));
            UbABR2TensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbABR2Offset + i * (ABR2buf_SIZE_ / 2));

            UbWforThreTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbWOffset_thre_ + i * (workspace_forThre_SIZE_ / 2)); // WforThre为计算阈值向量的工作空间

            UbWforCR1TensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbWOffset_R1_ + i * (workspace_forR1_SIZE_  / 2)); // WforCR1为计算Ce累加聚合的工作空间
            UbWforCR2TensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbWOffset_R2_ + i * (workspace_forR2_SIZE_  / 2)); // WforCR2为计算Cr2累积聚合的工作空间
            

            // Assign event ID for each stages
            UbInCREventList[i] = i;
            UbInABRedEventList[i] = i + STAGES;
            UbInABREventList[i] = i + STAGES * 2;

            UbOutEventList[i] = i + 3 * STAGES;
            UbOutZEventList[i] = i + STAGES;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInCREventList[i]);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInABRedEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInABREventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutZEventList[i]);
        }
        // AscendC::SetFlag<AscendC::HardEvent::V_S>(UbInBRedEvent);
    }

    /// Construct
    CATLASS_DEVICE
    BlockFTVerifyNoSplitKPreload(Arch::ResourceAIV<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        uint32_t UbCR1Offset = UBufAddrStart;
        uint32_t UbCR2Offset = UBufAddrStart + CR1buf_SIZE_;

        uint32_t UbAMeanOffset = UBufAddrStart + CR1buf_SIZE_ + CR2buf_SIZE_;
        uint32_t UbAMinOffset_forY = UBufAddrStart + CR1buf_SIZE_ + CR2buf_SIZE_ + AMeanbuf_SIZE_;
        uint32_t UbAMaxOffset_forY = UBufAddrStart + CR1buf_SIZE_ + CR2buf_SIZE_ + AMeanbuf_SIZE_ + AMinbuf_SIZE_forY_;
        uint32_t UbThreOffset = UBufAddrStart + CR1buf_SIZE_ + CR2buf_SIZE_ + AMeanbuf_SIZE_ + AMinbuf_SIZE_forY_ + AMaxbuf_SIZE_forY_;

        uint32_t UbZOffset = UBufAddrStart + CR1buf_SIZE_ + CR2buf_SIZE_ + AMeanbuf_SIZE_ + AMinbuf_SIZE_forY_ + AMaxbuf_SIZE_forY_ + Threbuf_SIZE_;
        uint32_t UbBMeanOffset = UBufAddrStart + CR1buf_SIZE_ + CR2buf_SIZE_ + AMeanbuf_SIZE_ + AMinbuf_SIZE_forY_ + AMaxbuf_SIZE_forY_ + Threbuf_SIZE_ + Zbuf_SIZE_;
        uint32_t UbBMeanSqaureOffset = UBufAddrStart + CR1buf_SIZE_ + CR2buf_SIZE_ + AMeanbuf_SIZE_ + AMinbuf_SIZE_forY_ + AMaxbuf_SIZE_forY_ + Threbuf_SIZE_ + Zbuf_SIZE_ + BMeanbuf_SIZE_;
        uint32_t UbBVarOffset = UBufAddrStart + CR1buf_SIZE_ + CR2buf_SIZE_ + AMeanbuf_SIZE_ + AMinbuf_SIZE_forY_ + AMaxbuf_SIZE_forY_ + Threbuf_SIZE_ + Zbuf_SIZE_ + BMeanbuf_SIZE_ + BMeanSquarebuf_SIZE_;
        uint32_t UbWOffset_thre_ = UBufAddrStart + CR1buf_SIZE_ + CR2buf_SIZE_ + AMeanbuf_SIZE_ + AMinbuf_SIZE_forY_ + AMaxbuf_SIZE_forY_ + Threbuf_SIZE_ + Zbuf_SIZE_ + BMeanbuf_SIZE_ + BMeanSquarebuf_SIZE_ + BVarbuf_SIZE_;
        uint32_t UbWOffset_R1_ = UBufAddrStart + CR1buf_SIZE_ + CR2buf_SIZE_ + AMeanbuf_SIZE_ + AMinbuf_SIZE_forY_ + AMaxbuf_SIZE_forY_ + Threbuf_SIZE_ + Zbuf_SIZE_ + BMeanbuf_SIZE_ + BMeanSquarebuf_SIZE_ + BVarbuf_SIZE_ + workspace_forThre_SIZE_;
        uint32_t UbWOffset_R2_ = UBufAddrStart + CR1buf_SIZE_ + CR2buf_SIZE_ + AMeanbuf_SIZE_ + AMinbuf_SIZE_forY_ + AMaxbuf_SIZE_forY_ + Threbuf_SIZE_ + Zbuf_SIZE_ + BMeanbuf_SIZE_ + BMeanSquarebuf_SIZE_ + BVarbuf_SIZE_ + workspace_forThre_SIZE_ + workspace_forR1_SIZE_;

        uint32_t UbABR1Offset = UBufAddrStart + CR1buf_SIZE_ + CR2buf_SIZE_ + AMeanbuf_SIZE_ + AMinbuf_SIZE_forY_ + AMaxbuf_SIZE_forY_ + Threbuf_SIZE_ + Zbuf_SIZE_ + BMeanbuf_SIZE_ + BMeanSquarebuf_SIZE_ + BVarbuf_SIZE_ + workspace_forThre_SIZE_ + workspace_forR1_SIZE_ + workspace_forR2_SIZE_;
        uint32_t UbABR2Offset = UBufAddrStart + CR1buf_SIZE_ + CR2buf_SIZE_ + AMeanbuf_SIZE_ + AMinbuf_SIZE_forY_ + AMaxbuf_SIZE_forY_ + Threbuf_SIZE_ + Zbuf_SIZE_ + BMeanbuf_SIZE_ + BMeanSquarebuf_SIZE_ + BVarbuf_SIZE_ + workspace_forThre_SIZE_ + workspace_forR1_SIZE_ + workspace_forR2_SIZE_ + ABR1buf_SIZE_;
        
        // Init buffers
        // UbInBRedEvent = 0;

        for (uint32_t i = 0; i < STAGES; i++) {

            UbCR1TensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbCR1Offset + i * (CR1buf_SIZE_ / 2)); // CR1结果矩阵
            UbCR2TensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbCR2Offset + i * (CR2buf_SIZE_ / 2)); // Cr2结果矩阵 

            UbThreTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbThreOffset + i * (Threbuf_SIZE_ / 2)); // Thre为阈值结果矩阵

            UbAMeanTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbAMeanOffset + i * (AMeanbuf_SIZE_ / 2)); // 存储 A 的行 mean 向量
            UbAMinTensorforYList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbAMinOffset_forY + i * (AMinbuf_SIZE_forY_ / 2)); //  存储 A 的行 min 向量（ElementY）
            UbAMaxTensorforYList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbAMaxOffset_forY + i * (AMaxbuf_SIZE_forY_ / 2)); //  存储 A 的行 max 向量 (ElementY)

            UbZTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementZ>(UbZOffset + i * (Zbuf_SIZE_ / 2)); // Z 为比较结果的向量

            /*
                B聚合结果输入向量矩阵
            */
            UbBMeanAbsTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbBMeanOffset + i * (BMeanbuf_SIZE_ / 2));
            UbBMeanSquareTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbBMeanSqaureOffset + i * (BMeanSquarebuf_SIZE_ / 2));
            UbBVarTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbBVarOffset + i * (BVarbuf_SIZE_ / 2));

            UbABR1TensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbABR1Offset + i * (ABR1buf_SIZE_ / 2));
            UbABR2TensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbABR2Offset + i * (ABR2buf_SIZE_ / 2));

            UbWforThreTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbWOffset_thre_ + i * (workspace_forThre_SIZE_ / 2)); // WforThre为计算阈值向量的工作空间

            UbWforCR1TensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbWOffset_R1_ + i * (workspace_forR1_SIZE_  / 2)); // WforCR1为计算Ce累加聚合的工作空间
            UbWforCR2TensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbWOffset_R2_ + i * (workspace_forR2_SIZE_  / 2)); // WforCR2为计算Cr2累积聚合的工作空间
            

            // Assign event ID for each stages
            UbInCREventList[i] = i;
            UbInABRedEventList[i] = i + STAGES;
            UbInABREventList[i] = i + STAGES * 2;

            UbOutEventList[i] = i + 3 * STAGES;
            UbOutZEventList[i] = i + STAGES;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInCREventList[i]);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInABRedEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInABREventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(UbOutZEventList[i]);
        }
        // AscendC::SetFlag<AscendC::HardEvent::V_S>(UbInBRedEvent);
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockFTVerifyNoSplitKPreload()
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInCREventList[i]);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInABRedEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInABREventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(UbOutZEventList[i]);
        }
        // AscendC::WaitFlag<AscendC::HardEvent::V_S>(UbInBRedEvent);
    }

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
        AscendC::GlobalTensor<ElementY> const &gmCR2Out, 
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
        Catlass::Arch::CrossCoreFlagWithReverse<> & flagAicFinishStore,
        // C pointer for online error correction (nullptr to disable)
        GM_ADDR gmC_ptr = nullptr)
    {
        // , AscendC::GlobalTensor<ElementY> const &gmABeOut
        TileMRound = RoundUp(UBTileShape::M, UBAlignHelperOut::ALIGN);
        TileNRound = RoundUp(UBTileShape::N, UBAlignHelperOut::ALIGN);

        ThreTileMRound = TileMRound;
        ThreTileNRound = TileNRound;
        
        BlockMRound = RoundUp(UBBlockShape::M, UBAlignHelperOut::ALIGN);
        BlockNRound = RoundUp(UBBlockShape::N, UBAlignHelperOut::ALIGN);

        ThreBlockMRound = BlockMRound;
        ThreBlockNRound = BlockNRound;

        uint32_t TileNRoundforZByte = (TileNRound + 8 - 1) / 8;
        
        TileNRoundforZ = (TileNRoundforZByte + sizeof(ElementZ) - 1) / sizeof(ElementZ);
        TileNRoundforZ = RoundUp(TileNRoundforZ, ELE_NUM_PER_BLK_OUTZ);

        strideCRCol = layoutCRIn.stride(1) * TileNRound;
        strideCRRow = layoutCRIn.stride(0) * TileMRound;

        strideCRSlice = layoutCRIn.shape(0) * layoutCRIn.shape(1);

        m_actual_total = (actualShape.m() < BlockMRound) ? actualShape.m() : BlockMRound;
        n_actual_total = (actualShape.n() < BlockNRound) ? actualShape.n() : BlockNRound;

        // uint32_t aiv_part_num = 1 * AscendC::GetTaskRation();

        n_actual_part = n_actual_total / aiv_part_num;
        out_z_actual_part = (n_actual_part + 8 - 1) / 8;

        uint32_t N_start_offset = AscendC::GetSubBlockIdx() * n_actual_part;
        uint32_t Z_start_offset = AscendC::GetSubBlockIdx() * out_z_actual_part;

        if(AscendC::GetSubBlockIdx() == (aiv_part_num - 1)) {
            n_actual_part = n_actual_total - N_start_offset;
            out_z_actual_part = (n_actual_part + 8 - 1) / 8;
        }

        uint32_t NloopCR = CeilDiv(n_actual_part, TileNRound);
        uint32_t NloopThre = CeilDiv(n_actual_part, TileNRound);        

        // auto BMeanTile = UbWforMeanTensorList[UbOutListId];

        m_actual = (m_actual_total < TileMRound) ? m_actual_total : TileMRound;
        n_actual = (n_actual_part < TileNRound) ? n_actual_part : TileNRound;

        BlockNPartRound = RoundUp(n_actual_part, UBAlignHelperOut::ALIGN);

        uint32_t BlockNPartRoundforZByte = (BlockNPartRound + 8 - 1) / 8;
        
        BlockNPartRoundforZ = (BlockNPartRoundforZByte + sizeof(ElementZ) - 1) / sizeof(ElementZ);
        BlockNPartRoundforZ = RoundUp(BlockNPartRoundforZ, ELE_NUM_PER_BLK_OUTZ);

        uint32_t nLoopOffset = N_start_offset;
        uint32_t zLoopOffset = Z_start_offset;

        /*
        1. 首先，计算CR的聚合结果
        */
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)(UbOutEventList[UbOutListId]));
        auto UbCR1Tensor = UbCR1TensorList[UbOutListId];
        auto UbCR2Tensor = UbCR2TensorList[UbOutListId];
        
        auto layoutCRInUbTotal = LayoutMY::template MakeLayoutInUb<ElementY>(MakeCoord(BlockMRound, BlockNPartRound));
        auto layoutTileCRTotal = layoutCRIn.GetTileLayout(MakeCoord(m_actual_total, n_actual_part));

        uint32_t CR_row_offset = 0;
        uint32_t CR_col_offset = nLoopOffset;
        uint32_t CR_block_offset = CR_row_offset * layoutCRIn.stride(0) + CR_col_offset * layoutCRIn.stride(1);

        matrixCopyGmToUbforCRIn(UbCR1Tensor, gmCR1Tmp[CR_block_offset], layoutCRInUbTotal, layoutTileCRTotal);
        matrixCopyGmToUbforCRIn(UbCR2Tensor, gmCR2Tmp[CR_block_offset], layoutCRInUbTotal, layoutTileCRTotal);
        
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbOutEventList[UbOutListId]));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbOutEventList[UbOutListId]));

        // AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>((event_t)(UbOutEventforAList[UbOutListId]));

        if(CRSliceNum > 1){
            uint32_t sliceOffset = strideCRSlice;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInCREventList[UbInListId]));

            auto layoutCRInUbSlice = LayoutMY::template MakeLayoutInUb<ElementY>(MakeCoord(BlockMRound, BlockNPartRound));
            auto layoutTileCRSlice = layoutCRIn.GetTileLayout(MakeCoord(m_actual_total, n_actual_part));

            matrixCopyGmToUbforCRIn(UbWforCR1TensorList[UbInListId], 
                    gmCR1Tmp[sliceOffset + CR_block_offset], layoutCRInUbSlice, layoutTileCRSlice);
            matrixCopyGmToUbforCRIn(UbWforCR2TensorList[UbInListId], 
                    gmCR2Tmp[sliceOffset + CR_block_offset], layoutCRInUbSlice, layoutTileCRSlice);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInCREventList[UbInListId]));
        }

        for(uint32_t SliceIdx = 1; SliceIdx < CRSliceNum; SliceIdx++){
            uint32_t sliceOffset = SliceIdx * strideCRSlice;
            uint32_t SliceIdxNext = SliceIdx + 1;
            uint32_t sliceOffsetNext = SliceIdxNext * strideCRSlice;
            uint32_t UbInListIdNext = ((UbInListId + 1) < STAGES) ? (UbInListId + 1) : 0;
            
            if(SliceIdx < (CRSliceNum - 1)){
                // Preload next K FT round data
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInCREventList[UbInListIdNext]));

                auto layoutCRInUbNext = LayoutMY::template MakeLayoutInUb<ElementY>(MakeCoord(BlockMRound, BlockNPartRound));
                auto layoutTileCRNext = layoutCRIn.GetTileLayout(MakeCoord(m_actual_total, n_actual_part));

                matrixCopyGmToUbforCRIn(UbWforCR1TensorList[UbInListIdNext], 
                    gmCR1Tmp[sliceOffsetNext + CR_block_offset], layoutCRInUbNext, layoutTileCRNext);
                matrixCopyGmToUbforCRIn(UbWforCR2TensorList[UbInListIdNext], 
                    gmCR2Tmp[sliceOffsetNext + CR_block_offset], layoutCRInUbNext, layoutTileCRNext);

                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInCREventList[UbInListIdNext]));
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInCREventList[UbInListId]));
            auto layoutComputeInUb = LayoutMY::template MakeLayoutInUb<ElementY>(MakeCoord(BlockMRound, BlockNPartRound));
            auto layoutTileCompute = layoutCRIn.GetTileLayout(MakeCoord(m_actual_total, n_actual_part));

            auto UbCR1InTensor = UbWforCR1TensorList[UbInListId];
            auto UbCR2InTensor = UbWforCR2TensorList[UbInListId];

            tileMatrixAddforCRReduce(UbCR1Tensor, UbCR1InTensor, UbCR1Tensor, layoutComputeInUb, layoutTileCompute);
            tileMatrixAddforCRReduce(UbCR2Tensor, UbCR2InTensor, UbCR2Tensor, layoutComputeInUb, layoutTileCompute);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInCREventList[UbInListId]));
            UbInListId = UbInListIdNext;
        }

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));
        AscendC::SetFlag<AscendC::HardEvent::V_V>((event_t)(UbOutEventList[UbOutListId]));
        
        /*
        2. 计算阈值
        */

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutZEventList[UbOutListId]));
        auto UbThreTensor = UbThreTensorList[UbOutListId];

        AscendC::Duplicate<ElementY>(UbThreTensor, (ElementY)0.0, m_actual_total * BlockNPartRound);

        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_V>((event_t)(UbOutZEventList[UbOutListId]));
            
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInABRedEventList[UbOutListId]));

        auto UbAMinTensor = UbAMinTensorforYList[UbOutListId];
        vecCopyGmToUbInY(UbAMinTensor, gmAMin[N_start_offset], n_actual_part);

        auto UbAMeanTensor = UbAMeanTensorList[UbOutListId];
        vecCopyGmToUbInY(UbAMeanTensor, gmAMean[N_start_offset], n_actual_part);

        auto UbAMaxTensor = UbAMaxTensorforYList[UbOutListId];
        vecCopyGmToUbInY(UbAMaxTensor, gmAMax[N_start_offset], n_actual_part);

        auto UbBMeanAbsTensor = UbBMeanAbsTensorList[UbOutListId];
        vecCopyGmToUbInY(UbBMeanAbsTensor, gmBMeanAbs, m_actual_total);

        auto UbBMeanSquareTensor = UbBMeanSquareTensorList[UbOutListId];
        vecCopyGmToUbInY(UbBMeanSquareTensor, gmBMeanSquare, m_actual_total);

        auto UbBVarTensor = UbBVarTensorList[UbOutListId];
        vecCopyGmToUbInY(UbBVarTensor, gmBVar, m_actual_total);
    
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInABRedEventList[UbOutListId]));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInABRedEventList[UbOutListId]));

        auto layoutStdInUb = layoutAforFT.GetTileLayout(MakeCoord(n_actual_part));
        auto layoutTileStd = layoutAforFT.GetTileLayout(MakeCoord(n_actual_part));

        /*
        (1) 计算Std
        */

        tileStdEstVector(UbAMaxTensor, UbAMeanTensor, UbAMaxTensor, UbAMinTensor, 
                layoutStdInUb, layoutTileStd);

        AscendC::PipeBarrier<PIPE_V>();

        /*
        (2) 计算阈值
        */

        auto layoutThreInUb = layoutThre.GetTileLayout(MakeCoord(BlockMRound, BlockNPartRound));
        auto layoutTileARed = layoutAforFT.GetTileLayout(MakeCoord(n_actual_part));
        auto layoutTileBRed = layoutBforFT.GetTileLayout(MakeCoord(m_actual_total));

            /*
            void operator()(
                AscendC::LocalTensor<ElementY> dstTensor,
                AscendC::LocalTensor<ElementX> srcMeanTensor,
                AscendC::LocalTensor<ElementY> srcStdTensor,
                AscendC::LocalTensor<ElementY> BMeanAbsTensor,
                AscendC::LocalTensor<ElementY> BMeanSquareTensor,
                AscendC::LocalTensor<ElementY> BVarTensor,
                AscendC::LocalTensor<ElementY> thre_workspace,
                LayoutDst const &layoutDst, LayoutSrc const &layoutSrcA, 
                LayoutSrc const &layoutSrcB,
                ElementY n_ratio_factor, ElementY n_sqrt_ratio_factor,
                ElementY n_square_ratio_factor,
                ElementY n_ratio_factor_tail, 
                ElementY n_sqrt_ratio_factor_tail, 
                ElementY n_square_ratio_factor_tail,
                ElementY e_max)
            */
        AscendC::WaitFlag<AscendC::HardEvent::V_V>((event_t)(UbOutZEventList[UbOutListId]));

        tileThreCalcChunk(
            UbThreTensor, UbAMeanTensor, UbAMaxTensor, 
            UbBMeanAbsTensor, UbBMeanSquareTensor,
            UbBVarTensor, UbWforThreTensorList[UbOutListId], 
            layoutThreInUb, layoutTileARed, layoutTileBRed,
            (ElementY)n_ratio_factor, (ElementY)n_sqrt_ratio_factor, 
            (ElementY)n_square_ratio_factor,
            (ElementY)n_ratio_factor_tail, (ElementY)n_sqrt_ratio_factor_tail, 
            (ElementY)n_square_ratio_factor_tail,
            (ElementY)e_max);

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::WaitFlag<AscendC::HardEvent::V_V>((event_t)(UbOutEventList[UbOutListId]));

        /*
        3. 最终进行比较即可
        */

        uint32_t ABR_row_offset = 0;
        uint32_t ABR_col_offset = nLoopOffset;
        uint32_t ABR_block_offset = ABR_row_offset * layoutABRIn.stride(0) + ABR_col_offset * layoutABRIn.stride(1);

        Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInABREventList[UbOutListId]));
        auto layoutABRInUb = layoutABRIn.GetTileLayout(MakeCoord(BlockMRound, BlockNPartRound));
        auto layoutTileABR = layoutABRIn.GetTileLayout(MakeCoord(m_actual_total, n_actual_part));
        matrixCopyGmToUbforABRIn(UbABR1TensorList[UbOutListId], gmABR1[ABR_block_offset], layoutABRInUb, layoutTileABR);
        matrixCopyGmToUbforABRIn(UbABR2TensorList[UbOutListId], gmABR2[ABR_block_offset], layoutABRInUb, layoutTileABR);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInABREventList[UbOutListId]));

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));

        auto layoutDstCROut = layoutCRIn.GetTileLayout(MakeCoord(m_actual_total, n_actual_part));
        auto layoutCROutInUb = LayoutMY::template MakeLayoutInUb<ElementY>(MakeCoord(BlockMRound, BlockNPartRound));

        matrixCopyUbToGmforCROut(gmCR1Out[CR_block_offset],
            UbCR1TensorList[UbOutListId],
            layoutDstCROut, layoutCROutInUb);

        matrixCopyUbToGmforCROut(gmCR2Out[CR_block_offset],
            UbCR2TensorList[UbOutListId],
            layoutDstCROut, layoutCROutInUb);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListId]));
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutEventList[UbOutListId]));

        auto layoutCompareInUb = layoutThre.GetTileLayout(MakeCoord(BlockMRound, BlockNPartRound));
        auto layoutTileCompare = layoutThre.GetTileLayout(MakeCoord(m_actual_total, n_actual_part));

        auto layoutCompareforZInUb = layoutZ.GetTileLayout(MakeCoord(BlockMRound, BlockNPartRoundforZ));

        /*
        CATLASS_DEVICE
        void operator()(
            AscendC::LocalTensor<ElementZ> dstTensor,
            AscendC::LocalTensor<ElementX> srcTensor_x,
            AscendC::LocalTensor<ElementY> srcTensor_y,
            AscendC::LocalTensor<ElementX> srcTensor_thre,
            LayoutDst const &layoutDst, LayoutDst const &layoutDstZ,
            LayoutSrc const &layoutSrc, ElementX threshold
        )
        */

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInABREventList[UbOutListId]));

        // Compute D1 = CR1 - ABR1 before tileCompare overwrites CR1.
        // Reuse UbWforCR1Tensor (freed after CR aggregation).
        {
            auto UbD1Tensor = UbWforCR1TensorList[UbOutListId];
            auto cr1_y = UbCR1TensorList[UbOutListId].template ReinterpretCast<ElementY>();
            auto d1_y = UbD1Tensor.template ReinterpretCast<ElementY>();
            uint32_t d1_count = BlockMRound * BlockNPartRound;
            AscendC::Sub(d1_y, cr1_y, UbABR1TensorList[UbOutListId], d1_count);
            AscendC::PipeBarrier<PIPE_V>();
        }

        tileCompare(
            UbZTensorList[UbOutListId],
            UbCR1TensorList[UbOutListId],
            UbABR1TensorList[UbOutListId],
            UbThreTensorList[UbOutListId],
            layoutCompareInUb,
            layoutCompareforZInUb,
            layoutTileCompare, (ElementY)0.002f);

        AscendC::PipeBarrier<PIPE_V>();

        // Scan Z bitmap for errors
        {
            bool has_error = false;
            for (uint32_t chunk = 0; chunk < m_actual_total && !has_error; chunk++) {
                for (uint32_t zi = 0; zi < out_z_actual_part; zi++) {
                    uint8_t z_byte = UbZTensorList[UbOutListId].GetValue(
                        chunk * BlockNPartRoundforZ + zi);
                    if (z_byte != 0xFF) { has_error = true; break; }
                }
            }
            error_detected_ = has_error;
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutZEventList[UbOutListId]));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutZEventList[UbOutListId]));

        uint32_t Thre_row_offset = 0;
        uint32_t Thre_col_offset = nLoopOffset;
        uint32_t Thre_block_offset = Thre_row_offset * layoutThre.stride(0) + Thre_col_offset * layoutThre.stride(1);

        if(outputThre){
            auto layoutDstYThre = layoutThre.GetTileLayout(MakeCoord(m_actual_total, n_actual_part));
            auto layoutComputeThreInUb = LayoutMY::template MakeLayoutInUb<ElementY>(MakeCoord(BlockMRound, BlockNPartRound));

            matrixCopyUbToGmforThreOut(gmThreZ[Thre_block_offset], 
                UbThreTensorList[UbOutListId], 
                layoutDstYThre, 
                layoutComputeThreInUb);
        }

        uint32_t Z_row_offset = 0;
        uint32_t Z_col_offset = zLoopOffset;
        uint32_t Z_block_offset = Z_row_offset * layoutThre.stride(0) + Z_col_offset * layoutThre.stride(1);

        auto layoutDstZ = layoutZ.GetTileLayout(MakeCoord(m_actual_total, out_z_actual_part));
        auto layoutComputeZInUb = LayoutMZ::template MakeLayoutInUb<ElementZ>(MakeCoord(BlockMRound, BlockNPartRoundforZ));
        // layoutZ.GetTileLayout(MakeCoord(BlockMRound, BlockNPartRoundforZ));

        matrixCopyUbToGmforZ(gmCOMPZ[Z_block_offset], 
            UbZTensorList[UbOutListId], layoutDstZ, layoutComputeZInUb);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(UbOutZEventList[UbOutListId]));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)(UbOutEventList[UbOutListId]));

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInABRedEventList[UbOutListId]));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInABREventList[UbOutListId]));

        last_stage_id_ = UbOutListId;
        UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
    }

    // Online error correction: locate faulty column and fix C matrix.
    // Separated from operator() to avoid affecting hot-path optimization.
    CATLASS_DEVICE
    void correctErrors(
        uint32_t stageId,
        GM_ADDR gmC_ptr, uint32_t n_total,
        uint32_t chunk_base, uint32_t row_offset,
        float lambda, float half_N, int chunk_N)
    {
        AscendC::GlobalTensor<ElementY> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementY *)gmC_ptr);
        auto d1_y = UbWforCR1TensorList[stageId].template ReinterpretCast<ElementY>();
        auto ubWork_y = UbWforThreTensorList[stageId].template ReinterpretCast<ElementY>();

        for (uint32_t chunk = 0; chunk < m_actual_total; chunk++) {
            for (uint32_t zi = 0; zi < out_z_actual_part; zi++) {
                uint8_t z_byte = UbZTensorList[stageId].GetValue(
                    chunk * BlockNPartRoundforZ + zi);
                if (z_byte == 0xFF) continue;

                for (uint32_t bit = 0; bit < 8; bit++) {
                    if ((z_byte >> bit) & 1) continue;
                    uint32_t c_row = zi * 8 + bit;
                    if (c_row >= n_actual_part) break;

                    uint32_t ub_idx = chunk * BlockNPartRound + c_row;
                    float d1 = d1_y.GetValue(ub_idx);
                    float abs_d1 = d1 > 0.0f ? d1 : -d1;
                    if (abs_d1 < 1e-10f) continue;

                    float cr2 = UbCR2TensorList[stageId].GetValue(ub_idx);
                    float abr2 = UbABR2TensorList[stageId].GetValue(ub_idx);
                    float d2 = cr2 - abr2;
                    float ratio = d2 / d1;

                    // arcsinh(ratio) = ln(ratio + sqrt(ratio² + 1))
                    ubWork_y.SetValue(0, ratio);
                    AscendC::Mul(ubWork_y[8], ubWork_y[0], ubWork_y[0], 8);
                    AscendC::PipeBarrier<PIPE_V>();

                    ubWork_y.SetValue(16, ubWork_y.GetValue(8) + 1.0f);
                    AscendC::Sqrt(ubWork_y[24], ubWork_y[16], 8);
                    AscendC::PipeBarrier<PIPE_V>();

                    ubWork_y.SetValue(32, ratio + ubWork_y.GetValue(24));
                    AscendC::Ln(ubWork_y[40], ubWork_y[32], 8);
                    AscendC::PipeBarrier<PIPE_V>();
                    float asinh_val = ubWork_y.GetValue(40);

                    float j_float = half_N + asinh_val / lambda;
                    int j = (int)(j_float + 0.5f);
                    if (j < 0) j = 0;
                    if (j >= chunk_N) j = chunk_N - 1;

                    uint32_t global_row = row_offset + c_row;
                    uint32_t global_col = (chunk_base + chunk) * chunk_N + j;
                    uint32_t aligned_col = (global_col / 8) * 8;
                    int64_t c_offset = (int64_t)global_row * (int64_t)n_total + (int64_t)aligned_col;
                    uint32_t lane_idx = global_col - aligned_col;

                    AscendC::DataCopy(ubWork_y, gmC[c_offset], 8);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(stageId));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(stageId));

                    float c_val = ubWork_y.GetValue(lane_idx);
                    c_val -= d1;
                    ubWork_y.SetValue(lane_idx, c_val);

                    AscendC::DataCopy(gmC[c_offset], ubWork_y, 8);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(stageId));
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(stageId));
                }
            }
        }
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> UbCR1TensorList[STAGES]; // Ce结果矩阵输入
    AscendC::LocalTensor<ElementY> UbCR2TensorList[STAGES]; // Cr2结果矩阵输入

    AscendC::LocalTensor<ElementY> UbThreTensorList[STAGES]; // Thre为阈值结果矩阵
    AscendC::LocalTensor<ElementY> UbABR1TensorList[STAGES]; // ABe的运算结果，共享Amean矩阵的存储空间
    AscendC::LocalTensor<ElementY> UbABR2TensorList[STAGES]; // ABr2的运算结果，共享Amin矩阵的存储空间

    AscendC::LocalTensor<ElementA> UbWforCR1TensorList[STAGES]; // WforCR1为计算Ce累加聚合的工作空间
    AscendC::LocalTensor<ElementY> UbWforCR2TensorList[STAGES]; // WforCR2为计算Cr2累积聚合的工作空间
    AscendC::LocalTensor<ElementY> UbWforThreTensorList[STAGES]; // WforThre为计算阈值向量的工作空间

    
    AscendC::LocalTensor<ElementY> UbAMeanTensorList[STAGES]; // 存储 A 的行 mean 向量
    AscendC::LocalTensor<ElementY> UbAMinTensorforYList[STAGES]; //  存储 A 的行 min 向量（ElementY）
    AscendC::LocalTensor<ElementY> UbAMaxTensorforYList[STAGES]; // 存储 A 的行 max 向量 (ElementY)
    AscendC::LocalTensor<ElementZ> UbZTensorList[STAGES]; // Z 为比较结果的向量

    /*
    B聚合结果输入向量矩阵
    */
    AscendC::LocalTensor<ElementY> UbBMeanAbsTensorList[STAGES];
    AscendC::LocalTensor<ElementY> UbBMeanSquareTensorList[STAGES];
    AscendC::LocalTensor<ElementY> UbBVarTensorList[STAGES];

    /*
    AscendC::GlobalTensor<ElementY> const &gmBMeanAbs,
        AscendC::GlobalTensor<ElementY> const &gmBMeanSquare, 
        AscendC::GlobalTensor<ElementY> const &gmBVar,
    */

    
    

    // Multi-stage event id list
    int32_t UbInCREventList[STAGES]; // CR1，CR2矩阵输入向量
    int32_t UbInABRedEventList[STAGES]; // AB聚合数据输入向量
    int32_t UbInABREventList[STAGES]; // ABR1，ABR2矩阵输入向量

    int32_t UbOutEventList[STAGES];
    int32_t UbOutZEventList[STAGES];

    // int32_t UbInBRedEvent;
    // int32_t UbInMaxEvent;

    // The id of current stage
    uint32_t UbOutListId{0};
public:
    bool error_detected_{false};
    uint32_t last_stage_id_{0};
    uint32_t UbZOutListId{0};
    uint32_t UbInListId{0};

    // ElementY B_slice_meanabs;
    // ElementY B_slice_meansquare;
    // ElementY B_slice_var;
    // ElementY B_slice_var_square;

    uint32_t m_actual, n_actual, x_actual, y_actual;
    uint32_t m_actual_total, n_actual_total, x_actual_total, y_actual_total;
    uint32_t m_actual_next_A, n_actual_next_A, x_actual_next_A, y_actual_next_A;
    uint32_t m_actual_total_next, n_actual_total_next, x_actual_total_next, y_actual_total_next;
    uint32_t n_actual_part;

    // uint32_t thre_n_actual_total;
    // uint32_t thre_n_actual;

    uint32_t dst_offset_ratio;
    uint32_t out_z_actual_part, out_z_actual;

    uint32_t TileMRound, TileNRound;
    uint32_t ThreTileMRound, ThreTileNRound;

    uint32_t TileNRoundforZ;
    uint32_t BlockNPartRoundforZ;

    uint32_t BlockMRound, BlockNRound;
    uint32_t BlockNPartRound;
    uint32_t ThreBlockMRound, ThreBlockNRound;

    uint32_t TaskSplit;
    uint32_t MatrixOffset;
    uint32_t strideCRRow, strideCRCol;

    uint32_t strideCRSlice;

    uint32_t strideOut;
    
    MatrixCopyGmToUb matrixCopyGmToUb;
    VecCopyGmToUb vecCopyGmToUb;
    VecCopyUbToGm vecCopyUbToGm;

    MatrixCopyGmToUbforThre matrixCopyGmToUbforThre; // 用来来数据做阈值计算
    VecCopyGmToUbInY vecCopyGmToUbInY; // 用来拉A聚合数据
    VecCopyUbToGmZ vecCopyUbToGmZ; // 用来输出比较结果

    VecCopyUbToGmforThre vecCopyUbToGmforThre; // 用来输出阈值结果

    TileThreCalc tileThreCalc;
    TileMatrixAdd tileMatrixAdd;

    TileMatrixAddforCRReduce tileMatrixAddforCRReduce;

    TileStdEstMatrix tileStdEstMatrix;
    TileStdEstVector tileStdEstVector;

    TileThreCalcMatrix tileThreCalcMatrix;
    TileThreCalcChunk tileThreCalcChunk;

    // Tile Compare
    TileCompare tileCompare;
    TileStdEst tileStdEst;

    MatrixCopyGmToUbforThreAIn matrixCopyGmToUbforThreAIn;
    MatrixCopyGmToUbforABRIn matrixCopyGmToUbforABRIn;
    MatrixCopyGmToUbforCRIn matrixCopyGmToUbforCRIn;

    MatrixCopyUbToGmforZ matrixCopyUbToGmforZ;
    MatrixCopyUbToGmforCROut matrixCopyUbToGmforCROut;

    MatrixCopyUbToGmforThreOut matrixCopyUbToGmforThreOut;
};

} // namespace Catlass::Gemv::Block

#endif // CATLASS_GEMV_BLOCK_BLOCK_GEMV_AIV_HPP
