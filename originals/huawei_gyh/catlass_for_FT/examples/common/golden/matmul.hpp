/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR dataA PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef EXAMPLES_COMMON_GOLDEN_MATMUL_HPP
#define EXAMPLES_COMMON_GOLDEN_MATMUL_HPP

#include <vector>

#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemv_coord.hpp"

namespace Catlass::golden {

// simple matmul
template<class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementGolden, class LayoutGolden>
void ComputeMatmul(
    const GemmCoord &problemShape,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    const std::vector<ElementB> &dataB, const LayoutB &layoutB,
    std::vector<ElementGolden> &dataGolden, const LayoutGolden &layoutGolden
)
{
    for (uint32_t i = 0; i < problemShape.m(); ++i) {
        for (uint32_t j = 0; j < problemShape.n(); ++j) {
            size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i, j));
            ElementGolden accumulator = 0;
            for (uint32_t k = 0; k < problemShape.k(); ++k) {
                size_t offsetA = layoutA.GetOffset(MakeCoord(i, k));
                size_t offsetB = layoutB.GetOffset(MakeCoord(k, j));
                accumulator += static_cast<ElementGolden>(dataA[offsetA]) * static_cast<ElementGolden>(dataB[offsetB]);
            }
            dataGolden[offsetGolden] = static_cast<ElementGolden>(accumulator);
        }
    }
}

template<class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
std::vector<uint64_t> ComputeAndCompareMatmul(
    const GemmCoord &problemShape,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    const std::vector<ElementB> &dataB, const LayoutB &layoutB,
    std::vector<ElementC> &dataC, const LayoutC &layoutC
)
{
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    float rtol = layoutA.shape(0) < computeNumThreshold ? rtolGeneral : rtolOverThreshold;

    std::vector<uint64_t> errorIndices;
    for (uint32_t i = 0; i < problemShape.m(); ++i) {
        if(errorIndices.size() >= 5) break;

        for (uint32_t j = 0; j < problemShape.n(); ++j) {
            if(errorIndices.size() >= 5) break;

            size_t offsetC = layoutC.GetOffset(MakeCoord(i, j));
            float actualValue = static_cast<float>(dataC[offsetC]);
            float expectValue = 0;

            for (uint32_t k = 0; k < problemShape.k(); ++k) {
                size_t offsetA = layoutA.GetOffset(MakeCoord(i, k));
                size_t offsetB = layoutB.GetOffset(MakeCoord(k, j));
                expectValue += static_cast<float>(dataA[offsetA]) * static_cast<float>(dataB[offsetB]);
            }

            float diff = std::fabs(actualValue - expectValue);
            if (diff > rtol * std::max(1.0f, std::fabs(expectValue))) {
                errorIndices.push_back(offsetC);
                std::cout << "Error at index " << offsetC << ": "
                        << "actual = " << actualValue << ", expect = " << expectValue
                        << ", diff = " << diff << std::endl;
            }
        }
    }
    return errorIndices;
}

////////////////////////////////////
// new add
// simple gemm
template<typename Element, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC, class ElementGolden, class LayoutGolden>
void ComputeGemm(
    const GemmCoord &problemShape,
    Element alpha, Element beta,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    const std::vector<ElementB> &dataB, const LayoutB &layoutB,
    const std::vector<ElementC> &dataC, const LayoutC &layoutC,
    std::vector<ElementGolden> &dataGolden, const LayoutGolden &layoutGolden
)
{
    for (uint32_t i = 0; i < problemShape.m(); ++i) {
        for (uint32_t j = 0; j < problemShape.n(); ++j) {
            size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i, j));
            ElementGolden accumulator = 0;
            for (uint32_t k = 0; k < problemShape.k(); ++k) {
                size_t offsetA = layoutA.GetOffset(MakeCoord(i, k));
                size_t offsetB = layoutB.GetOffset(MakeCoord(k, j));
                accumulator += static_cast<ElementGolden>(alpha) * static_cast<ElementGolden>(dataA[offsetA]) * static_cast<ElementGolden>(dataB[offsetB]);
            }
            dataGolden[offsetGolden] = static_cast<ElementGolden>(beta) * static_cast<ElementGolden>(dataC[offsetGolden]) + static_cast<ElementGolden>(accumulator);
        }
    }
}

template<typename Element, class ElementA, class LayoutA, class ElementX, class LayoutX, class ElementY, class LayoutY, class ElementGolden, class LayoutGolden>
void ComputeGemv(
    const Catlass::GemvCoord &problemShape,
    Element alpha, Element beta,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    const std::vector<ElementX> &dataX, const LayoutX &layoutX,
    const std::vector<ElementY> &dataY, const LayoutY &layoutY,
    std::vector<ElementGolden> &dataGolden, const LayoutGolden &layoutGolden
)
{
    for (uint32_t i = 0; i < problemShape.m(); ++i) {
        size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i));
        ElementGolden accumulator = 0;
        for (uint32_t k = 0; k < problemShape.n(); ++k) {
            size_t offsetA = layoutA.GetOffset(MakeCoord(i, k));
            size_t offsetX = layoutX.GetOffset(MakeCoord(k));
            accumulator += static_cast<ElementGolden>(alpha) *
                          static_cast<ElementGolden>(dataA[offsetA]) *
                          static_cast<ElementGolden>(dataX[offsetX]);
        }
        size_t offsetY = layoutY.GetOffset(MakeCoord(i));
        dataGolden[offsetGolden] = static_cast<ElementGolden>(beta) *
                                  static_cast<ElementGolden>(dataY[offsetY]) +
                                  static_cast<ElementGolden>(accumulator);
    }
}

template<typename Element, class ElementA, class LayoutA, class ElementX, class LayoutX, class ElementY, class LayoutY, class ElementGolden>
void ComputeGemvSlice(
    const Catlass::GemvCoord &problemShape,
    const Catlass::GemvCoord &sliceShape,
    Element alpha, Element beta,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    const std::vector<ElementX> &dataX, const LayoutX &layoutX,
    const std::vector<ElementY> &dataY, const LayoutY &layoutY,
    std::vector<ElementGolden> &dataGolden
)
{
    uint32_t splitNnum = (problemShape.n() + sliceShape.n() - 1) / sliceShape.n();
    uint32_t sliceNums = sliceShape.m();
    uint32_t sliceOutStride = splitNnum * problemShape.m();
    uint32_t sliceInStride = problemShape.m();
    uint32_t sliceRemain = problemShape.n()  - (splitNnum - 1) * sliceShape.n();

    for(uint32_t slice_i=0; slice_i < sliceNums; slice_i++){
        uint32_t slice_i_offset_x = slice_i * problemShape.n();

        for(uint32_t split_j=0; split_j < splitNnum; split_j++){
            uint32_t split_j_offset_x = split_j * sliceShape.n();

            uint32_t now_init_offset_x = slice_i_offset_x + split_j_offset_x;

            for (uint32_t i = 0; i < problemShape.m(); ++i) {
                size_t offsetGolden = split_j * sliceInStride + sliceOutStride * slice_i + i;
                float accumulator = 0.0;
                uint32_t sliceScale = sliceShape.n();
                if(split_j == splitNnum - 1){
                    sliceScale = sliceRemain;
                }else{
                    sliceScale = sliceShape.n();
                }
                for (uint32_t k = 0; k < sliceScale; ++k) {
                    size_t offsetA = layoutA.GetOffset(MakeCoord(i, split_j_offset_x + k));
                    size_t offsetX = k + now_init_offset_x;
                    accumulator += (float)(static_cast<ElementGolden>(alpha) *
                          static_cast<ElementGolden>(dataA[offsetA]) *
                          static_cast<ElementGolden>(dataX[offsetX]));
                }
                size_t offsetY = offsetGolden;
                // layoutY.GetOffset(MakeCoord(i));
                dataGolden[offsetGolden] = static_cast<ElementGolden>(beta) *
                                  static_cast<ElementGolden>(dataY[offsetY]) +
                                  static_cast<ElementGolden>(accumulator);
            }
        }
    } 
}

template<typename Element, class ElementA, class LayoutA, class ElementX, class LayoutX, class ElementY, class LayoutY, class ElementGolden>
void ComputeGemvSliceMultiVector(
    const Catlass::GemvCoord &problemShape,
    const Catlass::GemvCoord &sliceShape,
    Element alpha, Element beta,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    const std::vector<ElementX> &dataX, const LayoutX &layoutX,
    const std::vector<ElementX> &dataXTail, const LayoutX &layoutXTail,
    const std::vector<ElementY> &dataYR1, const std::vector<ElementY> &dataYR2,
    const LayoutY &layoutY,
    std::vector<ElementGolden> &dataGoldenR1, std::vector<ElementGolden> &dataGoldenR2
)
{
    uint32_t splitNnum = (problemShape.n() + sliceShape.n() - 1) / sliceShape.n();
    uint32_t sliceNums = sliceShape.m();
    uint32_t sliceOutStride = splitNnum * problemShape.m();
    uint32_t sliceInStride = problemShape.m();
    uint32_t sliceRemain = problemShape.n()  - (splitNnum - 1) * sliceShape.n();

    for(uint32_t slice_i=0; slice_i < sliceNums; slice_i++){
        uint32_t slice_i_offset_x = slice_i * sliceShape.n() * 2;

        for(uint32_t split_j=0; split_j < splitNnum; split_j++){
            uint32_t split_j_offset_x_r1 = 0;
            uint32_t split_j_offset_x_r2 = sliceShape.n();

            uint32_t split_j_offet_A = split_j * sliceShape.n();

            if(split_j == (splitNnum - 1)){
                split_j_offset_x_r2 = sliceRemain;
            }
            // split_j * sliceShape.n();

            uint32_t now_init_offset_x_r1 = slice_i_offset_x + split_j_offset_x_r1;
            uint32_t now_init_offset_x_r2 = slice_i_offset_x + split_j_offset_x_r2;

            for (uint32_t i = 0; i < problemShape.m(); ++i) {
                size_t offsetGolden = split_j * sliceInStride + sliceOutStride * slice_i + i;
                float accumulator_r1 = 0.0;
                float accumulator_r2 = 0.0;
                uint32_t sliceScale = sliceShape.n();
                if(split_j == splitNnum - 1){
                    sliceScale = sliceRemain;
                }else{
                    sliceScale = sliceShape.n();
                }
                for (uint32_t k = 0; k < sliceScale; ++k) {
                    size_t offsetA = layoutA.GetOffset(MakeCoord(i, split_j_offet_A + k));
                    size_t offsetXR1 = k + now_init_offset_x_r1;
                    size_t offsetXR2 = k + now_init_offset_x_r2;

                    accumulator_r1 += (float)(static_cast<ElementGolden>(alpha) *
                          static_cast<ElementGolden>(dataA[offsetA]) *
                          static_cast<ElementGolden>(dataX[offsetXR1]));
                    
                    accumulator_r2 += (float)(static_cast<ElementGolden>(alpha) *
                          static_cast<ElementGolden>(dataA[offsetA]) *
                          static_cast<ElementGolden>(dataX[offsetXR2]));
                }
                size_t offsetYR1 = offsetGolden;
                size_t offsetYR2 = offsetGolden;

                // layoutY.GetOffset(MakeCoord(i));
                dataGoldenR1[offsetYR1] = static_cast<ElementGolden>(beta) *
                                  static_cast<ElementGolden>(dataYR1[offsetYR1]) +
                                  static_cast<ElementGolden>(accumulator_r1);

                dataGoldenR2[offsetYR2] = static_cast<ElementGolden>(beta) *
                                  static_cast<ElementGolden>(dataYR2[offsetYR2]) +
                                  static_cast<ElementGolden>(accumulator_r2);                
            }
        }
    } 
}

template<typename Element, class ElementA, class LayoutA, class ElementX, class LayoutX, class ElementY, class LayoutY, class ElementGolden>
void ComputeGemvSliceSplitK(
    const Catlass::GemvCoord &problemShape,
    const Catlass::GemvCoord &sliceShape,
    const std::vector<uint32_t> &actualKSliceSize,
    uint32_t SplitKNum,
    Element alpha, Element beta,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    const std::vector<ElementX> &dataX, const LayoutX &layoutX,
    const std::vector<ElementY> &dataY, const LayoutY &layoutY,
    std::vector<ElementGolden> &dataGolden
)
{
    uint32_t TileKNum = (problemShape.n() + sliceShape.n() - 1) / sliceShape.n();
    uint32_t HeadKSliceNum = TileKNum % SplitKNum;
    uint32_t sliceNums = sliceShape.m();
    uint32_t KSliceOutStride = sliceNums * problemShape.m();
    uint32_t sliceInStride = problemShape.m();

    for(uint32_t KSlice_i = 0; KSlice_i < SplitKNum; KSlice_i++){
        uint32_t kActual = actualKSliceSize[0];
        uint32_t KOffset = 0;

        if(KSlice_i >= HeadKSliceNum){
            kActual = (KSlice_i == (SplitKNum - 1)) ? actualKSliceSize[2] : actualKSliceSize[1];

            uint32_t RemainKSliceNum = KSlice_i - HeadKSliceNum;
            KOffset = HeadKSliceNum * actualKSliceSize[0] + RemainKSliceNum * actualKSliceSize[1];
        }else{
            KOffset = KSlice_i * actualKSliceSize[0];
        }

        for(uint32_t MSlice_i=0; MSlice_i < sliceNums; MSlice_i++){
            uint32_t slice_i_offset_row = MSlice_i * problemShape.n();
            uint32_t now_init_offset_x = slice_i_offset_row + KOffset;

            for (uint32_t i = 0; i < problemShape.m(); ++i) {
                size_t offsetGolden = KSliceOutStride * KSlice_i + MSlice_i * sliceInStride + i;
                float accumulator = 0.0;
                uint32_t sliceScale = kActual;

                for (uint32_t k = 0; k < sliceScale; ++k) {
                    size_t offsetA = layoutA.GetOffset(MakeCoord(i, KOffset + k));
                    size_t offsetX = k + now_init_offset_x;
                    accumulator += (float)(static_cast<ElementGolden>(alpha) *
                          static_cast<ElementGolden>(dataA[offsetA]) *
                          static_cast<ElementGolden>(dataX[offsetX]));
                }
                size_t offsetY = offsetGolden;
                // layoutY.GetOffset(MakeCoord(i));
                dataGolden[offsetGolden] = static_cast<ElementGolden>(beta) *
                                  static_cast<ElementGolden>(dataY[offsetY]) +
                                  static_cast<ElementGolden>(accumulator);
            }
        }
    }  
}



// simple grouped gemm
template<typename Element, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC, class ElementGolden, class LayoutGolden>
void ComputeGroupGemm(
    uint32_t problemCount,
    const std::vector<GemmCoord> &problemShapeList,
    const std::vector<Element> &alphaList,
    const std::vector<Element> &betaList,
    const std::vector<ElementA> &dataA, const std::vector<LayoutA> &layoutAList,
    const std::vector<ElementB> &dataB, const std::vector<LayoutB> &layoutBList,
    const std::vector<ElementC> &dataC, const std::vector<LayoutC> &layoutCList,
    std::vector<ElementGolden> &dataGolden, const std::vector<LayoutGolden> &layoutGoldenList
)
{
    size_t inGroupOffsetA = 0;
    size_t inGroupOffsetB = 0;
    size_t inGroupOffsetC = 0;
    size_t inGroupOffsetGolden = 0;

    for (uint32_t inGroupId = 0; inGroupId < problemCount; ++inGroupId) {
        GemmCoord problemShape = problemShapeList[inGroupId];
        Element alpha = alphaList[inGroupId];
        Element beta = betaList[inGroupId];

        for (uint32_t i = 0; i < problemShape.m(); ++i) {
            for (uint32_t j = 0; j < problemShape.n(); ++j) {
                size_t offsetGolden = inGroupOffsetGolden + layoutGoldenList[inGroupId].GetOffset(MakeCoord(i, j));
                ElementGolden accumulator = 0;

                for (uint32_t k = 0; k < problemShape.k(); ++k) {
                    size_t offsetA = inGroupOffsetA + layoutAList[inGroupId].GetOffset(MakeCoord(i, k));
                    size_t offsetB = inGroupOffsetB + layoutBList[inGroupId].GetOffset(MakeCoord(k, j));
                    accumulator += static_cast<ElementGolden>(alpha) * static_cast<ElementGolden>(dataA[offsetA]) * static_cast<ElementGolden>(dataB[offsetB]);
                }

                size_t offsetC = inGroupOffsetC + layoutCList[inGroupId].GetOffset(MakeCoord(i, j));
                dataGolden[offsetGolden] = static_cast<ElementGolden>(beta) * static_cast<ElementGolden>(dataC[offsetC]) + static_cast<ElementGolden>(accumulator);
            }
        }

        inGroupOffsetA += static_cast<size_t>(problemShape.m()) * problemShape.k();
        inGroupOffsetB += static_cast<size_t>(problemShape.k()) * problemShape.n();
        inGroupOffsetC += static_cast<size_t>(problemShape.m()) * problemShape.n();
        inGroupOffsetGolden += static_cast<size_t>(problemShape.m()) * problemShape.n();
    }
}
////////////////////////////////////

// simple batched matmul
template<class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementGolden, class LayoutGolden>
void ComputeBatchedMatmul(
    const uint32_t batchedCount, const GemmCoord &problemShape,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    const std::vector<ElementB> &dataB, const LayoutB &layoutB,
    std::vector<ElementGolden> &dataC, const LayoutGolden &layoutGolden
)
{
    for (uint32_t batchId = 0; batchId < batchedCount; ++batchId) {
        size_t batchOffsetA = static_cast<size_t>(problemShape.m()) * problemShape.k() * batchId;
        size_t batchOffsetB = static_cast<size_t>(problemShape.k()) * problemShape.n() * batchId;
        size_t batchoffsetGolden = static_cast<size_t>(problemShape.m()) * problemShape.n() * batchId;
        for (uint32_t i = 0; i < problemShape.m(); ++i) {
            for (uint32_t j = 0; j < problemShape.n(); ++j) {
                size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i, j)) + batchoffsetGolden;
                ElementGolden accumulator = 0;
                for (uint32_t k = 0; k < problemShape.k(); ++k) {
                    size_t offsetA = layoutA.GetOffset(MakeCoord(i, k)) + batchOffsetA;
                    size_t offsetB = layoutB.GetOffset(MakeCoord(k, j)) + batchOffsetB;
                    accumulator += static_cast<ElementGolden>(dataA[offsetA]) *
                        static_cast<ElementGolden>(dataB[offsetB]);
                }
                dataC[offsetGolden] = static_cast<ElementGolden>(accumulator);
            }
        }
    }
}

// simple grouped matmul
template<class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementGolden, class LayoutGolden>
void ComputeGroupedMatmul(
    uint32_t problemCount,
    const std::vector<GemmCoord> &problemShapeList,
    const std::vector<ElementA> &dataA, const std::vector<LayoutA> &layoutAList,
    const std::vector<ElementB> &dataB, const std::vector<LayoutB> &layoutBList,
    std::vector<ElementGolden> &dataGolden, const std::vector<LayoutGolden> &layoutGoldenList
)
{
    size_t inGroupOffsetA = 0;
    size_t inGroupOffsetB = 0;
    size_t inGroupOffsetGolden = 0;
    for (uint32_t inGroupId = 0; inGroupId < problemCount; ++inGroupId) {
        GemmCoord problemShape = problemShapeList[inGroupId];
        for (uint32_t i = 0; i < problemShape.m(); ++i) {
            for (uint32_t j = 0; j < problemShape.n(); ++j) {
                size_t offsetGolden = inGroupOffsetGolden + layoutGoldenList[inGroupId].GetOffset(MakeCoord(i, j));
                ElementGolden accumulator = 0;
                for (uint32_t k = 0; k < problemShape.k(); ++k) {
                    size_t offsetA = inGroupOffsetA + layoutAList[inGroupId].GetOffset(MakeCoord(i, k));
                    size_t offsetB = inGroupOffsetB + layoutBList[inGroupId].GetOffset(MakeCoord(k, j));
                    accumulator += static_cast<ElementGolden>(dataA[offsetA]) *
                        static_cast<ElementGolden>(dataB[offsetB]);
                }
                dataGolden[offsetGolden] = static_cast<ElementGolden>(accumulator);
            }
        }
        inGroupOffsetA += static_cast<size_t>(problemShape.m()) * problemShape.k();
        inGroupOffsetB += static_cast<size_t>(problemShape.k()) * problemShape.n();
        inGroupOffsetGolden += static_cast<size_t>(problemShape.m()) * problemShape.n();
    }
}

// matmul add
template<
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementX,  // Layout X must be same as LayoutGolden
    class ElementGolden, class LayoutGolden
>
void ComputeMatmulElemWiseAdd(
    const GemmCoord &problemShape,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    const std::vector<ElementB> &dataB, const LayoutB &layoutB,
    const std::vector<ElementX> &dataX,  // layoutX must be same as layoutGolden
    std::vector<ElementGolden> &dataGolden, const LayoutGolden &layoutGolden
)
{
    for (uint32_t i = 0; i < problemShape.m(); ++i) {
        for (uint32_t j = 0; j < problemShape.n(); ++j) {
            ElementGolden accumulator = 0;
            for (uint32_t k = 0; k < problemShape.k(); ++k) {
                size_t offsetA = layoutA.GetOffset(MakeCoord(i, k));
                size_t offsetB = layoutB.GetOffset(MakeCoord(k, j));
                accumulator += static_cast<ElementGolden>(dataA[offsetA]) * static_cast<ElementGolden>(dataB[offsetB]);
            }
            size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i, j));
            dataGolden[offsetGolden] = accumulator + static_cast<ElementGolden>(dataX[offsetGolden]);
        }
    }
}

template <
    class ElementGroupList, class ElementScale,
    class LayoutB, class LayoutScale, class LayoutPerTokenScale
>
void ComputeGroupedMatmulPerTokenDequant(
    const GemmCoord &problemShape, uint32_t problemCount, const std::vector<ElementGroupList> &groupList,
    const std::vector<int8_t> &dataA, const layout::RowMajor &layoutA,
    const std::vector<int8_t> &dataB, const LayoutB &layoutB,
    const std::vector<ElementScale> &dataScale, const LayoutScale &,
    const std::vector<ElementScale> &dataPerTokenScale, const LayoutPerTokenScale &,
    std::vector<float> &dataGolden, const layout::RowMajor &layoutGolden
)
{
    size_t groupOffsetB = 0;
    size_t groupOffsetScale = 0;
    uint32_t startRow = 0;
    for (uint32_t inGroupId = 0; inGroupId < problemCount; ++inGroupId) {
        for (uint32_t i = startRow; i < groupList[inGroupId]; ++i) {
            for (uint32_t j = 0; j < problemShape.n(); ++j) {
                size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i, j));
                int32_t accumulator = 0;
                for (uint32_t k = 0; k < problemShape.k(); ++k) {
                    size_t offsetA = layoutA.GetOffset(MakeCoord(i, k));
                    size_t offsetB = groupOffsetB + layoutB.GetOffset(MakeCoord(k, j));
                    accumulator += static_cast<int32_t>(dataA[offsetA]) * static_cast<int32_t>(dataB[offsetB]);
                }
                dataGolden[offsetGolden] = static_cast<float>(accumulator) *
                    static_cast<float>(dataScale[groupOffsetScale + j]) *
                    static_cast<float>(dataPerTokenScale[i]);
            }
        }
        groupOffsetB += static_cast<size_t>(problemShape.k()) * problemShape.n();
        groupOffsetScale += static_cast<size_t>(problemShape.n());
        startRow = groupList[inGroupId];
    }
}

template <
    class LayoutA,
    class LayoutB,
    class ElementScale
>
void QuantMatmul(
    const GemmCoord &problemShape,
    const std::vector<int8_t> &dataA, const LayoutA &layoutA,
    const std::vector<int8_t> &dataB, const LayoutB &layoutB,
    const std::vector<ElementScale> &dataScale, const layout::VectorLayout &layoutScale,
    const std::vector<ElementScale> &dataPerTokenScale, const layout::VectorLayout &layoutPerTokenScale,
    std::vector<float> &dataGolden, const layout::RowMajor &layoutGolden
)
{
    for (uint32_t i = 0; i < problemShape.m(); ++i) {
        for (uint32_t j = 0; j < problemShape.n(); ++j) {
            int32_t accumulator = 0;
            for (uint32_t k = 0; k < problemShape.k(); ++k) {
                size_t offsetA = layoutA.GetOffset(MakeCoord(i, k));
                size_t offsetB = layoutB.GetOffset(MakeCoord(k, j));
                accumulator += static_cast<int32_t>(dataA[offsetA]) * static_cast<int32_t>(dataB[offsetB]);
            }
            size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i, j));
            dataGolden[offsetGolden] = static_cast<float>(accumulator) *
                static_cast<float>(dataScale[j]) *
                static_cast<float>(dataPerTokenScale[i]);
        }
    }
}

template <
    class ElementGroupList, class ElementScale, class LayoutScale, class LayoutPerTokenScale
>
void ComputeGroupedMatmulSliceKPerTokenDequant(
    const GemmCoord &problemShape, uint32_t problemCount, const std::vector<ElementGroupList> &groupList,
    const std::vector<int8_t> &dataA, const layout::ColumnMajor &layoutA,
    const std::vector<int8_t> &dataB, const layout::RowMajor &layoutB,
    const std::vector<ElementScale> &dataScale, const LayoutScale &,
    const std::vector<ElementScale> &dataPerTokenScale, const LayoutPerTokenScale &,
    std::vector<float> &dataGolden, const layout::RowMajor &layoutGolden
)
{
    size_t groupOffsetD = 0;
    size_t groupOffsetScale = 0;
    size_t groupOffsetPerTokenScale = 0;
    uint32_t startRow = 0;
    for (uint32_t inGroupId = 0; inGroupId < problemCount; ++inGroupId) {
        for (uint32_t i = 0; i < problemShape.m(); ++i) {
            for (uint32_t j = 0; j < problemShape.n(); ++j) {
                size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i, j));
                int32_t accumulator = 0;
                for (uint32_t k = startRow; k < groupList[inGroupId]; ++k) {
                    size_t offsetA = layoutA.GetOffset(MakeCoord(i, k));
                    size_t offsetB = layoutB.GetOffset(MakeCoord(k, j));
                    accumulator += static_cast<int32_t>(dataA[offsetA]) * static_cast<int32_t>(dataB[offsetB]);
                }
                dataGolden[groupOffsetD+offsetGolden] = static_cast<float>(accumulator) *
                    static_cast<float>(dataScale[groupOffsetScale + j]) *
                    static_cast<float>(dataPerTokenScale[groupOffsetPerTokenScale + i]);
            }
        }

        groupOffsetD += static_cast<size_t>(problemShape.m()) * problemShape.n();
        groupOffsetScale += static_cast<size_t>(problemShape.n());
        groupOffsetPerTokenScale += static_cast<size_t>(problemShape.m());
        startRow = groupList[inGroupId];
    }
}

template<class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementGolden, class LayoutGolden, class ElementBias>
void ComputeMatmulBias(
    const GemmCoord &problemShape,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    const std::vector<ElementB> &dataB, const LayoutB &layoutB,
    const std::vector<ElementBias> &dataBias,
    std::vector<ElementGolden> &dataGolden, const LayoutGolden &layoutGolden
)
{
    for (uint32_t i = 0; i < problemShape.m(); ++i) {
        for (uint32_t j = 0; j < problemShape.n(); ++j) {
            size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i, j));
            ElementGolden accumulator = static_cast<ElementGolden>(dataBias[j]);
            for (uint32_t k = 0; k < problemShape.k(); ++k) {
                size_t offsetA = layoutA.GetOffset(MakeCoord(i, k));
                size_t offsetB = layoutB.GetOffset(MakeCoord(k, j));
                accumulator += static_cast<ElementGolden>(dataA[offsetA]) * static_cast<ElementGolden>(dataB[offsetB]);
            }
            dataGolden[offsetGolden] = accumulator;
        }
    }
}

} // namespace Catlass::golden

#endif // EXAMPLES_COMMON_GOLDEN_MATMUL_HPP
