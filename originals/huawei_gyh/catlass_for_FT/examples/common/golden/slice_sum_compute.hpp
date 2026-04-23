#ifndef EXAMPLES_COMMON_GOLDEN_SLICESUM_DATA_HPP
#define EXAMPLES_COMMON_GOLDEN_SLICESUM_DATA_HPP

#include <cmath>
#include <vector>
#include <string>
#include <cstdio>
#include <algorithm> // std::max_element

#include "catlass/gemm_coord.hpp"
#include "catlass/gemv/helper.hpp"

/*
const std::vector<ElementX> &dataX, const LayoutX &layoutX,
const std::vector<ElementY> &dataY, const LayoutY &layoutY,
*/

// class ElementX, class LayoutX, class ElementY, 
    // class LayoutY, 

namespace Catlass::golden {
template<class ElementA, class LayoutA, 
    class ElementGolden, class LayoutGolden>
void ComputeSliceSum(
    const Catlass::GemvCoord &problemShape,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataGolden, 
    const LayoutGolden &layoutGolden)
{

    uint32_t m = problemShape.m();
    uint32_t n = problemShape.n();

    for (uint32_t i = 0; i < n; ++i) {
        size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i));
        dataGolden[offsetGolden] = (ElementGolden)0.0;
        ElementGolden accumulator = (ElementGolden)0.0;
        for(uint32_t j = 0; j < m; j++){
            size_t offsetA = layoutA.GetOffset(MakeCoord(j,i));
            accumulator += static_cast<ElementGolden>(dataA[offsetA]);
        }
        dataGolden[offsetGolden] = accumulator;
    }
}

template<class ElementA, class LayoutA, 
    class ElementGolden, class LayoutGolden>
void ComputeSliceSumSplitK(
    const Catlass::GemvCoord &problemShape,
    uint32_t SplitKNum,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataGolden, 
    const LayoutGolden &layoutGolden)
{

    uint32_t m = problemShape.m();
    uint32_t n = problemShape.n();

    uint32_t KSliceOutStride = m * n;
    for (uint32_t i = 0; i < m; ++i) {
        for(uint32_t j = 0; j < n; j++){
            size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i,j));
            dataGolden[offsetGolden] = (ElementGolden)0.0;
            ElementGolden accumulator = (ElementGolden)0.0;
            size_t offsetA = layoutA.GetOffset(MakeCoord(i,j));
            for(uint32_t KSlice_i=0; KSlice_i < SplitKNum; KSlice_i++){
                size_t slice_offsetA = KSlice_i * KSliceOutStride + offsetA;
                accumulator += static_cast<ElementGolden>(dataA[slice_offsetA]);
            }
            dataGolden[offsetGolden] = accumulator;
        }       
    } 
}


}
#endif