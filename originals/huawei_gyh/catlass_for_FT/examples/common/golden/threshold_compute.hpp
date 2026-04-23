#ifndef EXAMPLES_COMMON_GOLDEN_THRESHOLD_DATA_HPP
#define EXAMPLES_COMMON_GOLDEN_THRESHOLD_DATA_HPP

#include <cmath>
#include <vector>
#include <string>
#include <cstdio>
// #include <cmath>
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
template<typename Element, class ElementA, class LayoutA, 
    class ElementGolden, class LayoutGolden>
void ComputeThresholds(
    const Catlass::GemvCoord &problemShape,
    Element round_exp, Element beta,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataGolden, 
    const LayoutGolden &layoutGolden, Catlass::Gemv::helper::FT_RCE_THRE_TYPE rce_thre_type 
)
{
    float input_exponent = (round_exp < 0.0f) ? round_exp : (0.0 - round_exp);

    float rounding_error = std::pow(2.0f,input_exponent);

    float row_sqrt = 1.0f;

    uint32_t m = problemShape.m();
    uint32_t n = problemShape.n();

    if(beta < 1.0f){
        row_sqrt = std::sqrt(n*1.0f);
    }else{
        row_sqrt = beta;
    }



    ElementGolden alpha = static_cast<ElementGolden>(row_sqrt * rounding_error);

    if(rce_thre_type == Catlass::Gemv::helper::FT_RCE_THRE_TYPE::ROUND_WITH_ACC){
        float acc_rounding_error = std::pow(2.0f, -23.0f);
        float acc_scaling_factor = 1.0f * n*(n+1)*(2*n+1) / 48.0f;
        acc_scaling_factor = std::sqrt(acc_scaling_factor);
        alpha = static_cast<ElementGolden>(row_sqrt * rounding_error + acc_rounding_error * acc_scaling_factor); 
    }
    
    for (uint32_t i = 0; i < m; ++i) {
        size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i));
        size_t offsetA_start = layoutA.GetOffset(MakeCoord(i,(uint32_t)0));
        size_t offsetA_end = layoutA.GetOffset(MakeCoord(i,n));

        // auto it = std::max_element(v.begin() + l, v.begin() + r);
        // static_cast<ElementGolden>
        auto tmp_max_it = (std::max_element(dataA.begin() + offsetA_start, dataA.begin() + offsetA_end));
        ElementGolden tmp_max_result = *tmp_max_it;
        tmp_max_result = std::abs(tmp_max_result);

        auto tmp_min_it = std::min_element(dataA.begin() + offsetA_start, dataA.begin() + offsetA_end);
        ElementGolden tmp_min_result = *tmp_min_it;
        tmp_min_result = std::abs(tmp_min_result);

        ElementGolden tmp_result = (tmp_max_result > tmp_min_result) ? tmp_max_result : tmp_min_result;

        dataGolden[offsetGolden] = tmp_result * alpha;

        // ElementGolden accumulator = 0;
        // for (uint32_t k = 0; k < problemShape.n(); ++k) {
        //     size_t offsetA = layoutA.GetOffset(MakeCoord(i, k));
        //     size_t offsetX = layoutX.GetOffset(MakeCoord(k));
        //     accumulator += static_cast<ElementGolden>(alpha) *
        //                   static_cast<ElementGolden>(dataA[offsetA]) *
        //                   static_cast<ElementGolden>(dataX[offsetX]);
        // }
        // size_t offsetY = layoutY.GetOffset(MakeCoord(i));
        // dataGolden[offsetGolden] = static_cast<ElementGolden>(beta) *
        //                           static_cast<ElementGolden>(dataY[offsetY]) +
        //                           static_cast<ElementGolden>(accumulator);
    }
}

template<class ElementA, class LayoutA, 
    class ElementGolden, class LayoutGolden>
void ComputeMatrixTransposeSimple(
    const Catlass::GemvCoord &problemShape,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataGolden, 
    const LayoutGolden &layoutGolden
)
{
    uint32_t m = problemShape.m();
    uint32_t n = problemShape.n();

    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            size_t offsetA = layoutA.GetOffset(MakeCoord(i, j));
            size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(j, i));

            dataGolden[offsetGolden] = static_cast<ElementGolden>(dataA[offsetA]);
        }
    }
}

template<class ElementA, class LayoutA, 
    class ElementGolden, class LayoutGolden>
void ComputeMatrixTransposeSplitKSimple(
    const Catlass::GemvCoord &problemShape,
    const std::vector<uint32_t> &actualKSliceSize,
    uint32_t SplitKNum, uint32_t HeadKSliceNum,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataGolden, 
    const LayoutGolden &layoutGolden
)
{
    uint32_t m = problemShape.m();
    uint32_t n = problemShape.n();

    uint32_t KSliceOutStride = m * n;

    for(uint32_t KSlice_i=0; KSlice_i < SplitKNum; KSlice_i++){
        for (uint32_t i = 0; i < m; ++i) {
            for (uint32_t j = 0; j < n; ++j) {
                size_t offsetA = layoutA.GetOffset(MakeCoord(i, j));
                size_t slice_offsetA = KSlice_i * KSliceOutStride + offsetA;

                size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(j, i));
                size_t slice_offsetGolden = KSlice_i * KSliceOutStride + offsetGolden;

                dataGolden[slice_offsetGolden] = static_cast<ElementGolden>(dataA[slice_offsetA]);
            }
        }
    }
}

template<class ElementA, class LayoutA, 
    class ElementGolden, class LayoutGolden>
void ComputeMatrixTransposeRobust(
    const Catlass::GemvCoord &problemShape,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataGolden,
    std::vector<ElementGolden> &dataMeanGolden,
    const LayoutGolden &layoutGolden, uint32_t aligned_n
)
{
    uint32_t m = problemShape.m();
    uint32_t n = problemShape.n();

    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            size_t offsetA = layoutA.GetOffset(MakeCoord(i, j));
            size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(j, i));

            dataGolden[offsetGolden] = static_cast<ElementGolden>(dataA[offsetA]);
        }
        size_t offsetAMean = layoutA.GetOffset(MakeCoord(i, aligned_n));
        size_t offsetMeanGolden = i;
        dataMeanGolden[offsetMeanGolden] = static_cast<ElementGolden>(dataA[offsetAMean]);
    }
}

template<class ElementA, class LayoutA, 
    class ElementGolden, class LayoutGolden>
void ComputeMatrixTransposeSplitKRobust(
    const Catlass::GemvCoord &problemShape,
    const std::vector<uint32_t> &actualKSliceSize,
    uint32_t SplitKNum, uint32_t HeadKSliceNum,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataGolden,
    std::vector<ElementGolden> &dataMeanGolden,
    const LayoutGolden &layoutGolden, uint32_t aligned_n
)
{
    uint32_t m = problemShape.m();
    uint32_t n = problemShape.n();

    uint32_t KSliceInStride = m * (aligned_n+1);
    uint32_t KSliceOutStride = m * n;
    uint32_t KSliceOutMeanStride = m;

    for(uint32_t KSlice_i=0; KSlice_i < SplitKNum; KSlice_i++){
        for (uint32_t i = 0; i < m; ++i) {
            for (uint32_t j = 0; j < n; ++j) {
                size_t offsetA = layoutA.GetOffset(MakeCoord(i, j));
                size_t slice_offsetA = KSlice_i * KSliceInStride + offsetA;

                size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(j, i));
                size_t slice_offsetGolden = KSlice_i * KSliceOutStride + offsetGolden;

                dataGolden[slice_offsetGolden] = static_cast<ElementGolden>(dataA[slice_offsetA]);
            }
            size_t offsetAMean = layoutA.GetOffset(MakeCoord(i, aligned_n));
            size_t slice_offsetAMean = KSlice_i * KSliceInStride + offsetAMean;
            size_t offsetMeanGolden = i;
            size_t slice_offsetMeanGolden = KSlice_i * KSliceOutMeanStride + offsetMeanGolden;
            dataMeanGolden[slice_offsetMeanGolden] = static_cast<ElementGolden>(dataA[slice_offsetAMean]);
        }
    } 
}

template<typename Element, class ElementA, class LayoutA, 
    class ElementGolden, class LayoutGolden>
void ComputeThresholdsAABFT(
    const Catlass::GemvCoord &problemShape,
    Element round_exp, Element beta,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataGolden, 
    const LayoutGolden &layoutGolden, Catlass::Gemv::helper::FT_RCE_THRE_TYPE rce_thre_type 
)
{
    float input_exponent = (round_exp < 0.0f) ? round_exp : (0.0 - round_exp);

    float rounding_error = std::pow(2.0f,input_exponent);

    float row_sqrt = 1.0f;

    uint32_t m = problemShape.m();
    uint32_t n = problemShape.n();

    if(beta < 1.0f){
        row_sqrt = std::sqrt(n*1.0f);
    }else{
        row_sqrt = beta;
    }



    ElementGolden alpha = static_cast<ElementGolden>(row_sqrt * rounding_error);

    if(rce_thre_type == Catlass::Gemv::helper::FT_RCE_THRE_TYPE::ROUND_WITH_ACC){
        float acc_rounding_error = std::pow(2.0f, -23.0f);
        float acc_scaling_factor = 1.0f * n*(n+1)*(2*n+1) / 48.0f;
        acc_scaling_factor = std::sqrt(acc_scaling_factor);
        alpha = static_cast<ElementGolden>(row_sqrt * rounding_error + acc_rounding_error * acc_scaling_factor); 
    }
    
    for (uint32_t i = 0; i < m; ++i) {
        size_t offsetGolden = layoutGolden.GetOffset(MakeCoord(i));
        size_t offsetA_start = layoutA.GetOffset(MakeCoord(i,(uint32_t)0));
        size_t offsetA_end = layoutA.GetOffset(MakeCoord(i,n));

        // auto it = std::max_element(v.begin() + l, v.begin() + r);
        // static_cast<ElementGolden>
        auto tmp_max_it = (std::max_element(dataA.begin() + offsetA_start, dataA.begin() + offsetA_end));
        ElementGolden tmp_max_result = *tmp_max_it;
        tmp_max_result = std::abs(tmp_max_result);

        auto tmp_min_it = std::min_element(dataA.begin() + offsetA_start, dataA.begin() + offsetA_end);
        ElementGolden tmp_min_result = *tmp_min_it;
        tmp_min_result = std::abs(tmp_min_result);

        ElementGolden tmp_result = (tmp_max_result > tmp_min_result) ? tmp_max_result : tmp_min_result;

        dataGolden[offsetGolden] = tmp_result * alpha;

        // ElementGolden accumulator = 0;
        // for (uint32_t k = 0; k < problemShape.n(); ++k) {
        //     size_t offsetA = layoutA.GetOffset(MakeCoord(i, k));
        //     size_t offsetX = layoutX.GetOffset(MakeCoord(k));
        //     accumulator += static_cast<ElementGolden>(alpha) *
        //                   static_cast<ElementGolden>(dataA[offsetA]) *
        //                   static_cast<ElementGolden>(dataX[offsetX]);
        // }
        // size_t offsetY = layoutY.GetOffset(MakeCoord(i));
        // dataGolden[offsetGolden] = static_cast<ElementGolden>(beta) *
        //                           static_cast<ElementGolden>(dataY[offsetY]) +
        //                           static_cast<ElementGolden>(accumulator);
    }
}

template<typename Element, class ElementA, class LayoutA, 
    class ElementGolden, class LayoutGolden>
void ComputeThresholdsAABFTSlice(
    const Catlass::GemvCoord &problemShape,
    const Catlass::GemvCoord &sliceShape,
    Element round_exp, Element beta,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataGolden, 
    const LayoutGolden &layoutGolden, Catlass::Gemv::helper::FT_RCE_THRE_TYPE rce_thre_type 
)
{
    uint32_t splitNnum = (problemShape.n() + sliceShape.n() - 1) / sliceShape.n();
    uint32_t splitMnum = (problemShape.m() + sliceShape.m() - 1) / sliceShape.m();

    float input_exponent = (round_exp < 0.0f) ? round_exp : (0.0 - round_exp);

    float rounding_error = std::pow(2.0f,input_exponent);

    float row_sqrt = 1.0f;

    uint32_t m = problemShape.m();
    uint32_t n = problemShape.n();

    uint32_t slice_n = sliceShape.n();
    uint32_t slice_m = sliceShape.m();

    if(beta < 1.0f){
        row_sqrt = std::sqrt(slice_n*1.0f);
    }else{
        row_sqrt = beta;
    }

    ElementGolden alpha = static_cast<ElementGolden>(row_sqrt * rounding_error);

    if(rce_thre_type == Catlass::Gemv::helper::FT_RCE_THRE_TYPE::ROUND_WITH_ACC){
        float acc_rounding_error = std::pow(2.0f, -23.0f);
        float acc_scaling_factor = 1.0f * slice_n*(slice_n+1)*(2*slice_n+1) / 48.0f;
        acc_scaling_factor = std::sqrt(acc_scaling_factor);
        alpha = static_cast<ElementGolden>(row_sqrt * rounding_error + acc_rounding_error * acc_scaling_factor); 
    }

    uint32_t sliceNStride = problemShape.m();
    uint32_t sliceMStride = sliceShape.m();

    for(uint32_t split_n_i=0; split_n_i < splitNnum; split_n_i++){
        uint32_t slice_n_start = split_n_i * sliceShape.n();
        uint32_t slice_n_end = (split_n_i + 1) * sliceShape.n();
        for(uint32_t split_m_j=0; split_m_j < splitMnum; split_m_j++){
            uint32_t slice_m_start = split_m_j * sliceShape.m();
            for(uint32_t i=0; i < slice_m; i++){
                size_t offsetGolden = split_n_i * sliceNStride + split_m_j * sliceMStride + i;
                size_t offsetA_start = layoutA.GetOffset(MakeCoord(slice_m_start + i,slice_n_start));
                size_t offsetA_end = layoutA.GetOffset(MakeCoord(slice_m_start + i, slice_n_end));

                auto tmp_max_it = (std::max_element(dataA.begin() + offsetA_start, dataA.begin() + offsetA_end));

                ElementGolden tmp_max_result = *tmp_max_it;
                tmp_max_result = std::abs(tmp_max_result);

                auto tmp_min_it = std::min_element(dataA.begin() + offsetA_start, dataA.begin() + offsetA_end);
                ElementGolden tmp_min_result = *tmp_min_it;
                tmp_min_result = std::abs(tmp_min_result);

                ElementGolden tmp_result = (tmp_max_result > tmp_min_result) ? tmp_max_result : tmp_min_result;
                dataGolden[offsetGolden] = tmp_result * alpha;
            }
        }
    }
}


template<class ElementA, class LayoutA, 
    class ElementGolden>
void ComputeMeanMaxSlice(
    const Catlass::GemvCoord &problemShape,
    const Catlass::GemvCoord &sliceShape,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataMeanGolden,
    std::vector<ElementGolden> &dataMaxGolden)
{
    uint32_t splitNnum = (problemShape.n() + sliceShape.n() - 1) / sliceShape.n();

    uint32_t split_remain = problemShape.n() % sliceShape.n();

    float common_size = 1.0f * problemShape.m() * sliceShape.n();

    float remain_size = common_size;

    if(split_remain > 0){
        remain_size = 1.0f * problemShape.m() * split_remain;
    }

    uint32_t sliceNStride = problemShape.m();
    uint32_t sliceMStride = sliceShape.m();

    for(uint32_t split_n_i=0; split_n_i < splitNnum; split_n_i++){
        uint32_t slice_n_start = split_n_i * sliceShape.n();
        uint32_t slice_n_end = (split_n_i + 1) * sliceShape.n();
        if(split_n_i == (splitNnum - 1)){
            slice_n_end = problemShape.n();
        }
        ElementGolden slice_max = 0.0f;
        ElementGolden slice_mean = 0.0f;

        size_t offsetGolden = split_n_i;

        for(uint32_t i=0; i < problemShape.m(); i++){
            size_t offsetA_start = layoutA.GetOffset(MakeCoord(i,slice_n_start));
            size_t offsetA_end = layoutA.GetOffset(MakeCoord(i, slice_n_end));

            for(uint32_t j=offsetA_start; j < offsetA_end; j++){
                slice_mean += static_cast<ElementGolden>(dataA[j]);
            }

            auto tmp_max_it = (std::max_element(dataA.begin() + offsetA_start, dataA.begin() + offsetA_end));
            ElementGolden tmp_max_result = *tmp_max_it;
            if(tmp_max_result > slice_max){
                slice_max = tmp_max_result;
            }
        }

        if(split_n_i < (splitNnum - 1)){
            slice_mean = slice_mean / static_cast<ElementGolden>(common_size);
        }else{
            slice_mean = slice_mean / static_cast<ElementGolden>(remain_size);
        }
        
        dataMaxGolden[split_n_i] = slice_max;
        dataMeanGolden[split_n_i] = slice_mean;
    }
}

template<class ElementA, class LayoutA, 
    class ElementGolden>
void ComputeMeanAbsSquareVarSliceRobust(
    const Catlass::GemvCoord &problemShape,
    const Catlass::GemvCoord &sliceShape,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataMeanAbsGolden,
    std::vector<ElementGolden> &dataMeanSquareGolden,
    std::vector<ElementGolden> &dataVarGolden)
{
    uint32_t splitNnum = (problemShape.n() + sliceShape.n() - 1) / sliceShape.n();

    uint32_t split_remain = problemShape.n() % sliceShape.n();

    float common_size = 1.0f * sliceShape.n();

    float remain_size = common_size;

    if(split_remain > 0){
        remain_size = 1.0f * split_remain;
    }

    uint32_t sliceNStride = problemShape.m();
    uint32_t sliceMStride = sliceShape.m();

    for(uint32_t split_n_i=0; split_n_i < splitNnum; split_n_i++){
        uint32_t slice_n_start = split_n_i * sliceShape.n();
        uint32_t slice_n_end = (split_n_i + 1) * sliceShape.n();
        if(split_n_i == (splitNnum - 1)){
            slice_n_end = problemShape.n();
        }
        ElementGolden slice_max = 0.0f;
        ElementGolden slice_min = 0.0f;
        ElementGolden slice_mean = 0.0f;
        
        ElementGolden slice_mean_abs = 0.0f;
        ElementGolden slice_mean_square = 0.0f;
        ElementGolden slice_var = 0.0f;

        size_t offsetGolden = split_n_i;

        for(uint32_t i=0; i < problemShape.m(); i++){
            size_t offsetA_start = layoutA.GetOffset(MakeCoord(i,slice_n_start));
            size_t offsetA_end = layoutA.GetOffset(MakeCoord(i, slice_n_end));
            ElementGolden row_mean = 0.0f;
            ElementGolden row_mean_abs = 0.0f;
            ElementGolden row_mean_square = 0.0f;

            ElementGolden row_var = 0.0f;
            ElementGolden row_max = 0.0f;
            ElementGolden row_min = 0.0f;

            for(uint32_t j=offsetA_start; j < offsetA_end; j++){
                row_mean += static_cast<ElementGolden>(dataA[j]);
            }
            if(split_n_i < (splitNnum - 1)){
                row_mean = row_mean / static_cast<ElementGolden>(common_size);
            }else{
                row_mean = row_mean / static_cast<ElementGolden>(remain_size);
            }
            
            if(row_mean < 0.0f){
                row_mean_abs = -row_mean;
            }else{
                row_mean_abs = row_mean;
            }

            row_mean_square = row_mean * row_mean;
            slice_mean += row_mean;
            slice_mean_abs += row_mean_abs;
            slice_mean_square += row_mean_square;

            auto tmp_max_it = (std::max_element(dataA.begin() + offsetA_start, dataA.begin() + offsetA_end));
            auto tmp_min_it = (std::min_element(dataA.begin() + offsetA_start, dataA.begin() + offsetA_end));

            row_max = *tmp_max_it;
            row_min = *tmp_min_it;

            row_var = (row_max - row_mean) * (row_mean - row_min);

            if(row_max > slice_max){
                slice_max = row_max;
            }

            if(row_min < slice_min){
                slice_min = row_min;
            }

            slice_var += row_var;
        }

        slice_mean = slice_mean / static_cast<ElementGolden>(problemShape.m());

        dataMeanAbsGolden[split_n_i] = slice_mean_abs;
        dataMeanSquareGolden[split_n_i] = slice_mean_square;
        dataVarGolden[split_n_i] = std::sqrt(slice_var * 1.0f);
    }
}

template<class ElementA, class LayoutA, 
    class ElementGolden>
void ComputeMeanAbsSquareVarSliceSimple(
    const Catlass::GemvCoord &problemShape,
    const Catlass::GemvCoord &sliceShape,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataMeanAbsGolden,
    std::vector<ElementGolden> &dataMeanSquareGolden,
    std::vector<ElementGolden> &dataVarGolden)
{
    uint32_t splitNnum = (problemShape.n() + sliceShape.n() - 1) / sliceShape.n();

    uint32_t split_remain = problemShape.n() % sliceShape.n();

    float common_size = 1.0f * sliceShape.n();

    float remain_size = common_size;

    if(split_remain > 0){
        remain_size = 1.0f * split_remain;
    }

    uint32_t sliceNStride = problemShape.m();
    uint32_t sliceMStride = sliceShape.m();

    for(uint32_t split_n_i=0; split_n_i < splitNnum; split_n_i++){
        uint32_t slice_n_start = split_n_i * sliceShape.n();
        uint32_t slice_n_end = (split_n_i + 1) * sliceShape.n();
        if(split_n_i == (splitNnum - 1)){
            slice_n_end = problemShape.n();
        }
        ElementGolden slice_max = 0.0f;
        ElementGolden slice_min = 0.0f;
        ElementGolden slice_mean = 0.0f;
        
        ElementGolden slice_mean_abs = 0.0f;
        ElementGolden slice_mean_square = 0.0f;
        ElementGolden slice_var = 0.0f;

        size_t offsetGolden = split_n_i;

        for(uint32_t i=0; i < problemShape.m(); i++){
            size_t offsetA_start = layoutA.GetOffset(MakeCoord(i,slice_n_start));
            size_t offsetA_end = layoutA.GetOffset(MakeCoord(i, slice_n_end));
            ElementGolden row_mean = 0.0f;
            ElementGolden row_mean_abs = 0.0f;
            ElementGolden row_mean_square = 0.0f;

            ElementGolden row_var = 0.0f;
            ElementGolden row_max = 0.0f;
            ElementGolden row_min = 0.0f;

            // for(uint32_t j=offsetA_start; j < offsetA_end; j++){
            //     row_mean += static_cast<ElementGolden>(dataA[j]);
            // }
            // if(split_n_i < (splitNnum - 1)){
            //     row_mean = row_mean / static_cast<ElementGolden>(common_size);
            // }else{
            //     row_mean = row_mean / static_cast<ElementGolden>(remain_size);
            // }
            
            if(row_mean < 0.0f){
                row_mean_abs = -row_mean;
            }else{
                row_mean_abs = row_mean;
            }

            row_mean_square = row_mean * row_mean;
            slice_mean += row_mean;
            slice_mean_abs += row_mean_abs;
            slice_mean_square += row_mean_square;

            auto tmp_max_it = (std::max_element(dataA.begin() + offsetA_start, dataA.begin() + offsetA_end));
            // auto tmp_min_it = (std::min_element(dataA.begin() + offsetA_start, dataA.begin() + offsetA_end));

            row_max = (ElementGolden)(*tmp_max_it);
            // row_min = *tmp_min_it;

            row_var = (row_max) * (row_max);

            if(row_max > slice_max){
                slice_max = row_max;
            }

            if(row_min < slice_min){
                slice_min = row_min;
            }

            slice_var += row_var;
        }

        slice_mean = slice_mean / static_cast<ElementGolden>(problemShape.m());

        dataMeanAbsGolden[split_n_i] = slice_mean_abs;
        dataMeanSquareGolden[split_n_i] = slice_mean_square;
        dataVarGolden[split_n_i] = std::sqrt(slice_var * 1.0f);
    }
}

template<class ElementA, class LayoutA, 
    class ElementGolden>
void ComputeMeanAbsSquareVarSliceSplitKSimple(
    const Catlass::GemvCoord &problemShape,
    const Catlass::GemvCoord &sliceShape,
    const std::vector<uint32_t> &actualKSliceSize,
    uint32_t SplitKNum, uint32_t HeadKSliceNum,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataMeanAbsGolden,
    std::vector<ElementGolden> &dataMeanSquareGolden,
    std::vector<ElementGolden> &dataVarGolden)
{
    uint32_t splitNnum = (problemShape.n() + sliceShape.n() - 1) / sliceShape.n();

    uint32_t align = 32 / sizeof(ElementGolden);
    uint32_t split_block_num = (splitNnum + align - 1) / align;

    uint32_t SplitNnumAligned = split_block_num * align + align;
    uint32_t KSliceOutStride = SplitNnumAligned;

    uint32_t split_remain = problemShape.n() % sliceShape.n();

    float common_size = 1.0f * sliceShape.n();

    float remain_size = common_size;

    if(split_remain > 0){
        remain_size = 1.0f * split_remain;
    }

    
    uint32_t sliceNStride = problemShape.m();
    uint32_t sliceMStride = sliceShape.m();

    for(uint32_t KSlice_i=0; KSlice_i < SplitKNum; KSlice_i++){
        uint32_t kActual = actualKSliceSize[0];
        uint32_t KOffset = 0;

        if(KSlice_i >= HeadKSliceNum){
            kActual = (KSlice_i == (SplitKNum - 1)) ? actualKSliceSize[2] : actualKSliceSize[1];

            uint32_t RemainKSliceNum = KSlice_i - HeadKSliceNum;
            KOffset = HeadKSliceNum * actualKSliceSize[0] + RemainKSliceNum * actualKSliceSize[1];
        }else{
            KOffset = KSlice_i * actualKSliceSize[0];
        }

        uint32_t KSliceOutOffset = KSlice_i * KSliceOutStride;

        for(uint32_t split_n_i=0; split_n_i < splitNnum; split_n_i++){

            uint32_t slice_n_start = split_n_i * sliceShape.n();
            uint32_t slice_n_end = (split_n_i + 1) * sliceShape.n();

            if(split_n_i == (splitNnum - 1)){
                slice_n_end = problemShape.n();
            }

            ElementGolden slice_max = 0.0f;
            ElementGolden slice_min = 0.0f;
            ElementGolden slice_mean = 0.0f;
        
            ElementGolden slice_mean_abs = 0.0f;
            ElementGolden slice_mean_square = 0.0f;
            ElementGolden slice_var = 0.0f;

            size_t offsetGoldenInSlice = split_n_i;
            size_t KSliceInOffset = KOffset;

            for(uint32_t i=0; i < kActual; i++){
                size_t offsetA_start = layoutA.GetOffset(MakeCoord((uint32_t)KSliceInOffset + i, slice_n_start));
                size_t offsetA_end = layoutA.GetOffset(MakeCoord((uint32_t)KSliceInOffset + i, slice_n_end));

                ElementGolden row_mean = 0.0f;
                ElementGolden row_mean_abs = 0.0f;
                ElementGolden row_mean_square = 0.0f;

                ElementGolden row_var = 0.0f;
                ElementGolden row_max = 0.0f;
                ElementGolden row_min = 0.0f;
            
                if(row_mean < 0.0f){
                    row_mean_abs = -row_mean;
                }else{
                    row_mean_abs = row_mean;
                }

                row_mean_square = row_mean * row_mean;
                slice_mean += row_mean;
                slice_mean_abs += row_mean_abs;
                slice_mean_square += row_mean_square;

                auto tmp_max_it = (std::max_element(dataA.begin() + offsetA_start, dataA.begin() + offsetA_end));
                // auto tmp_min_it = (std::min_element(dataA.begin() + offsetA_start, dataA.begin() + offsetA_end));

                row_max = (ElementGolden)(*tmp_max_it);
                // row_min = *tmp_min_it;

                row_var = (row_max) * (row_max);

                if(row_max > slice_max){
                    slice_max = row_max;
                }

                if(row_min < slice_min){
                    slice_min = row_min;
                }

                slice_var += row_var;
            }

            slice_mean = slice_mean / static_cast<ElementGolden>(kActual);

            uint32_t slice_total_offset = KSliceOutOffset + split_n_i;

            dataMeanAbsGolden[slice_total_offset] = slice_mean_abs;
            dataMeanSquareGolden[slice_total_offset] = slice_mean_square;
            dataVarGolden[slice_total_offset] = std::sqrt(slice_var * 1.0f);
        }
    }
}

template<class ElementA, class LayoutA, 
    class ElementGolden>
void ComputeMeanAbsSquareVarSliceSplitKRobust(
    const Catlass::GemvCoord &problemShape,
    const Catlass::GemvCoord &sliceShape,
    const std::vector<uint32_t> &actualKSliceSize,
    uint32_t SplitKNum, uint32_t HeadKSliceNum,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataMeanAbsGolden,
    std::vector<ElementGolden> &dataMeanSquareGolden,
    std::vector<ElementGolden> &dataVarGolden)
{
    uint32_t splitNnum = (problemShape.n() + sliceShape.n() - 1) / sliceShape.n();

    uint32_t align = 32 / sizeof(ElementGolden);
    uint32_t split_block_num = (splitNnum + align - 1) / align;

    uint32_t SplitNnumAligned = split_block_num * align + align;
    uint32_t KSliceOutStride = SplitNnumAligned;

    uint32_t split_remain = problemShape.n() % sliceShape.n();

    float common_size = 1.0f * sliceShape.n();

    float remain_size = common_size;

    if(split_remain > 0){
        remain_size = 1.0f * split_remain;
    }

    uint32_t sliceNStride = problemShape.m();
    uint32_t sliceMStride = sliceShape.m();

    for(uint32_t KSlice_i=0; KSlice_i < SplitKNum; KSlice_i++){
        uint32_t kActual = actualKSliceSize[0];
        uint32_t KOffset = 0;

        if(KSlice_i >= HeadKSliceNum){
            kActual = (KSlice_i == (SplitKNum - 1)) ? actualKSliceSize[2] : actualKSliceSize[1];

            uint32_t RemainKSliceNum = KSlice_i - HeadKSliceNum;
            KOffset = HeadKSliceNum * actualKSliceSize[0] + RemainKSliceNum * actualKSliceSize[1];
        }else{
            KOffset = KSlice_i * actualKSliceSize[0];
        }

        uint32_t KSliceOutOffset = KSlice_i * KSliceOutStride;

        for(uint32_t split_n_i=0; split_n_i < splitNnum; split_n_i++){

            uint32_t slice_n_start = split_n_i * sliceShape.n();
            uint32_t slice_n_end = (split_n_i + 1) * sliceShape.n();

            if(split_n_i == (splitNnum - 1)){
                slice_n_end = problemShape.n();
            }

            ElementGolden slice_max = 0.0f;
            ElementGolden slice_min = 0.0f;
            ElementGolden slice_mean = 0.0f;
        
            ElementGolden slice_mean_abs = 0.0f;
            ElementGolden slice_mean_square = 0.0f;
            ElementGolden slice_var = 0.0f;

            size_t offsetGoldenInSlice = split_n_i;
            size_t KSliceInOffset = KOffset;

            for(uint32_t i=0; i < kActual; i++){
                size_t offsetA_start = layoutA.GetOffset(MakeCoord((uint32_t)KSliceInOffset + i, slice_n_start));
                size_t offsetA_end = layoutA.GetOffset(MakeCoord((uint32_t)KSliceInOffset + i, slice_n_end));

                ElementGolden row_mean = 0.0f;
                ElementGolden row_mean_abs = 0.0f;
                ElementGolden row_mean_square = 0.0f;

                ElementGolden row_var = 0.0f;
                ElementGolden row_max = 0.0f;
                ElementGolden row_min = 0.0f;

                for(uint32_t j=offsetA_start; j < offsetA_end; j++){
                    row_mean += static_cast<ElementGolden>(dataA[j]);
                }
                if(split_n_i < (splitNnum - 1)){
                    row_mean = row_mean / static_cast<ElementGolden>(common_size);
                }else{
                    row_mean = row_mean / static_cast<ElementGolden>(remain_size);
                }
            
                if(row_mean < 0.0f){
                    row_mean_abs = -row_mean;
                }else{
                    row_mean_abs = row_mean;
                }

                row_mean_square = row_mean * row_mean;
                slice_mean += row_mean;
                slice_mean_abs += row_mean_abs;
                slice_mean_square += row_mean_square;

                auto tmp_max_it = (std::max_element(dataA.begin() + offsetA_start, dataA.begin() + offsetA_end));
                auto tmp_min_it = (std::min_element(dataA.begin() + offsetA_start, dataA.begin() + offsetA_end));

                row_max = *tmp_max_it;
                row_min = *tmp_min_it;

                row_var = (row_max - row_mean) * (row_mean - row_min);

                if(row_max > slice_max){
                    slice_max = row_max;
                }

                if(row_min < slice_min){
                    slice_min = row_min;
                }

                slice_var += row_var;
            }

            slice_mean = slice_mean / static_cast<ElementGolden>(kActual);

            uint32_t slice_total_offset = KSliceOutOffset + split_n_i;

            dataMeanAbsGolden[slice_total_offset] = slice_mean_abs;
            dataMeanSquareGolden[slice_total_offset] = slice_mean_square;
            dataVarGolden[slice_total_offset] = std::sqrt(slice_var * 1.0f);
        }
    }
}

template<class ElementA, class LayoutA, 
    class ElementGolden>
void ComputeMeanAbsSquareVarSliceSimpleExt(
    const Catlass::GemvCoord &problemShape,
    const Catlass::GemvCoord &sliceShape,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataMeanAbsGolden,
    std::vector<ElementGolden> &dataMeanSquareGolden,
    std::vector<ElementGolden> &dataVarGolden, 
    uint32_t simpling_stride, uint32_t blockMSize,uint32_t elementSize)
{
    uint32_t elem_align_size = 32 / elementSize;
    uint32_t splitNnum = (problemShape.n() + sliceShape.n() - 1) / sliceShape.n();

    uint32_t split_remain = problemShape.n() % sliceShape.n();

    float common_size = 1.0f * sliceShape.n();

    float remain_size = common_size;

    if(split_remain > 0){
        remain_size = 1.0f * split_remain;
    }

    
    uint32_t sliceNStride = problemShape.m();
    uint32_t sliceMStride = sliceShape.m();

    uint32_t UBBlockKRoundforB = ((blockMSize + elem_align_size - 1) / elem_align_size) * elem_align_size;

    uint32_t BlockKforBNumAlign = problemShape.m() / UBBlockKRoundforB;
    uint32_t TotalKforBAlign = BlockKforBNumAlign * UBBlockKRoundforB;
    uint32_t ChunkKforBAlign = simpling_stride * UBBlockKRoundforB;

    uint32_t OutKforBAlign = 0;
    //uint32_t UBTileKRound = 1;
    //uint32_t UBTileMRound = 1;

    uint32_t loopsNumKforB = (TotalKforBAlign + ChunkKforBAlign - 1) / ChunkKforBAlign;

    if(loopsNumKforB < 2){
        OutKforBAlign = (TotalKforBAlign < UBBlockKRoundforB) ? TotalKforBAlign : UBBlockKRoundforB;
    }else{
        OutKforBAlign = (loopsNumKforB - 1) * UBBlockKRoundforB;
        uint32_t OutKforBAlignRemain = TotalKforBAlign - OutKforBAlign;
        OutKforBAlignRemain = (OutKforBAlignRemain < UBBlockKRoundforB) ? OutKforBAlignRemain : UBBlockKRoundforB;
        OutKforBAlign = OutKforBAlign + OutKforBAlignRemain;
    }

    for(uint32_t split_n_i=0; split_n_i < splitNnum; split_n_i++){
        uint32_t slice_n_start = split_n_i * sliceShape.n();
        uint32_t slice_n_end = (split_n_i + 1) * sliceShape.n();
        if(split_n_i == (splitNnum - 1)){
            slice_n_end = problemShape.n();
        }
        ElementGolden slice_max = 0.0f;
        ElementGolden slice_min = 0.0f;
        ElementGolden slice_mean = 0.0f;
        
        ElementGolden slice_mean_abs = 0.0f;
        ElementGolden slice_mean_square = 0.0f;
        ElementGolden slice_var = 0.0f;

        size_t offsetGolden = split_n_i;

        for(uint32_t chunk_i=0; chunk_i < loopsNumKforB; chunk_i++){
            uint32_t offsetA_start_row = ChunkKforBAlign * chunk_i;
            uint32_t slice_m_size = UBBlockKRoundforB;

            if(chunk_i == (loopsNumKforB - 1)){
                slice_m_size = TotalKforBAlign - offsetA_start_row;
                slice_m_size = (slice_m_size < UBBlockKRoundforB) ? slice_m_size : UBBlockKRoundforB;
            }

            for(uint32_t i=0; i < slice_m_size; i++){
                size_t offsetA_start = layoutA.GetOffset(MakeCoord(offsetA_start_row + i,slice_n_start));
                size_t offsetA_end = layoutA.GetOffset(MakeCoord(offsetA_start_row + i, slice_n_end));

                ElementGolden row_mean = 0.0f;
                ElementGolden row_mean_abs = 0.0f;
                ElementGolden row_mean_square = 0.0f;

                ElementGolden row_var = 0.0f;
                ElementGolden row_max = 0.0f;
                ElementGolden row_min = 0.0f;

                // for(uint32_t j=offsetA_start; j < offsetA_end; j++){
                //     row_mean += static_cast<ElementGolden>(dataA[j]);
                // }
                // if(split_n_i < (splitNnum - 1)){
                //     row_mean = row_mean / static_cast<ElementGolden>(common_size);
                // }else{
                //     row_mean = row_mean / static_cast<ElementGolden>(remain_size);
                // }
            
                if(row_mean < 0.0f){
                    row_mean_abs = -row_mean;
                }else{
                    row_mean_abs = row_mean;
                }

                row_mean_square = row_mean * row_mean;
                slice_mean += row_mean;
                slice_mean_abs += row_mean_abs;
                slice_mean_square += row_mean_square;

                auto tmp_max_it = (std::max_element(dataA.begin() + offsetA_start, dataA.begin() + offsetA_end));
                // auto tmp_min_it = (std::min_element(dataA.begin() + offsetA_start, dataA.begin() + offsetA_end));

                row_max = (ElementGolden)(*tmp_max_it);
                // row_min = *tmp_min_it;

                row_var = (row_max) * (row_max);
                if(chunk_i == 0 && i == 0){
                    slice_max = row_max;
                    slice_min = row_min;
                }else{
                    if(row_max > slice_max){
                        slice_max = row_max;
                    }

                    if(row_min < slice_min){
                        slice_min = row_min;
                    }
                }

                slice_var += row_var;
            }   
        }

        slice_mean = slice_mean * simpling_stride / static_cast<ElementGolden>(problemShape.m());
        
        slice_mean_abs = slice_mean_abs *1.0f * simpling_stride;
        slice_mean_square = slice_mean_square * 1.0f * simpling_stride;
        slice_var = slice_var * 1.0f * simpling_stride;

        dataMeanAbsGolden[split_n_i] = slice_mean_abs;
        dataMeanSquareGolden[split_n_i] = slice_mean_square;
        dataVarGolden[split_n_i] = std::sqrt(slice_var * 1.0f);
    }
}

template<class ElementA, class LayoutA, 
    class ElementGolden>
void ComputeMeanMaxMinRow(
    const Catlass::GemvCoord &problemShape,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataMeanGolden,
    std::vector<ElementGolden> &dataMaxGolden,
    std::vector<ElementGolden> &dataMinGolden)
{
    std::cout << "dataMaxGolden.size() = " << dataMaxGolden.size() << "\n";
    std::cout << "dataMeanGolden.size() = " << dataMeanGolden.size() << "\n";
    std::cout << "dataMinGolden.size() = " << dataMinGolden.size() << "\n";
    std::cout << "problemShape.m() = " << problemShape.m() << "\n";
    uint32_t splitMnum = problemShape.m();

    float common_size = 1.0f * problemShape.n();

    float remain_size = common_size;

    uint32_t sliceNStride = problemShape.m();

    for(uint32_t i=0; i < problemShape.m(); i++){

        ElementGolden slice_max = 0.0f;
        ElementGolden slice_min = 0.0f;
        ElementGolden slice_mean = 0.0f;

        size_t offsetA_start = layoutA.GetOffset(MakeCoord(i,(uint32_t)0));
        size_t offsetA_end = layoutA.GetOffset(MakeCoord(i, problemShape.n()));

        for(uint32_t j=offsetA_start; j < offsetA_end; j++){
            slice_mean += static_cast<ElementGolden>(dataA[j]);
        }

        auto tmp_max_it = (std::max_element(dataA.begin() + offsetA_start, dataA.begin() + offsetA_end));
        auto tmp_min_it = (std::min_element(dataA.begin() + offsetA_start, dataA.begin() + offsetA_end));

        ElementA tmp_max_result = *tmp_max_it;
        ElementA tmp_min_result = *tmp_min_it;
        // printf("tmp_max_result: %f\n", tmp_max_result);
        slice_max = (ElementGolden)tmp_max_result;
        slice_min = (ElementGolden)tmp_min_result;
        slice_mean = slice_mean / static_cast<ElementGolden>(common_size);
        dataMaxGolden[i] = (ElementGolden)tmp_max_result;
        // printf("dataMaxGolden[i]: %f\n", dataMaxGolden[i]);
        dataMeanGolden[i] = slice_mean;
        dataMinGolden[i] = slice_min;
    }
}

template<class ElementA, class LayoutA, 
    class ElementGolden>
void ComputeMeanMaxMinRowSplitK(
    const Catlass::GemvCoord &problemShape,
    const std::vector<uint32_t> &actualKSliceSize,
    uint32_t SplitKNum, uint32_t HeadKSliceNum,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataMeanGolden,
    std::vector<ElementGolden> &dataMaxGolden,
    std::vector<ElementGolden> &dataMinGolden)
{
    uint32_t splitMnum = problemShape.m();

    uint32_t KSliceOutStride = problemShape.m();

    float common_size = 1.0f * problemShape.n();

    float remain_size = common_size;

    uint32_t sliceNStride = problemShape.m();

    for(uint32_t KSlice_i=0; KSlice_i < SplitKNum; KSlice_i++){
        uint32_t kActual = actualKSliceSize[0];
        uint32_t KOffset = 0;

        if(KSlice_i >= HeadKSliceNum){
            kActual = (KSlice_i == (SplitKNum - 1)) ? actualKSliceSize[2] : actualKSliceSize[1];

            uint32_t RemainKSliceNum = KSlice_i - HeadKSliceNum;
            KOffset = HeadKSliceNum * actualKSliceSize[0] + RemainKSliceNum * actualKSliceSize[1];
        }else{
            KOffset = KSlice_i * actualKSliceSize[0];
        }

        uint32_t KSliceOutOffset = KSlice_i * KSliceOutStride;
        uint32_t KSliceInOffset = KOffset;

        for(uint32_t i=0; i < problemShape.m(); i++){

            ElementGolden slice_max = 0.0f;
            ElementGolden slice_min = 0.0f;
            ElementGolden slice_mean = 0.0f;

            size_t offsetA_start = layoutA.GetOffset(MakeCoord(i, KSliceInOffset));
            size_t offsetA_end = layoutA.GetOffset(MakeCoord(i, (KSliceInOffset + kActual)));

            for(uint32_t j=offsetA_start; j < offsetA_end; j++){
                slice_mean += static_cast<ElementGolden>(dataA[j]);
            }

            auto tmp_max_it = (std::max_element(dataA.begin() + offsetA_start, dataA.begin() + offsetA_end));
            auto tmp_min_it = (std::min_element(dataA.begin() + offsetA_start, dataA.begin() + offsetA_end));

            ElementA tmp_max_result = *tmp_max_it;
            ElementA tmp_min_result = *tmp_min_it;
            slice_max = (ElementGolden)tmp_max_result;
            slice_min = (ElementGolden)tmp_min_result;
            slice_mean = slice_mean / static_cast<ElementGolden>((kActual * 1.0f));
            
            uint32_t slice_total_offset = KSliceOutOffset + i;
            dataMaxGolden[slice_total_offset] = slice_max;
            dataMeanGolden[slice_total_offset] = slice_mean;
            dataMinGolden[slice_total_offset] = slice_min;
        }
    }
}


template<class ElementA, class LayoutA, 
    class ElementGolden>
void ComputeMeanMaxMinRowSimplingC(
    const Catlass::GemvCoord &problemShape,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataMeanGolden,
    std::vector<ElementGolden> &dataMaxGolden,
    std::vector<ElementGolden> &dataMinGolden,
    uint32_t simpling_stride, uint32_t simpling_window)
{
    uint32_t RowNSize = problemShape.n();

    uint32_t simpling_window_head = simpling_window / simpling_stride;
    uint32_t simpling_window_num = (RowNSize + simpling_window - 1) / simpling_window;
    uint32_t splitMnum = problemShape.m();

    float common_size = 1.0f * problemShape.n();

    float remain_size = common_size;

    uint32_t sliceNStride = problemShape.m();

    for(uint32_t i=0; i < problemShape.m(); i++){

        ElementGolden row_max = 0.0f;
        ElementGolden row_min = 0.0f;
        ElementGolden row_mean = 0.0f;
        for(uint32_t j=0; j<simpling_window_num; j++){
            ElementGolden slice_max = 0.0f;
            ElementGolden slice_min = 0.0f;

            size_t offsetA_start = layoutA.GetOffset(MakeCoord(i,(uint32_t)(j*simpling_window)));
            size_t offsetA_end = layoutA.GetOffset(MakeCoord(i, (uint32_t)(j*simpling_window+simpling_window_head)));
            
            if(j == (simpling_window_num - 1)){
                uint32_t remain_window_size = problemShape.n() - j*simpling_window;
                if(remain_window_size < simpling_window_head){
                    offsetA_end = layoutA.GetOffset(MakeCoord(i, problemShape.n()));   
                }else{
                    offsetA_end = layoutA.GetOffset(MakeCoord(i, (uint32_t)(j*simpling_window+simpling_window_head)));
                } 
            }

            auto tmp_max_it = (std::max_element(dataA.begin() + offsetA_start, dataA.begin() + offsetA_end));
            auto tmp_min_it = (std::min_element(dataA.begin() + offsetA_start, dataA.begin() + offsetA_end));

            ElementA tmp_max_result = *tmp_max_it;
            ElementA tmp_min_result = *tmp_min_it;

            slice_max = (ElementGolden)tmp_max_result;
            slice_min = (ElementGolden)tmp_min_result;

            if(j == 0){

                row_max = slice_max;
                row_min = slice_min;

            }else{
                if(slice_max > row_max){
                    row_max = slice_max;
                }

                if(slice_min < row_min){
                    row_min = slice_min;
                }
            }
        }

        size_t row_offsetA_start = layoutA.GetOffset(MakeCoord(i,(uint32_t)0));
        size_t row_offsetA_end = layoutA.GetOffset(MakeCoord(i, problemShape.n()));
        for(uint32_t k=row_offsetA_start; k < row_offsetA_end; k++){
            row_mean += static_cast<ElementGolden>(dataA[k]);
        }

        row_mean = row_mean / static_cast<ElementGolden>(common_size);
        dataMaxGolden[i] = row_max;
        dataMeanGolden[i] = row_mean;
        dataMinGolden[i] = row_min;
    }
}

template<class ElementA, class LayoutA, 
    class ElementGolden>
void ComputeMeanMaxMinRowSimple(
    const Catlass::GemvCoord &problemShape,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataMeanGolden,
    std::vector<ElementGolden> &dataMaxGolden,
    std::vector<ElementGolden> &dataMinGolden)
{
    uint32_t splitMnum = problemShape.m();

    float common_size = 1.0f * problemShape.n();

    float remain_size = common_size;

    uint32_t sliceNStride = problemShape.m();

    for(uint32_t i=0; i < problemShape.m(); i++){

        ElementGolden slice_max = 0.0f;
        ElementGolden slice_min = 0.0f;
        ElementGolden slice_mean = 0.0f;

        size_t offsetA_start = layoutA.GetOffset(MakeCoord(i,(uint32_t)0));
        size_t offsetA_end = layoutA.GetOffset(MakeCoord(i, problemShape.n()));

        // for(uint32_t j=offsetA_start; j < offsetA_end; j++){
        //     slice_mean += static_cast<ElementGolden>(dataA[j]);
        // }

        auto tmp_max_it = (std::max_element(dataA.begin() + offsetA_start, dataA.begin() + offsetA_end));
        // auto tmp_min_it = (std::min_element(dataA.begin() + offsetA_start, dataA.begin() + offsetA_end));

        ElementA tmp_max_result = *tmp_max_it;
        // ElementA tmp_min_result = *tmp_min_it;
        slice_max = (ElementGolden)tmp_max_result;
        // slice_min = (ElementGolden)tmp_min_result;
        slice_mean = slice_mean / static_cast<ElementGolden>(common_size);
        dataMaxGolden[i] = slice_max;
        dataMeanGolden[i] = slice_mean;
        dataMinGolden[i] = slice_min;
    }
}

template<class ElementA, class LayoutA, 
    class ElementGolden>
void ComputeMeanMaxMinRowSimpleSplitK(
    const Catlass::GemvCoord &problemShape,
    const std::vector<uint32_t> &actualKSliceSize,
    uint32_t SplitKNum, uint32_t HeadKSliceNum,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataMeanGolden,
    std::vector<ElementGolden> &dataMaxGolden,
    std::vector<ElementGolden> &dataMinGolden)
{
    uint32_t splitMnum = problemShape.m();

    uint32_t KSliceOutStride = problemShape.m();

    float common_size = 1.0f * problemShape.n();

    float remain_size = common_size;

    uint32_t sliceNStride = problemShape.m();

    for(uint32_t KSlice_i=0; KSlice_i < SplitKNum; KSlice_i++){
        uint32_t kActual = actualKSliceSize[0];
        uint32_t KOffset = 0;

        if(KSlice_i >= HeadKSliceNum){
            kActual = (KSlice_i == (SplitKNum - 1)) ? actualKSliceSize[2] : actualKSliceSize[1];

            uint32_t RemainKSliceNum = KSlice_i - HeadKSliceNum;
            KOffset = HeadKSliceNum * actualKSliceSize[0] + RemainKSliceNum * actualKSliceSize[1];
        }else{
            KOffset = KSlice_i * actualKSliceSize[0];
        }

        uint32_t KSliceOutOffset = KSlice_i * KSliceOutStride;
        uint32_t KSliceInOffset = KOffset;

        for(uint32_t i=0; i < problemShape.m(); i++){

            ElementGolden slice_max = 0.0f;
            ElementGolden slice_min = 0.0f;
            ElementGolden slice_mean = 0.0f;

            size_t offsetA_start = layoutA.GetOffset(MakeCoord(i, KSliceInOffset));
            size_t offsetA_end = layoutA.GetOffset(MakeCoord(i, (KSliceInOffset + kActual)));

            // for(uint32_t j=offsetA_start; j < offsetA_end; j++){
            //     slice_mean += static_cast<ElementGolden>(dataA[j]);
            // }

            auto tmp_max_it = (std::max_element(dataA.begin() + offsetA_start, dataA.begin() + offsetA_end));
            // auto tmp_min_it = (std::min_element(dataA.begin() + offsetA_start, dataA.begin() + offsetA_end));

            ElementA tmp_max_result = *tmp_max_it;
            // ElementA tmp_min_result = *tmp_min_it;
            slice_max = (ElementGolden)tmp_max_result;
            // slice_min = (ElementGolden)tmp_min_result;
            slice_mean = slice_mean / static_cast<ElementGolden>((kActual * 1.0f));
            
            uint32_t slice_total_offset = KSliceOutOffset + i;
            dataMaxGolden[slice_total_offset] = slice_max;
            dataMeanGolden[slice_total_offset] = slice_mean;
            dataMinGolden[slice_total_offset] = slice_min;    
        }
    }
}

template<class ElementA, class LayoutA, 
    class ElementGolden>
void ComputeMeanMaxMinRowSimpleSimplingC(
    const Catlass::GemvCoord &problemShape,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataMeanGolden,
    std::vector<ElementGolden> &dataMaxGolden,
    std::vector<ElementGolden> &dataMinGolden, 
    uint32_t simpling_stride, uint32_t simpling_window)
{
    uint32_t splitMnum = problemShape.m();
    uint32_t RowNSize = problemShape.n();

    uint32_t simpling_window_head = simpling_window / simpling_stride;
    uint32_t simpling_window_num = (RowNSize + simpling_window - 1) / simpling_window;
    float common_size = 1.0f * problemShape.n();

    float remain_size = common_size;

    uint32_t sliceNStride = problemShape.m();

    for(uint32_t i=0; i < problemShape.m(); i++){
        ElementGolden row_max = 0.0f;
        ElementGolden row_min = 0.0f;
        ElementGolden row_mean = 0.0f;
        for(uint32_t j=0; j<simpling_window_num; j++){
            ElementGolden slice_max = 0.0f;
            ElementGolden slice_min = 0.0f;
            ElementGolden slice_mean = 0.0f;
            size_t offsetA_start = layoutA.GetOffset(MakeCoord(i,(uint32_t)(j*simpling_window)));
            size_t offsetA_end = layoutA.GetOffset(MakeCoord(i, (uint32_t)(j*simpling_window+simpling_window_head)));
            if(j == (simpling_window_num - 1)){
                uint32_t remain_window_size = problemShape.n() - j*simpling_window;
                if(remain_window_size < simpling_window_head){
                    offsetA_end = layoutA.GetOffset(MakeCoord(i, problemShape.n()));   
                }else{
                    offsetA_end = layoutA.GetOffset(MakeCoord(i, (uint32_t)(j*simpling_window+simpling_window_head)));
                }    
            }
            // for(uint32_t j=offsetA_start; j < offsetA_end; j++){
            //     slice_mean += static_cast<ElementGolden>(dataA[j]);
            // }
            auto tmp_max_it = (std::max_element(dataA.begin() + offsetA_start, dataA.begin() + offsetA_end));
            // auto tmp_min_it = (std::min_element(dataA.begin() + offsetA_start, dataA.begin() + offsetA_end));
            ElementA tmp_max_result = *tmp_max_it;
            // ElementA tmp_min_result = *tmp_min_it;

            slice_max = (ElementGolden)tmp_max_result;
            // slice_min = (ElementGolden)tmp_min_result;
            slice_mean = slice_mean / static_cast<ElementGolden>(common_size);
            if(j == 0){
                row_max = slice_max;
            }else{
                if(slice_max > row_max){
                    row_max = slice_max;
                }
            }
        }
        dataMaxGolden[i] = row_max;
        dataMeanGolden[i] = row_mean;
        dataMinGolden[i] = row_min;
    }
}


template<class ElementA, class LayoutA, 
    class ElementGolden>
void ComputeMeanMaxRow(
    const Catlass::GemvCoord &problemShape,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataMeanGolden,
    std::vector<ElementGolden> &dataMaxGolden)
{
    uint32_t splitMnum = problemShape.m();

    float common_size = 1.0f * problemShape.n();

    float remain_size = common_size;

    uint32_t sliceNStride = problemShape.m();

    for(uint32_t i=0; i < problemShape.m(); i++){

        ElementGolden slice_max = 0.0f;
        ElementGolden slice_mean = 0.0f;

        size_t offsetA_start = layoutA.GetOffset(MakeCoord(i,(uint32_t)0));
        size_t offsetA_end = layoutA.GetOffset(MakeCoord(i, problemShape.n()));

        for(uint32_t j=offsetA_start; j < offsetA_end; j++){
            slice_mean += static_cast<ElementGolden>(dataA[j]);
        }

        auto tmp_max_it = (std::max_element(dataA.begin() + offsetA_start, dataA.begin() + offsetA_end));
        ElementA tmp_max_result = *tmp_max_it;
        slice_max = (ElementGolden)tmp_max_result;
        slice_mean = slice_mean / static_cast<ElementGolden>(common_size);
        dataMaxGolden[i] = slice_max;
        dataMeanGolden[i] = slice_mean;
    }
}

template<class ElementX, class ElementA, class LayoutA, 
    class ElementGolden>
void ComputeThresholdsASVARTSlice(
    const Catlass::GemvCoord &problemShape,
    uint32_t splitNnum,
    const std::vector<ElementX> &dataBMean,
    const std::vector<ElementX> &dataBMax,
    uint32_t B_N_size, uint32_t B_N_tile,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataAMean,
    std::vector<ElementGolden> &dataAMax,
    std::vector<ElementGolden> &dataAStd,
    std::vector<ElementGolden> &dataGolden, 
    float e_max,
    Catlass::Gemv::helper::FT_RCE_THRE_TYPE rce_thre_type 
)
{
    /*
    void ComputeMeanMaxRow(
    const Catlass::GemvCoord &problemShape,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataMeanGolden,
    std::vector<ElementGolden> &dataMaxGolden)
    */
    ComputeMeanMaxRow(
        problemShape,
        dataA, layoutA,
        dataAMean,
        dataAMax
    );

    // uint32_t splitNnum = sliceShape.m();
    uint32_t B_N_remain = B_N_size % B_N_tile;

    uint32_t k2 = problemShape.n();

    float std_est_ratios[2];
    float kn_ratios[2];
    float kn_scale_ratios[2];
    float kn_sqrt_ratios[2];
    float k_sqrt_n_ratios[2];

    uint32_t n_remain_split = B_N_size % B_N_tile;

    float common_size = k2 * B_N_tile * 1.0f;
    float remain_size = k2 * n_remain_split * 1.0f;

    float common_std_factor = std::sqrt(2.0f * logf(common_size));
    float remain_std_factor = common_std_factor;

    float std_est_ratio_common = (1.0f / common_std_factor);
    float std_est_ratio_remain = std_est_ratio_common;

    float common_kn_ratio = (1.0f * common_size);
    float remain_kn_ratio = common_kn_ratio;

    float common_kn_sqrt_ratio = (std::sqrt(common_kn_ratio));
    float remain_kn_sqrt_ratio = common_kn_sqrt_ratio;

    float common_k_sqrt_n_ratio = (std::sqrt(k2 * 1.0f) *(B_N_tile * 1.0f));
    float remain_k_sqrt_n_ratio = common_k_sqrt_n_ratio;

    float std_est_A_row_factor = std::sqrt(2.0f * logf((k2 * 1.0f)));
    float std_est_A_row_ratio = (1.0f / std_est_A_row_factor);
    float A_row_scale_ratio = 1.0f / (1.0f * k2);

    if(n_remain_split > 0){
        remain_std_factor = std::sqrt(2.0f * logf(remain_size));
        std_est_ratio_remain = (1.0f / remain_std_factor);

        remain_kn_ratio = (1.0f * remain_size);
        remain_kn_sqrt_ratio = (std::sqrt(n_remain_split));
        remain_k_sqrt_n_ratio = (std::sqrt(k2*1.0f) *(n_remain_split * 1.0f));
    }

    std_est_ratios[0] = std_est_ratio_common;
    std_est_ratios[1] = std_est_ratio_remain;

    kn_ratios[0] = common_kn_ratio;
    kn_ratios[1] = remain_kn_ratio;

    kn_scale_ratios[0] = (1.0f / common_kn_ratio);
    kn_scale_ratios[1] = (1.0f / remain_kn_ratio);

    kn_sqrt_ratios[0] = common_kn_sqrt_ratio;
    kn_sqrt_ratios[1] = remain_kn_sqrt_ratio;

    k_sqrt_n_ratios[0] = common_k_sqrt_n_ratio;
    k_sqrt_n_ratios[1] = remain_k_sqrt_n_ratio;
    
    for(uint32_t i=0; i < splitNnum; i++){
        uint32_t goldenOffsetInit = i * problemShape.m();
        ElementGolden B_slice_max = (ElementGolden)dataBMax[i];
        ElementGolden B_slice_mean = (ElementGolden)dataBMean[i];

        ElementGolden std_est_B_ratio = (ElementGolden)std_est_ratios[0];
        float kn_ratio_factor = kn_ratios[0];
        float kn_sqrt_ratio_factor = kn_sqrt_ratios[0];
        float k_sqrt_n_ratio_factor = k_sqrt_n_ratios[0];
        if(i == splitNnum - 1){
            std_est_B_ratio = (ElementGolden)std_est_ratios[1];
            kn_ratio_factor = kn_ratios[1];
            kn_sqrt_ratio_factor = kn_sqrt_ratios[1];
            k_sqrt_n_ratio_factor = k_sqrt_n_ratios[1];
        }

        ElementGolden B_slice_mean_abs = (B_slice_mean < (ElementGolden)0.0) ? ((ElementGolden)0.0 - B_slice_mean) : B_slice_mean; 
        ElementGolden B_slice_std = (B_slice_max - B_slice_mean) * std_est_B_ratio;

        /*
            float kn_ratio_factor = params.kn_ratios[0];
            float kn_sqrt_ratio_factor = params.kn_sqrt_ratios[0];
            float k_sqrt_n_ratio_factor = params.k_sqrt_n_ratios[0];
            if(splitNIdx>=(splitNnum - 1)){
                std_est_B_ratio = params.std_est_ratios[1];
                kn_ratio_factor = params.kn_ratios[1];
                kn_sqrt_ratio_factor = params.kn_sqrt_ratios[1];
                k_sqrt_n_ratio_factor = params.k_sqrt_n_ratios[1];
            }

        */ 
        for(uint32_t j=0; j < problemShape.m(); j++){
            uint32_t goldenOffset = goldenOffsetInit + j;
            ElementGolden A_mean = (ElementGolden)dataAMean[j];
            ElementGolden A_max = (ElementGolden)dataAMax[j];
            
            ElementGolden A_std = (A_max - A_mean) * static_cast<ElementGolden>(std_est_A_row_ratio);
            dataAStd[j] = A_std;
            
            ElementGolden A_mean_abs = (A_mean < (ElementGolden)0.0) ? (ElementGolden)0.0 - A_mean : A_mean;
            ElementGolden tmp_thre = static_cast<ElementGolden>(kn_ratio_factor) * A_mean_abs * B_slice_mean_abs;
            tmp_thre = tmp_thre + static_cast<ElementGolden>(4.0 * kn_sqrt_ratio_factor) * A_mean_abs * B_slice_std;
            tmp_thre = tmp_thre + static_cast<ElementGolden>(4.0 * k_sqrt_n_ratio_factor) * A_std * B_slice_mean_abs;
            tmp_thre = tmp_thre + static_cast<ElementGolden>(4.0 * kn_sqrt_ratio_factor) * B_slice_std * A_std;

            dataGolden[goldenOffset] = static_cast<ElementGolden>(e_max) * tmp_thre;
        }
    }
}

template<class ElementX, class ElementA, class LayoutA, 
    class ElementGolden>
void ComputeThresholdsASVARRobustTSlice(
    const Catlass::GemvCoord &problemShape,
    uint32_t splitNnum,
    const std::vector<ElementX> &dataBMeanabs,
    const std::vector<ElementX> &dataBMeanSquare,
    const std::vector<ElementX> &dataBVar,
    uint32_t B_N_size, uint32_t B_N_tile,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataAMean,
    std::vector<ElementGolden> &dataAMax,
    std::vector<ElementGolden> &dataAMin,
    std::vector<ElementGolden> &dataAStd,
    std::vector<ElementGolden> &dataAMeanAbs,
    std::vector<ElementGolden> &dataGolden, 
    float e_max,
    Catlass::Gemv::helper::FT_RCE_THRE_TYPE rce_thre_type 
)
{
    
    /*
    void ComputeMeanMaxMinRow(
    const Catlass::GemvCoord &problemShape,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataMeanGolden,
    std::vector<ElementGolden> &dataMaxGolden,
    std::vector<ElementGolden> &dataMinGolden)
    */

    // ComputeMeanMaxMinRow(
    //     problemShape,
    //     dataA, layoutA,
    //     dataAMean,
    //     dataAMax,
    //     dataAMin);

    float common_size_row = 1.0f * problemShape.n();

    float remain_size_row = common_size_row;

    uint32_t sliceNStride = problemShape.m();

    for(uint32_t i=0; i < problemShape.m(); i++){

        ElementGolden slice_max = 0.0f;
        ElementGolden slice_min = 0.0f;
        ElementGolden slice_mean = 0.0f;

        size_t offsetA_start = layoutA.GetOffset(MakeCoord(i,(uint32_t)0));
        size_t offsetA_end = layoutA.GetOffset(MakeCoord(i, problemShape.n()));

        for(uint32_t j=offsetA_start; j < offsetA_end; j++){
            slice_mean += static_cast<ElementGolden>(dataA[j]);
        }

        auto tmp_max_it = (std::max_element(dataA.begin() + offsetA_start, dataA.begin() + offsetA_end));
        auto tmp_min_it = (std::min_element(dataA.begin() + offsetA_start, dataA.begin() + offsetA_end));

        ElementA tmp_max_result = *tmp_max_it;
        ElementA tmp_min_result = *tmp_min_it;
        // printf("tmp_max_result: %f\n", tmp_max_result);
        slice_max = (ElementGolden)tmp_max_result;
        slice_min = (ElementGolden)tmp_min_result;
        slice_mean = slice_mean / static_cast<ElementGolden>(common_size_row);
        // dataAMax[i] = (ElementGolden)*tmp_max_it;
        // printf("dataMaxGolden[i]: %f\n", dataAMax[i]);
        dataAMean[i] = slice_mean;
        dataAMin[i] = slice_min;
    }

    // uint32_t splitNnum = sliceShape.m();
    uint32_t B_N_remain = B_N_size % B_N_tile;

    uint32_t k2 = problemShape.n();

    float std_est_ratios[2];
    float kn_ratios[2];
    float kn_scale_ratios[2];
    float kn_sqrt_ratios[2];
    float k_sqrt_n_ratios[2];

    float n_ratios[2];
    float n_sqrt_ratios[2];
    float n_square_ratios[2];

    uint32_t n_remain_split = B_N_size % B_N_tile;

    float common_size = k2 * B_N_tile * 1.0f;
    float remain_size = k2 * n_remain_split * 1.0f;

    float common_std_factor = std::sqrt(2.0f * logf(common_size));
    float remain_std_factor = common_std_factor;

    float std_est_ratio_common = (1.0f / common_std_factor);
    float std_est_ratio_remain = std_est_ratio_common;

    float common_kn_ratio = (1.0f * common_size);
    float remain_kn_ratio = common_kn_ratio;

    float common_kn_sqrt_ratio = (std::sqrt(common_kn_ratio));
    float remain_kn_sqrt_ratio = common_kn_sqrt_ratio;

    float common_k_sqrt_n_ratio = (std::sqrt(k2 * 1.0f) *(B_N_tile * 1.0f));
    float remain_k_sqrt_n_ratio = common_k_sqrt_n_ratio;

    float std_est_A_row_factor = std::sqrt(2.0f * logf((k2 * 1.0f)));
    float std_est_A_row_ratio = (1.0f / std_est_A_row_factor);
    float A_row_scale_ratio = 1.0f / (1.0f * k2);

    float common_n_ratio = (1.0f * B_N_tile);
    float remain_n_ratio = common_n_ratio;

    float common_n_sqrt_ratio = std::sqrt(common_n_ratio);
    float remain_n_sqrt_ratio = common_n_sqrt_ratio;

    float common_n_square_ratio = (1.0f * B_N_tile * B_N_tile);
    float remain_n_square_ratio = common_n_square_ratio;

    if(n_remain_split > 0){
        remain_std_factor = std::sqrt(2.0f * logf(remain_size));
        std_est_ratio_remain = (1.0f / remain_std_factor);

        remain_kn_ratio = (1.0f * remain_size);
        remain_kn_sqrt_ratio = (std::sqrt(n_remain_split));
        remain_k_sqrt_n_ratio = (std::sqrt(k2*1.0f) *(n_remain_split * 1.0f));

        remain_n_ratio = (1.0f * n_remain_split);
        remain_n_sqrt_ratio = std::sqrt(remain_n_ratio);
        remain_n_square_ratio = (1.0f * n_remain_split * n_remain_split);
    }

    std_est_ratios[0] = std_est_ratio_common;
    std_est_ratios[1] = std_est_ratio_remain;

    kn_ratios[0] = common_kn_ratio;
    kn_ratios[1] = remain_kn_ratio;

    kn_scale_ratios[0] = (1.0f / common_kn_ratio);
    kn_scale_ratios[1] = (1.0f / remain_kn_ratio);

    kn_sqrt_ratios[0] = common_kn_sqrt_ratio;
    kn_sqrt_ratios[1] = remain_kn_sqrt_ratio;

    k_sqrt_n_ratios[0] = common_k_sqrt_n_ratio;
    k_sqrt_n_ratios[1] = remain_k_sqrt_n_ratio;

    n_ratios[0] = common_n_ratio;
    n_ratios[1] = remain_n_ratio;

    n_sqrt_ratios[0] = common_n_sqrt_ratio;
    n_sqrt_ratios[1] = remain_n_sqrt_ratio;

    n_square_ratios[0] = common_n_square_ratio;
    n_square_ratios[1] = remain_n_square_ratio;
    
    for(uint32_t i=0; i < splitNnum; i++){
        uint32_t goldenOffsetInit = i * problemShape.m();
        ElementGolden B_slice_meanabs = (ElementGolden)dataBMeanabs[i];
        ElementGolden B_slice_meansquare = (ElementGolden)dataBMeanSquare[i];
        ElementGolden B_slice_var = (ElementGolden)dataBVar[i];

        float n_ratio_factor = n_ratios[0];
        float n_sqrt_ratio_factor = n_sqrt_ratios[0];
        float n_square_ratio_factor = n_square_ratios[0];
        if(i == splitNnum - 1){
            n_ratio_factor = n_ratios[1];
            n_sqrt_ratio_factor = n_sqrt_ratios[1];
            n_square_ratio_factor = n_square_ratios[1];
        }
        /*
            float kn_ratio_factor = params.kn_ratios[0];
            float kn_sqrt_ratio_factor = params.kn_sqrt_ratios[0];
            float k_sqrt_n_ratio_factor = params.k_sqrt_n_ratios[0];
            if(splitNIdx>=(splitNnum - 1)){
                std_est_B_ratio = params.std_est_ratios[1];
                kn_ratio_factor = params.kn_ratios[1];
                kn_sqrt_ratio_factor = params.kn_sqrt_ratios[1];
                k_sqrt_n_ratio_factor = params.k_sqrt_n_ratios[1];
            }

        */ 
        for(uint32_t j=0; j < problemShape.m(); j++){
            uint32_t goldenOffset = goldenOffsetInit + j;
            ElementGolden A_mean = (ElementGolden)dataAMean[j];
            // printf("A_mean: %f\n", (float)A_mean);
            ElementGolden A_max = (ElementGolden)dataAMax[j];
            ElementGolden A_min = (ElementGolden)dataAMin[j];
            // printf("A_min: %f\n", (float)A_min);
            
            ElementGolden A_std = std::sqrt((A_max - A_mean) * (A_mean - A_min));
            dataAStd[j] = A_std;
            
            ElementGolden A_mean_abs = (A_mean < (ElementGolden)0.0) ? (ElementGolden)0.0 - A_mean : A_mean;
            ElementGolden tmp_thre = static_cast<ElementGolden>(n_ratio_factor) * A_mean_abs * B_slice_meanabs;
            ElementGolden tmp_thre_s = static_cast<ElementGolden>(n_ratio_factor) * 16.0 * A_mean_abs * A_mean_abs * B_slice_var * B_slice_var;
            tmp_thre_s = tmp_thre_s + static_cast<ElementGolden>(n_square_ratio_factor) * 16.0 * A_std * A_std * B_slice_meansquare;
            tmp_thre_s = std::sqrt(tmp_thre_s);

            tmp_thre = tmp_thre  + tmp_thre_s;
            tmp_thre = tmp_thre + static_cast<ElementGolden>(n_sqrt_ratio_factor) * A_std * B_slice_var;

            dataGolden[goldenOffset] = static_cast<ElementGolden>(e_max) * tmp_thre;
        }
    }

    // std::cout << "dataMaxGolden.size() = " << dataAMax[0] << "\n";
    // std::cout << "dataMeanGolden.size() = " << dataAMean.size() << "\n";
    // std::cout << "dataMinGolden.size() = " << dataAMin.size() << "\n";
    // std::cout << "problemShape.m() = " << problemShape.m() << "\n";
    // uint32_t splitMnum = problemShape.m();

}


template<class ElementX, class ElementA, class LayoutA, 
    class ElementGolden>
void ComputeThresholdsASVARRobustTSliceSplitK(
    const Catlass::GemvCoord &problemShape,
    uint32_t splitNnum,
    const std::vector<uint32_t> &actualKSliceSize,
    uint32_t SplitKNum, uint32_t HeadKSliceNum,
    const std::vector<ElementX> &dataBMeanabs,
    const std::vector<ElementX> &dataBMeanSquare,
    const std::vector<ElementX> &dataBVar,
    uint32_t B_N_size, uint32_t B_N_tile,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataAMean,
    std::vector<ElementGolden> &dataAMax,
    std::vector<ElementGolden> &dataAMin,
    std::vector<ElementGolden> &dataAStd,
    std::vector<ElementGolden> &dataAMeanAbs,
    std::vector<ElementGolden> &dataGolden, 
    float e_max,
    Catlass::Gemv::helper::FT_RCE_THRE_TYPE rce_thre_type 
)
{
    /*
    template<class ElementA, class LayoutA, 
    class ElementGolden>
    void ComputeMeanMaxMinRowSplitK(
        const Catlass::GemvCoord &problemShape,
        const std::vector<uint32_t> &actualKSliceSize,
        uint32_t SplitKNum, uint32_t HeadKSliceNum,
        const std::vector<ElementA> &dataA, const LayoutA &layoutA,
        std::vector<ElementGolden> &dataMeanGolden,
        std::vector<ElementGolden> &dataMaxGolden,
        std::vector<ElementGolden> &dataMinGolden)
    */
    ComputeMeanMaxMinRowSplitK(
        problemShape,
        actualKSliceSize,
        SplitKNum, HeadKSliceNum,
        dataA, layoutA,
        dataAMean,
        dataAMax,
        dataAMin);

    // uint32_t splitNnum = sliceShape.m();
    uint32_t B_N_remain = B_N_size % B_N_tile;

    uint32_t k2 = problemShape.n();

    float n_ratios[2];
    float n_sqrt_ratios[2];
    float n_square_ratios[2];

    uint32_t n_remain_split = B_N_size % B_N_tile;

    float common_n_ratio = (1.0f * B_N_tile);
    float remain_n_ratio = common_n_ratio;

    float common_n_sqrt_ratio = std::sqrt(common_n_ratio);
    float remain_n_sqrt_ratio = common_n_sqrt_ratio;

    float common_n_square_ratio = (1.0f * B_N_tile * B_N_tile);
    float remain_n_square_ratio = common_n_square_ratio;

    if(n_remain_split > 0){

        remain_n_ratio = (1.0f * n_remain_split);
        remain_n_sqrt_ratio = std::sqrt(remain_n_ratio);
        remain_n_square_ratio = (1.0f * n_remain_split * n_remain_split);
    }

    n_ratios[0] = common_n_ratio;
    n_ratios[1] = remain_n_ratio;

    n_sqrt_ratios[0] = common_n_sqrt_ratio;
    n_sqrt_ratios[1] = remain_n_sqrt_ratio;

    n_square_ratios[0] = common_n_square_ratio;
    n_square_ratios[1] = remain_n_square_ratio;

    uint32_t KSliceOutStride = splitNnum * problemShape.m();

    int32_t align = 32 / sizeof(ElementGolden);
    uint32_t split_block_num = (splitNnum + align - 1) / align;

    uint32_t SplitNnumAligned = split_block_num * align + align;
    uint32_t KSliceInStrideforB = SplitNnumAligned;
    uint32_t KSliceInStrideforA = problemShape.m();

    for(uint32_t KSlice_i=0; KSlice_i < SplitKNum; KSlice_i++){
        uint32_t kActual = actualKSliceSize[0];
        uint32_t KOffset = 0;

        if(KSlice_i >= HeadKSliceNum){
            kActual = (KSlice_i == (SplitKNum - 1)) ? actualKSliceSize[2] : actualKSliceSize[1];

            uint32_t RemainKSliceNum = KSlice_i - HeadKSliceNum;
            KOffset = HeadKSliceNum * actualKSliceSize[0] + RemainKSliceNum * actualKSliceSize[1];
        }else{
            KOffset = KSlice_i * actualKSliceSize[0];
        }

        uint32_t KSliceOutOffset = KSlice_i * KSliceOutStride;
        uint32_t KSliceInOffset = KOffset;

        uint32_t KSliceInOffsetforB = KSlice_i * KSliceInStrideforB;
        uint32_t KSliceInOffsetforA = KSlice_i * KSliceInStrideforA;
        

        for(uint32_t i=0; i < splitNnum; i++){
            uint32_t goldenOffsetInitOut = KSliceOutOffset + i * problemShape.m();
            uint32_t goldenOffsetInitIn = KSliceInOffsetforA;

            ElementGolden B_slice_meanabs = (ElementGolden)dataBMeanabs[KSliceInOffsetforB + i];
            ElementGolden B_slice_meansquare = (ElementGolden)dataBMeanSquare[KSliceInOffsetforB + i];
            ElementGolden B_slice_var = (ElementGolden)dataBVar[KSliceInOffsetforB + i];

            float n_ratio_factor = n_ratios[0];
            float n_sqrt_ratio_factor = n_sqrt_ratios[0];
            float n_square_ratio_factor = n_square_ratios[0];
            if(i == splitNnum - 1){
                n_ratio_factor = n_ratios[1];
                n_sqrt_ratio_factor = n_sqrt_ratios[1];
                n_square_ratio_factor = n_square_ratios[1];
            }
           
            for(uint32_t j=0; j < problemShape.m(); j++){
                uint32_t goldenOffsetIn = goldenOffsetInitIn + j;
                uint32_t goldenOffsetOut = goldenOffsetInitOut + j;
                ElementGolden A_mean = (ElementGolden)dataAMean[goldenOffsetIn];
                ElementGolden A_max = (ElementGolden)dataAMax[goldenOffsetIn];
                ElementGolden A_min = (ElementGolden)dataAMin[goldenOffsetIn];
            
                ElementGolden A_std = std::sqrt((A_max - A_mean) * (A_mean - A_min));
                dataAStd[goldenOffsetIn] = A_std;
            
                ElementGolden A_mean_abs = (A_mean < (ElementGolden)0.0) ? (ElementGolden)0.0 - A_mean : A_mean;
                ElementGolden tmp_thre = static_cast<ElementGolden>(n_ratio_factor) * A_mean_abs * B_slice_meanabs;
                ElementGolden tmp_thre_s = static_cast<ElementGolden>(n_ratio_factor) * 16.0 * A_mean_abs * A_mean_abs * B_slice_var * B_slice_var;
                tmp_thre_s = tmp_thre_s + static_cast<ElementGolden>(n_square_ratio_factor) * 16.0 * A_std * A_std * B_slice_meansquare;
                tmp_thre_s = std::sqrt(tmp_thre_s);

                tmp_thre = tmp_thre  + tmp_thre_s;
                tmp_thre = tmp_thre + static_cast<ElementGolden>(n_sqrt_ratio_factor) * A_std * B_slice_var;

                dataGolden[goldenOffsetOut] = static_cast<ElementGolden>(e_max) * tmp_thre;
            }
        }
    }
    
    
}


template<class ElementX, class ElementA, class LayoutA, 
    class ElementGolden>
void ComputeThresholdsASVARSimpleTSlice(
    const Catlass::GemvCoord &problemShape,
    uint32_t splitNnum,
    const std::vector<ElementX> &dataBMeanabs,
    const std::vector<ElementX> &dataBMeanSquare,
    const std::vector<ElementX> &dataBVar,
    uint32_t B_N_size, uint32_t B_N_tile,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataAMean,
    std::vector<ElementGolden> &dataAMax,
    std::vector<ElementGolden> &dataAMin,
    std::vector<ElementGolden> &dataAStd,
    std::vector<ElementGolden> &dataAMeanAbs,
    std::vector<ElementGolden> &dataGolden, 
    float e_max,
    Catlass::Gemv::helper::FT_RCE_THRE_TYPE rce_thre_type,
    int32_t useLogRatio)
{
    /*
    void ComputeMeanMaxMinRow(
    const Catlass::GemvCoord &problemShape,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataMeanGolden,
    std::vector<ElementGolden> &dataMaxGolden,
    std::vector<ElementGolden> &dataMinGolden)
    */
    ComputeMeanMaxMinRowSimple(
        problemShape,
        dataA, layoutA,
        dataAMean,
        dataAMax,
        dataAMin);

    // uint32_t splitNnum = sliceShape.m();
    uint32_t B_N_remain = B_N_size % B_N_tile;

    uint32_t k2 = problemShape.n();

    float std_est_ratios[2];
    float kn_ratios[2];
    float kn_scale_ratios[2];
    float kn_sqrt_ratios[2];
    float k_sqrt_n_ratios[2];

    float n_ratios[2];
    float n_sqrt_ratios[2];
    float n_square_ratios[2];

    float simple_std_est_A_row_factor = std::sqrt(2.0f * logf(k2 * 1.0f));
    if(useLogRatio < 1){
        simple_std_est_A_row_factor = 1.0f;
    }
    float simple_std_est_A_row_ratio = 1.0f / simple_std_est_A_row_factor;

    float simple_std_est_ratios[2];
    

    uint32_t n_remain_split = B_N_size % B_N_tile;

    float common_size = k2 * B_N_tile * 1.0f;
    float remain_size = k2 * n_remain_split * 1.0f;

    float common_std_factor = std::sqrt(2.0f * logf(common_size));
    float remain_std_factor = common_std_factor;

    float std_est_ratio_common = (1.0f / common_std_factor);
    float std_est_ratio_remain = std_est_ratio_common;

    
    float simple_common_std_factor = std::sqrt(2.0f * logf(B_N_tile * 1.0f));
    float simple_remain_std_factor = simple_common_std_factor;

    if(useLogRatio < 1){
        simple_common_std_factor = 1.0f;
        simple_remain_std_factor = 1.0f;
    }

    float simple_std_est_ratio_common = (1.0f / simple_common_std_factor);
    float simple_std_est_ratio_remain = simple_std_est_ratio_common;

    float common_kn_ratio = (1.0f * common_size);
    float remain_kn_ratio = common_kn_ratio;

    float common_kn_sqrt_ratio = (std::sqrt(common_kn_ratio));
    float remain_kn_sqrt_ratio = common_kn_sqrt_ratio;

    float common_k_sqrt_n_ratio = (std::sqrt(k2 * 1.0f) *(B_N_tile * 1.0f));
    float remain_k_sqrt_n_ratio = common_k_sqrt_n_ratio;

    float std_est_A_row_factor = std::sqrt(2.0f * logf((k2 * 1.0f)));
    float std_est_A_row_ratio = (1.0f / std_est_A_row_factor);
    float A_row_scale_ratio = 1.0f / (1.0f * k2);

    float common_n_ratio = (1.0f * B_N_tile);
    float remain_n_ratio = common_n_ratio;

    float common_n_sqrt_ratio = std::sqrt(common_n_ratio);
    float remain_n_sqrt_ratio = common_n_sqrt_ratio;

    float common_n_square_ratio = (1.0f * B_N_tile * B_N_tile);
    float remain_n_square_ratio = common_n_square_ratio;

    if(n_remain_split > 0){
        remain_std_factor = std::sqrt(2.0f * logf(remain_size));
        std_est_ratio_remain = (1.0f / remain_std_factor);

        remain_kn_ratio = (1.0f * remain_size);
        remain_kn_sqrt_ratio = (std::sqrt(n_remain_split));
        remain_k_sqrt_n_ratio = (std::sqrt(k2*1.0f) *(n_remain_split * 1.0f));

        remain_n_ratio = (1.0f * n_remain_split);
        remain_n_sqrt_ratio = std::sqrt(remain_n_ratio);
        remain_n_square_ratio = (1.0f * n_remain_split * n_remain_split);
        
        simple_remain_std_factor = std::sqrt(2.0f * logf(n_remain_split * 1.0f));

        if(useLogRatio < 1){
            simple_remain_std_factor = 1.0f;
        }

        simple_std_est_ratio_remain = (1.0f / simple_remain_std_factor);
    }

    std_est_ratios[0] = std_est_ratio_common;
    std_est_ratios[1] = std_est_ratio_remain;

    simple_std_est_ratios[0] = simple_std_est_ratio_common;
    simple_std_est_ratios[0] = simple_std_est_ratio_remain;

    kn_ratios[0] = common_kn_ratio;
    kn_ratios[1] = remain_kn_ratio;

    kn_scale_ratios[0] = (1.0f / common_kn_ratio);
    kn_scale_ratios[1] = (1.0f / remain_kn_ratio);

    kn_sqrt_ratios[0] = common_kn_sqrt_ratio;
    kn_sqrt_ratios[1] = remain_kn_sqrt_ratio;

    k_sqrt_n_ratios[0] = common_k_sqrt_n_ratio;
    k_sqrt_n_ratios[1] = remain_k_sqrt_n_ratio;

    n_ratios[0] = common_n_ratio;
    n_ratios[1] = remain_n_ratio;

    n_sqrt_ratios[0] = common_n_sqrt_ratio;
    n_sqrt_ratios[1] = remain_n_sqrt_ratio;

    n_square_ratios[0] = common_n_square_ratio;
    n_square_ratios[1] = remain_n_square_ratio;
    
    for(uint32_t i=0; i < splitNnum; i++){
        uint32_t goldenOffsetInit = i * problemShape.m();
        ElementGolden B_slice_meanabs = (ElementGolden)dataBMeanabs[i];
        ElementGolden B_slice_meansquare = (ElementGolden)dataBMeanSquare[i];
        ElementGolden B_slice_var = (ElementGolden)dataBVar[i];

        float n_ratio_factor = n_ratios[0];
        float n_sqrt_ratio_factor = n_sqrt_ratios[0];
        float n_square_ratio_factor = n_square_ratios[0];
        float simple_std_est_B_ratio = simple_std_est_ratios[0];

        if(i == splitNnum - 1){
            n_ratio_factor = n_ratios[1];
            n_sqrt_ratio_factor = n_sqrt_ratios[1];
            n_square_ratio_factor = n_square_ratios[1];
            simple_std_est_B_ratio = simple_std_est_ratios[1];
        }
        /*
            float kn_ratio_factor = params.kn_ratios[0];
            float kn_sqrt_ratio_factor = params.kn_sqrt_ratios[0];
            float k_sqrt_n_ratio_factor = params.k_sqrt_n_ratios[0];
            if(splitNIdx>=(splitNnum - 1)){
                std_est_B_ratio = params.std_est_ratios[1];
                kn_ratio_factor = params.kn_ratios[1];
                kn_sqrt_ratio_factor = params.kn_sqrt_ratios[1];
                k_sqrt_n_ratio_factor = params.k_sqrt_n_ratios[1];
            }

        */ 
        for(uint32_t j=0; j < problemShape.m(); j++){
            uint32_t goldenOffset = goldenOffsetInit + j;
            ElementGolden A_mean = (ElementGolden)dataAMean[j];
            ElementGolden A_max = (ElementGolden)dataAMax[j];
            ElementGolden A_min = (ElementGolden)dataAMin[j];
            
            ElementGolden A_std = A_max * simple_std_est_A_row_ratio;
            // std::sqrt((A_max) * (A_mean - A_min));
            dataAStd[j] = A_std;
            ElementGolden A_mean_abs = (A_mean < (ElementGolden)0.0) ? (ElementGolden)0.0 - A_mean : A_mean;
            ElementGolden tmp_thre = static_cast<ElementGolden>(n_sqrt_ratio_factor) * A_std * B_slice_var;
            tmp_thre = tmp_thre * static_cast<ElementGolden>(simple_std_est_B_ratio);

            // static_cast<ElementGolden>(n_ratio_factor) * A_mean_abs * B_slice_meanabs;
            // ElementGolden tmp_thre_s = static_cast<ElementGolden>(n_ratio_factor) * 16.0 * A_mean_abs * A_mean_abs * B_slice_var * B_slice_var;
            // tmp_thre_s = tmp_thre_s + static_cast<ElementGolden>(n_square_ratio_factor) * 16.0 * A_std * A_std * B_slice_meansquare;
            // tmp_thre_s = std::sqrt(tmp_thre_s);

            // tmp_thre = tmp_thre  + tmp_thre_s;
            // tmp_thre = tmp_thre + 
            dataGolden[goldenOffset] = static_cast<ElementGolden>(e_max) * tmp_thre;
        }
    }
}


template<class ElementX, class ElementA, class LayoutA, 
    class ElementGolden>
void ComputeThresholdsASVARSimpleTSliceSplitK(
    const Catlass::GemvCoord &problemShape,
    uint32_t splitNnum,
    const std::vector<uint32_t> &actualKSliceSize,
    uint32_t SplitKNum, uint32_t HeadKSliceNum,
    const std::vector<ElementX> &dataBMeanabs,
    const std::vector<ElementX> &dataBMeanSquare,
    const std::vector<ElementX> &dataBVar,
    uint32_t B_N_size, uint32_t B_N_tile,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataAMean,
    std::vector<ElementGolden> &dataAMax,
    std::vector<ElementGolden> &dataAMin,
    std::vector<ElementGolden> &dataAStd,
    std::vector<ElementGolden> &dataAMeanAbs,
    std::vector<ElementGolden> &dataGolden, 
    float e_max,
    Catlass::Gemv::helper::FT_RCE_THRE_TYPE rce_thre_type,
    int32_t useLogRatio)
{
    /*
    void ComputeMeanMaxMinRowSimpleSplitK(
    const Catlass::GemvCoord &problemShape,
    const std::vector<uint32_t> &actualKSliceSize,
    uint32_t SplitKNum, uint32_t HeadKSliceNum,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataMeanGolden,
    std::vector<ElementGolden> &dataMaxGolden,
    std::vector<ElementGolden> &dataMinGolden)
    */
    ComputeMeanMaxMinRowSimpleSplitK(
        problemShape,
        actualKSliceSize,
        SplitKNum, HeadKSliceNum,
        dataA, layoutA,
        dataAMean,
        dataAMax,
        dataAMin);

    // uint32_t splitNnum = sliceShape.m();
    uint32_t B_N_remain = B_N_size % B_N_tile;

    uint32_t k2 = problemShape.n();

    float n_ratios[2];
    float n_sqrt_ratios[2];
    float n_square_ratios[2];

    float simple_std_est_A_row_ratios[3];

    if(useLogRatio < 1){
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


    float simple_std_est_ratios[2];
    

    uint32_t n_remain_split = B_N_size % B_N_tile;

    
    float simple_common_std_factor = std::sqrt(2.0f * logf(B_N_tile * 1.0f));
    float simple_remain_std_factor = simple_common_std_factor;

    if(useLogRatio < 1){
        simple_common_std_factor = 1.0f;
        simple_remain_std_factor = 1.0f;
    }

    float simple_std_est_ratio_common = (1.0f / simple_common_std_factor);
    float simple_std_est_ratio_remain = simple_std_est_ratio_common;


    float common_n_ratio = (1.0f * B_N_tile);
    float remain_n_ratio = common_n_ratio;

    float common_n_sqrt_ratio = std::sqrt(common_n_ratio);
    float remain_n_sqrt_ratio = common_n_sqrt_ratio;

    float common_n_square_ratio = (1.0f * B_N_tile * B_N_tile);
    float remain_n_square_ratio = common_n_square_ratio;

    if(n_remain_split > 0){

        remain_n_ratio = (1.0f * n_remain_split);
        remain_n_sqrt_ratio = std::sqrt(remain_n_ratio);
        remain_n_square_ratio = (1.0f * n_remain_split * n_remain_split);
        
        simple_remain_std_factor = std::sqrt(2.0f * logf(n_remain_split * 1.0f));

        if(useLogRatio < 1){
            simple_remain_std_factor = 1.0f;
        }

        simple_std_est_ratio_remain = (1.0f / simple_remain_std_factor);
    }

    simple_std_est_ratios[0] = simple_std_est_ratio_common;
    simple_std_est_ratios[0] = simple_std_est_ratio_remain;

    n_ratios[0] = common_n_ratio;
    n_ratios[1] = remain_n_ratio;

    n_sqrt_ratios[0] = common_n_sqrt_ratio;
    n_sqrt_ratios[1] = remain_n_sqrt_ratio;

    n_square_ratios[0] = common_n_square_ratio;
    n_square_ratios[1] = remain_n_square_ratio;

    uint32_t KSliceOutStride = splitNnum * problemShape.m();

    int32_t align = 32 / sizeof(ElementGolden);
    uint32_t split_block_num = (splitNnum + align - 1) / align;

    uint32_t SplitNnumAligned = split_block_num * align + align;
    uint32_t KSliceInStrideforB = SplitNnumAligned;
    uint32_t KSliceInStrideforA = problemShape.m();

    for(uint32_t KSlice_i=0; KSlice_i < SplitKNum; KSlice_i++){
        uint32_t kActual = actualKSliceSize[0];
        uint32_t KOffset = 0;

        if(KSlice_i >= HeadKSliceNum){
            kActual = (KSlice_i == (SplitKNum - 1)) ? actualKSliceSize[2] : actualKSliceSize[1];

            uint32_t RemainKSliceNum = KSlice_i - HeadKSliceNum;
            KOffset = HeadKSliceNum * actualKSliceSize[0] + RemainKSliceNum * actualKSliceSize[1];
        }else{
            KOffset = KSlice_i * actualKSliceSize[0];
        }

        uint32_t KSliceOutOffset = KSlice_i * KSliceOutStride;
        uint32_t KSliceInOffset = KOffset;

        uint32_t KSliceInOffsetforB = KSlice_i * KSliceInStrideforB;
        uint32_t KSliceInOffsetforA = KSlice_i * KSliceInStrideforA;

        float simple_std_est_A_row_ratio_use = simple_std_est_A_row_ratios[0]; 

        if(KSlice_i >= HeadKSliceNum){
            simple_std_est_A_row_ratio_use = (KSlice_i ==(SplitKNum -1)) ? simple_std_est_A_row_ratios[2] : simple_std_est_A_row_ratios[1];
        }

        for(uint32_t i=0; i < splitNnum; i++){
            uint32_t goldenOffsetInitOut = KSliceOutOffset + i * problemShape.m();
            uint32_t goldenOffsetInitIn = KSliceInOffsetforA;

            ElementGolden B_slice_meanabs = (ElementGolden)dataBMeanabs[KSliceInOffsetforB + i];
            ElementGolden B_slice_meansquare = (ElementGolden)dataBMeanSquare[KSliceInOffsetforB + i];
            ElementGolden B_slice_var = (ElementGolden)dataBVar[KSliceInOffsetforB + i];

            float n_ratio_factor = n_ratios[0];
            float n_sqrt_ratio_factor = n_sqrt_ratios[0];
            float n_square_ratio_factor = n_square_ratios[0];
            float simple_std_est_B_ratio = simple_std_est_ratios[0];

            if(i == splitNnum - 1){
                n_ratio_factor = n_ratios[1];
                n_sqrt_ratio_factor = n_sqrt_ratios[1];
                n_square_ratio_factor = n_square_ratios[1];
                simple_std_est_B_ratio = simple_std_est_ratios[1];
            }
            
            for(uint32_t j=0; j < problemShape.m(); j++){
                uint32_t goldenOffsetIn = goldenOffsetInitIn + j;
                uint32_t goldenOffsetOut = goldenOffsetInitOut + j;
                ElementGolden A_mean = (ElementGolden)dataAMean[goldenOffsetIn];
                ElementGolden A_max = (ElementGolden)dataAMax[goldenOffsetIn];
                ElementGolden A_min = (ElementGolden)dataAMin[goldenOffsetIn];
            
                ElementGolden A_std = A_max * simple_std_est_A_row_ratio_use;
                // std::sqrt((A_max) * (A_mean - A_min));
                dataAStd[goldenOffsetIn] = A_std;
                ElementGolden A_mean_abs = (A_mean < (ElementGolden)0.0) ? (ElementGolden)0.0 - A_mean : A_mean;
                ElementGolden tmp_thre = static_cast<ElementGolden>(n_sqrt_ratio_factor) * A_std * B_slice_var;
                tmp_thre = tmp_thre * static_cast<ElementGolden>(simple_std_est_B_ratio);

                dataGolden[goldenOffsetOut] = static_cast<ElementGolden>(e_max) * tmp_thre;
        }
    }
    }


    
    
}

template<class ElementX, class ElementA, class LayoutA, 
    class ElementGolden>
void ComputeThresholdsASVARSimpleTSliceSimplingC(
    const Catlass::GemvCoord &problemShape,
    uint32_t splitNnum,
    const std::vector<ElementX> &dataBMeanabs,
    const std::vector<ElementX> &dataBMeanSquare,
    const std::vector<ElementX> &dataBVar,
    uint32_t B_N_size, uint32_t B_N_tile,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataAMean,
    std::vector<ElementGolden> &dataAMax,
    std::vector<ElementGolden> &dataAMin,
    std::vector<ElementGolden> &dataAStd,
    std::vector<ElementGolden> &dataAMeanAbs,
    std::vector<ElementGolden> &dataGolden, 
    float e_max,
    Catlass::Gemv::helper::FT_RCE_THRE_TYPE rce_thre_type,
    int32_t useLogRatio, uint32_t simpling_stride, 
    uint32_t simpling_window)
{
    /*
    void ComputeMeanMaxMinRowSimpleSimplingC(
    const Catlass::GemvCoord &problemShape,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    std::vector<ElementGolden> &dataMeanGolden,
    std::vector<ElementGolden> &dataMaxGolden,
    std::vector<ElementGolden> &dataMinGolden, 
    uint32_t simpling_stride, uint32_t simpling_window)
    */
    ComputeMeanMaxMinRowSimpleSimplingC(
        problemShape,
        dataA, layoutA,
        dataAMean,
        dataAMax,
        dataAMin, 
        simpling_stride, simpling_window);

    // uint32_t splitNnum = sliceShape.m();
    uint32_t B_N_remain = B_N_size % B_N_tile;

    uint32_t k2 = problemShape.n();

    float std_est_ratios[2];
    float kn_ratios[2];
    float kn_scale_ratios[2];
    float kn_sqrt_ratios[2];
    float k_sqrt_n_ratios[2];

    float n_ratios[2];
    float n_sqrt_ratios[2];
    float n_square_ratios[2];

    float simple_std_est_A_row_factor = std::sqrt(2.0f * logf(k2 * 1.0f));
    if(useLogRatio < 1){
        simple_std_est_A_row_factor = 1.0f;
    }
    float simple_std_est_A_row_ratio = 1.0f / simple_std_est_A_row_factor;

    float simple_std_est_ratios[2];
    
    uint32_t n_remain_split = B_N_size % B_N_tile;

    float common_size = k2 * B_N_tile * 1.0f;
    float remain_size = k2 * n_remain_split * 1.0f;

    float common_std_factor = std::sqrt(2.0f * logf(common_size));
    float remain_std_factor = common_std_factor;

    float std_est_ratio_common = (1.0f / common_std_factor);
    float std_est_ratio_remain = std_est_ratio_common;

    
    float simple_common_std_factor = std::sqrt(2.0f * logf(B_N_tile * 1.0f));
    float simple_remain_std_factor = simple_common_std_factor;

    if(useLogRatio < 1){
        simple_common_std_factor = 1.0f;
        simple_remain_std_factor = 1.0f;
    }

    float simple_std_est_ratio_common = (1.0f / simple_common_std_factor);
    float simple_std_est_ratio_remain = simple_std_est_ratio_common;

    float common_kn_ratio = (1.0f * common_size);
    float remain_kn_ratio = common_kn_ratio;

    float common_kn_sqrt_ratio = (std::sqrt(common_kn_ratio));
    float remain_kn_sqrt_ratio = common_kn_sqrt_ratio;

    float common_k_sqrt_n_ratio = (std::sqrt(k2 * 1.0f) *(B_N_tile * 1.0f));
    float remain_k_sqrt_n_ratio = common_k_sqrt_n_ratio;

    float std_est_A_row_factor = std::sqrt(2.0f * logf((k2 * 1.0f)));
    float std_est_A_row_ratio = (1.0f / std_est_A_row_factor);
    float A_row_scale_ratio = 1.0f / (1.0f * k2);

    float common_n_ratio = (1.0f * B_N_tile);
    float remain_n_ratio = common_n_ratio;

    float common_n_sqrt_ratio = std::sqrt(common_n_ratio);
    float remain_n_sqrt_ratio = common_n_sqrt_ratio;

    float common_n_square_ratio = (1.0f * B_N_tile * B_N_tile);
    float remain_n_square_ratio = common_n_square_ratio;

    if(n_remain_split > 0){
        remain_std_factor = std::sqrt(2.0f * logf(remain_size));
        std_est_ratio_remain = (1.0f / remain_std_factor);

        remain_kn_ratio = (1.0f * remain_size);
        remain_kn_sqrt_ratio = (std::sqrt(n_remain_split));
        remain_k_sqrt_n_ratio = (std::sqrt(k2*1.0f) *(n_remain_split * 1.0f));

        remain_n_ratio = (1.0f * n_remain_split);
        remain_n_sqrt_ratio = std::sqrt(remain_n_ratio);
        remain_n_square_ratio = (1.0f * n_remain_split * n_remain_split);
        
        simple_remain_std_factor = std::sqrt(2.0f * logf(n_remain_split * 1.0f));

        if(useLogRatio < 1){
            simple_remain_std_factor = 1.0f;
        }

        simple_std_est_ratio_remain = (1.0f / simple_remain_std_factor);
    }

    std_est_ratios[0] = std_est_ratio_common;
    std_est_ratios[1] = std_est_ratio_remain;

    simple_std_est_ratios[0] = simple_std_est_ratio_common;
    simple_std_est_ratios[0] = simple_std_est_ratio_remain;

    kn_ratios[0] = common_kn_ratio;
    kn_ratios[1] = remain_kn_ratio;

    kn_scale_ratios[0] = (1.0f / common_kn_ratio);
    kn_scale_ratios[1] = (1.0f / remain_kn_ratio);

    kn_sqrt_ratios[0] = common_kn_sqrt_ratio;
    kn_sqrt_ratios[1] = remain_kn_sqrt_ratio;

    k_sqrt_n_ratios[0] = common_k_sqrt_n_ratio;
    k_sqrt_n_ratios[1] = remain_k_sqrt_n_ratio;

    n_ratios[0] = common_n_ratio;
    n_ratios[1] = remain_n_ratio;

    n_sqrt_ratios[0] = common_n_sqrt_ratio;
    n_sqrt_ratios[1] = remain_n_sqrt_ratio;

    n_square_ratios[0] = common_n_square_ratio;
    n_square_ratios[1] = remain_n_square_ratio;
    
    for(uint32_t i=0; i < splitNnum; i++){
        uint32_t goldenOffsetInit = i * problemShape.m();
        ElementGolden B_slice_meanabs = (ElementGolden)dataBMeanabs[i];
        ElementGolden B_slice_meansquare = (ElementGolden)dataBMeanSquare[i];
        ElementGolden B_slice_var = (ElementGolden)dataBVar[i];

        float n_ratio_factor = n_ratios[0];
        float n_sqrt_ratio_factor = n_sqrt_ratios[0];
        float n_square_ratio_factor = n_square_ratios[0];
        float simple_std_est_B_ratio = simple_std_est_ratios[0];

        if(i == splitNnum - 1){
            n_ratio_factor = n_ratios[1];
            n_sqrt_ratio_factor = n_sqrt_ratios[1];
            n_square_ratio_factor = n_square_ratios[1];
            simple_std_est_B_ratio = simple_std_est_ratios[1];
        }
        /*
            float kn_ratio_factor = params.kn_ratios[0];
            float kn_sqrt_ratio_factor = params.kn_sqrt_ratios[0];
            float k_sqrt_n_ratio_factor = params.k_sqrt_n_ratios[0];
            if(splitNIdx>=(splitNnum - 1)){
                std_est_B_ratio = params.std_est_ratios[1];
                kn_ratio_factor = params.kn_ratios[1];
                kn_sqrt_ratio_factor = params.kn_sqrt_ratios[1];
                k_sqrt_n_ratio_factor = params.k_sqrt_n_ratios[1];
            }

        */ 
        for(uint32_t j=0; j < problemShape.m(); j++){
            uint32_t goldenOffset = goldenOffsetInit + j;
            ElementGolden A_mean = (ElementGolden)dataAMean[j];
            ElementGolden A_max = (ElementGolden)dataAMax[j];
            ElementGolden A_min = (ElementGolden)dataAMin[j];
            
            ElementGolden A_std = A_max * simple_std_est_A_row_ratio;
            // std::sqrt((A_max) * (A_mean - A_min));
            dataAStd[j] = A_std;
            ElementGolden A_mean_abs = (A_mean < (ElementGolden)0.0) ? (ElementGolden)0.0 - A_mean : A_mean;
            ElementGolden tmp_thre = static_cast<ElementGolden>(n_sqrt_ratio_factor) * A_std * B_slice_var;
            tmp_thre = tmp_thre * static_cast<ElementGolden>(simple_std_est_B_ratio);

            // static_cast<ElementGolden>(n_ratio_factor) * A_mean_abs * B_slice_meanabs;
            // ElementGolden tmp_thre_s = static_cast<ElementGolden>(n_ratio_factor) * 16.0 * A_mean_abs * A_mean_abs * B_slice_var * B_slice_var;
            // tmp_thre_s = tmp_thre_s + static_cast<ElementGolden>(n_square_ratio_factor) * 16.0 * A_std * A_std * B_slice_meansquare;
            // tmp_thre_s = std::sqrt(tmp_thre_s);

            // tmp_thre = tmp_thre  + tmp_thre_s;
            // tmp_thre = tmp_thre + 
            dataGolden[goldenOffset] = static_cast<ElementGolden>(e_max) * tmp_thre;
        }
    }
}

uint32_t GetSplitkFactorForABFT(uint32_t k, 
    uint32_t SliceKUnit, uint32_t SplitKNumLimit,
    uint32_t TileKSize, 
    std::vector<uint32_t> &actualKSliceSize)
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

        uint32_t TileKNumUpper = ((SliceKUnit + TileKSize - 1) / TileKSize);
        uint32_t SliceKUnitRound = TileKNumUpper * TileKSize;
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

    uint32_t TileKNum = (k + TileKSize - 1) / TileKSize;
    uint32_t BasicSliceTileKNum = (TileKNum / splitkFactor);

    if(TileKNum % splitkFactor == 0){
        // (TileKNum / splitkFactor)
        actualKSliceSize[0] =  BasicSliceTileKNum * TileKSize;
        // (TileKNum / splitkFactor)
        actualKSliceSize[1] = BasicSliceTileKNum * TileKSize;
    }else{
        // (TileKNum / splitkFactor)
        actualKSliceSize[0] = (BasicSliceTileKNum + 1) * TileKSize;
        // (TileKNum / splitkFactor)
        actualKSliceSize[1] = BasicSliceTileKNum * TileKSize;
    }
    // (TileKNum / splitkFactor)
    actualKSliceSize[2] = k - (((TileKNum % splitkFactor) + BasicSliceTileKNum * (splitkFactor - 1)) * TileKSize);
    return splitkFactor;
}

uint32_t GetSplitkHeadSliceNum(uint32_t k, 
    uint32_t SplitKNum, 
    uint32_t TileKSize)
{
    uint32_t TileKNum = (k + TileKSize - 1) / TileKSize;
    uint32_t BasicSliceTileKNum = (TileKNum / SplitKNum);

    uint32_t HeadKSliceNum = TileKNum % SplitKNum;
    return HeadKSliceNum;
}

float ComputeSinhValue(uint32_t idj, uint32_t ChunkNSize, float lambda_v)
{
    // std:: std::
    float base_half = (float)(ChunkNSize * 1.0f) / 2;
    float exp_idj = lambda_v * (idj * 1.0f - base_half);
    float sinh_value = 0.5 *(expf(exp_idj) - expf((0.0f - exp_idj)));
    return sinh_value;
}
}
#endif