/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef EXAMPLES_COMMON_GOLDEN_COMPARE_DATA_HPP
#define EXAMPLES_COMMON_GOLDEN_COMPARE_DATA_HPP

#include <cmath>
#include <vector>
#include <string>
#include <cstdio>
#include <iterator>  // ← 必须包含！用于 std::back_inserter
#include <unordered_set>
#include <algorithm>

#include "catlass/gemv_coord.hpp"
#include "catlass/gemm_coord.hpp"

namespace Catlass::golden {

/// @brief 检测性能统计指标：检测率（DR）、行级误检率（RFP）、块级误检率（BFP）
struct DetectionMetrics {
    float detection_rate;           ///< [0.0f, 1.0f] 检测率（如 0.972f → 97.2%）
    float row_false_positive_rate;  ///< [0.0f, 1.0f] 行级误检率（越低越好）
    float block_false_positive_rate;///< [0.0f, 1.0f] 块级误检率（越低越好）
    uint32_t total_inject_errs;   ///< 注入错误总数（用于计算检测率的分母）
    uint32_t total_detected_errs; ///< 检测到的错误总数（用于计算检测率的分子）
    uint32_t false_alarm_rows;     ///< 误检的行数（用于计算行级误检率的分子）
    uint32_t false_alarm_blocks;   ///< 误检的块数（用于计算块级误检率的分子）
    uint32_t total_rows;          ///< 总行数（用于计算行级误检率的分母）
    uint32_t total_blocks;        ///< 总块数（用于计算块级误检率的分母）

    // 默认构造（显式初始化，避免未定义值）
    constexpr DetectionMetrics()
        : detection_rate(0.0f)
        , row_false_positive_rate(0.0f)
        , block_false_positive_rate(0.0f)
        , total_inject_errs(0)
        , total_detected_errs(0)
        , false_alarm_rows(0)
        , false_alarm_blocks(0)
        , total_rows(0)
        , total_blocks(0) {}

    // 带参构造（支持 {dr, rfp, bfp} 初始化）
    constexpr DetectionMetrics(
        float dr,
        float rfp,
        float bfp,
        uint32_t inject_errs,
        uint32_t detected_errs,
        uint32_t fa_rows,
        uint32_t fa_blocks,
        uint32_t total_rows,
        uint32_t total_blocks
    )
        : detection_rate(dr)
        , row_false_positive_rate(rfp)
        , block_false_positive_rate(bfp)
        , total_inject_errs(inject_errs)
        , total_detected_errs(detected_errs)
        , false_alarm_rows(fa_rows)
        , false_alarm_blocks(fa_blocks)
        , total_rows(total_rows)
        , total_blocks(total_blocks) {}

    // 语义化别名函数（全部 constexpr + noexcept，零开销！）
    [[nodiscard]] constexpr float dr() const noexcept { return detection_rate; }
    [[nodiscard]] constexpr float rfp() const noexcept { return row_false_positive_rate; }
    [[nodiscard]] constexpr float bfp() const noexcept { return block_false_positive_rate; }
    [[nodiscard]] constexpr uint32_t inject_errs() const noexcept { return total_inject_errs; }
    [[nodiscard]] constexpr uint32_t detected_errs() const noexcept { return total_detected_errs; }
    [[nodiscard]] constexpr uint32_t fa_rows() const noexcept { return false_alarm_rows; }
    [[nodiscard]] constexpr uint32_t fa_blocks() const noexcept { return false_alarm_blocks; }
    [[nodiscard]] constexpr uint32_t total_rows_count() const noexcept { return total_rows; }
    [[nodiscard]] constexpr uint32_t total_blocks_count() const noexcept { return total_blocks; }

    // 可选：反向别名（读写兼容，但通常只读就够了）
    [[nodiscard]] constexpr float& dr() noexcept { return detection_rate; }
    [[nodiscard]] constexpr float& rfp() noexcept { return row_false_positive_rate; }
    [[nodiscard]] constexpr float& bfp() noexcept { return block_false_positive_rate; }
    [[nodiscard]] constexpr uint32_t& inject_errs() noexcept { return total_inject_errs; }
    [[nodiscard]] constexpr uint32_t& detected_errs() noexcept { return total_detected_errs; }
    [[nodiscard]] constexpr uint32_t& fa_rows() noexcept { return false_alarm_rows; }
    [[nodiscard]] constexpr uint32_t& fa_blocks() noexcept { return false_alarm_blocks; }
    [[nodiscard]] constexpr uint32_t& total_rows_count() noexcept { return total_rows; }
    [[nodiscard]] constexpr uint32_t& total_blocks_count() noexcept { return total_blocks; }

    // 可选：支持结构化绑定（C++17+），让解包更自然
    // （无需额外代码 —— 因为是 public 成员，天然支持！）
    // auto [dr, rfp, bfp] = metrics; // 直接工作！
};

template<class ElementResult, class ElementCompare>
std::vector<uint64_t> CompareData(const std::vector<ElementResult>& result, const std::vector<ElementCompare>& expect,
    uint32_t computeNum)
{
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    for (uint64_t i = 0; i < result.size(); ++i) {
        if(errorIndices.size() >= 64) break;
        ElementCompare actualValue = static_cast<ElementCompare>(result[i]);
        ElementCompare expectValue = expect[i];
        ElementCompare diff = static_cast<ElementCompare>(std::fabs(static_cast<float>(actualValue) - static_cast<float>(expectValue)));
        ElementCompare threshold = 0;
        threshold = static_cast<ElementCompare>(rtol * (std::max(1.0f, std::fabs(static_cast<float>(expectValue)))));
        if (diff > threshold) {
            errorIndices.push_back(i);
            std::cout << "Error at index " << i << ": "
                      << "actual = " << actualValue << ", expect = " << expectValue
                      << ", diff = " << diff << std::endl;
        }
    }
    return errorIndices;
}

template<>
std::vector<uint64_t> CompareData(const std::vector<half>& result, const std::vector<half>& expect,
    uint32_t computeNum)
{
    // class ElementResult, class ElementCompare
    using ElementCompare = half;
    using ElementResult = half;
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    for (uint64_t i = 0; i < result.size(); ++i) {
        if(errorIndices.size() >= 64) break;
        ElementCompare actualValue = static_cast<ElementCompare>(result[i]);
        ElementCompare expectValue = expect[i];
        ElementCompare diff = static_cast<ElementCompare>(std::fabs(static_cast<float>(actualValue) - static_cast<float>(expectValue)));
        ElementCompare threshold = 0;
        threshold = static_cast<ElementCompare>(rtol * (std::max(1.0f, std::fabs(static_cast<float>(expectValue)))));
        if (diff > threshold) {
            errorIndices.push_back(i);
            std::cout << "Error at index " << i << ": "
                      << "actual = " << (float)actualValue << ", expect = " << (float)expectValue
                      << ", diff = " << (float)diff << std::endl;
        }
    }
    return errorIndices;
}

template<>
std::vector<uint64_t> CompareData(const std::vector<uint8_t>& result, const std::vector<uint8_t>& expect,
    uint32_t computeNum)
{
    using ElementCompare = uint8_t;
    using ElementResult = uint8_t;
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    for (uint64_t i = 0; i < result.size(); ++i) {
        if(errorIndices.size() >= 64) break;
        ElementCompare actualValue = static_cast<ElementCompare>(result[i]);
        ElementCompare expectValue = expect[i];
        ElementCompare diff = std::fabs(actualValue - expectValue);
        ElementCompare threshold = 0;
        // threshold = rtol * std::max(1.0f, std::fabs(expectValue));
        if (diff > threshold) {
            errorIndices.push_back(i);
            std::cout << "Error at index " << i << ": "
                      << "actual = " << static_cast<unsigned int>(actualValue) << ", expect = " << static_cast<unsigned int>(expectValue)
                      << ", diff = " << static_cast<unsigned int>(diff) << std::endl;
        }
    }
    return errorIndices;
}

template<class ElementIndex>
void ComputeErrorIndex(uint64_t basic_id, ElementIndex raw_index,
    std::vector<uint64_t> & error_indies)
{
    uint64_t bit_size = sizeof(ElementIndex) * 8;
    uint64_t start_base = basic_id * bit_size;

    error_indies.clear();
    std::cout<<"Bit Test: ";
    for (int i = 0; i < 8; i++) {
        std::cout << ((raw_index >> i) & 1);
        if (i < 7) std::cout<<" ";
        if ((raw_index & (1 << i)) == 0) {
            error_indies.push_back((uint64_t)i + start_base);
        }
    }
    std::cout<<std::endl;
}

template<class ElementIndex>
void GetErrorIndexPart(uint64_t basic_id, ElementIndex raw_index,
    std::vector<uint64_t> & error_indies,bool do_print)
{
    uint64_t bit_size = sizeof(ElementIndex) * 8;
    uint64_t start_base = basic_id * bit_size;

    error_indies.clear();
    if(do_print){
        std::cout<<"Bit Test: ";
    }
    
    for (int i = 0; i < 8; i++) {
        
        if(do_print){
            std::cout << ((raw_index >> i) & 1);
            if (i < 7) std::cout<<" ";
        }
        
        if ((raw_index & (1 << i)) == 0) {
            error_indies.push_back((uint64_t)i + start_base);
        }
    }
    if(do_print){
        std::cout<<std::endl;
    }
}

template<class ElementData>
std::vector<uint64_t> CompareDataWithIndex(
    const std::vector<uint8_t>& result, const std::vector<uint8_t>& expect,
    const std::vector<ElementData> &actualdata, const std::vector<ElementData> &expectdata, 
    uint32_t computeNum, const char* IdNameAct, const char* IdNameExp)
{
    using ElementCompare = uint8_t;
    using ElementResult = uint8_t;
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    uint64_t bit_size = sizeof(uint8_t) * 8;
    uint64_t start_base = 0;
    uint64_t end_base = 0;
    std::vector<uint64_t> unit_error_indices;
    for (uint64_t i = 0; i < result.size(); ++i) {
        if(errorIndices.size() >= 64) break;
        ElementCompare actualValue = static_cast<ElementCompare>(result[i]);
        ElementCompare expectValue = expect[i];
        ElementCompare diff = std::fabs(actualValue - expectValue);
        ElementCompare threshold = 0;

        // threshold = rtol * std::max(1.0f, std::fabs(expectValue));
        if (diff > threshold) {
            errorIndices.push_back(i);
            std::cout << "Error at index " << i << ": "
                      << "actual = " << static_cast<unsigned int>(actualValue) << ", expect = " << static_cast<unsigned int>(expectValue)
                      << ", diff = " << static_cast<unsigned int>(diff) << std::endl;
            
            start_base = i*bit_size;
            end_base = start_base + bit_size - 1;
            std::cout<<IdNameAct<<" ("<<start_base<<"~"<<end_base<<"): ";
            for(uint64_t idx = start_base; idx<=end_base; idx++){
                printf("%f ",actualdata[idx]);
            }
            printf("\n");
            std::cout<<IdNameExp<<"("<<start_base<<"~"<<end_base<<"): ";
            for(uint64_t idx = start_base; idx<=end_base; idx++){
                printf("%f ",expectdata[idx]);
            }
            printf("\n");
            ComputeErrorIndex<ElementCompare>(i, actualValue, unit_error_indices);
            printf("Detected Index: ");
            for(int32_t j = 0; j < unit_error_indices.size(); j++){
                std::cout<<unit_error_indices[j];
                if(j < unit_error_indices.size() - 1){
                    std::cout<<" ";
                }
            }
            printf("\n");
        }
    }
    return errorIndices;
}


template<class ElementData>
std::vector<uint64_t> GetErrorDataWithIndexTotal(
    const std::vector<uint8_t>& result, const std::vector<uint8_t>& expect,
    const std::vector<ElementData> &actualdata, const std::vector<ElementData> &expectdata, 
    uint32_t computeNum, const char* IdNameAct, const char* IdNameExp,
    std::vector<uint64_t>& total_error_idies, std::vector<ElementData>& total_error_data 
)
{
    using ElementCompare = uint8_t;
    using ElementResult = uint8_t;
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    uint64_t bit_size = sizeof(uint8_t) * 8;
    uint64_t start_base = 0;
    uint64_t end_base = 0;
    std::vector<uint64_t> unit_error_indices;
    for (uint64_t i = 0; i < result.size(); ++i) {
        // if(errorIndices.size() >= 64) break;
        ElementCompare actualValue = static_cast<ElementCompare>(result[i]);
        ElementCompare expectValue = expect[i];
        ElementCompare diff = std::fabs(actualValue - expectValue);
        ElementCompare threshold = 0;

        // threshold = rtol * std::max(1.0f, std::fabs(expectValue));
        if (diff > threshold) {
            errorIndices.push_back(i);
            if(errorIndices.size() < 2){
                std::cout << "Error at index " << i << ": "
                      << "actual = " << static_cast<unsigned int>(actualValue) << ", expect = " << static_cast<unsigned int>(expectValue)
                      << ", diff = " << static_cast<unsigned int>(diff) << std::endl;
            }
            
            start_base = i*bit_size;
            end_base = start_base + bit_size - 1;
            if(errorIndices.size() < 2){
                std::cout<<IdNameAct<<" ("<<start_base<<"~"<<end_base<<"): ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",actualdata[idx]);
                }
                printf("\n");
                std::cout<<IdNameExp<<"("<<start_base<<"~"<<end_base<<"): ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",expectdata[idx]);
                }
                printf("\n");
            }
            /*
            GetErrorIndexPart(uint64_t basic_id, ElementIndex raw_index,
            std::vector<uint64_t> & error_indies,bool do_print)
            */
            if(errorIndices.size() < 2){
                GetErrorIndexPart<ElementCompare>(i, actualValue, unit_error_indices,true);
                printf("Detected Index: ");
                for(int32_t j = 0; j < unit_error_indices.size(); j++){
                    std::cout<<unit_error_indices[j];
                    if(j < unit_error_indices.size() - 1){
                        std::cout<<" ";
                    }
                }
                printf("\n");
            }else{
                GetErrorIndexPart<ElementCompare>(i, actualValue, unit_error_indices,false);
            }
            total_error_idies.insert(total_error_idies.end(), unit_error_indices.begin(), unit_error_indices.end()); 
            for(int32_t j = 0; j < unit_error_indices.size(); j++){
                total_error_data.push_back(actualdata[unit_error_indices[j]]);
            }
        }
    }
    return errorIndices;
}

template<class ElementData>
std::vector<uint64_t> GetErrorDataAndIndexTotalWithThreshold(
    const std::vector<uint8_t>& result, const std::vector<uint8_t>& expect,
    const std::vector<ElementData> &actualdata, 
    const std::vector<ElementData> &expectdata,
    const std::vector<ElementData> &thresholddata, 
    uint32_t computeNum, const char* IdNameAct, const char* IdNameExp,
    std::vector<uint64_t>& total_error_idies, 
    std::vector<ElementData>& total_error_data,
    std::vector<ElementData>& total_fail_threshold_data 
)
{
    using ElementCompare = uint8_t;
    using ElementResult = uint8_t;
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    uint64_t bit_size = sizeof(uint8_t) * 8;
    uint64_t start_base = 0;
    uint64_t end_base = 0;
    std::vector<uint64_t> unit_error_indices;
    uint64_t show_cases = 0;
    for (uint64_t i = 0; i < result.size(); ++i) {
        // if(errorIndices.size() >= 64) break;
        ElementCompare actualValue = static_cast<ElementCompare>(result[i]);
        ElementCompare expectValue = expect[i];
        ElementCompare diff = std::fabs(actualValue - expectValue);
        ElementCompare threshold = 0;

        // threshold = rtol * std::max(1.0f, std::fabs(expectValue));
        if (diff > threshold) {
            errorIndices.push_back(i);
            if(errorIndices.size() < 8){
                std::cout << "Error at index " << i << ": "
                      << "actual = " << static_cast<unsigned int>(actualValue) << ", expect = " << static_cast<unsigned int>(expectValue)
                      << ", diff = " << static_cast<unsigned int>(diff) << std::endl;
            }
            
            start_base = i*bit_size;
            end_base = start_base + bit_size - 1;
            if(errorIndices.size() < 8){
                std::cout<<IdNameAct<<" ("<<start_base<<"~"<<end_base<<"): ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",actualdata[idx]);
                }
                printf("\n");
                std::cout<<IdNameExp<<"("<<start_base<<"~"<<end_base<<"): ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",expectdata[idx]);
                }
                printf("\n");
                std::cout<<"Threshold"<<"("<<start_base<<"~"<<end_base<<"): ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ", thresholddata[idx]);
                }
                printf("\n");
            }
            /*
            GetErrorIndexPart(uint64_t basic_id, ElementIndex raw_index,
            std::vector<uint64_t> & error_indies,bool do_print)
            */
            if(errorIndices.size() < 8){
                GetErrorIndexPart<ElementCompare>(i, actualValue, unit_error_indices,true);
                printf("Detected Index: ");
                for(int32_t j = 0; j < unit_error_indices.size(); j++){
                    std::cout<<unit_error_indices[j];
                    if(j < unit_error_indices.size() - 1){
                        std::cout<<" ";
                    }
                }
                printf("\n");
            }else{
                GetErrorIndexPart<ElementCompare>(i, actualValue, unit_error_indices,false);
            }
            total_error_idies.insert(total_error_idies.end(), unit_error_indices.begin(), unit_error_indices.end()); 
            for(int32_t j = 0; j < unit_error_indices.size(); j++){
                total_error_data.push_back(actualdata[unit_error_indices[j]]);
                total_fail_threshold_data.push_back(thresholddata[unit_error_indices[j]]);
            }
        }
        else{
            if(show_cases < 1){
                std::cout << "Correct at Index " << i << ": "
                      << "actual = " << static_cast<unsigned int>(actualValue) << ", expect = " << static_cast<unsigned int>(expectValue)
                      << ", diff = " << static_cast<unsigned int>(diff) << std::endl;
                start_base = i*bit_size;
                end_base = start_base + bit_size - 1;
                std::cout<<IdNameAct<<" ("<<start_base<<"~"<<end_base<<"): ";
                
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",actualdata[idx]);
                }

                printf("\n");
                
                std::cout<<IdNameExp<<"("<<start_base<<"~"<<end_base<<"): ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",expectdata[idx]);
                }
                
                printf("\n");
                
                std::cout<<"Threshold"<<"("<<start_base<<"~"<<end_base<<"): ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ", thresholddata[idx]);
                }
                printf("\n");
                std::cout<<"Bit Test: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%d ", 1);
                }
                printf("\n");
            }
            show_cases += 1;
        }
    }
    return errorIndices;
}


template<class ElementData>
std::vector<uint64_t> GetErrorDataAndIndexSliceWithThreshold(
    const Catlass::GemvCoord &problemShape,
    const std::vector<uint8_t>& result, 
    const std::vector<uint8_t>& expect,
    const std::vector<ElementData> &actualdata, 
    const std::vector<ElementData> &expectdata,
    const std::vector<ElementData> &thresholddata, 
    uint32_t computeNum, const char* IdNameAct, const char* IdNameExp,
    std::vector<uint64_t>& total_error_idies,
    std::vector<uint64_t>& total_error_idies_m,
    std::vector<uint64_t>& total_error_idies_n, 
    std::vector<ElementData>& total_error_data,
    std::vector<ElementData>& total_fail_threshold_data)
{
    using ElementCompare = uint8_t;
    using ElementResult = uint8_t;
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    uint32_t slice_row_size = problemShape.m();

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    uint64_t bit_size = sizeof(uint8_t) * 8;
    uint64_t start_base = 0;
    uint64_t end_base = 0;
    uint64_t start_row_base = 0;
    uint64_t start_slice_col_base = 0;
    uint64_t end_row_base = 0;
    uint64_t end_slice_col_base = 0;
    std::vector<uint64_t> unit_error_indices;
    uint64_t show_cases = 0;
    for (uint64_t i = 0; i < result.size(); ++i) {
        // if(errorIndices.size() >= 64) break;
        ElementCompare actualValue = static_cast<ElementCompare>(result[i]);
        ElementCompare expectValue = expect[i];
        ElementCompare diff = std::fabs(actualValue - expectValue);
        ElementCompare threshold = 0;

        // threshold = rtol * std::max(1.0f, std::fabs(expectValue));
        if (diff > threshold) {
            errorIndices.push_back(i);
            if(errorIndices.size() < 8){
                std::cout << "Error at index " << i << ": "
                      << "actual = " << static_cast<unsigned int>(actualValue) << ", expect = " << static_cast<unsigned int>(expectValue)
                      << ", diff = " << static_cast<unsigned int>(diff) << std::endl;
            }
            
            start_base = i*bit_size;
            end_base = start_base + bit_size - 1;

            start_row_base = start_base % slice_row_size;
            start_slice_col_base = start_base / slice_row_size;

            end_row_base = end_base % slice_row_size;
            end_slice_col_base = end_base / slice_row_size;

            if(errorIndices.size() < 8){
                std::cout<<IdNameAct<<"["<<"("<<start_row_base<<","<<start_slice_col_base<<")"<<"~("<<end_row_base<<","<<end_slice_col_base<<")]: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",actualdata[idx]);
                }
                printf("\n");
                std::cout<<IdNameExp<<"["<<"("<<start_row_base<<","<<start_slice_col_base<<")"<<"~("<<end_row_base<<","<<end_slice_col_base<<")]: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",expectdata[idx]);
                }
                printf("\n");
                std::cout<<"Threshold"<<"["<<"("<<start_row_base<<","<<start_slice_col_base<<")"<<"~("<<end_row_base<<","<<end_slice_col_base<<")]: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ", thresholddata[idx]);
                }
                printf("\n");
            }
            /*
            GetErrorIndexPart(uint64_t basic_id, ElementIndex raw_index,
            std::vector<uint64_t> & error_indies,bool do_print)
            */
            if(errorIndices.size() < 8){
                GetErrorIndexPart<ElementCompare>(i, actualValue, unit_error_indices,true);
                printf("Detected Index: ");

                for(int32_t j = 0; j < unit_error_indices.size(); j++){
                    uint32_t tmp_err_index = unit_error_indices[j];
                    uint32_t tmp_err_row_idx = tmp_err_index % slice_row_size;
                    uint32_t tmp_err_col_idx = tmp_err_index / slice_row_size;
                    std::cout<<"("<<tmp_err_row_idx<<","<<tmp_err_col_idx<<")";
                    if(j < unit_error_indices.size() - 1){
                        std::cout<<" ";
                    }
                }
                printf("\n");
            }else{
                GetErrorIndexPart<ElementCompare>(i, actualValue, unit_error_indices,false);
            }
            total_error_idies.insert(total_error_idies.end(), unit_error_indices.begin(), unit_error_indices.end()); 
            for(int32_t j = 0; j < unit_error_indices.size(); j++){

                uint32_t tmp_err_index = unit_error_indices[j];
                uint32_t tmp_err_row_idx = tmp_err_index % slice_row_size;
                uint32_t tmp_err_col_idx = tmp_err_index / slice_row_size;

                total_error_idies_m.push_back(tmp_err_row_idx);
                total_error_idies_n.push_back(tmp_err_col_idx);

                total_error_data.push_back(actualdata[unit_error_indices[j]]);
                total_fail_threshold_data.push_back(thresholddata[unit_error_indices[j]]);
            }
        }
        else{
            if(show_cases < 1){
                std::cout << "Correct at Index " << i << ": "
                      << "actual = " << static_cast<unsigned int>(actualValue) << ", expect = " << static_cast<unsigned int>(expectValue)
                      << ", diff = " << static_cast<unsigned int>(diff) << std::endl;
                start_base = i*bit_size;
                end_base = start_base + bit_size - 1;

                start_row_base = start_base % slice_row_size;
                start_slice_col_base = start_base / slice_row_size;

                end_row_base = end_base % slice_row_size;
                end_slice_col_base = end_base / slice_row_size;

                std::cout<<IdNameAct<<"["<<"("<<start_row_base<<","<<start_slice_col_base<<")"<<"~("<<end_row_base<<","<<end_slice_col_base<<")]: ";
                
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",actualdata[idx]);
                }

                printf("\n");
                
                std::cout<<IdNameExp<<"["<<"("<<start_row_base<<","<<start_slice_col_base<<")"<<"~("<<end_row_base<<","<<end_slice_col_base<<")]: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",expectdata[idx]);
                }
                
                printf("\n");
                
                std::cout<<"Threshold"<<"["<<"("<<start_row_base<<","<<start_slice_col_base<<")"<<"~("<<end_row_base<<","<<end_slice_col_base<<")]: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ", thresholddata[idx]);
                }
                printf("\n");
                std::cout<<"Bit Test: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%d ", 1);
                }
                printf("\n");
            }
            show_cases += 1;
        }
    }
    return errorIndices;
}

template<class ElementData>
std::vector<uint64_t> GetErrorDataAndIndexSliceSplitKWithThreshold(
    const Catlass::GemvCoord &problemShape,
    const std::vector<uint32_t> &actualKSliceSize,
    uint32_t SplitNnum,
    uint32_t SplitKNum, uint32_t HeadKSliceNum,
    const std::vector<uint8_t>& result, 
    const std::vector<uint8_t>& expect,
    const std::vector<ElementData> &actualdata, 
    const std::vector<ElementData> &expectdata,
    const std::vector<ElementData> &thresholddata, 
    uint32_t computeNum, const char* IdNameAct, const char* IdNameExp,
    std::vector<uint64_t>& total_error_idies,
    std::vector<uint64_t>& total_error_idies_m,
    std::vector<uint64_t>& total_error_idies_n, 
    std::vector<uint64_t>& total_error_idies_k,
    std::vector<ElementData>& total_error_data,
    std::vector<ElementData>& total_fail_threshold_data)
{
    using ElementCompare = uint8_t;
    using ElementResult = uint8_t;
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    uint32_t slice_row_size = problemShape.m();
    uint32_t slice_total_size = problemShape.m() * SplitNnum;

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    uint64_t bit_size = sizeof(uint8_t) * 8;
    uint64_t start_base = 0;
    uint64_t end_base = 0;
    uint64_t start_KSlice_base = 0;
    uint64_t end_KSlice_base = 0;
    uint64_t start_base_inSlice = 0;
    uint64_t end_base_inSlice = 0; 
    uint64_t start_row_base = 0;
    uint64_t start_slice_col_base = 0;
    uint64_t end_row_base = 0;
    uint64_t end_slice_col_base = 0;
    std::vector<uint64_t> unit_error_indices;
    uint64_t show_cases = 0;
    for (uint64_t i = 0; i < result.size(); ++i) {
        // if(errorIndices.size() >= 64) break;
        ElementCompare actualValue = static_cast<ElementCompare>(result[i]);
        ElementCompare expectValue = expect[i];
        ElementCompare diff = std::fabs(actualValue - expectValue);
        ElementCompare threshold = 0;

        // threshold = rtol * std::max(1.0f, std::fabs(expectValue));
        if (diff > threshold) {
            errorIndices.push_back(i);
            if(errorIndices.size() < 8){
                std::cout << "Error at index " << i << ": "
                      << "actual = " << static_cast<unsigned int>(actualValue) << ", expect = " << static_cast<unsigned int>(expectValue)
                      << ", diff = " << static_cast<unsigned int>(diff) << std::endl;
            }
            
            start_base = i * bit_size;
            end_base = start_base + bit_size - 1;

            start_KSlice_base = start_base  / slice_total_size;
            start_base_inSlice = start_base % slice_total_size;

            start_row_base = start_base_inSlice % slice_row_size;
            start_slice_col_base = start_base_inSlice / slice_row_size;

            end_KSlice_base = end_base  / slice_total_size;
            end_base_inSlice = end_base % slice_total_size;

            end_row_base = end_base_inSlice % slice_row_size;
            end_slice_col_base = end_base_inSlice / slice_row_size;

            if(errorIndices.size() < 8){
                std::cout<<IdNameAct<<"["<<"("<<start_KSlice_base<<","<<start_row_base<<","<<start_slice_col_base<<")"<<"~("<<end_KSlice_base<<","<<end_row_base<<","<<end_slice_col_base<<")]: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",actualdata[idx]);
                }
                printf("\n");
                std::cout<<IdNameExp<<"["<<"("<<start_KSlice_base<<","<<start_row_base<<","<<start_slice_col_base<<")"<<"~("<<end_KSlice_base<<","<<end_row_base<<","<<end_slice_col_base<<")]: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",expectdata[idx]);
                }
                printf("\n");
                std::cout<<"Threshold"<<"["<<"("<<start_KSlice_base<<","<<start_row_base<<","<<start_slice_col_base<<")"<<"~("<<end_KSlice_base<<","<<end_row_base<<","<<end_slice_col_base<<")]: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ", thresholddata[idx]);
                }
                printf("\n");
            }
            /*
            GetErrorIndexPart(uint64_t basic_id, ElementIndex raw_index,
            std::vector<uint64_t> & error_indies,bool do_print)
            */
            if(errorIndices.size() < 8){
                GetErrorIndexPart<ElementCompare>(i, actualValue, unit_error_indices,true);
                printf("Detected Index: ");

                for(int32_t j = 0; j < unit_error_indices.size(); j++){
                    uint32_t tmp_err_index = unit_error_indices[j];
                    uint32_t tmp_err_kslice_idx = tmp_err_index / slice_total_size;
                    uint32_t tmp_err_idx_mn = tmp_err_index % slice_total_size;

                    uint32_t tmp_err_row_idx = tmp_err_idx_mn % slice_row_size;
                    uint32_t tmp_err_col_idx = tmp_err_idx_mn / slice_row_size;
                    std::cout<<"("<<tmp_err_kslice_idx<<","<<tmp_err_row_idx<<","<<tmp_err_col_idx<<")";
                    if(j < unit_error_indices.size() - 1){
                        std::cout<<" ";
                    }
                }
                printf("\n");
            }else{
                GetErrorIndexPart<ElementCompare>(i, actualValue, unit_error_indices,false);
            }

            total_error_idies.insert(total_error_idies.end(), unit_error_indices.begin(), unit_error_indices.end()); 
            for(int32_t j = 0; j < unit_error_indices.size(); j++){

                uint32_t tmp_err_index = unit_error_indices[j];
                uint32_t tmp_err_kslice_idx = tmp_err_index / slice_total_size;
                uint32_t tmp_err_idx_mn = tmp_err_index % slice_total_size;

                uint32_t tmp_err_row_idx = tmp_err_idx_mn % slice_row_size;
                uint32_t tmp_err_col_idx = tmp_err_idx_mn / slice_row_size;

                total_error_idies_m.push_back(tmp_err_row_idx);
                total_error_idies_n.push_back(tmp_err_col_idx);
                total_error_idies_k.push_back(tmp_err_kslice_idx);

                total_error_data.push_back(actualdata[unit_error_indices[j]]);
                total_fail_threshold_data.push_back(thresholddata[unit_error_indices[j]]);
            }
        }
        else{
            if(show_cases < 1){
                std::cout << "Correct at Index " << i << ": "
                      << "actual = " << static_cast<unsigned int>(actualValue) << ", expect = " << static_cast<unsigned int>(expectValue)
                      << ", diff = " << static_cast<unsigned int>(diff) << std::endl;
                

                end_row_base = end_base % slice_row_size;
                end_slice_col_base = end_base / slice_row_size;

                start_base = i*bit_size;
                end_base = start_base + bit_size - 1;

                start_KSlice_base = start_base  / slice_total_size;
                start_base_inSlice = start_base % slice_total_size;

                start_row_base = start_base_inSlice % slice_row_size;
                start_slice_col_base = start_base_inSlice / slice_row_size;

                end_KSlice_base = end_base  / slice_total_size;
                end_base_inSlice = end_base % slice_total_size;

                end_row_base = end_base_inSlice % slice_row_size;
                end_slice_col_base = end_base_inSlice / slice_row_size;

                std::cout<<IdNameAct<<"["<<"("<<start_KSlice_base<<","<<start_row_base<<","<<start_slice_col_base<<")"<<"~("<<end_KSlice_base<<","<<end_row_base<<","<<end_slice_col_base<<")]: ";
                
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",actualdata[idx]);
                }

                printf("\n");
                
                std::cout<<IdNameExp<<"["<<"("<<start_KSlice_base<<","<<start_row_base<<","<<start_slice_col_base<<")"<<"~("<<end_KSlice_base<<","<<end_row_base<<","<<end_slice_col_base<<")]: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",expectdata[idx]);
                }
                
                printf("\n");
                
                std::cout<<"Threshold"<<"["<<"("<<start_KSlice_base<<","<<start_row_base<<","<<start_slice_col_base<<")"<<"~("<<end_KSlice_base<<","<<end_row_base<<","<<end_slice_col_base<<")]: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ", thresholddata[idx]);
                }
                printf("\n");
                std::cout<<"Bit Test: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%d ", 1);
                }
                printf("\n");
            }
            show_cases += 1;
        }
    }
    return errorIndices;
}


template<class ElementData>
std::vector<uint64_t> CompareDataAndIndexSliceWithThreshold(
    const Catlass::GemvCoord &problemShape,
    const std::vector<ElementData> &actualdata, 
    const std::vector<ElementData> &expectdata,
    const std::vector<ElementData> &thresholddata, 
    uint32_t computeNum, const char* IdNameAct, const char* IdNameExp,
    std::vector<uint64_t>& total_error_idies,
    std::vector<uint64_t>& total_error_idies_m,
    std::vector<uint64_t>& total_error_idies_n, 
    std::vector<ElementData>& total_error_data,
    std::vector<ElementData>& total_fail_threshold_data)
{
    // using ElementCompare = uint8_t;
    // using ElementResult = uint8_t;
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    uint32_t slice_row_size = problemShape.m();

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    uint64_t bit_size = 1;
    uint64_t start_base = 0;
    uint64_t end_base = 0;
    uint64_t start_row_base = 0;
    uint64_t start_slice_col_base = 0;
    uint64_t end_row_base = 0;
    uint64_t end_slice_col_base = 0;
    std::vector<uint64_t> unit_error_indices;
    uint64_t show_cases = 0;
    for (uint64_t i = 0; i < expectdata.size(); ++i) {
        // if(errorIndices.size() >= 64) break;
        ElementData actualValue = (actualdata[i]);
        ElementData expectValue = expectdata[i];
        ElementData diff = std::fabs(actualValue - expectValue);
        ElementData tmp_threshold = thresholddata[i];

        // threshold = rtol * std::max(1.0f, std::fabs(expectValue));
        if (diff > tmp_threshold) {
            errorIndices.push_back(i);
            if(errorIndices.size() < 8){
                std::cout << "Error at index " << i << ": "
                      << "actual = " << (actualValue) << ", expect = " << (expectValue)
                      << ", diff = " << (diff) << ", threshold = "<<(tmp_threshold)<< std::endl;
            }
            
            start_base = i;
            end_base = start_base + bit_size - 1;

            start_row_base = start_base % slice_row_size;
            start_slice_col_base = start_base / slice_row_size;

            end_row_base = end_base % slice_row_size;
            end_slice_col_base = end_base / slice_row_size;

            if(errorIndices.size() < 8){
                std::cout<<IdNameAct<<"("<<start_row_base<<","<<start_slice_col_base<<")"<<": ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",actualdata[idx]);
                }
                printf("\n");
                std::cout<<IdNameExp<<"("<<start_row_base<<","<<start_slice_col_base<<")"<<": ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",expectdata[idx]);
                }
                printf("\n");
                std::cout<<"Threshold"<<"("<<start_row_base<<","<<start_slice_col_base<<")"<<": ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ", thresholddata[idx]);
                }
                printf("\n");
            }
            /*
            GetErrorIndexPart(uint64_t basic_id, ElementIndex raw_index,
            std::vector<uint64_t> & error_indies,bool do_print)
            */
            if(errorIndices.size() < 8){
                // GetErrorIndexPart<ElementCompare>(i, actualValue, unit_error_indices,true);
                // printf("Detected Index: ");
                std::cout<<"Detected Index: "<<i<<"("<<start_row_base<<","<<start_slice_col_base<<")";
                printf("\n");
            }else{
                // GetErrorIndexPart<ElementCompare>(i, actualValue, unit_error_indices,false);
            }
            total_error_idies.push_back(i);
            total_error_idies_m.push_back(start_row_base);
            total_error_idies_n.push_back(start_slice_col_base);

            total_error_data.push_back(actualdata[i]);
            total_fail_threshold_data.push_back(thresholddata[i]);
        }
        else{
            if(show_cases < 1){
                std::cout << "Correct at Index " << i << ": "
                      << "actual = " << (actualValue) << ", expect = " << (expectValue)
                      << ", diff = " << (diff)<< ", threshold = " <<(tmp_threshold) << std::endl;
                start_base = i*bit_size;
                end_base = start_base + bit_size - 1;

                start_row_base = start_base % slice_row_size;
                start_slice_col_base = start_base / slice_row_size;

                end_row_base = end_base % slice_row_size;
                end_slice_col_base = end_base / slice_row_size;

                std::cout<<IdNameAct<<"("<<start_row_base<<","<<start_slice_col_base<<")"<<": ";
                
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",actualdata[idx]);
                }

                printf("\n");
                
                std::cout<<IdNameExp<<"("<<start_row_base<<","<<start_slice_col_base<<")"<<": ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",expectdata[idx]);
                }
                
                printf("\n");
                
                std::cout<<"Threshold"<<"("<<start_row_base<<","<<start_slice_col_base<<")"<<": ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ", thresholddata[idx]);
                }
                printf("\n");
                std::cout<<"Bit Test: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%d ", 1);
                }
                printf("\n");
            }
            show_cases += 1;
        }
    }
    return errorIndices;
}

template<class ElementData>
std::vector<uint64_t> CompareDataAndIndexSliceSplitKWithThreshold(
    const Catlass::GemvCoord &problemShape,
    const std::vector<uint32_t> &actualKSliceSize,
    uint32_t SplitNnum,
    uint32_t SplitKNum, uint32_t HeadKSliceNum,
    const std::vector<ElementData> &actualdata, 
    const std::vector<ElementData> &expectdata,
    const std::vector<ElementData> &thresholddata, 
    uint32_t computeNum, const char* IdNameAct, const char* IdNameExp,
    std::vector<uint64_t>& total_error_idies,
    std::vector<uint64_t>& total_error_idies_m,
    std::vector<uint64_t>& total_error_idies_n,
    std::vector<uint64_t>& total_error_idies_k, 
    std::vector<ElementData>& total_error_data,
    std::vector<ElementData>& total_fail_threshold_data)
{
    // using ElementCompare = uint8_t;
    // using ElementResult = uint8_t;
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    uint32_t slice_row_size = problemShape.m();
    uint32_t slice_total_size = problemShape.m() * SplitNnum;

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    uint64_t bit_size = 1;
    uint64_t start_base = 0;
    uint64_t end_base = 0;
    uint64_t start_KSlice_base = 0;
    uint64_t end_KSlice_base = 0;
    uint64_t start_base_inSlice = 0;
    uint64_t end_base_inSlice = 0; 
    uint64_t start_row_base = 0;
    uint64_t start_slice_col_base = 0;
    uint64_t end_row_base = 0;
    uint64_t end_slice_col_base = 0;
    std::vector<uint64_t> unit_error_indices;
    uint64_t show_cases = 0;
    for (uint64_t i = 0; i < expectdata.size(); ++i) {
        // if(errorIndices.size() >= 64) break;
        ElementData actualValue = (actualdata[i]);
        ElementData expectValue = expectdata[i];
        ElementData diff = std::fabs(actualValue - expectValue);
        ElementData tmp_threshold = thresholddata[i];

        // threshold = rtol * std::max(1.0f, std::fabs(expectValue));
        if (diff > tmp_threshold) {
            errorIndices.push_back(i);
            if(errorIndices.size() < 8){
                std::cout << "Error at index " << i << ": "
                      << "actual = " << (actualValue) << ", expect = " << (expectValue)
                      << ", diff = " << (diff) << ", threshold = "<<(tmp_threshold)<< std::endl;
            }

            start_base = i;
            end_base = start_base + bit_size - 1;

            start_KSlice_base = start_base  / slice_total_size;
            start_base_inSlice = start_base % slice_total_size;

            start_row_base = start_base_inSlice % slice_row_size;
            start_slice_col_base = start_base_inSlice / slice_row_size;

            end_KSlice_base = end_base  / slice_total_size;
            end_base_inSlice = end_base % slice_total_size;

            end_row_base = end_base_inSlice % slice_row_size;
            end_slice_col_base = end_base_inSlice / slice_row_size;

            if(errorIndices.size() < 8){
                std::cout<<IdNameAct<<"("<<start_KSlice_base<<","<<start_row_base<<","<<start_slice_col_base<<")"<<": ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",actualdata[idx]);
                }
                printf("\n");
                std::cout<<IdNameExp<<"("<<start_KSlice_base<<","<<start_row_base<<","<<start_slice_col_base<<")"<<": ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",expectdata[idx]);
                }
                printf("\n");
                std::cout<<"Threshold"<<"("<<start_KSlice_base<<","<<start_row_base<<","<<start_slice_col_base<<")"<<": ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ", thresholddata[idx]);
                }
                printf("\n");
            }
            /*
            GetErrorIndexPart(uint64_t basic_id, ElementIndex raw_index,
            std::vector<uint64_t> & error_indies,bool do_print)
            */
            if(errorIndices.size() < 8){
                // GetErrorIndexPart<ElementCompare>(i, actualValue, unit_error_indices,true);
                // printf("Detected Index: ");
                std::cout<<"Detected Index: "<<i<<"("<<start_KSlice_base<<","<<start_row_base<<","<<start_slice_col_base<<")";
                printf("\n");
            }else{
                // GetErrorIndexPart<ElementCompare>(i, actualValue, unit_error_indices,false);
            }
            total_error_idies.push_back(i);
            total_error_idies_m.push_back(start_row_base);
            total_error_idies_n.push_back(start_slice_col_base);
            total_error_idies_k.push_back(start_KSlice_base);

            total_error_data.push_back(actualdata[i]);
            total_fail_threshold_data.push_back(thresholddata[i]);
        }
        else{
            if(show_cases < 1){
                std::cout << "Correct at Index " << i << ": "
                      << "actual = " << (actualValue) << ", expect = " << (expectValue)
                      << ", diff = " << (diff)<< ", threshold = " <<(tmp_threshold) << std::endl;

                start_base = i*bit_size;
                end_base = start_base + bit_size - 1;

                start_KSlice_base = start_base  / slice_total_size;
                start_base_inSlice = start_base % slice_total_size;

                start_row_base = start_base_inSlice % slice_row_size;
                start_slice_col_base = start_base_inSlice / slice_row_size;

                end_KSlice_base = end_base  / slice_total_size;
                end_base_inSlice = end_base % slice_total_size;

                end_row_base = end_base_inSlice % slice_row_size;
                end_slice_col_base = end_base_inSlice / slice_row_size;

                std::cout<<IdNameAct<<"("<<start_KSlice_base<<","<<start_row_base<<","<<start_slice_col_base<<")"<<": ";
                
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",actualdata[idx]);
                }

                printf("\n");
                
                std::cout<<IdNameExp<<"("<<start_KSlice_base<<","<<start_row_base<<","<<start_slice_col_base<<")"<<": ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",expectdata[idx]);
                }
                
                printf("\n");
                
                std::cout<<"Threshold"<<"("<<start_KSlice_base<<","<<start_row_base<<","<<start_slice_col_base<<")"<<": ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ", thresholddata[idx]);
                }
                printf("\n");
                std::cout<<"Bit Test: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%d ", 1);
                }
                printf("\n");
            }
            show_cases += 1;
        }
    }
    return errorIndices;
}


template<>
std::vector<uint64_t> CompareData(const std::vector<int32_t>& result, const std::vector<int32_t>& expect,
    uint32_t computeNum)
{
    using ElementCompare = int32_t;
    using ElementResult = int32_t;
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    for (uint64_t i = 0; i < result.size(); ++i) {
        if(errorIndices.size() >= 1000) break;
        ElementCompare actualValue = static_cast<ElementCompare>(result[i]);
        ElementCompare expectValue = expect[i];
        ElementCompare diff = std::fabs(actualValue - expectValue);
        ElementCompare threshold = 0;
        // threshold = rtol * std::max(1.0f, std::fabs(expectValue));
        if (diff > threshold) {
            errorIndices.push_back(i);
            std::cout << "Error at index " << i << ": "
                      << "actual = " << actualValue << ", expect = " << expectValue
                      << ", diff = " << diff << std::endl;
        }
    }
    return errorIndices;
}

// Compare for GroupedMatmul slicing M
template<class ElementResult, class ElementCompare>
std::vector<uint64_t> CompareData(const std::vector<ElementResult>& result, const std::vector<ElementCompare>& expect,
    uint32_t computeNum, uint32_t validNum)
{
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    for (uint64_t i = 0; i < validNum; ++i) {
        // if(errorIndices.size() >= 64) break;
        ElementCompare actualValue = static_cast<ElementCompare>(result[i]);
        ElementCompare expectValue = expect[i];
        ElementCompare diff = std::fabs(actualValue - expectValue);
        if (diff > rtol * std::max(1.0f, std::fabs(expectValue))) {
            errorIndices.push_back(i);
            std::cout << "Error at index " << i << ": "
                      << "actual = " << actualValue << ", expect = " << expectValue
                      << ", diff = " << diff << std::endl;
        }
    }
    return errorIndices;
}

// Compare for GroupedMatmul slicing K
template<class ElementResult, class ElementCompare, class T>
std::vector<uint64_t> CompareData(const std::vector<ElementResult>& result, const std::vector<ElementCompare>& expect,
    uint32_t computeNum, const std::vector<T>& groupList, uint32_t stride)
{
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    T prevGroupValue = 0;
    uint64_t currentIndex = 0;
    for (const auto& groupValue : groupList) {
        // if(errorIndices.size() >= 64) break;
        if (groupValue == prevGroupValue) {
            currentIndex += stride;
            prevGroupValue = groupValue;
            continue;
        }
        for (uint64_t i = 0; i < stride; ++i) {
            if (currentIndex >= result.size()) break;
            // if(errorIndices.size() >= 64) break;
            ElementCompare actualValue = static_cast<ElementCompare>(result[currentIndex]);
            ElementCompare expectValue = expect[currentIndex];
            ElementCompare diff = std::fabs(actualValue - expectValue);
            if (diff > rtol * std::max(1.0f, std::fabs(expectValue))) {
                errorIndices.push_back(i);
                std::cout << "Error at index " << i << ": "
                        << "actual = " << actualValue << ", expect = " << expectValue
                        << ", diff = " << diff << std::endl;
            }
            currentIndex++;
        }
        prevGroupValue = groupValue;
    }
    return errorIndices;
}

template<class ElementData>
std::vector<uint64_t> GetErrorDataAndIndexSliceWithThresholdFI(
    const Catlass::GemvCoord &problemShape,
    const std::vector<uint8_t>& result, 
    const std::vector<uint8_t>& expect,
    const std::vector<ElementData> &actualdata, 
    const std::vector<ElementData> &expectdata,
    const std::vector<ElementData> &thresholddata, 
    const std::vector<ElementData> &rawOutputdata,
    uint32_t computeNum, const char* IdNameAct, const char* IdNameExp,
    std::vector<uint64_t>& total_error_idies,
    std::vector<int32_t>& total_error_idies_for_FI,
    std::vector<int32_t>& total_error_blk_idies_for_FI,
    std::vector<uint64_t>& total_error_idies_m,
    std::vector<uint64_t>& total_error_idies_n, 
    std::vector<ElementData>& total_error_data,
    std::vector<ElementData>& total_fail_threshold_data,
    std::vector<ElementData>& total_original_data,
    uint32_t BlockMSize, uint32_t BlockNSize,
    uint32_t FI_col_pos,
    bool ifShowError)
{
    using ElementCompare = uint8_t;
    using ElementResult = uint8_t;
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    uint32_t slice_row_size = problemShape.m();
    uint32_t slice_row_blk_num = (slice_row_size + BlockMSize - 1) / BlockMSize;

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    uint64_t bit_size = sizeof(uint8_t) * 8;
    uint64_t start_base = 0;
    uint64_t end_base = 0;
    uint64_t start_row_base = 0;
    uint64_t start_slice_col_base = 0;
    uint64_t end_row_base = 0;
    uint64_t end_slice_col_base = 0;
    uint64_t start_slice_row_base = 0;
    uint64_t end_slice_row_base = 0;
    std::vector<uint64_t> unit_error_indices;
    uint64_t show_cases = 0;
    for (uint64_t i = 0; i < result.size(); ++i) {
        // if(errorIndices.size() >= 64) break;
        ElementCompare actualValue = static_cast<ElementCompare>(result[i]);
        ElementCompare expectValue = expect[i];
        ElementCompare diff = std::fabs(actualValue - expectValue);
        ElementCompare threshold = 0;

        // threshold = rtol * std::max(1.0f, std::fabs(expectValue));
        if (diff > threshold) {
            errorIndices.push_back(i);
            if((errorIndices.size() < 8)&&ifShowError){
                std::cout << "Error at index " << i << ": "
                      << "actual = " << static_cast<unsigned int>(actualValue) << ", expect = " << static_cast<unsigned int>(expectValue)
                      << ", diff = " << static_cast<unsigned int>(diff) << std::endl;
            }
            
            start_base = i*bit_size;
            end_base = start_base + bit_size - 1;

            start_row_base = start_base % slice_row_size;
            start_slice_row_base = start_row_base / BlockMSize;
            start_slice_col_base = start_base / slice_row_size;

            end_row_base = end_base % slice_row_size;
            end_slice_row_base = end_row_base / BlockMSize;
            end_slice_col_base = end_base / slice_row_size;

            if((errorIndices.size() < 8)&&ifShowError){
                std::cout<<IdNameAct<<"["<<"("<<start_row_base<<","<<start_slice_row_base<<","<<start_slice_col_base<<")"<<"~("<<end_row_base<<","<<end_slice_row_base<<","<<end_slice_col_base<<")]: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",actualdata[idx]);
                }
                printf("\n");
                std::cout<<IdNameExp<<"["<<"("<<start_row_base<<","<<start_slice_row_base<<","<<start_slice_col_base<<")"<<"~("<<end_row_base<<","<<end_slice_row_base<<","<<end_slice_col_base<<")]: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",expectdata[idx]);
                }
                printf("\n");
                std::cout<<"Threshold"<<"["<<"("<<start_row_base<<","<<start_slice_row_base<<","<<start_slice_col_base<<")"<<"~("<<end_row_base<<","<<end_slice_row_base<<","<<end_slice_col_base<<")]: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ", thresholddata[idx]);
                }
                printf("\n");
            }
            /*
            GetErrorIndexPart(uint64_t basic_id, ElementIndex raw_index,
            std::vector<uint64_t> & error_indies,bool do_print)
            */
            if((errorIndices.size() < 8)&&ifShowError){
                GetErrorIndexPart<ElementCompare>(i, actualValue, unit_error_indices,true);
                printf("Detected Index: ");

                for(int32_t j = 0; j < unit_error_indices.size(); j++){
                    uint32_t tmp_err_index = unit_error_indices[j];
                    uint32_t tmp_err_row_idx = tmp_err_index % slice_row_size;
                    uint32_t tmp_error_row_block_idx = tmp_err_row_idx / BlockMSize;
                    uint32_t tmp_err_col_idx = tmp_err_index / slice_row_size;
                    std::cout<<"("<<tmp_err_row_idx<<","<<tmp_error_row_block_idx<<","<<tmp_err_col_idx<<")";
                    if(j < unit_error_indices.size() - 1){
                        std::cout<<" ";
                    }
                }
                printf("\n");
            }else{
                GetErrorIndexPart<ElementCompare>(i, actualValue, unit_error_indices,false);
            }
            total_error_idies.insert(total_error_idies.end(), unit_error_indices.begin(), unit_error_indices.end()); 

            for(int32_t j = 0; j < unit_error_indices.size(); j++){

                uint32_t tmp_err_index = unit_error_indices[j];
                uint32_t tmp_err_row_idx = tmp_err_index % slice_row_size;
                uint32_t tmp_err_col_idx = tmp_err_index / slice_row_size;
                uint32_t tmp_err_col_element_idx = tmp_err_col_idx * BlockNSize + FI_col_pos;
                uint32_t tmp_error_row_block_idx = tmp_err_row_idx / BlockMSize;

                int32_t tmp_err_blk_index_for_FI = (int32_t)tmp_err_col_idx * (int32_t)slice_row_blk_num + (int32_t)tmp_error_row_block_idx;
                
                int32_t tmp_err_index_for_FI = (int32_t)unit_error_indices[j];

                uint64_t tmp_original_data_index = (uint64_t)tmp_err_row_idx * problemShape.n() + tmp_err_col_element_idx;

                ElementData tmp_original_data = rawOutputdata[tmp_original_data_index];
                
                total_error_idies_for_FI.push_back(tmp_err_index_for_FI);
                total_error_blk_idies_for_FI.push_back(tmp_err_blk_index_for_FI);

                total_error_idies_m.push_back(tmp_err_row_idx);
                total_error_idies_n.push_back(tmp_err_col_idx);

                total_error_data.push_back(actualdata[unit_error_indices[j]]);
                total_fail_threshold_data.push_back(thresholddata[unit_error_indices[j]]);
                total_original_data.push_back(tmp_original_data);

            }
        }
        else{
            if((show_cases < 1)&&ifShowError){
                std::cout << "Correct at Index " << i << ": "
                      << "actual = " << static_cast<unsigned int>(actualValue) << ", expect = " << static_cast<unsigned int>(expectValue)
                      << ", diff = " << static_cast<unsigned int>(diff) << std::endl;
                start_base = i*bit_size;
                end_base = start_base + bit_size - 1;

                start_row_base = start_base % slice_row_size;
                start_slice_row_base = start_row_base / BlockMSize;
                start_slice_col_base = start_base / slice_row_size;

                end_row_base = end_base % slice_row_size;
                end_slice_row_base = end_row_base / BlockMSize;
                end_slice_col_base = end_base / slice_row_size;

                std::cout<<IdNameAct<<"["<<"("<<start_row_base<<","<<start_slice_row_base<<","<<start_slice_col_base<<")"<<"~("<<end_row_base<<","<<end_slice_row_base<<","<<end_slice_col_base<<")]: ";
                
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",actualdata[idx]);
                }

                printf("\n");
                
                std::cout<<IdNameExp<<"["<<"("<<start_row_base<<","<<start_slice_row_base<<","<<start_slice_col_base<<")"<<"~("<<end_row_base<<","<<end_slice_row_base<<","<<end_slice_col_base<<")]: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",expectdata[idx]);
                }
                
                printf("\n");
                
                std::cout<<"Threshold"<<"["<<"("<<start_row_base<<","<<start_slice_row_base<<","<<start_slice_col_base<<")"<<"~("<<end_row_base<<","<<end_slice_row_base<<","<<end_slice_col_base<<")]: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ", thresholddata[idx]);
                }
                printf("\n");
                std::cout<<"Bit Test: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%d ", 1);
                }
                printf("\n");
            }
            show_cases += 1;
        }
    }
    return errorIndices;
}

// 取出>=0 且不重复的元素
template<class ElementData>
std::vector<ElementData> nonNegativeUniqueforFI(const std::vector<ElementData>& raw_vector){
    std::unordered_set<ElementData> seen;
    std::vector<ElementData> result;
    result.reserve(raw_vector.size()); // 预分配，避免多次扩容

    for (ElementData x : raw_vector) {
        if (x >= (ElementData)(0) && seen.insert(x).second) {
            result.push_back(x);
        }
    }
    return result;
}


// 排序后且去除掉重复的元素

template<class ElementData>
std::vector<ElementData> unique_sorted(std::vector<ElementData> v) {
    std::sort(v.begin(), v.end());
    auto last = std::unique(v.begin(), v.end());
    v.erase(last, v.end());
    return v;
}

// 求两个集合的交集
template<class ElementData>
std::vector<ElementData> intersect_sorted_unique(
    const std::vector<ElementData>& a,
    const std::vector<ElementData>& b) 
{
    // 提前返回空结果（可选优化）
    if (a.empty() || b.empty()) return {};

    // assert(std::is_sorted(a.begin(), a.end()));
    // assert(std::is_sorted(b.begin(), b.end()));

    std::vector<ElementData> result;
    result.reserve(std::min(a.size(), b.size()));

    std::set_intersection(
        a.begin(), a.end(),
        b.begin(), b.end(),
        std::back_inserter(result)
    );
    return result;
}

// 求集合a中存在但不存在于集合b中的元素，用于筛选误检的情况
template<class ElementData>
std::vector<ElementData> set_difference_sorted_unique(
    const std::vector<ElementData>& a,
    const std::vector<ElementData>& b)
{
    // assert(std::is_sorted(a.begin(), a.end()));
    // assert(std::is_sorted(b.begin(), b.end()));

    std::vector<ElementData> result;
    result.reserve(a.size()); // 最多情况是a中元素全部保留，全部不在b内

    // a - b
    std::set_difference(
        a.begin(), a.end(), // 被减数（左操作数）
        b.begin(), b.end(),  // 减数（右操作数）
        std::back_inserter(result)
    );
    return result;
}

// 将a中元素重新放缩为Blk Index 来求解block级别的误检率
std::vector<int32_t> convert_to_blk_index(const std::vector<int32_t>& a,
    uint32_t slice_row_size, 
    uint32_t BlockMSize, uint32_t BlockNSize)
{
    uint32_t slice_row_blk_num = (slice_row_size + BlockMSize - 1) / BlockMSize;
    std::vector<int32_t> blk_index_result;
    for(int32_t idx : a){
        uint32_t tmp_err_row_idx = idx % slice_row_size;
        uint32_t tmp_error_row_block_idx = tmp_err_row_idx / BlockMSize;
        uint32_t tmp_err_col_idx = idx / slice_row_size;
        uint32_t tmp_err_col_element_idx = tmp_err_col_idx * BlockNSize;
        uint32_t tmp_err_blk_index = tmp_err_col_idx * slice_row_blk_num + tmp_error_row_block_idx;
        blk_index_result.push_back((int32_t)tmp_err_blk_index);
    }
    return blk_index_result;
}

// 计算每个block所注入错误的预期数量
uint32_t calculate_expected_errors_per_blk(
    uint32_t KSize, uint32_t KInjectUnit,
    uint32_t inject_num_blk_limit, uint32_t BlockMSize
)
{
    uint32_t inject_num_blk_aim = (KSize + KInjectUnit - 1) / KInjectUnit;

    uint32_t inject_num_blk = (inject_num_blk_aim <= inject_num_blk_limit) ? inject_num_blk_aim : inject_num_blk_limit;

    inject_num_blk = (inject_num_blk <= BlockMSize) ? inject_num_blk : BlockMSize;
        
    inject_num_blk = (inject_num_blk <= 1) ? 1 : inject_num_blk;

    return inject_num_blk;
}

// 对齐每个Block所注入错误的预期数量，确保每个Block注入的错误数量为 32 byte 的整数倍
uint32_t compute_align_expected_errors_per_blk(
    uint32_t expected_errors_per_blk, uint32_t element_size_in_byte)
{
    uint32_t elements_aligned_num = 32 / element_size_in_byte;
    uint32_t aligned_expected_errors_per_blk = ((expected_errors_per_blk + elements_aligned_num - 1) / elements_aligned_num) * elements_aligned_num;
    return aligned_expected_errors_per_blk;
}

// 计算存储ground 结果所需的vector的大小，单位为element数量
uint32_t calculate_ground_vector_size(
    const Catlass::GemvCoord &problemShape,
    uint32_t expected_errors_per_blk, uint32_t BlockMSize, 
    uint32_t BlockNSize, uint32_t TileMSize, uint32_t element_size_in_byte)
{
    uint32_t total_row_size = problemShape.m();
    uint32_t total_col_size = problemShape.n();

    uint32_t SplitNnum = (total_col_size + BlockNSize - 1) / BlockNSize;
    uint32_t SplitMnum = (total_row_size + BlockMSize - 1) / BlockMSize;

    uint32_t total_blocks = SplitNnum * SplitMnum;

    uint32_t aligned_expected_errors_per_blk = compute_align_expected_errors_per_blk(
        expected_errors_per_blk, element_size_in_byte);

    uint32_t tiles_per_blk = (BlockMSize + TileMSize - 1) / TileMSize;
    uint32_t total_tiles = tiles_per_blk * total_blocks;

    uint32_t ground_vector_size = total_tiles * aligned_expected_errors_per_blk;

    return ground_vector_size;
}


// 统计检测结果，将结果保存为DetectionMetrics结构体

DetectionMetrics CalculateDetectionMetrics(
    const std::vector<int32_t>& detected_error_idies_for_FI, 
    const std::vector<int32_t>& ground_error_idies_for_FI,
    const Catlass::GemvCoord &problemShape, 
    std::vector<int32_t>& true_detected_error_idies_set,
    std::vector<int32_t>& false_detected_error_idies_set,
    std::vector<int32_t>& false_detected_error_blks_set,
    uint32_t BlockMSize, uint32_t BlockNSize)
{
    // DetectionMetrics metrics;

    float detection_rate;           ///< [0.0f, 1.0f] 检测率（如 0.972f → 97.2%）
    float row_false_positive_rate;  ///< [0.0f, 1.0f] 行级误检率（越低越好）
    float block_false_positive_rate;///< [0.0f, 1.0f] 块级误检率（越低越好）
    uint32_t total_inject_errs;   ///< 注入错误总数（用于计算检测率的分母）
    uint32_t total_detected_errs; ///< 检测到的错误总数（用于计算检测率的分子）
    uint32_t false_alarm_rows;     ///< 误检的行数（用于计算行级误检率的分子）
    uint32_t false_alarm_blocks;   ///< 误检的块数（用于计算块级误检率的分子）
    uint32_t total_rows;          ///< 总行数（用于计算行级误检率的分母）
    uint32_t total_blocks;        ///< 总块数（用于计算块级误检率的分母）

    uint32_t slice_row_size = problemShape.m();
    uint32_t SplitNnum = (problemShape.n() + BlockNSize - 1) / BlockNSize;
    uint32_t SplitMnum = (problemShape.m() + BlockMSize - 1) / BlockMSize;
    
    total_blocks = SplitNnum * SplitMnum;
    total_rows  = SplitNnum * slice_row_size;

    std::vector<int32_t> ground_error_idies_set = nonNegativeUniqueforFI(ground_error_idies_for_FI);
    std::vector<int32_t> detected_error_idies_set = nonNegativeUniqueforFI(detected_error_idies_for_FI);

    ground_error_idies_set = unique_sorted(ground_error_idies_set);
    detected_error_idies_set = unique_sorted(detected_error_idies_set);

    total_inject_errs = ground_error_idies_set.size();

    true_detected_error_idies_set.clear();
    true_detected_error_idies_set = intersect_sorted_unique(
        detected_error_idies_set, ground_error_idies_set);
    
    true_detected_error_idies_set = unique_sorted(true_detected_error_idies_set);
    
    total_detected_errs = true_detected_error_idies_set.size();

    false_detected_error_idies_set.clear();
    false_detected_error_idies_set = set_difference_sorted_unique(
        detected_error_idies_set, ground_error_idies_set);

    
    false_detected_error_idies_set = unique_sorted(false_detected_error_idies_set);

    false_alarm_rows = false_detected_error_idies_set.size();

    false_detected_error_blks_set.clear();
    false_detected_error_blks_set = convert_to_blk_index(false_detected_error_idies_set,
        slice_row_size, BlockMSize, BlockNSize);
    
    false_detected_error_blks_set = unique_sorted(false_detected_error_blks_set);

    false_alarm_blocks = false_detected_error_blks_set.size();

    uint32_t uninjected_row_nums = (total_rows > total_inject_errs) ? total_rows - total_inject_errs : 0;


    detection_rate = (total_inject_errs > 0) ? static_cast<float>(total_detected_errs*1.0f) / static_cast<float>(total_inject_errs*1.0f) : 1.0f;

    detection_rate = (detection_rate > 1.0f) ? 1.0f : detection_rate;

    row_false_positive_rate = (uninjected_row_nums > 0) ? static_cast<float>(false_alarm_rows*1.0f) / static_cast<float>(uninjected_row_nums*1.0f) : 0.0f;

    row_false_positive_rate = (row_false_positive_rate > 1.0f) ? 1.0f : row_false_positive_rate;

    block_false_positive_rate = (total_blocks > 0) ? static_cast<float>(false_alarm_blocks*1.0f) / static_cast<float>(total_blocks*1.0f) : 0.0f;
    block_false_positive_rate = (block_false_positive_rate > 1.0f) ? 1.0f : block_false_positive_rate;


    

    DetectionMetrics metrics{
        detection_rate,
        row_false_positive_rate,
        block_false_positive_rate,
        total_inject_errs,
        total_detected_errs,
        false_alarm_rows,
        false_alarm_blocks,
        total_rows,
        total_blocks
    };
    
    return metrics;
}

DetectionMetrics UpdateDetectionMetrics(
    DetectionMetrics existing_metrics,
    DetectionMetrics add_metrics
    )
{
    // DetectionMetrics metrics;

    float update_detection_rate;           ///< [0.0f, 1.0f] 检测率（如 0.972f → 97.2%）
    float update_row_false_positive_rate;  ///< [0.0f, 1.0f] 行级误检率（越低越好）
    float update_block_false_positive_rate;///< [0.0f, 1.0f] 块级误检率（越低越好）
    uint32_t update_total_inject_errs;   ///< 注入错误总数（用于计算检测率的分母）
    uint32_t update_total_detected_errs; ///< 检测到的错误总数（用于计算检测率的分子）
    uint32_t update_false_alarm_rows;     ///< 误检的行数（用于计算行级误检率的分子）
    uint32_t update_false_alarm_blocks;   ///< 误检的块数（用于计算块级误检率的分子）
    uint32_t update_total_rows;          ///< 总行数（用于计算行级误检率的分母）
    uint32_t update_total_blocks;        ///< 总块数（用于计算块级误检率的分母）

    
    uint32_t add_blocks = add_metrics.total_blocks_count();
    uint32_t add_rows  = add_metrics.total_rows_count();
    uint32_t add_inject_errs = add_metrics.inject_errs();
    uint32_t add_detected_errs = add_metrics.detected_errs();
    uint32_t add_false_alarm_rows = add_metrics.fa_rows();
    uint32_t add_false_alarm_blocks = add_metrics.fa_blocks();

    update_total_blocks = existing_metrics.total_blocks_count() + add_blocks;
    update_total_rows = existing_metrics.total_rows_count() + add_rows;
    update_total_inject_errs = existing_metrics.inject_errs() + add_inject_errs;
    update_total_detected_errs = existing_metrics.detected_errs() + add_detected_errs;
    update_false_alarm_rows = existing_metrics.fa_rows() + add_false_alarm_rows;
    update_false_alarm_blocks = existing_metrics.fa_blocks() + add_false_alarm_blocks;

    uint32_t uninjected_row_nums = (update_total_rows > update_total_inject_errs) ? update_total_rows - update_total_inject_errs : 0;

    update_detection_rate = (update_total_inject_errs > 0) ? static_cast<float>(update_total_detected_errs*1.0f) / static_cast<float>(update_total_inject_errs*1.0f) : 1.0f;

    update_detection_rate = (update_detection_rate > 1.0f) ? 1.0f : update_detection_rate;

    update_row_false_positive_rate = (uninjected_row_nums > 0) ? static_cast<float>(update_false_alarm_rows*1.0f) / static_cast<float>(uninjected_row_nums*1.0f) : 0.0f;

    update_row_false_positive_rate = (update_row_false_positive_rate > 1.0f) ? 1.0f : update_row_false_positive_rate;

    update_block_false_positive_rate = (update_total_blocks > 0) ? static_cast<float>(update_false_alarm_blocks*1.0f) / static_cast<float>(update_total_blocks*1.0f) : 0.0f;
    update_block_false_positive_rate = (update_block_false_positive_rate > 1.0f) ? 1.0f : update_block_false_positive_rate;

    DetectionMetrics updated_metrics{
        update_detection_rate,
        update_row_false_positive_rate,
        update_block_false_positive_rate,
        update_total_inject_errs,
        update_total_detected_errs,
        update_false_alarm_rows,
        update_false_alarm_blocks,
        update_total_rows,
        update_total_blocks
    };
    
    return updated_metrics;
}

}  // namespace Catlass::golden

#endif  // EXAMPLES_COMMON_GOLDEN_COMPARE_DATA_HPP
