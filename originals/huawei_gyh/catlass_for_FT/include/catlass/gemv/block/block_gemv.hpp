/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMV_BLOCK_BLOCK_GEMV_HPP
#define CATLASS_GEMV_BLOCK_BLOCK_GEMV_HPP

#include "catlass/catlass.hpp"
#include "catlass/gemv/helper.hpp"
namespace Catlass::Gemv::Block {

template <
    class DispatchPolicy,
    class... Args
>
struct BlockGemv {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockGemv is not implemented for this DispatchPolicy");
};

/*
struct BlockSumMaxNoSplitK <
    Gemm::GemvAtlasA2,
    Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR,
    UBTileShape_,
    AType_,
    XType_,
    YType_,
    BiasType_,
    TileCopy_,
    TileFaultSum_
>
*/
template <
    class DispatchPolicy,
    Gemv::helper::FT_THRESHOLD_ALGORITHM ALGO_TYPE,
    class... Args 
>
struct BlockSumMaxNoSplitK {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockSumMaxNoSplitK is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    Gemv::helper::FT_THRESHOLD_ALGORITHM ALGO_TYPE,
    class... Args 
>
struct BlockSumMaxNoSplitKBF {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockSumMaxNoSplitKBF is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    Gemv::helper::FT_THRESHOLD_ALGORITHM ALGO_TYPE_,
    Gemv::helper::FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE_,
    class... Args
>
struct BlockFTSumNoSplitK {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockFTSumNoSplitK is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    Gemv::helper::FT_THRESHOLD_ALGORITHM ALGO_TYPE_,
    Gemv::helper::FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE_,
    Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    Gemv::helper::FT_COMP_TYPE COMP_TYPE_,
    class... Args
>
struct BlockFTGemvNoSplitK {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockFTGemvNoSplitK is not implemented for this DispatchPolicy");
};

// Gemv::helper::FT_ABE_TYPE ABE_TYPE_,
template <
    class DispatchPolicy,
    Gemv::helper::FT_THRESHOLD_ALGORITHM ALGO_TYPE_,
    Gemv::helper::FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE_,
    Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    Gemv::helper::FT_COMP_TYPE COMP_TYPE_,
    Gemv::helper::FT_ABE_TYPE ABE_TYPE_,
    class... Args
>
struct BlockFTGemvCENoSplitK {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockFTGemvCENoSplitK is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    Gemv::helper::FT_THRESHOLD_ALGORITHM ALGO_TYPE_,
    Gemv::helper::FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE_,
    Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    Gemv::helper::FT_COMP_TYPE COMP_TYPE_,
    Gemv::helper::FT_ABE_TYPE ABE_TYPE_,
    class... Args
>
struct BlockFTGemvCENoSplitKPreload {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockFTGemvCENoSplitKPreload is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    Gemv::helper::FT_THRESHOLD_ALGORITHM ALGO_TYPE_,
    Gemv::helper::FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE_,
    Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    Gemv::helper::FT_COMP_TYPE COMP_TYPE_,
    Gemv::helper::FT_ABE_TYPE ABE_TYPE_,
    class... Args
>
struct BlockFTVerifyNoSplitKPreload {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockFTVerifyNoSplitKPreload is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    Gemv::helper::FT_THRESHOLD_ALGORITHM ALGO_TYPE_,
    Gemv::helper::FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE_,
    Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    Gemv::helper::FT_COMP_TYPE COMP_TYPE_,
    Gemv::helper::FT_ABE_TYPE ABE_TYPE_,
    class... Args
>
struct BlockFTGemvCENoSplitKPreloadFI {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockFTGemvCENoSplitKPreloadFI is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    class... Args
>
struct BlockSumGemv {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockSumGemv is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    Gemv::helper::FT_COMP_TYPE COMP_TYPE_,
    class... Args
>
struct BlockCompare {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockCompare is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    class... Args
>
struct BlockThresholdCalc {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockThresholdCalc is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    Gemv::helper::FT_THRESHOLD_ALGORITHM ALGO_TYPE_,
    Gemv::helper::FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE_,
    Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    Gemv::helper::FT_COMP_TYPE COMP_TYPE_,
    class... Args
>
struct BlockThresholdCalcFused {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockThresholdCalc is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    class... Args
>
struct BlockSliceSum {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockSliceSum is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    class... Args
>
struct BlockFTGemv {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockFTGemv is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    class... Args
>
struct BlockFTGemvDouble {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockFTGemv is not implemented for this DispatchPolicy");
};


template <
    class DispatchPolicy,
    Gemv::helper::FT_AIC_BE_SCHEME BE_SCHEME_,
    class... Args
>
struct BlockFTGemvBe {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockFTGemvBe is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    Gemv::helper::FT_AIC_BE_SCHEME BE_SCHEME_,
    class... Args
>
struct BlockFTGemvBr {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockFTGemvBr is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    class... Args
>
struct BlockSumGemvPingPong {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockSumGemvPingPong is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    bool  VECTORIZED_TRANS_,
    class... Args
>
struct BlockFTGemvVectorized {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockFTGemvVectorized is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    class... Args
>
struct BlockMatrixAdd{
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMatrixAdd is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    class... Args
>
struct BlockMatrixTranspose{
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMatrixTranspose is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    class... Args
>
struct BlockMatrixAddVectorized{
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMatrixAddVectorized is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    Gemv::helper::FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE_,
    class... Args
>
struct BlockSliceKMNSum {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockSliceKMNSum is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    Gemv::helper::FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE_,
    class... Args
>
struct BlockSliceKMNSumVectorized {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockSliceKMNSumVectorized is not implemented for this DispatchPolicy");
};

}  // namespace Catlass::Gemv::Block

#include "catlass/gemv/block/block_gemv_aiv.hpp"
// #include "catlass/gemv/block/block_sum_gemv_aiv_pingpong.hpp"
#include "catlass/gemv/block/block_threshold_compare_fused.hpp"
#include "catlass/gemv/block/block_gemv_aic_FT_double.hpp"
#include "catlass/gemv/block/block_sum_gemv_aiv.hpp"
#include "catlass/gemv/block/block_gemv_aic.hpp"
#include "catlass/gemv/block/block_gemv_aic_FT.hpp"
#include "catlass/gemv/block/block_slice_reduce_sum.hpp"
#include "catlass/gemv/block/block_threshold_compute.hpp"
#include "catlass/gemv/block/block_gemv_aic_FT_double_vectorized.hpp"
#include "catlass/gemv/block/block_sum_aiv_no_splitk.hpp"
#include "catlass/gemv/block/block_gemv_aiv_no_splitk.hpp"
#include "catlass/gemv/block/block_sum_aiv_beft_no_splitk.hpp"
#include "catlass/gemv/block/block_std_max_reduce_aiv.hpp"
#include "catlass/gemv/block/block_gemv_aiv_no_splitk_asvar.hpp"
#include "catlass/gemv/block/block_gemv_aiv_no_splitk_asvar_relieve.hpp"
#include "catlass/gemv/block/block_matrix_add_elem.hpp"
#include "catlass/gemv/block/block_slicekmn_reduce_sum.hpp"
#include "catlass/gemv/block/block_gemv_aic_FT_BE_col_complete.hpp"
#include "catlass/gemv/block/block_gemv_aic_FT_BE_row_complete.hpp"
#include "catlass/gemv/block/block_b_max_a_sum_red_no_splitk.hpp"
#include "catlass/gemv/block/block_slicekmn_reduce_sum_fused.hpp"
#include "catlass/gemv/block/block_sum_aiv_ceft_no_splitk_thre_complete.hpp"
#include "catlass/gemv/block/block_slicekmn_reduce_sum_fused_robust.hpp"
#include "catlass/gemv/block/block_b_a_red_no_splitk_robust.hpp"
#include "catlass/gemv/block/block_gemv_aic_FT_BE_row_complete_bf.hpp"
#include "catlass/gemv/block/block_b_a_red_no_splitk_robust_bf.hpp"
#include "catlass/gemv/block/block_std_max_reduce_aiv_robust.hpp"
#include "catlass/gemv/block/block_sum_aiv_ceft_no_splitk_thre_robust.hpp"
#include "catlass/gemv/block/block_b_a_red_no_splitk_simplified_bf.hpp"
#include "catlass/gemv/block/block_b_a_red_no_splitk_simplified.hpp"
#include "catlass/gemv/block/block_slicekmn_reduce_sum_fused_simplified.hpp"
#include "catlass/gemv/block/block_std_max_reduce_aiv_simplified.hpp"
#include "catlass/gemv/block/block_sum_aiv_ceft_no_splitk_thre_simplified.hpp"
#include "catlass/gemv/block/block_sum_aiv_ceft_no_splitk_thre_tiling_simplified.hpp"
#include "catlass/gemv/block/block_sum_aiv_ceft_no_splitk_thre_tiling_robust.hpp"
#include "catlass/gemv/block/block_matrix_vec_add_elem.hpp"
#include "catlass/gemv/block/block_matrix_transpose.hpp"
#include "catlass/gemv/block/block_sum_aiv_ceft_no_splitk_thre_wait_simplified.hpp"
#include "catlass/gemv/block/block_sum_aiv_ceft_no_splitk_thre_robust_preload.hpp"
#include "catlass/gemv/block/block_sum_aiv_ceft_no_splitk_thre_simplified_preload.hpp"
#include "catlass/gemv/block/block_sum_aiv_ceft_no_splitk_thre_robust_preload_FI.hpp"
#include "catlass/gemv/block/block_b_a_red_no_splitk_robust_chunk.hpp"
#include "catlass/gemv/block/block_b_a_red_no_splitk_robust_chunk_bf.hpp"
#include "catlass/gemv/block/block_b_a_red_no_splitk_robust_chunk_f16.hpp"
#include "catlass/gemv/block/block_gemv_aic_FT_BR_row_complete.hpp"
#include "catlass/gemv/block/block_gemv_aic_FT_BR_row_complete_bf.hpp"
#include "catlass/gemv/block/block_sum_aiv_ceft_no_splitk_thre_robust_chunk_f16.hpp"
#include "catlass/gemv/block/block_sum_aiv_ceft_no_splitk_thre_robust_chunk.hpp"
#include "catlass/gemv/block/block_sum_aiv_verify_no_splitk_thre_robust_preload.hpp"
// #include "catlass/gemv/block/block_slicekmn_reduce_sum_fused_robust_vec.hpp"
// #include "catlass/gemv/block/block_slicekmn_reduce_sum_fused_simplified_vec.hpp"
#endif
