# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# examples/18_matmul_ft_abe_aic_thre_no_splitk_asvar_ft_relieve_fp16
# examples/18_matmul_ft_abe_aic_thre_no_splitk_asvar_ft_spec_fp16


set_source_files_properties(FI_test.cpp PROPERTIES LANGUAGE ASCEND)
catlass_example_add_executable(24_FI_test_bf16 dav-c220 FI_test.cpp)
