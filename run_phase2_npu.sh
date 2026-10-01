#!/bin/bash
# Phase 2 — <npu-host> NPU 全量复现 P1 实验
# nohup bash run_phase2_npu.sh > phase2.log 2>&1 &

set +e  # 不要因单个失败中断
export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export ASCEND_OPP_PATH=$ASCEND_HOME_PATH/opp
export ASCEND_AICPU_PATH=$ASCEND_HOME_PATH
export TOOLCHAIN_HOME=$ASCEND_HOME_PATH/toolkit
export ASCEND_TOOLKIT_HOME=$ASCEND_HOME_PATH
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/lib64/plugin/opskernel:$ASCEND_HOME_PATH/lib64/plugin/nnengine:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export PATH=$ASCEND_HOME_PATH/bin:$PATH
export PYTHONPATH=$ASCEND_HOME_PATH/python/site-packages:$PYTHONPATH

PY=python3
DIR=$VABFT_ROOT
RES=$DIR/results_phase2
mkdir -p $RES
cd $DIR

ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(ts)] Phase 2 START on NPU"

# ============================================================
# Section B.2 — NPU e_max 标定
# ============================================================
echo "[$(ts)] [B.2a] NPU e_max baseline (test_emax_v3.py)"
$PY test_emax_v3.py > $RES/B2a_emax_npu_v3.log 2>&1

echo "[$(ts)] [B.2b] NPU e_max scaling (test_emax_scaling_v2.py)"
$PY test_emax_scaling_v2.py > $RES/B2b_emax_npu_scaling.log 2>&1

# ============================================================
# Section D — Figure 2 端到端注入
# ============================================================
echo "[$(ts)] [D.1] BF16 V-ABFT detection (bf16_detection_test.py)"
$PY bf16_detection_test.py > $RES/D1_bf16_vabft.log 2>&1

echo "[$(ts)] [D.2] FP32 V-ABFT detection (fp32_detection_test.py)"
$PY fp32_detection_test.py > $RES/D2_fp32_vabft.log 2>&1

echo "[$(ts)] [D.3] BF16/FP32 A-ABFT (原版公式) detection"
$PY run_e2e_npu_original_aabft.py > $RES/D3_aabft_original.log 2>&1

# ============================================================
# Section E — FPR 测试
# ============================================================
echo "[$(ts)] [E.1] V-ABFT FPR (test_fpr.py)"
$PY test_fpr.py > $RES/E1_fpr_test.log 2>&1

# ============================================================
# Section accuracy — V-ABFT/A-ABFT NPU 精度对比
# ============================================================
echo "[$(ts)] [Acc.1] V-ABFT NPU 精度 (vabft_acc.py)"
$PY vabft_acc.py > $RES/Acc1_vabft_npu.log 2>&1

echo "[$(ts)] [Acc.2] A-ABFT NPU 精度 (aabft_acc.py)"
$PY aabft_acc.py > $RES/Acc2_aabft_npu.log 2>&1

echo "[$(ts)] Phase 2 COMPLETE — results in $RES"
ls -la $RES/
