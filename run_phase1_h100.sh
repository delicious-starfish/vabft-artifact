#!/usr/bin/env bash
# Phase 1 — <gpu-host> 全量复现 P0 实验
# 在 <gpu-host> 上后台运行：
#   nohup bash run_phase1_h100.sh > phase1.log 2>&1 &

set -e
PY=python
DIR=$VABFT_ROOT
RES=$DIR/results
mkdir -p $RES

cd $DIR

ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(ts)] Phase 1 START"

# ============================================================
# Section A — Table 1 阈值紧度
# ============================================================
echo "[$(ts)] [A.1] FP64 tightness (500 trials, sizes 128-2048)"
$PY test_final_comparison.py --device cuda --dtype float64 --trials 500 \
    > $RES/A1_tightness_fp64.log 2>&1 || echo "  [WARN] A.1 failed"

echo "[$(ts)] [A.2] FP32 tightness (500 trials)"
$PY test_final_comparison.py --device cuda --dtype float32 --trials 500 \
    > $RES/A2_tightness_fp32.log 2>&1 || echo "  [WARN] A.2 failed"

echo "[$(ts)] [A.3] BF16 tightness (100 trials)"
$PY test_lowprec_comparison.py --dtype bfloat16 --trials 100 \
    --sizes 128,256,512,1024,2048 \
    > $RES/A3_tightness_bf16.log 2>&1 || echo "  [WARN] A.3 failed"

# ============================================================
# Section B — e_max 标定
# ============================================================
echo "[$(ts)] [B.1a] GPU FP64 e_max sweep (50k trials, K=256-16384)"
$PY test_emax_cpu_gpu_v2.py --device cuda --dtype float64 --trials 50000 \
    --K-values 256,512,1024,2048,4096,8192,16384 \
    > $RES/B1_emax_gpu_fp64.log 2>&1 || echo "  [WARN] B.1a failed"

echo "[$(ts)] [B.1b] GPU FP32 e_max sweep (50k trials)"
$PY test_emax_cpu_gpu_v2.py --device cuda --dtype float32 --trials 50000 \
    --K-values 256,512,1024,2048,4096,8192,16384 \
    > $RES/B1_emax_gpu_fp32.log 2>&1 || echo "  [WARN] B.1b failed"

echo "[$(ts)] [B.1c] CPU FP64 e_max sweep (5k trials)"
$PY test_emax_cpu_gpu_v2.py --device cpu --dtype float64 --trials 5000 \
    --K-values 256,512,1024,2048,4096,8192,16384 \
    > $RES/B1_emax_cpu_fp64.log 2>&1 || echo "  [WARN] B.1c failed"

echo "[$(ts)] [B.1d] CPU FP32 e_max sweep (5k trials)"
$PY test_emax_cpu_gpu_v2.py --device cpu --dtype float32 --trials 5000 \
    --K-values 256,512,1024,2048,4096,8192,16384 \
    > $RES/B1_emax_cpu_fp32.log 2>&1 || echo "  [WARN] B.1d failed"

echo "[$(ts)] [B.1e] GPU BF16 e_max"
$PY test_emax_gpu_lowprec.py --dtype bfloat16 \
    > $RES/B1_emax_gpu_bf16.log 2>&1 || echo "  [WARN] B.1e failed"

echo "[$(ts)] [B.1f] GPU FP16 e_max"
$PY test_emax_gpu_lowprec.py --dtype float16 \
    > $RES/B1_emax_gpu_fp16.log 2>&1 || echo "  [WARN] B.1f failed"

# ============================================================
# Section C — A-ABFT 公式验证
# ============================================================
echo "[$(ts)] [C.1] A-ABFT paper Table II reproduction (FP64, sizes 128-2048)"
$PY test_aabft_vs_vabft_corrected.py --device cuda --dtype float64 --trials 200 \
    --sizes 128,256,512,1024,2048 \
    > $RES/C1_aabft_validation.log 2>&1 || echo "  [WARN] C.1 failed"

echo "[$(ts)] Phase 1 COMPLETE — results in $RES"
ls -la $RES/
