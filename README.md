# V-ABFT Artifact Reproduction Package

复现论文 V-ABFT 的实验数据。脚本按运行硬件分为两部分：

- **`gpu_h100/`** — 在 NVIDIA H100 GPU 上跑（PyTorch CUDA）
- **`npu_ascend/`** — 在 Ascend 910B NPU 上跑（PyTorch + torch_npu + CANN）

---

## 论文图表 → 脚本映射

| 论文元素 | 脚本 | 运行环境 | 服务器 |
|---------|------|---------|------|
| **Table 1 (FP64/FP32 GPU 行)** | `gpu_h100/test_final_comparison.py` | CUDA | <gpu-host> |
| **Table 1 (BF16 GPU 行)** | `gpu_h100/test_lowprec_comparison.py` | CUDA | <gpu-host> |
| **Figure 3 (tightness)** | `plots/plot_tightness.py` | CPU 任意 | 本地 |
| **Appendix Table 2 (CPU/GPU FP64/FP32)** | `gpu_h100/test_emax_cpu_gpu_v2.py` | CUDA + CPU | <gpu-host> |
| **Appendix Table 2 (GPU BF16/FP16)** | `gpu_h100/test_emax_gpu_lowprec.py` | CUDA | <gpu-host> |
| **Appendix Table 2 (NPU 行)** | `npu_ascend/test_emax_v3.py`, `test_emax_scaling_v2.py` | NPU | <npu-host> |
| **Appendix C (A-ABFT 论文 Table II 对齐)** | `gpu_h100/test_aabft_vs_vabft_corrected.py` | CUDA | <gpu-host> |
| **Figure 2 (e2e-heatmap, NPU)** | `npu_ascend/bf16_detection_test.py`, `fp32_detection_test.py`, `run_e2e_npu_original_aabft.py` | NPU | <npu-host> |
| **Figure 4 (performance, NPU)** | 由生产 CANN GEMM 算子产生（私有） | — | — |
| **e_max 拟合公式 (Eq 6-7)** | `gpu_h100/fit_emax_growth.py` | CPU | 任意 |

---

## 目录结构

```
artifact_code/
├── gpu_h100/                # → <gpu-host>: $VABFT_ROOT/
│   ├── test_final_comparison.py        # Table 1 FP64/FP32
│   ├── test_lowprec_comparison.py      # Table 1 BF16
│   ├── test_aabft_vs_vabft_corrected.py # Appendix C
│   ├── test_emax_cpu_gpu_v2.py         # Appendix Table 2 CPU/GPU
│   ├── test_emax_gpu_lowprec.py        # Appendix Table 2 BF16/FP16
│   └── fit_emax_growth.py              # e_max 公式拟合
├── npu_ascend/              # → <npu-host>: $VABFT_ROOT/
│   ├── test_emax_v3.py                 # NPU e_max
│   ├── test_emax_scaling_v2.py         # NPU e_max scaling
│   ├── vabft_acc.py                    # V-ABFT NPU 精度
│   ├── aabft_acc.py                    # A-ABFT NPU 精度
│   ├── bf16_detection_test.py          # BF16 端到端
│   ├── fp32_detection_test.py          # FP32 端到端
│   ├── run_e2e_npu_original_aabft.py   # 端到端 (原版 A-ABFT)
│   ├── run_endtoend.sh                 # CANN 8.0 启动
│   ├── run_endtoend_cann85.sh          # CANN 8.5 启动
│   ├── utils.py                        # 工具函数
│   └── encoding_utils.py               # 编码工具
├── plots/                   # 任意机器
│   ├── plot_tightness.py               # Figure 3
│   └── plot_emax_scaling.py            # e_max scaling 图
└── results/                 # 输出目录
```

---

## 环境要求

### GPU 部分 (<gpu-host>)
```
Python ≥ 3.9, PyTorch ≥ 2.0 (CUDA), NumPy, mpmath, scipy, matplotlib
```
<gpu-host> conda 环境: `$CONDA_ENV`

### NPU 部分 (<npu-host>)
```
Python ≥ 3.9, PyTorch + torch_npu, CANN ≥ 8.0
```
NPU 启动: `source npu_ascend/run_endtoend.sh` (设置 ASCEND_HOME_PATH 等环境变量)

---

## 运行流程

### Step 1 — 同步代码到对应服务器

```bash
# GPU 部分
scp -r artifact_code/gpu_h100 <gpu-host>:$VABFT_ROOT/

# NPU 部分（已在 <npu-host> 上，本地为镜像备份）
# 如有更新需 rsync 回去:
# rsync -av artifact_code/npu_ascend/ <npu-host>:$VABFT_ROOT/
```

### Step 2 — 在 <gpu-host> 上跑 GPU 实验

```bash
ssh <gpu-host>
PY=python
cd $VABFT_ROOT

# Appendix Table 2 (e_max)
$PY test_emax_cpu_gpu_v2.py --device cuda --dtype float64 --num_trials 5000
$PY test_emax_cpu_gpu_v2.py --device cuda --dtype float32 --num_trials 5000
$PY test_emax_gpu_lowprec.py --dtype bfloat16
$PY test_emax_gpu_lowprec.py --dtype float16

# Table 1 (tightness)
$PY test_final_comparison.py --dtype float64
$PY test_final_comparison.py --dtype float32
$PY test_lowprec_comparison.py --dtype bfloat16

# Appendix C 验证
$PY test_aabft_vs_vabft_corrected.py --dtype float64
```

### Step 3 — 在 <npu-host> 上跑 NPU 实验

```bash
ssh <npu-host>
cd $VABFT_ROOT
source run_endtoend.sh   # 设环境

# Appendix Table 2 NPU 行
python test_emax_v3.py
python test_emax_scaling_v2.py

# Figure 2 端到端
python bf16_detection_test.py
python fp32_detection_test.py
python run_e2e_npu_original_aabft.py
```

### Step 4 — 本地绘图

```bash
cd plots
python plot_tightness.py
python plot_emax_scaling.py
```

---

## 验证方式

每个脚本运行后预期值见原论文：
- **Table 1 V-Tight**: FP64 ~9-16×, FP32 ~8-25×, BF16 ~53×
- **Appendix Table 2** $e_{\max}/u$: GPU FP64 3.6-7.2, FP32 2.6-6.0, BF16/FP16 ~1.4 (常数)
- **A-ABFT 验证**: N=512 FP64 处与原论文 $1.68\times10^{-11}$ 误差 ≤1%

---

## 备注

- **DOI**: 提交 Stage 2 前需上传 `artifact_code/` 到 Zenodo
- **NPU 端到端 Figure 2 数据**: 需运行约 100 trials/bit × 16 bits × 4 distributions × 2 K 值 ≈ 12800 次注入实验，单次约 5-10 秒
- **GPU $e_{\max}$ 标定**: 5000 trials × 7 K 值 × 2 精度 ≈ 70 min on H100
