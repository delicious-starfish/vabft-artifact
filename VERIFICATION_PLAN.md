# V-ABFT Paper Data Verification Plan

完整覆盖论文 `main_ieee.tex` 中的所有数值 claim，按可复现性分类。

**分类标记**：
- 🟢 **GPU 公开可复现** (在 <gpu-host> 上重跑即可)
- 🟡 **NPU 私有硬件** (在 <npu-host> 上重跑，论文 PDF 内含数据，AD 附录写 N/A)
- 🔵 **生产数据** (厂商生产环境 NPU 日志或 CANN 算子，无法复现)
- ⚪ **理论 claim** (数学证明，无需运行代码)
- 🟣 **小规模已验证** (本次会话已抽样验证通过)

---

## A. 阈值紧度对比 (Table 1) — 论文 Section 5.3 + Fig. 3

**配置**：方阵 N×N, N∈{128,256,512,1024,2048}, 分布 U(-1,1), GPU H100

### A.1 FP64 (500 trials/size) 🟢

| Size | A-ABFT 阈值 | V-ABFT 阈值 | A-Tight | V-Tight |
|------|------------:|------------:|--------:|--------:|
| 128² | 1.81e-12 | 2.36e-13 | 122× | 16× |
| 256² | 7.85e-12 | 5.40e-13 | 197× | 14× |
| 512² | 3.33e-11 | 1.26e-12 | 305× | 12× |
| 1024² | 1.42e-10 | 3.03e-12 | 468× | 10× |
| 2048² | 5.95e-10 | 7.46e-12 | 692× | 9× |

**复现命令** (<gpu-host>):
```bash
python test_final_comparison.py --dtype float64
```

### A.2 FP32 (500 trials/size) 🟢

| Size | A-ABFT 阈值 | V-ABFT 阈值 | A-Tight | V-Tight |
|------|------------:|------------:|--------:|--------:|
| 128² | 1.96e-3 | 1.09e-4 | 284× | 16× |
| 256² | 8.40e-3 | 2.51e-4 | 824× | 25× |
| 512² | 3.62e-2 | 5.93e-4 | 1282× | 21× |
| 1024² | 1.52e-1 | 1.43e-3 | 971× | 9× |
| 2048² | 6.40e-1 | 3.56e-3 | 1451× | 8× |

**复现命令** (<gpu-host>):
```bash
python test_final_comparison.py --dtype float32
```

### A.3 BF16 (100 trials/size) 🟣 (小规模已验证)

| Size | A-ABFT 阈值 | V-ABFT 阈值 | A-Tight | V-Tight |
|------|------------:|------------:|--------:|--------:|
| 128² | 6.42e+1 | 4.15e+0 | 825× | 53× |
| 256² | 2.71e+2 | 8.40e+0 | 1699× | 53× |
| 512² | 1.16e+3 | 1.69e+1 | 3668× | 53× |
| 1024² | 5.00e+3 | 3.40e+1 | 7835× | 53× |
| 2048² | 2.11e+4 | 6.80e+1 | 16542× | 53× |

**已验证 (10 trials)**: 128² → 859×/54×, 256² → 1700×/53× ✓

**复现命令** (<gpu-host>):
```bash
python test_lowprec_comparison.py --dtype bfloat16 --trials 100
```

### A.4 总体声明
- 整体改进倍数：**8–300×** (Abstract、Section 5.3)
- V-ABFT 范围：FP64 9–16×, FP32 8–25×, BF16 ~53×
- A-ABFT 范围：122–16,542×
- FPR：两方法均 **0%**

---

## B. e_max 标定 (Appendix B) — Eq.(6)-(7) + Tables

### B.1 CPU/GPU $e_{\max}$ 标定 — Appendix Table 2 🟢

**配置**：M=1024, N=256, K∈{256,512,1024,2048,4096,8192,16384}

| Platform | Precision | $e_{\max}/u$ 范围 | CV | R²(√K) | Scaling |
|---------|-----------|------------------|------|--------|---------|
| CPU | FP64 | 4.6–10.7 | 33.5% | 0.96 | ∝√K |
| CPU | FP32 | 6.1–11.9 | 26.3% | 0.95 | ∝√K |
| GPU H100 | FP64 | 3.6–7.2 | 23.1% | 0.92 | ∝√K |
| GPU H100 | FP32 | 2.6–6.0 | 24.2% | 0.85 | ∝√K |
| GPU H100 | BF16 | ~1.4 | — | — | constant |
| GPU H100 | FP16 | ~1.4 | — | — | constant |

**复现命令** (<gpu-host>):
```bash
# GPU/CPU FP64/FP32 (50k GPU trials, 5k CPU trials)
python test_emax_cpu_gpu_v2.py --device cuda --dtype float64 --num_trials 50000
python test_emax_cpu_gpu_v2.py --device cuda --dtype float32 --num_trials 50000
python test_emax_cpu_gpu_v2.py --device cpu  --dtype float64 --num_trials 5000
python test_emax_cpu_gpu_v2.py --device cpu  --dtype float32 --num_trials 5000
# GPU 低精度
python test_emax_gpu_lowprec.py --dtype bfloat16
python test_emax_gpu_lowprec.py --dtype float16
```

### B.2 NPU $e_{\max}$ 标定 — Table emax-scaling-npu (论文中) 🟡

| Precision | Recommended $e_{\max}$ | $e_{\max}/u$ | K Dependency |
|-----------|----------------------:|-------------:|--------------|
| BF16 | 8×10⁻³ | ~2.0 | 无 (constant) |
| FP16 | 1×10⁻³ | ~2.0 | 无 (constant) |
| FP32 | 2×10⁻⁶ √(K/1024) | ~34√(K/1024) | 是 (∝√K) |

**复现命令** (<npu-host>):
```bash
cd $VABFT_ROOT
source run_endtoend.sh
python test_emax_v3.py
python test_emax_scaling_v2.py
```

### B.3 拟合公式 — Eq.(6)-(7) 🟢

$$e_{\max}^{FP64} = 3.3\times10^{-18}\sqrt{K} + 3.7\times10^{-16}$$
$$e_{\max}^{FP32} = 1.5\times10^{-9}\sqrt{K} + 1.7\times10^{-7}$$

**复现命令**:
```bash
python fit_emax_growth.py  # 输入 B.1 数据，输出 a, b, R²
```

---

## C. A-ABFT 公式验证 (Appendix C) 🟢

### C.1 与 A-ABFT 论文 Table II 对齐
- 配置: N=512, FP64, y=21 (论文经验值)
- 论文 Table II 报告值: **1.68×10⁻¹¹**
- 复现值预期: **1.66×10⁻¹¹** (0.99× 匹配)

### C.2 V-ABFT vs A-ABFT 紧度对比 (Table comparison-appendix)
| Aspect | A-ABFT | V-ABFT |
|--------|-------:|-------:|
| FP64 紧度 | 122–692× | 9–16× |
| FP32 紧度 | 284–1451× | 8–25× |
| 复杂度 | O(pn) | O(n) |

**复现命令** (<gpu-host>):
```bash
python test_aabft_vs_vabft_corrected.py --dtype float64
```

---

## D. NPU 端到端检测 (Figure 2) 🟡

**配置**：BF16, Ascend 910B, (M,K,N)=(256,K,256), K∈{4096,8192}, 100 trials/bit, 4 distributions

### D.1 K=4096 关键数据点
- bit 2 U(-1,1): V-ABFT **71%** vs A-ABFT **18%** (端到端成功率)
- bit 1 U(-1,1): V-ABFT **38%** vs A-ABFT **1%**

### D.2 K=8192 关键数据点
- bit 0 (零均值分布): V-ABFT ≥**57%** vs A-ABFT **0%**

### D.3 完整数据
- 检测率: 4 distributions × 16 bits × 100 trials = 6400 注入实验
- 端到端成功率: 同上规模

**复现命令** (<npu-host>):
```bash
cd $VABFT_ROOT
source run_endtoend.sh
python bf16_detection_test.py             # V-ABFT
python run_e2e_npu_original_aabft.py      # A-ABFT (原版公式)
python fp32_detection_test.py             # FP32 端到端
```

---

## E. 真实模型 FPR 验证 (Section 5.2) 🟡

**论文 claim**: V-ABFT 在所有模型上 **0% FPR**

| Model | 测试矩阵数 | V-ABFT FPR | A-ABFT FPR |
|-------|-----------:|------:|------:|
| LLaMA-7B | 111 (A-ABFT 误检矩阵) | 0% | (基线) |
| GPT-2 | 5,379 | 0% | — |
| ViT-B/32 | 5,937 (50 epochs 1% 抽样) | 0% | — |

**数据位置** (<npu-host>):
- `$LLM_DATA_ROOT/{atte_c,atte_r,mlp_r}/` — LLM 矩阵 dump
- `$LLM_DATA_ROOT_FAULTY/`

**复现命令** (<npu-host>):
```bash
cd $VABFT_ROOT
python test_fpr.py  # 模拟数据 FPR 测试 (3 distributions)
# 真实模型 FPR 测试需运行单独脚本（待补 - 见下方 Gap 列表）
```

---

## F. 编码方法消融 (Table 3) 🟡

**配置**: (M,K,N)=(256,1024,256), Ascend 910B, N(0,1) 分布, FP32 累加, 3000 trials, 5 columns

| \|Δ\| | Linear 最坏 | Linear 范围 | Sinh 最坏 | Sinh 范围 |
|-------|-----------:|----------:|---------:|--------:|
| 0.05 | 58.1% | 28.8% | 78.7% | 2.2% |
| 0.07 | 74.3% | 21.0% | 90.7% | 1.7% |
| 0.10 | 88.2% | 11.0% | 98.2% | 0.3% |

**复现命令** (<npu-host>):
```bash
# 需要确认对应脚本（<npu-host> 上未直接找到 sinh ablation 脚本）
# 推测：encoding_utils.py + 自编 ablation runner
```

⚠ **Gap**: <npu-host> 未直接发现编码消融脚本，可能需补写或定位到旧版本。

---

## G. 性能开销 (Figure 4) 🔵

**配置**: Ascend 910B, M=N=K∈{1024, 2048, 4096, 8192}, vendor aclnn baseline

| 精度 | 大矩阵 TFLOPS | 平均开销 |
|-----|--------------:|---------:|
| FP32 | 72–76 | **12.8%** |
| BF16 | 208–220 | **27.2%** |
| DMR (对照) | — | >200% |

**数据来源**: `performance.md` (生产 CANN GEMM 算子，私有)

**详细数据 (Chunk=512)**:
- FP32: 1024³ → 13.49 TF, 2048³ → 47.01 TF, 4096³ → 74.49 TF, 8192³ → 77.83 TF
- BF16: 1024³ → 15.03 TF, 2048³ → 83.84 TF, 4096³ → 205.76 TF

⚠ **不可独立复现**: 需要 Huawei 生产 CANN GEMM 实现，AD 附录写 N/A。

---

## H. NPU 生产错误类型分析 (Section 2) 🔵

**Section 2.1 paragraph "Real-world evidence"**:
- 矩阵规模: 4096×2048 — 6144×12288
- 收集错误输出: **10,000+** 元素
- 错误类型分布:
  - 单比特翻转: **46.8%**
  - 错误进位: **25.1%**
  - 缺失进位: **24.1%**
  - 复合错误: **4.0%**

**数据来源**: 厂商提供的生产环境 NPU 故障日志归档（非公开数据，不可再分发）

**位置**: 内部归档，未随 artifact 发布

⚠ **不可独立复现**: 需访问厂商故障 NPU 卡与日志归档，数据不可再分发。AD 附录写 N/A。

---

## I. 理论 claims (Appendix A) ⚪

无需运行代码，但需验证数学正确性：

- **Theorem 1**: λ*N ≈ **2.821** (sinh(x)=3x 唯一正根)
- **极值方差界**: σ² ≤ (m-μ)(μ-l) (凸性证明)
- **最优编码 difficulty**: 0.354 N^1.5 (sinh) vs 0.577 N^1.5 (linear)
- **改进倍数**: 0.577/0.354 = **1.63×**
- **置信系数**: c_σ=2.5 → ≥84% (Chebyshev) / ~99% (Gaussian)
- **复杂度**: O(n) (V-ABFT) vs O(pn) (A-ABFT)

**验证方式**: 数学推导审阅 + 数值代入 (sinh 根可用 `scipy.optimize.brentq` 验证)

---

## J. 背景统计 (Section 1) 🔵

- 训练规模: 10²⁵+ FLOPs (引用 GPT-4, Llama-3 405B)
- SDC 速率: 10⁻⁷ — 10⁻³ per chip-hour (引用 sdc-at-scale, sdc-taxonomy, google-mercurial)
- A-ABFT 现有阈值范围: 120–16,500× (引用 + 本工作分析)

**性质**: 引用其他论文，无需复现。

---

## 复现优先级总览

| 等级 | 类别 | 命令 | 服务器 | 预估耗时 |
|------|-----|------|--------|---------|
| **P0** | A.1, A.2, A.3 (Table 1) | `test_final_comparison.py + test_lowprec_comparison.py` | <gpu-host> | ~30 min |
| **P0** | B.1, B.3 (Eq 6-7 + Appendix B) | `test_emax_cpu_gpu_v2.py + fit_emax_growth.py` | <gpu-host> | ~90 min |
| **P0** | C.1, C.2 (Appendix C) | `test_aabft_vs_vabft_corrected.py` | <gpu-host> | ~5 min |
| **P1** | B.2 (NPU emax) | `test_emax_v3.py` | <npu-host> | ~30 min |
| **P1** | D (Figure 2) | `bf16_detection_test.py + run_e2e_npu_original_aabft.py` | <npu-host> | ~3 hr |
| **P1** | E (FPR test) | `test_fpr.py` + LLM 数据脚本 (待补) | <npu-host> | ~30 min |
| **P2** | F (Table 3 编码消融) | 需定位/补写 | <npu-host> | TBD |
| **N/A** | G, H (生产数据) | 不可独立复现 | — | — |
| **N/A** | I (理论) | 数学审阅 | — | — |

---

## 已发现的 Gap (需要补充)

1. **F (编码消融)**: <npu-host> 上无直接的 sinh-vs-linear ablation 脚本，可能需要定位旧脚本或补写
2. **E.1-E.3 (真实模型 FPR)**: `test_fpr.py` 只做了模拟数据 (3 distributions × N(1e-6,1)/N(1,1)/U(-1,1))，未直接读取 LLaMA/GPT-2/ViT 矩阵 dump。需要补写一个加载 `$LLM_DATA_ROOT/` 矩阵的 FPR 测试脚本
3. **H (生产错误分析)**: 需要确认 厂商故障日志归档 是否包含错误类型分类的原始数据

---

## 建议执行顺序

**阶段 1 — <gpu-host> GPU 全量复现 (P0, ~2hr)**
1. e_max 标定 → 拟合公式
2. Table 1 全精度紧度 (FP64/FP32/BF16)
3. A-ABFT 公式验证

**阶段 2 — <npu-host> NPU 端到端 (P1, ~3-4hr)**
1. NPU e_max 标定
2. Figure 2 端到端 (BF16/FP32)
3. test_fpr.py 模拟 FPR

**阶段 3 — Gap 填补**
1. 定位/补写编码消融脚本
2. 补写真实模型 FPR 加载脚本
3. 确认生产错误分析数据

**阶段 4 — 文档化**
所有结果汇总到 `results/` 目录，对照本文档逐项打 ✓
