# V-ABFT 实时指数位错误注入测试规格

## 1. 目标与范围

本测试用于评估 Ascend NPU 上 V-ABFT 对 GEMM 中实时单 bit 指数位错误的：

1. block 级检出率；
2. row 级检出率；
3. row 级和 block 级误检率；
4. 行列定位成功率；
5. 注错后 NaN/Inf 的出现率。

本规格是新的统一实验，不再分别复制 `past_experiment` 中的 offset、specialLH 和 onlyHI 注错路径。所有指数位一律使用 XOR，同时包含 `0→1` 和 `1→0`，不按翻转方向分组统计。

## 2. 运行环境

- 后端：`torch_npu`，实际 Ascend NPU。
- 设备：默认 `npu:0`，应允许通过命令行参数修改。
- 每次运行必须记录 PyTorch、`torch_npu`、CANN、设备型号、代码 commit 和随机 seed。
- 不允许用 CPU 回退结果代替 NPU 正式结果；CPU 只可用于位操作和参考值计算。

## 3. GEMM 与测试矩阵

计算：

```text
C = A[M,K] @ B[K,N]
M = 1024
N = 1024
K ∈ {1024, 2048, 3072, 4096, 6144, 8192}
dtype ∈ {torch.bfloat16, torch.float32}
```

指数位按 IEEE 位模式的最低位为 bit 0 编号：

| dtype | 尾数位 | 待测指数位 | 符号位 |
|---|---:|---:|---:|
| BF16 | 0–6 | 7–14 | 15 |
| FP32 | 0–22 | 23–30 | 31 |

每个 `(K, dtype, distribution, bit)` 为一个独立测试条件，共：

```text
6 K × 2 dtype × 4 distribution × 8 bit = 384 个条件
```

## 4. 输入数据分布

A 和 B 独立生成，且在同一测试条件中使用同一分布：

| 名称 | 生成规则 |
|---|---|
| `N(1,1)` | `randn * 1 + 1` |
| `N(1e-6,1)` | `randn * 1 + 1e-6` |
| `TruncN` | 先生成 `N(1,1)`，再 clamp 到 `[0,2]` |
| `U(-1,1)` | `rand * 2 - 1` |

生成后直接转为当前测试 dtype，不得先用 FP32 生成再在主机端量化，除非将该变体单独标记。

> 说明：上述 `TruncN` 定义按历史实验参数 `mean=1, std=1` 与当前 `npu_ascend/utils.py::generate_matrice_clamp` 确定。项目中部分新脚本使用 `mean=0` 的 `[-1,1]` 变体，不属于本规格的 `TruncN`。

## 5. block 和重复次数

- 逻辑 GEMM block 尺寸沿用项目实现：`block_m=128`、`block_n=256`、`KInjectUnit=1024`。
- 边界不足一块时补 0，统计时只考虑原始 `M×N` 范围。本规格的 M/N 均已对齐，因此不会产生 M/N padding。
- 参数 `C` 表示对同一组 A/B 重复执行的注错 trial 数，默认 `C=10000`，允许用户设置。
- 每次 trial 恢复未注错的 C/中间状态，禁止错误在 trial 间累积。
- 沿用项目的 block、block 内 row/column 和 K 分块调度。每个参与 block 是否注错由 Bernoulli 概率 `p_inject` 决定，默认 `p_inject=0.5`，必须写入结果元数据。
- 一个 block 最多注入一个错误；不注错的 block 用于误检统计，不另跑 clean-only 实验。
- 为避免不同 bit 共享完全相同的输入造成相关性，每个测试条件使用独立但可重现的 seed。

## 6. 错误注入

### 6.1 操作

对被选中的实时计算值 `x` 的指定指数位 `b` 执行：

```text
raw_after = raw_before XOR (1 << b)
```

位操作必须使用等宽整数视图：BF16 使用 16 bit，FP32 使用 32 bit。不得使用加偏置模拟 bit flip。

### 6.2 有效性

- XOR 后无论是 `0→1` 还是 `1→0` 都是有效注错，不过滤、不重试。
- 每次注错记录：trial ID、block ID、K-slice ID、row、column、bit、翻转方向、原始位模式、翻转后位模式和翻转后数值类别（finite/Inf/NaN）。
- 翻转方向只用于可审计记录，不作为主结果的分组维度。

## 7. V-ABFT 检测

对每个输出行 `i` 计算：

```text
D1[i] = ABe[i] - Ce[i]
ABe   = A @ sum(B, dim=1)
Ce    = sum(C, dim=1)
```

使用当前 `npu_ascend/utils.py::my_bound_improve_robust` 定义的逐行 V-ABFT 方差阈值 `T[i]`。为避免代码改动导致实验口径静默变化，正式运行必须记录：

```text
BF16: e_max = 8e-3
FP32: e_max = 1.377e-6 * sqrt(K / 1024)
c_sigma = 5.39
```

判定规则：

```text
detected_row[i] = (abs(D1[i]) >= T[i]) OR isinf(D1[i])
```

`D1[i]` 为 NaN 时不强制判为检出，按第 10 节独立处理。

## 8. 行列定位

项目已在 `npu_ascend/encoding_utils.py` 中提供列定位，因此本测试必须统计完整行列定位，不只统计 row detection。

默认使用零中心 linear encoding：

```text
w[j]  = j - (N - 1) / 2
D2[i] = A @ (B @ w)[i] - (C @ w)[i]
j_est = round(D2[i] / D1[i] + (N - 1) / 2)
```

`j_est` 限制在 `[0, N-1]`。若后续使用 sinh encoding，必须作为单独实验变体，不得与 linear 结果混合。

对每个已注错且非 NaN 的样本：

```text
row_success      = 注错 row 被 detected_row 标记
column_success   = row_success AND (j_est == injected_column)
localize_success = row_success AND column_success
```

因为本规格限制每 block 最多一个错误，`D2/D1` 的单错误定位假设成立。

## 9. 计数与指标

所有百分比先累加整数分子/分母，最后一次性计算；不得平均每个 trial 的百分比。

### 9.1 注错样本

```text
injected       = 发生 XOR 的非 NaN 错误数
row_detected   = 其注错 row 被报告的错误数
localized      = row 和 column 均精确命中的错误数
row_DR         = row_detected / injected
localization_R = localized / injected
conditional_localization_R = localized / row_detected
```

### 9.2 block 级

```text
injected_blocks = 至少有一次注错的 block 数
detected_blocks = 被报错的 injected block 数
block_DR        = detected_blocks / injected_blocks
```

### 9.3 误检

不注错 block 直接构成 clean 样本：

```text
clean_rows = 未注错的 row 数
false_rows = 被报错的 clean row 数
row_FPR    = false_rows / clean_rows

clean_blocks = 未注错的 block 数
false_blocks = 被报错的 clean block 数
block_FPR    = false_blocks / clean_blocks
```

若分母为 0，输出 `null/NA`，不得输出 `-1%`或伪造为 0%。

## 10. NaN 和 Inf

- `nan_injected`：XOR 后错误值为 NaN 的注错数。
- `inf_injected`：XOR 后错误值为 `+Inf/-Inf` 的注错数。
- `nan_rate = nan_injected / all_injected`，`inf_rate = inf_injected / all_injected`。
- 主检出率和主定位率的分母排除 NaN 注错；NaN 无论是否报错都不计入分子或分母。
- Inf 不从主检出率中排除；但应同时输出 finite-only 检出率，便于区分有限数错误和特殊值错误。
- 列定位需要有限的 `D1`/`D2`；无法反演列号的 Inf 样本计为定位失败。

## 11. 随机性

- 接受用户 seed，默认 `seed=0`。
- 同时设置 Python `random`、NumPy、PyTorch CPU 和 `torch_npu` seed。
- 每个条件的 seed 由一个稳定的条件 ID 派生，不得使用 Python 进程内随机化的 `hash()`。
- 随机流至少要覆盖：A/B 数据、block 是否注错、block 内位置和调度 seed。

## 12. 输出

每个测试条件至少输出一行 CSV/JSONL，字段包括：

```text
run_id, timestamp, commit, device, torch_version, torch_npu_version, cann_version,
seed, M, N, K, dtype, distribution, bit, C, p_inject,
all_injected, nan_injected, inf_injected, finite_injected,
injected_rows, detected_rows, false_rows, clean_rows,
injected_blocks, detected_blocks, false_blocks, clean_blocks,
localized, row_DR, block_DR, row_FPR, block_FPR,
localization_R, conditional_localization_R, nan_rate, inf_rate,
e_max, c_sigma, encoding
```

另保存逐注错审计日志，以支持对 NaN、Inf、列定位失败和 `0→1`/`1→0` 的事后排查。

## 13. 验收要求

1. 384 个测试条件全部存在，无因 bit 原值为 1 而跳过的条件。
2. 任取样本的 `raw_before XOR raw_after` 必须等于 `1 << bit`。
3. 每 block 最多一个注错。
4. 主检出/定位率不包含 NaN 样本，NaN/Inf 计数可由审计日志重算。
5. 同一环境、同一 seed 的连续两次运行应产生相同的注错坐标和 bit 方向；若 NPU 并行规约不具备 bitwise determinism，必须在元数据中声明。
6. 每个百分比均能由同行整数计数重算，且分母为 0 时为 `NA`。

## 14. 与历史结果的关系

`past_experiment` 仅用于恢复 K、dtype、分布、bit 和统计维度。新结果不应被要求与三个历史文件数值相等，原因是：

- 新规格对所有指数位统一使用 XOR；
- 不再为最高/最低位使用特殊 offset 路径；
- `0→1` 和 `1→0` 均计入同一主结果；
- M/N 固定为 1024；
- 新规格增加了列定位和更完整的 NaN/Inf 审计。
