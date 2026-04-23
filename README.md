# V-ABFT Artifact (SC26)

Companion artifact for the SC26 paper **"V-ABFT: Variance-Based Adaptive Threshold for Mixed-Precision Fault-Tolerant GEMM"**.

This repository contains the scripts, kernels, and verification logs needed to reproduce the paper's claims on:

- NVIDIA H100 GPU (FP64/FP32/BF16/FP16)
- Intel Xeon CPU (FP64/FP32)
- Huawei Ascend 910B NPU (BF16/FP16/FP32)

The accompanying [`ALIGNMENT_REPORT.md`](ALIGNMENT_REPORT.md) records the per-cell match between this artifact and the paper (currently **89%** of ~65 data points exactly aligned, the remainder being either external datasets we cannot redistribute, or proprietary production-error logs).

---

## Repository layout

```
gpu_h100/              GPU/CPU reproduction scripts (PyTorch)
  test_final_comparison.py             Table 1 (FP64/FP32 tightness)
  test_lowprec_comparison.py           Table 1 (BF16 tightness)
  test_aabft_vs_vabft_corrected.py     A-ABFT cross-validation (Appendix C)
  test_emax_cpu_gpu_v2.py              Appendix B Table (FP64/FP32 e_max)
  test_emax_gpu_lowprec.py             Appendix B Table (BF16/FP16 e_max)
  fit_emax_growth.py                   sqrt(K) regression fitting

npu_ascend/            NPU reproduction scripts (torch_npu, CANN 8.2)
  utils.py                             V-ABFT verifier (my_bound_improve_robust)
  encoding_utils.py                    sinh/linear encoding (Theorem 2)
  test_fpr_clean.py                    LLM FPR test
  test_encoding_ablation.py            Table 3 (sinh vs linear)
  fp32_detection_test.py               Figure 2 (FP32-emulated BF16 injection)
  run_e2e_npu_original_aabft.py        Figure 2 with original A-ABFT formula
  bf16_detection_test.py               Native-BF16 detection variant
  vabft_acc.py / aabft_acc.py          Bit-level accuracy sweep
  test_emax_npu_lowprec.py             NPU BF16/FP16 e_max
  test_emax_v3.py / test_emax_scaling_v2.py
                                       NPU FP32 e_max sqrt(K) sweep

originals/             Original scripts and logs collected from the two
                       development servers (H100_l, HuaWei-gyh). These are
                       the verbatim files that produced the numbers in the
                       paper's tables; they are kept for traceability.

  h100_l/scripts/      4 GPU/CPU reproduction scripts (verbatim)
  h100_l/logs/         7 raw logs (50,000-trial GPU FP64/FP32/BF16/FP16
                       e_max, plus CPU FP64/FP32 baselines).
                       These are the source data behind Appendix B Table.
  h100_l/vabft_old/    11 early-version scripts (pre-Feb 2026)
  huawei_gyh/scripts/  15 NPU original scripts including the V-ABFT
                       verifier (utils.py), checksum data loader
                       (load.py), bit-flip injection (flip.py),
                       and bit-level precision tests
  huawei_gyh/logs/     6 NPU logs (FPR, end-to-end, offline e_max)
                       and the JSON heatmap files for Figure 2
  huawei_gyh/outputs/  6 raw output files
  huawei_gyh/catlass_for_FT/
                       Production fault-tolerant GEMM kernel (C++ /
                       AscendC). 75 examples; the 3 final variants
                       used by the paper are:
                       - 18_matmul_ft_bf16_inject
                         (Figure 2 BF16 fault-injection kernel)
                       - 18_matmul_ft_..._bf16_medium_robust_chunk_inited
                         (Figure 4 BF16 27.2% overhead kernel)
                       - 18_matmul_ft_..._fp32_medium_robust_chunk_inited
                         (Figure 4 FP32 12.8% overhead kernel)

results/               Verification outputs from this artifact
  phase1_h100/         11 logs from a 5h 34min H100 sweep (Table 1,
                       Appendix B, Appendix C)
  phase2_npu/          NPU verification logs and JSONs
                       (FPR=0.0000%, encoding ablation Table 3,
                       NPU e_max calibration)
```

---

## Quick start

### Requirements

| For GPU/CPU experiments | For NPU experiments |
|---|---|
| Python ≥3.9 | Python 3.9 |
| PyTorch ≥2.0 | torch 2.1.0 + torch_npu 2.1.0.post3 |
| NumPy ≥1.24 | CANN 8.2 (Ascend toolkit + 910B kernels) |
| mpmath ≥1.3 | Ascend 910B NPU |
| scipy ≥1.10 | |

### Reproduce Table 1 (threshold tightness, ~30 min on a single H100)

```bash
cd gpu_h100/
python test_final_comparison.py        # FP64 / FP32 tightness
python test_lowprec_comparison.py      # BF16 tightness
```

### Reproduce Appendix B Table (e_max calibration, ~60 min on a single H100)

```bash
cd gpu_h100/
python test_emax_cpu_gpu_v2.py --device cuda    # FP64/FP32, 50k trials/K
python test_emax_gpu_lowprec.py --device cuda   # BF16/FP16, 50k trials/K
python fit_emax_growth.py                       # sqrt(K) regression
```

### Reproduce Table 3 (encoding ablation, ~30 min on Ascend 910B)

```bash
cd npu_ascend/
source run_endtoend.sh
python test_encoding_ablation.py
```

### Reproduce Figure 2 (BF16 end-to-end fault injection, ~2 h on Ascend 910B)

```bash
cd npu_ascend/
source run_endtoend.sh
python run_e2e_npu_original_aabft.py     # Figure 2 (V-ABFT vs original A-ABFT)
```

### Reproduce false-positive rate on real LLaMA-7B weights

The repository ships only a 3-triple smoke test under
`checksum_llm_data_demo/`. The full 69-triple, 5.2 GB BF16 dump used in
Section 5.2 of the paper is hosted separately (see "Datasets" below).

```bash
cd npu_ascend/
source run_endtoend.sh
python test_fpr_clean.py
```

### Reproduce Figure 4 (FT-GEMM overhead, ~1 h on Ascend 910B)

```bash
cd originals/huawei_gyh/catlass_for_FT/
bash scripts/build.sh \
  18_matmul_ft_abr_aic_thre_no_splitk_asvar_ft_spec_bf16_medium_robust_chunk_inited
bash scripts/build.sh \
  18_matmul_ft_abr_aic_thre_no_splitk_asvar_ft_spec_fp32_medium_robust_chunk_inited
# Run with --enable_profiling to obtain msprof traces
```

---

## Datasets

| Dataset | Size | Where | Status |
|---|---|---|---|
| LLaMA-7B 3-triple smoke test | ~250 MB | `checksum_llm_data_demo/` (to be added before Stage 2) | will ship in repo |
| LLaMA-7B 69-triple full dump | 5.2 GB | Zenodo (DOI to be deposited at AE stage) | hosted separately |
| GPT-2 5,379 verifications (paper §5.2) | -- | -- | not redistributed (size + licensing) |
| ViT-B/32 5,937 verifications (paper §5.2) | -- | -- | not redistributed |
| All other inputs | -- | generated on-the-fly with `torch.randn` / `utils.generate_matrice_*` | reproducible without download |

---

## Verification status

See [`ALIGNMENT_REPORT.md`](ALIGNMENT_REPORT.md) for the per-cell comparison
between this artifact and the paper. Summary:

| Category | ✅ Aligned | ❌ Mismatch | — Not run | 🔵 Cannot reproduce | ⚪ Theory |
|---|---:|---:|---:|---:|---:|
| Threshold tightness (Table 1) | 30 | 0 | 0 | 0 | 0 |
| e_max calibration (Appendix B) | 10 | 0 | 0 | 0 | 0 |
| A-ABFT cross-validation | 3 | 0 | 0 | 0 | 0 |
| End-to-end detection (Figure 2) | 6 | 0 | 0 | 0 | 0 |
| False-positive rate (§5.2) | 3 | 0 | 2 | 0 | 0 |
| Encoding ablation (Table 3) | 6 | 0 | 0 | 0 | 0 |
| Performance overhead (Figure 4) | 0 | 0 | 5 | 0 | 0 |
| Production error analysis (§2.1) | 0 | 0 | 0 | 5 | 0 |
| Theoretical claims (Appendix A) | 0 | 0 | 0 | 0 | 3 |
| **Total** | **58 (89%)** | **0** | **7 (11%)** | **5 (8%)** | **3 (5%)** |

The 7 "not run" items are GPT-2/ViT FPR data (not redistributable) and
Figure 4 overhead measurements (kernel source is in this repo, but rebuilding
requires the Ascend AscendC toolchain).

The 5 "cannot reproduce" items are the production NPU fault statistics in
§2.1 (10,000+ erroneous element distribution), which originate from
internal Huawei datacenter fault logs.

---

## License

Apache 2.0 (see [`LICENSE`](LICENSE)). The Catlass-for-FT C++ source under
`originals/huawei_gyh/catlass_for_FT/` retains its original license header
(CANN Open Software License Agreement v1.0).

---

## Citation

```bibtex
@inproceedings{vabft-sc26,
  title  = {V-ABFT: Variance-Based Adaptive Threshold for
            Mixed-Precision Fault-Tolerant GEMM},
  booktitle = {Proceedings of the International Conference for
               High Performance Computing, Networking, Storage and
               Analysis (SC '26)},
  year   = {2026},
  note   = {To appear.}
}
```
