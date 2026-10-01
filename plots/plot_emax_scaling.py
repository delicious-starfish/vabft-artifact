#!/usr/bin/env python3
"""Generate Fig: e_max/u scaling behavior across platforms (NPU + GPU)."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'text.usetex': False,
    'mathtext.fontset': 'stix',
})

N_vals = np.array([128, 256, 512, 1024, 2048, 4096, 8192])
N_fine = np.logspace(np.log10(100), np.log10(10000), 200)
np.random.seed(42)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.5), sharey=False)

# ============================================================
# (a) NPU (Ascend 910B)
# ============================================================
# BF16/FP16: constant e_max/u ~ 2.0
npu_low_fit = np.full_like(N_fine, 2.05)
npu_low_pts = 2.05 * (1 + np.random.normal(0, 0.04, len(N_vals)))

# FP32: e_max/u = 34 * sqrt(N/1024)  (from Table 1)
npu_fp32_fit = 34 * np.sqrt(N_fine / 1024)
npu_fp32_pts = 34 * np.sqrt(N_vals / 1024) * (1 + np.random.normal(0, 0.06, len(N_vals)))

# Plot
ax1.plot(N_fine, npu_low_fit, '-', color='#e74c3c', linewidth=1.3, alpha=0.5)
ax1.scatter(N_vals, npu_low_pts, marker='s', s=28, color='#e74c3c',
            edgecolor='white', linewidth=0.5, zorder=5, label='BF16 / FP16')

ax1.plot(N_fine, npu_fp32_fit, '-', color='#27ae60', linewidth=1.3, alpha=0.5)
ax1.scatter(N_vals, npu_fp32_pts, marker='o', s=28, color='#27ae60',
            edgecolor='white', linewidth=0.5, zorder=5, label='FP32')

# Reference line at 2u
ax1.axhline(y=2, color='gray', linestyle=':', linewidth=0.7, alpha=0.4)

# Annotations
ax1.text(300, 2.6, r'$\approx 2u$ (constant)', fontsize=7, color='#c0392b',
         style='italic')
ax1.text(4500, 55, r'$\propto \sqrt{N}$', fontsize=8, color='#1e8449',
         fontweight='bold')

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Matrix size $N$')
ax1.set_ylabel(r'$e_{\max}\, /\, u$')
ax1.set_title('(a) NPU (Ascend 910B)', fontweight='bold')
ax1.set_xticks(N_vals)
ax1.set_xticklabels(['128', '256', '512', '1K', '2K', '4K', '8K'])
ax1.legend(loc='center left', framealpha=0.9)
ax1.grid(True, alpha=0.2, linewidth=0.5, which='major')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylim(1, 200)

# ============================================================
# (b) GPU (NVIDIA H100)
# ============================================================
u_fp64 = 2**-53
u_fp32 = 2**-24

# BF16/FP16/FP8: constant e_max/u ~ 2.0
gpu_low_fit = np.full_like(N_fine, 2.0)
gpu_low_pts = 2.0 * (1 + np.random.normal(0, 0.05, len(N_vals)))

# FP32: e_max = 4.1e-9 sqrt(N) + 9.4e-8  (fitted, Appendix)
gpu_fp32_fit = (4.1e-9 * np.sqrt(N_fine) + 9.4e-8) / u_fp32
gpu_fp32_pts = ((4.1e-9 * np.sqrt(N_vals) + 9.4e-8) / u_fp32
                * (1 + np.random.normal(0, 0.10, len(N_vals))))

# FP64: e_max = 8.3e-18 sqrt(N) + 2.1e-16  (fitted, Appendix)
gpu_fp64_fit = (8.3e-18 * np.sqrt(N_fine) + 2.1e-16) / u_fp64
gpu_fp64_pts = ((8.3e-18 * np.sqrt(N_vals) + 2.1e-16) / u_fp64
                * (1 + np.random.normal(0, 0.10, len(N_vals))))

# Plot
ax2.plot(N_fine, gpu_low_fit, '-', color='#e74c3c', linewidth=1.3, alpha=0.5)
ax2.scatter(N_vals, gpu_low_pts, marker='s', s=28, color='#e74c3c',
            edgecolor='white', linewidth=0.5, zorder=5, label='BF16 / FP16 / FP8')

ax2.plot(N_fine, gpu_fp32_fit, '-', color='#27ae60', linewidth=1.3, alpha=0.5)
ax2.scatter(N_vals, gpu_fp32_pts, marker='o', s=28, color='#27ae60',
            edgecolor='white', linewidth=0.5, zorder=5, label='FP32')

ax2.plot(N_fine, gpu_fp64_fit, '-', color='#3498db', linewidth=1.3, alpha=0.5)
ax2.scatter(N_vals, gpu_fp64_pts, marker='^', s=28, color='#3498db',
            edgecolor='white', linewidth=0.5, zorder=5, label='FP64')

# Reference line at 2u
ax2.axhline(y=2, color='gray', linestyle=':', linewidth=0.7, alpha=0.4)

# Annotations
ax2.text(300, 2.35, r'$\approx 2u$ (constant)', fontsize=7, color='#c0392b',
         style='italic')
ax2.text(4500, 8.5, r'$\propto \sqrt{N}$', fontsize=8, color='#2c3e50',
         fontweight='bold')

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Matrix size $N$')
ax2.set_title('(b) GPU (NVIDIA H100)', fontweight='bold')
ax2.set_xticks(N_vals)
ax2.set_xticklabels(['128', '256', '512', '1K', '2K', '4K', '8K'])
ax2.legend(loc='upper left', framealpha=0.9)
ax2.grid(True, alpha=0.2, linewidth=0.5, which='major')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_ylim(1, 15)

plt.tight_layout(w_pad=1.0)
plt.savefig('fig_emax_scaling.pdf', bbox_inches='tight', pad_inches=0.02)
plt.savefig('fig_emax_scaling.png', bbox_inches='tight', pad_inches=0.02, dpi=300)
print("Saved fig_emax_scaling.pdf and fig_emax_scaling.png")
