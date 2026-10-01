#!/usr/bin/env python3
"""Generate Fig 2: Threshold Tightness Comparison (V-ABFT vs A-ABFT)."""

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
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'text.usetex': False,
    'mathtext.fontset': 'stix',
})

sizes = [128, 256, 512, 1024, 2048]
size_labels = ['128', '256', '512', '1024', '2048']

# Tightness ratios from Tables 3, 4, 5
data = {
    'FP64': {
        'A-ABFT': [122, 197, 305, 468, 692],
        'V-ABFT': [16, 14, 12, 10, 9],
    },
    'FP32': {
        'A-ABFT': [284, 824, 1282, 971, 1451],
        'V-ABFT': [16, 25, 21, 9, 8],
    },
    'BF16': {
        'A-ABFT': [825, 1699, 3668, 7835, 16542],
        'V-ABFT': [53, 53, 53, 53, 53],
    },
}

fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2), sharey=False)
# 7.16 inches = IEEE double-column width

colors = {'A-ABFT': '#d62728', 'V-ABFT': '#1f77b4'}
markers = {'A-ABFT': 's', 'V-ABFT': 'o'}

x = np.arange(len(sizes))
bar_width = 0.32

for idx, (precision, vals) in enumerate(data.items()):
    ax = axes[idx]

    a_bars = ax.bar(x - bar_width / 2, vals['A-ABFT'], bar_width,
                    label='A-ABFT', color=colors['A-ABFT'], alpha=0.8,
                    edgecolor='white', linewidth=0.5)
    v_bars = ax.bar(x + bar_width / 2, vals['V-ABFT'], bar_width,
                    label='V-ABFT', color=colors['V-ABFT'], alpha=0.8,
                    edgecolor='white', linewidth=0.5)

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(size_labels)
    ax.set_xlabel('Matrix size ($N$)')
    ax.set_title(precision, fontweight='bold')

    # Add value labels on top of bars
    for bar in a_bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h * 1.15,
                f'{int(h)}', ha='center', va='bottom', fontsize=6.5,
                color=colors['A-ABFT'])
    for bar in v_bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h * 1.15,
                f'{int(h)}', ha='center', va='bottom', fontsize=6.5,
                color=colors['V-ABFT'])

    # Add ideal line at y=1
    ax.axhline(y=1, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.text(len(sizes) - 0.5, 1.3, 'ideal', fontsize=7, color='gray',
            ha='right', va='bottom', style='italic')

    ax.set_ylim(0.8, None)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].set_ylabel('Tightness ratio (threshold / actual, log scale)')
axes[0].legend(loc='upper left', framealpha=0.9)

plt.tight_layout(w_pad=0.8)
plt.savefig('fig_tightness.pdf', bbox_inches='tight', pad_inches=0.02)
plt.savefig('fig_tightness.png', bbox_inches='tight', pad_inches=0.02, dpi=300)
print("Saved fig_tightness.pdf and fig_tightness.png")
