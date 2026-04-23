"""
拟合 e_max 随 K 增长的规律
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 测量数据
K = np.array([128, 256, 512, 1024, 2048, 4096, 8192])
e_max = np.array([6.259773e-07, 7.995144e-07, 8.760367e-07,
                  1.377460e-06, 1.699135e-06, 2.347423e-06, 3.195341e-06])

# 候选拟合模型
def sqrt_model(x, a, b):
    """e_max = a * sqrt(K) + b"""
    return a * np.sqrt(x) + b

def log_model(x, a, b):
    """e_max = a * log(K) + b"""
    return a * np.log(x) + b

def linear_model(x, a, b):
    """e_max = a * K + b"""
    return a * x + b

def power_model(x, a, n):
    """e_max = a * K^n"""
    return a * np.power(x, n)

# 拟合各模型
print("=" * 60)
print("e_max vs K 拟合分析")
print("=" * 60)

# 1. sqrt(K) 模型
try:
    popt_sqrt, _ = curve_fit(sqrt_model, K, e_max)
    pred_sqrt = sqrt_model(K, *popt_sqrt)
    r2_sqrt = 1 - np.sum((e_max - pred_sqrt)**2) / np.sum((e_max - np.mean(e_max))**2)
    print(f"\n1. sqrt(K) 模型: e_max = {popt_sqrt[0]:.4e} * sqrt(K) + {popt_sqrt[1]:.4e}")
    print(f"   R² = {r2_sqrt:.6f}")
except Exception as e:
    print(f"sqrt 模型拟合失败: {e}")

# 2. log(K) 模型
try:
    popt_log, _ = curve_fit(log_model, K, e_max)
    pred_log = log_model(K, *popt_log)
    r2_log = 1 - np.sum((e_max - pred_log)**2) / np.sum((e_max - np.mean(e_max))**2)
    print(f"\n2. log(K) 模型: e_max = {popt_log[0]:.4e} * log(K) + {popt_log[1]:.4e}")
    print(f"   R² = {r2_log:.6f}")
except Exception as e:
    print(f"log 模型拟合失败: {e}")

# 3. linear 模型
try:
    popt_lin, _ = curve_fit(linear_model, K, e_max)
    pred_lin = linear_model(K, *popt_lin)
    r2_lin = 1 - np.sum((e_max - pred_lin)**2) / np.sum((e_max - np.mean(e_max))**2)
    print(f"\n3. linear 模型: e_max = {popt_lin[0]:.4e} * K + {popt_lin[1]:.4e}")
    print(f"   R² = {r2_lin:.6f}")
except Exception as e:
    print(f"linear 模型拟合失败: {e}")

# 4. power 模型 (e_max = a * K^n)
try:
    # 使用对数变换进行线性拟合来获得初始值
    log_K = np.log(K)
    log_emax = np.log(e_max)
    n_init, log_a_init = np.polyfit(log_K, log_emax, 1)
    a_init = np.exp(log_a_init)

    popt_pow, _ = curve_fit(power_model, K, e_max, p0=[a_init, n_init])
    pred_pow = power_model(K, *popt_pow)
    r2_pow = 1 - np.sum((e_max - pred_pow)**2) / np.sum((e_max - np.mean(e_max))**2)
    print(f"\n4. power 模型: e_max = {popt_pow[0]:.4e} * K^{popt_pow[1]:.4f}")
    print(f"   R² = {r2_pow:.6f}")
except Exception as e:
    print(f"power 模型拟合失败: {e}")

# 总结
print("\n" + "=" * 60)
print("拟合结果总结")
print("=" * 60)

results = [
    ("sqrt(K)", r2_sqrt, f"e_max = {popt_sqrt[0]:.4e} * sqrt(K) + {popt_sqrt[1]:.4e}"),
    ("log(K)", r2_log, f"e_max = {popt_log[0]:.4e} * log(K) + {popt_log[1]:.4e}"),
    ("linear", r2_lin, f"e_max = {popt_lin[0]:.4e} * K + {popt_lin[1]:.4e}"),
    ("power", r2_pow, f"e_max = {popt_pow[0]:.4e} * K^{popt_pow[1]:.4f}"),
]

results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
print(f"\n按 R² 排序:")
for i, (name, r2, formula) in enumerate(results_sorted):
    marker = "  <-- 最佳" if i == 0 else ""
    print(f"  {name:10s}: R² = {r2:.6f}  {formula}{marker}")

# 验证 sqrt(K) 假设
print("\n" + "=" * 60)
print("sqrt(K) 模型验证")
print("=" * 60)

# 计算 e_max / sqrt(K) 的一致性
ratio = e_max / np.sqrt(K)
print(f"\ne_max / sqrt(K) 值:")
for k, e, r in zip(K, e_max, ratio):
    print(f"  K={k:5d}: e_max={e:.4e}, e_max/sqrt(K)={r:.4e}")

print(f"\n  均值: {np.mean(ratio):.4e}")
print(f"  标准差: {np.std(ratio):.4e}")
print(f"  变异系数 (CV): {np.std(ratio)/np.mean(ratio)*100:.2f}%")

# 推荐公式
print("\n" + "=" * 60)
print("推荐公式")
print("=" * 60)

if popt_pow[1] > 0.45 and popt_pow[1] < 0.55:
    # 接近 0.5，使用 sqrt 模型
    coef = popt_sqrt[0]
    # 归一化到 K=1024
    e_max_1024 = sqrt_model(1024, *popt_sqrt)
    print(f"\n推荐使用 sqrt(K) 模型:")
    print(f"  e_max = {coef:.4e} * sqrt(K)")
    print(f"  或: e_max = {e_max_1024:.4e} * sqrt(K/1024)")
else:
    print(f"\n推荐使用 power 模型:")
    print(f"  e_max = {popt_pow[0]:.4e} * K^{popt_pow[1]:.4f}")

# 绘图
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.scatter(K, e_max, label='Measured', s=100, zorder=5)
K_dense = np.linspace(100, 10000, 100)
plt.plot(K_dense, sqrt_model(K_dense, *popt_sqrt), 'r-', label=f'sqrt(K), R²={r2_sqrt:.4f}')
plt.plot(K_dense, log_model(K_dense, *popt_log), 'g--', label=f'log(K), R²={r2_log:.4f}')
plt.plot(K_dense, power_model(K_dense, *popt_pow), 'b:', label=f'K^{popt_pow[1]:.3f}, R²={r2_pow:.4f}')
plt.xlabel('K')
plt.ylabel('e_max')
plt.title('e_max vs K (Linear Scale)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.loglog(K, e_max, 'ko', label='Measured', markersize=10)
plt.loglog(K_dense, sqrt_model(K_dense, *popt_sqrt), 'r-', label=f'sqrt(K)')
plt.loglog(K_dense, power_model(K_dense, *popt_pow), 'b:', label=f'K^{popt_pow[1]:.3f}')
plt.xlabel('K')
plt.ylabel('e_max')
plt.title('e_max vs K (Log-Log Scale)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/henrygao/Desktop/25fall/cuhk/10-29-/V-ABFT/emax_fit.png', dpi=150)
print(f"\n图表已保存到: emax_fit.png")
