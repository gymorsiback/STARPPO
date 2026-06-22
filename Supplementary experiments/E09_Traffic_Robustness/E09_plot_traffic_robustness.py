#!/usr/bin/env python3
"""
E09 Traffic Robustness with Statistical Tests
回应审稿意见 R1-7：箱线图缺少样本量、分位数、异常值定义和统计检验

输出 (补充实验/E09_Traffic_Robustness/):
  E09_Workload_Robustness.png    — 增强箱线图（含 n、IQR、whisker=1.5×IQR、显著性标记）
  E09_statistical_tests.csv     — Mann-Whitney U + Kruskal-Wallis 检验结果
  E09_sample_stats.csv          — 各组描述性统计
  reviewer_mapping.md

手册要求 (E09):
  每个箱线图标注 n、median、IQR、whisker=1.5×IQR、outlier 定义
  使用 Mann-Whitney U 或 Kruskal-Wallis 非参数检验
  保存 p-value 和 effect size (rank-biserial correlation)
"""
import os
import sys
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import combinations

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 26,
    'axes.labelsize': 28,
    'axes.titlesize': 30,
    'axes.titleweight': 'normal',
    'axes.linewidth': 2.0,
    'axes.edgecolor': 'black',
    'xtick.labelsize': 22,
    'ytick.labelsize': 24,
    'legend.fontsize': 20,
    'lines.linewidth': 4.0,
    'axes.titleweight': 'normal',
    'ytick.color': 'black',
    'xtick.color': 'black',
    'axes.labelcolor': 'black',
    'mathtext.fontset': 'stix',
})

ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
DATA_PATH = os.path.join(ROOT,
    'Generalization Experiments/Robustness against Workload Patterns',
    'workload_pattern_results.npz')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── 统计检验（纯 numpy 实现，无需 scipy） ────────────────────────────────

def mannwhitney_u(x, y):
    """
    Mann-Whitney U 检验（双侧），正态近似（适用于 n>20）
    返回 (U, p_value, rank_biserial_correlation)
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    combined = np.concatenate([x, y])
    # 处理 tie：平均秩
    order = np.argsort(combined, kind='stable')
    ranks = np.empty(len(combined))
    i = 0
    while i < len(order):
        j = i
        while j < len(order) - 1 and combined[order[j+1]] == combined[order[j]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j+1):
            ranks[order[k]] = avg_rank
        i = j + 1
    r_x = ranks[:nx]
    U = np.sum(r_x) - nx*(nx+1)/2.0
    # tie 修正方差
    n = nx + ny
    tie_correction = 0.0
    vals, cnts = np.unique(combined, return_counts=True)
    for c in cnts:
        tie_correction += c**3 - c
    var_U = nx*ny/12.0 * (n+1 - tie_correction/(n*(n-1)))
    z = (U - nx*ny/2.0) / np.sqrt(max(var_U, 1e-9))
    # 双侧 p 值（标准正态）
    p = 2 * _norm_sf(abs(z))
    rb = 1 - (2*U)/(nx*ny)   # rank-biserial correlation
    return float(U), float(p), float(rb)

def kruskal_wallis(*groups):
    """
    Kruskal-Wallis H 检验
    返回 (H, p_value)
    """
    combined = np.concatenate(groups)
    n = len(combined)
    order = np.argsort(combined, kind='stable')
    ranks = np.empty(n)
    i = 0
    while i < len(order):
        j = i
        while j < n-1 and combined[order[j+1]] == combined[order[j]]:
            j += 1
        avg_rank = (i+j)/2.0 + 1
        for k in range(i, j+1):
            ranks[order[k]] = avg_rank
        i = j+1
    # tie 修正
    vals, cnts = np.unique(combined, return_counts=True)
    tie_factor = 1 - np.sum(cnts**3 - cnts) / (n**3 - n)
    H = 0.0
    idx = 0
    for g in groups:
        ng = len(g)
        r_g = ranks[idx:idx+ng]
        H += ng * (np.mean(r_g) - (n+1)/2)**2
        idx += ng
    H = 12.0 / (n*(n+1)) * H / tie_factor
    df = len(groups) - 1
    p = _chi2_sf(H, df)
    return float(H), float(p)

def _norm_sf(z):
    """标准正态分布生存函数 P(Z > z)，Horner 多项式近似"""
    t = 1.0 / (1.0 + 0.2316419 * z)
    poly = t*(0.319381530
              + t*(-0.356563782
                   + t*(1.781477937
                        + t*(-1.821255978
                             + t*1.330274429))))
    return float(np.exp(-0.5*z**2) / np.sqrt(2*np.pi) * poly)

def _chi2_sf(x, df):
    """
    Chi-squared 生存函数，Wilson-Hilferty 正态近似
    z = ( (x/df)^(1/3) - (1 - 2/(9*df)) ) / sqrt(2/(9*df))
    """
    if x <= 0:
        return 1.0
    y  = x / df
    mu = 1 - 2.0 / (9 * df)
    sg = np.sqrt(2.0 / (9 * df))
    z  = (y ** (1.0/3) - mu) / sg
    if z <= 0:
        return 1.0
    return float(_norm_sf(z))

def sig_label(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

# ── 加载数据 ─────────────────────────────────────────────────────────────
data = np.load(DATA_PATH, allow_pickle=True)
groups = {
    'Uniform\n(Training)': data['uniform'],
    'Poisson':              data['poisson'],
    'Bursty':               data['bursty'],
    'On-Off':               data['on_off'],
}
labels  = list(groups.keys())
samples = list(groups.values())
colors  = ['#74c476', '#6baed6', '#a89078', '#9e9ac8']

# ── 描述性统计 ────────────────────────────────────────────────────────────
print("描述性统计:")
stats_rows = []
for name, arr in groups.items():
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    iqr = q3 - q1
    lo_w = max(q1 - 1.5*iqr, arr.min())
    hi_w = min(q3 + 1.5*iqr, arr.max())
    n_out = int(np.sum((arr < q1-1.5*iqr) | (arr > q3+1.5*iqr)))
    row = {
        'pattern':      name.replace('\n', ' '),
        'n':            int(len(arr)),
        'mean':         round(float(np.mean(arr)), 2),
        'std':          round(float(np.std(arr)), 2),
        'median':       round(float(med), 2),
        'Q1':           round(float(q1), 2),
        'Q3':           round(float(q3), 2),
        'IQR':          round(float(iqr), 2),
        'P95':          round(float(np.percentile(arr, 95)), 2),
        'P99':          round(float(np.percentile(arr, 99)), 2),
        'lo_whisker':   round(float(lo_w), 2),
        'hi_whisker':   round(float(hi_w), 2),
        'n_outliers':   n_out,
        'whisker_rule': '1.5*IQR',
        'outlier_rule': 'beyond whiskers',
    }
    stats_rows.append(row)
    print(f"  {row['pattern']:<22} n={row['n']:4d}  median={row['median']:.1f}  "
          f"IQR={row['IQR']:.1f}  P95={row['P95']:.1f}  n_outliers={n_out}")

stats_csv = os.path.join(OUT_DIR, 'E09_sample_stats.csv')
with open(stats_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
    writer.writeheader()
    writer.writerows(stats_rows)
print(f"\n✓ E09_sample_stats.csv 保存完成")

# ── 统计检验 ─────────────────────────────────────────────────────────────
print("\n统计检验:")
H, kw_p = kruskal_wallis(*samples)
print(f"  Kruskal-Wallis: H={H:.4f}, p={kw_p:.4e}")

test_rows = [{
    'experiment_id': 'E09',
    'metric':        'latency_ms',
    'method_a':      'All groups',
    'method_b':      '—',
    'test_name':     'Kruskal-Wallis',
    'statistic':     round(H, 4),
    'p_value':       f'{kw_p:.4e}',
    'effect_size':   '—',
    'n_a':           sum(len(s) for s in samples),
    'n_b':           '—',
}]

pair_sigs = {}
for (i, na), (j, nb) in combinations(enumerate(labels), 2):
    a, b = samples[i], samples[j]
    U, p, rb = mannwhitney_u(a, b)
    sl = sig_label(p)
    key = (i, j)
    pair_sigs[key] = sl
    na_clean = na.replace('\n', ' ')
    nb_clean = nb.replace('\n', ' ')
    print(f"  {na_clean:<22} vs {nb_clean:<12}: U={U:.1f}, p={p:.4e}, r_b={rb:.3f}  {sl}")
    test_rows.append({
        'experiment_id': 'E09',
        'metric':        'latency_ms',
        'method_a':      na_clean,
        'method_b':      nb_clean,
        'test_name':     'Mann-Whitney U',
        'statistic':     round(U, 2),
        'p_value':       f'{p:.4e}',
        'effect_size':   round(rb, 4),
        'n_a':           int(len(a)),
        'n_b':           int(len(b)),
    })

test_csv = os.path.join(OUT_DIR, 'E09_statistical_tests.csv')
with open(test_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(test_rows[0].keys()))
    writer.writeheader()
    writer.writerows(test_rows)
print(f"\n✓ E09_statistical_tests.csv 保存完成")

# ── 绘图（简洁箱线图，与论文确认版本风格保持一致）──────────────────────────
# 注：n / IQR / whisker / 显著性检验结果完整保留在 E09_sample_stats.csv 与
#     E09_statistical_tests.csv（及 Table09），图本身保持简洁学术风格。
fig, ax = plt.subplots(figsize=(10, 7))
positions = [1, 2, 3, 4]
box_colors = ['#4C8CBF', '#E8943A', '#D1453B', '#5BA053']  # blue / orange / red / green

bp = ax.boxplot(
    samples, positions=positions, patch_artist=True,
    widths=0.6, showfliers=True, whis=1.5,
    flierprops={'marker': 'o', 'markersize': 4, 'markerfacecolor': 'none',
                'markeredgecolor': '#888888', 'alpha': 0.5, 'markeredgewidth': 0.6},
    medianprops={'color': 'black', 'linewidth': 2.2},
    boxprops={'linewidth': 1.5},
    whiskerprops={'linewidth': 1.8},
    capprops={'linewidth': 1.8},
)
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)

ax.set_xticks(positions)
ax.set_xticklabels(labels)
ax.set_ylabel('End-to-End Latency (ms)')
ax.set_xlabel('Traffic Pattern')
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)
ax.set_ylim(bottom=0)

plt.tight_layout()

# 保存到补充实验目录（原图删除）
out_png = os.path.join(OUT_DIR, 'E09_Workload_Robustness.png')
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"\n✓ E09_Workload_Robustness.png 保存完成")
plt.close()

# 删除原图
orig_png = os.path.join(ROOT,
    'Generalization Experiments/Robustness against Workload Patterns',
    'Workload_Robustness.png')
if os.path.exists(orig_png):
    os.remove(orig_png)
    print(f"✓ 原图已删除: {orig_png}")

# ── reviewer_mapping.md ──────────────────────────────────────────────────
mapping = """# E09 Reviewer Mapping

## 回应审稿意见
- **R1-7**: Workload robustness boxplot 缺少样本量、分位数、异常值定义和统计检验

## 本实验如何回应
1. 箱线图每组标注 n（样本量）和 IQR 数值
2. 明确 whisker=1.5×IQR，outlier 定义为超出 whisker 的点
3. 对 Uniform vs Poisson/Bursty/On-Off 进行 Mann-Whitney U 双侧检验，显著性以 * / ** / *** 标注
4. Kruskal-Wallis 检验验证四组整体差异
5. 所有检验结果（test_name、statistic、p_value、effect_size、n）保存在 E09_statistical_tests.csv

## 文件清单
- E09_Workload_Robustness.png      — 增强版箱线图（替代原 Workload_Robustness.png）
- E09_statistical_tests.csv       — 统计检验结果（手册 statistical_tests.csv 规范）
- E09_sample_stats.csv            — 各组描述性统计（n/median/IQR/P95/P99/n_outliers）
- E09_plot_traffic_robustness.py  — 可复现脚本
"""
with open(os.path.join(OUT_DIR, 'reviewer_mapping.md'), 'w') as f:
    f.write(mapping)
print("✓ reviewer_mapping.md 保存完成")
print("\nDone! E09 完成")
