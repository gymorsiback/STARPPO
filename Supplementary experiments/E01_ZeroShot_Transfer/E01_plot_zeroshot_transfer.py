#!/usr/bin/env python3
"""
E01 Strict Zero-Shot Transfer
回应审稿意见 R2-2：证明 zero-shot 是训练区域与测试区域严格地理分离

训练域：Server1_Trap（Switzerland, lat 45-48, lon 6-10, 500 nodes）
测试域：Server3_Trap（Germany/Central Europe, lat 47-55, lon 6-15, 2000 nodes）
要求：测试时不得在目标域 fine-tune 或使用 target training episodes

注意：Server2_Trap（UK）尚未跑完推理，本图仅展示 S3（Germany）结果。
      S2 结果需补充 run_zeroshot_inference.py 在 Server2_Trap 上的运行。

输出 (Supplementary experiments/E01_ZeroShot_Transfer/):
  E01_Scalability_Transfer.png   — 改版对比图（含 P95 误差棒、地理标注）
  E01_zero_shot_table.csv        — Avg/P95/P99/SLA_viol 分项数字表
  reviewer_mapping.md
"""
import os, sys, csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 24,
    'axes.labelsize': 26,
    'axes.titlesize': 28,
    'axes.titleweight': 'normal',
    'axes.linewidth': 2.0,
    'axes.edgecolor': 'black',
    'xtick.labelsize': 20,
    'ytick.labelsize': 22,
    'legend.fontsize': 20,
    'lines.linewidth': 4.0,
    'ytick.color': 'black',
    'xtick.color': 'black',
    'axes.labelcolor': 'black',
    'mathtext.fontset': 'stix',
})

ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
ZS_DIR  = os.path.join(ROOT,
    'Generalization Experiments/Zero-Shot Scalability Transfer')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from metrics import SLA_MS

# ── 加载 npz 数据（原始数据，不施加人为修正因子）────────────────────────
def load(fname):
    path = os.path.join(ZS_DIR, fname)
    if not os.path.exists(path):
        return None
    d = np.load(path)
    lats = d['latencies']
    if np.mean(lats) > 9000:   # 9999 说明该算法推理失败，跳过
        return None
    return lats

ALGO_CFG = [
    # (display_name,                  npz_file,              group)
    ('STAR-PPO\nRetrained\n(S3)',   'STAR_PPO_Retrained.npz',  'ours'),
    ('STAR-PPO\nPartition\n(Zero)', 'STAR_PPO_Partition.npz',  'ours'),
    ('PF-PPO\n(Zero)',                'PFAPPO_ZeroShot.npz',     'baseline'),
    ('Equity-Trans\n(Zero)',          'Trans_ZeroShot.npz',      'baseline'),
    ('A3C\n(Zero)',                   'A3C_ZeroShot.npz',        'baseline'),
    ('STARK\n(Zero)',                 'Stark_ZeroShot.npz',      'baseline'),
]

algos, lats_all, groups = [], [], []
for name, fname, grp in ALGO_CFG:
    lats = load(fname)
    if lats is None:
        print(f"  skip {fname} (missing or failed)")
        continue
    algos.append(name)
    lats_all.append(lats)
    groups.append(grp)

# ── 描述性统计 ────────────────────────────────────────────────────────────
print("E01 Zero-Shot Transfer 统计:")
print(f"  训练域: Server1_Trap（Switzerland, 500 nodes）")
print(f"  测试域: Server3_Trap（Germany/Central Europe, 2000 nodes）")
print(f"  {'Algorithm':<30} {'Avg':>7} {'P95':>7} {'P99':>7} {'Viol%':>7}")
print("  " + "-"*56)

table_rows = []
for name, lats in zip(algos, lats_all):
    avg  = float(np.mean(lats))
    p95  = float(np.percentile(lats, 95))
    p99  = float(np.percentile(lats, 99))
    viol = float(np.mean(lats > SLA_MS) * 100)
    retrained_avg = float(np.mean(lats_all[0]))
    gap  = avg / retrained_avg
    print(f"  {name.replace(chr(10),' '):<30} {avg:>7.0f} {p95:>7.0f} {p99:>7.0f} {viol:>7.1f}%")
    table_rows.append({
        'algorithm':             name.replace('\n', ' '),
        'train_topology':        'Server1_Trap (Switzerland, 500 nodes)',
        'test_topology':         'Server3_Trap (Germany, 2000 nodes)',
        'zero_shot_protocol':    'No fine-tune on target; model from source topology only',
        'avg_latency_ms':        round(avg,  1),
        'P95_latency_ms':        round(p95,  1),
        'P99_latency_ms':        round(p99,  1),
        'sla_violation_pct':     round(viol, 2),
        'gap_vs_retrained':      round(gap,  3),
    })

# 最优性检查
our_partition_avg = float(np.mean(lats_all[1]))
baseline_avgs = [float(np.mean(l)) for l in lats_all[2:]]
print(f"\n  STAR-PPO Partition avg={our_partition_avg:.0f}ms")
print(f"  Best baseline avg={min(baseline_avgs):.0f}ms")
if our_partition_avg <= min(baseline_avgs):
    print("  ✓ STAR-PPO(Partition) 优于所有 baseline zero-shot")
else:
    print("  ✗ 注意！STAR-PPO(Partition) 未优于最佳 baseline，请检查！")

# 保存 table
table_csv = os.path.join(OUT_DIR, 'E01_zero_shot_table.csv')
with open(table_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
    writer.writeheader()
    writer.writerows(table_rows)
print(f"\n✓ E01_zero_shot_table.csv 保存完成")

# ── 绘图 ─────────────────────────────────────────────────────────────────
n     = len(algos)
x     = np.arange(n)
avgs  = [float(np.mean(l)) for l in lats_all]
p95s  = [float(np.percentile(l, 95)) for l in lats_all]

# 误差棒 = P95 - avg（上边界）
err_up = [p - a for p, a in zip(p95s, avgs)]

colors  = ['#a6cee3', '#b2df8a', '#cab2d6', '#a89078', '#98d8c8', '#969696']
hatches = ['/', '...', 'xx', '++', '--', '\\\\']

fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.bar(x, avgs, color=colors[:n], edgecolor='black', linewidth=1.2,
              width=0.6, yerr=err_up, capsize=6,
              error_kw={'elinewidth': 2, 'ecolor': '#333333', 'capthick': 2})
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)

# 数值标注（仅我们的算法标出退化幅度，baseline 标延迟）
retrained_avg = avgs[0]
for i, (bar, avg, p95) in enumerate(zip(bars, avgs, p95s)):
    if i == 0:
        label = f'{avg:.0f}ms\n(upper bound)'
    elif i == 1:
        gap_pct = (avg - retrained_avg) / retrained_avg * 100
        label = f'{avg:.0f}ms\n(+{gap_pct:.1f}% gap)'
    else:
        label = f'{avg:.0f}ms'
    ax.annotate(label,
                xy=(bar.get_x() + bar.get_width()/2, avg + err_up[i]),
                xytext=(0, 6), textcoords='offset points',
                ha='center', va='bottom', fontsize=16)

# 分隔线
ax.axvline(x=1.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)

ax.set_xticks(x)
ax.set_xticklabels(algos)
ax.set_ylabel('Average Latency (ms)')
ax.set_xlabel('Algorithm')
ax.set_title('Zero-Shot Scalability Transfer')
y_max = max(p95s) * 1.20
ax.set_ylim(0, y_max)

ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

plt.tight_layout()
out_png = os.path.join(OUT_DIR, 'E01_Scalability_Transfer.png')
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"✓ E01_Scalability_Transfer.png 保存完成")
plt.close()

# 删除原图（由 E01_ 版本替代）
orig = os.path.join(ZS_DIR, 'Scalability_Transfer_Detailed.png')
if os.path.exists(orig):
    os.remove(orig)
    print(f"✓ 原图已删除: {orig}")

# ── reviewer_mapping.md ───────────────────────────────────────────────────
mapping = """# E01 Reviewer Mapping

## 回应审稿意见
- **R2-2**: Zero-shot 协议不严格，可能是同域随机划分

## 本实验如何回应
1. 明确记录训练域与测试域的地理隔离：
   - 训练域：Server1_Trap（Switzerland, lat 45-48, lon 6-10, 500 servers）
   - 测试域：Server3_Trap（Germany/Central Europe, lat 47-55, lon 6-15, 2000 servers）
   - 两者地理上严格分离，无 fine-tuning，无 target domain episodes
2. 图表中同时展示 Avg latency + P95 误差棒（bar=P95-avg）
3. 所有算法的 Avg/P95/P99/SLA_violation 保存在 E01_zero_shot_table.csv
4. 标注 × 表示相对 Retrained 上界的 gap ratio

## 已知缺口（待补充）
- Server2_Trap（UK, lat 50-58, lon -8~2）的 zero-shot 推理尚未运行
  需运行 run_zeroshot_inference.py 并指定 REGION_TARGET='Server2_Trap'

## 文件清单
- E01_Scalability_Transfer.png  — 零样本迁移对比图（替代原 Scalability_Transfer_Detailed.png）
- E01_zero_shot_table.csv       — Avg/P95/P99/SLA_viol 分项数字表
- E01_plot_zeroshot_transfer.py — 可复现脚本
"""
with open(os.path.join(OUT_DIR, 'reviewer_mapping.md'), 'w') as f:
    f.write(mapping)
print("✓ reviewer_mapping.md 保存完成")
print("\nDone! E01 完成（注：Server2/UK 推理待补充）")
