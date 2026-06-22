#!/usr/bin/env python3
"""
生成 DAG 泛化实验的可视化图表
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from metrics import SLA_MS

# 设置字体 - 论文格式（大字体，适合双栏论文）
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 26,
    'axes.labelsize': 28,
    'axes.titlesize': 30,
    'axes.titleweight': 'normal',
    'axes.linewidth': 2.0,
    'axes.edgecolor': 'black',
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 20,
    'lines.linewidth': 4.0,
    'mathtext.fontset': 'stix',
})

# ============================================================
# 加载数据
# ============================================================
RESULT_DIR = os.path.dirname(os.path.abspath(__file__))
result_path = os.path.join(RESULT_DIR, 'dag_generalization_results.npz')

data = np.load(result_path, allow_pickle=True)
topologies = list(data['topologies'])
star_ppo_latencies = data['star_ppo']
greedy_latencies = data['greedy']
random_latencies = data['random']

print("加载的实验结果:")
for i, topo in enumerate(topologies):
    print(f"  {topo}: STAR-PPO={star_ppo_latencies[i]:.2f}, Greedy={greedy_latencies[i]:.2f}, Random={random_latencies[i]:.2f}")

# ============================================================
# 准备绘图数据
# ============================================================
algorithms = ['STAR_PPO', 'Greedy', 'Random']
# 三种对比色：浅蓝、浅绿、灰棕
colors = {'STAR_PPO': '#a6cee3', 'Greedy': '#b2df8a', 'Random': '#a89078'}
hatches = {'STAR_PPO': '/', 'Greedy': '...', 'Random': 'xx'}
labels = {'STAR_PPO': 'TopoFreeRL (Ours)', 'Greedy': 'Greedy', 'Random': 'Random'}

latencies = {
    'STAR_PPO': star_ppo_latencies,
    'Greedy': greedy_latencies,
    'Random': random_latencies
}

# ============================================================
# 图1: 分组柱状图
# ============================================================
fig, ax = plt.subplots(figsize=(9, 7))

x = np.arange(len(topologies))
width = 0.25

for i, algo in enumerate(algorithms):
    offset = (i - 1) * width
    bars = ax.bar(x + offset, latencies[algo], width,
                  label=labels[algo], color=colors[algo],
                  edgecolor='black', linewidth=1.2, hatch=hatches[algo])

    # 在柱子上添加数值
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=15)

ax.set_xlabel('DAG Topology')
ax.set_ylabel('Average Latency (ms)')
ax.set_title('Generalization to Unseen DAG Topologies')
ax.set_xticks(x)
ax.set_xticklabels(topologies)
ax.legend(fontsize=18, loc='upper center', ncol=1, bbox_to_anchor=(0.5, 1.0))
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

# SLA 阈值线
ax.axhline(y=SLA_MS, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'SLA = {SLA_MS:.0f} ms')
ax.legend(fontsize=16, loc='upper center', ncol=1, bbox_to_anchor=(0.5, 1.0))

# 设置 Y 轴范围
ax.set_ylim(0, max(max(latencies['Random']), max(latencies['Greedy'])) * 1.15)

plt.tight_layout()
output_path = os.path.join(RESULT_DIR, 'DAG_Generalization.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n已保存: {output_path}")
plt.close()

# ============================================================
# 计算改进率（用于打印汇总）
# ============================================================
improvements_vs_greedy = []
improvements_vs_random = []

for i in range(len(topologies)):
    star = star_ppo_latencies[i]
    greedy = greedy_latencies[i]
    random = random_latencies[i]

    imp_greedy = (greedy - star) / greedy * 100
    imp_random = (random - star) / random * 100

    improvements_vs_greedy.append(imp_greedy)
    improvements_vs_random.append(imp_random)

# ============================================================
# 打印汇总
# ============================================================
print("\n" + "="*60)
print("📊 实验结果汇总")
print("="*60)
print(f"\n{'拓扑':<12} {'STAR-PPO':<12} {'Greedy':<12} {'Random':<12} {'vs Greedy':<12} {'vs Random':<12}")
print("-"*72)

for i, topo in enumerate(topologies):
    star = star_ppo_latencies[i]
    greedy = greedy_latencies[i]
    random = random_latencies[i]
    imp_g = improvements_vs_greedy[i]
    imp_r = improvements_vs_random[i]
    print(f"{topo:<12} {star:<12.2f} {greedy:<12.2f} {random:<12.2f} {imp_g:>+.1f}%{'':<6} {imp_r:>+.1f}%")

print("\n✅ 关键发现:")
print(f"   - STAR-PPO 比 Greedy 平均提升: {np.mean(improvements_vs_greedy):.1f}%")
print(f"   - STAR-PPO 比 Random 平均提升: {np.mean(improvements_vs_random):.1f}%")
print(f"   - 无需重新训练，模型直接泛化到复杂 DAG 拓扑")
print("\nDone!")

# ============================================================
# 图2: 雷达图 (恢复默认边框，雷达图不需要加粗边框)
# ============================================================
matplotlib.rcParams['axes.linewidth'] = 1.0  # 恢复默认边框
fig, ax = plt.subplots(figsize=(9, 7), subplot_kw=dict(polar=True))

# 准备数据
categories = topologies
N = len(categories)

# 计算角度
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # 闭合

# 归一化数据 (使用 Random 作为基准 = 1.0，越低越好)
star_ppo_norm = [random_latencies[i] / star_ppo_latencies[i] for i in range(N)]
greedy_norm = [random_latencies[i] / greedy_latencies[i] for i in range(N)]
random_norm = [1.0] * N  # Random 作为基准

# 闭合数据
star_ppo_norm += star_ppo_norm[:1]
greedy_norm += greedy_norm[:1]
random_norm += random_norm[:1]

# 绘制雷达图
ax.plot(angles, star_ppo_norm, 'o-', linewidth=2.5, label='TopoFreeRL (Ours)', color='#2E86AB', markersize=8)
ax.fill(angles, star_ppo_norm, alpha=0.25, color='#2E86AB')

ax.plot(angles, greedy_norm, 's-', linewidth=2.5, label='Greedy', color='#28A745', markersize=8)
ax.fill(angles, greedy_norm, alpha=0.25, color='#28A745')

ax.plot(angles, random_norm, '^-', linewidth=2.5, label='Random', color='#DC3545', markersize=8)
ax.fill(angles, random_norm, alpha=0.25, color='#DC3545')

# 设置雷达图的标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# 设置标题和图例
ax.set_title('Generalization to Unseen DAG Topologies',
             pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.05), fontsize=18)

# 设置 y 轴范围
ax.set_ylim(0, max(max(star_ppo_norm), max(greedy_norm)) * 1.1)

# 设置雷达图圆圈上的数字（y轴刻度）字体小一点
ax.tick_params(axis='y', labelsize=14)

plt.tight_layout()
output_path = os.path.join(RESULT_DIR, 'DAG_Generalization_Radar.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"已保存: {output_path}")
plt.close()
