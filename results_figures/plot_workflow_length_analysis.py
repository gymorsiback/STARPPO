#!/usr/bin/env python3
"""
工作流长度分析图：展示 STAR-PPO 的长期规划能力
- X轴：Workflow Length (2 steps, 3 steps, 5 steps)
- Y轴：Average Latency (ms)
- 核心展示：STAR-PPO 优势随工作流长度增加而更加明显
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import SLA_MS, sla_violation as _sla

# 配置matplotlib样式 - 论文格式（大字体，适合双栏论文）
plt.rcParams.update({
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
    'legend.fontsize': 18,
    'lines.linewidth': 4.0,
    'ytick.color': 'black',
    'xtick.color': 'black',
    'axes.labelcolor': 'black',
    'mathtext.fontset': 'stix',
})

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 10个算法
ALGORITHMS = ['STAR_PPO', 'A3C', 'Trans', 'PPO', 'PPO_CN', 'PFAPPO', 'PPO_GNN', 'Greedy', 'Stark', 'Random']

# 10种不同颜色，每个算法唯一（浅色调）
COLORS = {
    'STAR_PPO': '#a6cee3',   # Light Blue - Ours
    'A3C': '#b2df8a',        # Light Green
    'Trans': '#cab2d6',      # Light Purple
    'PPO': '#a89078',        # Brown
    'PPO_CN': '#98d8c8',     # Mint
    'PFAPPO': '#c7e9c0',     # Lighter Green
    'PPO_GNN': '#dadaeb',    # Lighter Purple
    'Greedy': '#bcbddc',     # Lavender
    'Stark': '#969696',      # Gray
    'Random': '#d9d9d9'      # Light Gray
}

HATCHES = {
    'STAR_PPO': '/',
    'A3C': '...',
    'Trans': 'xx',
    'PPO': '++',
    'PPO_CN': '--',
    'PFAPPO': '\\\\',
    'PPO_GNN': 'oo',
    'Greedy': '//',
    'Stark': '**',
    'Random': 'OO'
}

DISPLAY_NAMES = {
    'STAR_PPO': 'STAR-PPO',
    'A3C': 'A3C',
    'Trans': 'Equity-Trans',
    'PPO': 'PPO-Std',
    'PPO_CN': 'PPO-CN',
    'PFAPPO': 'PF-PPO',
    'PPO_GNN': 'GA-PPO',
    'Greedy': 'Greedy',
    'Stark': 'STARK',
    'Random': 'Random'
}

def load_workflow_length_results():
    """加载不同工作流长度的推理结果"""
    results_dir = 'inference/results_500'
    data = {}

    for algo in ALGORITHMS:
        data[algo] = {}

        for steps in [2, 3, 5]:
            path = os.path.join(results_dir, f"{algo}_workflow_{steps}steps.npz")
            if os.path.exists(path):
                npz = np.load(path)
                data[algo][steps] = {
                    'latencies': npz['latencies'],
                    'costs': npz['costs']
                }
                viol = np.mean(_sla(npz['latencies'])) * 100
                print(f"  Loaded {algo} {steps}步: AvgLat={np.mean(npz['latencies']):.1f}ms  SLA={viol:.1f}%")

    return data


def main():
    print("=" * 60)
    print("Workflow Length Analysis - Long-term Planning Capability")
    print("=" * 60)

    # 加载数据
    print("\n加载推理结果...")
    data = load_workflow_length_results()

    # 构建 DataFrame
    records = []
    categories = ['Short\n(2 steps)', 'Medium\n(3 steps)', 'Long\n(5 steps)']
    step_map = {2: 'Short\n(2 steps)', 3: 'Medium\n(3 steps)', 5: 'Long\n(5 steps)'}

    for algo in ALGORITHMS:
        if algo not in data or not data[algo]:
            continue

        for steps in [2, 3, 5]:
            if steps not in data[algo]:
                continue

            latencies = data[algo][steps]['latencies']

            records.append({
                'Algorithm': DISPLAY_NAMES[algo],
                'AlgoKey': algo,
                'Category': step_map[steps],
                'Steps': steps,
                'Latency': np.mean(latencies),
                'Std': np.std(latencies)
            })

    df = pd.DataFrame(records)

    # 打印统计
    print("\n" + "=" * 60)
    print("=== Performance Summary ===")
    for cat in categories:
        print(f"\n{cat}:")
        cat_df = df[df['Category'] == cat].sort_values('Latency')
        for _, row in cat_df.iterrows():
            print(f"  {row['Algorithm']}: {row['Latency']:.1f}ms")

    # 计算长期规划优势
    print("\n" + "=" * 60)
    print("=== Long-term Planning Analysis ===")
    print("(Degradation = (Long - Short) / Short × 100%)")
    print("Lower degradation = Better long-term planning\n")

    degradation_data = []
    for algo in ALGORITHMS:
        display = DISPLAY_NAMES[algo]
        short_data = df[(df['Algorithm'] == display) & (df['Steps'] == 2)]['Latency'].values
        long_data = df[(df['Algorithm'] == display) & (df['Steps'] == 5)]['Latency'].values

        if len(short_data) > 0 and len(long_data) > 0:
            short_lat = short_data[0]
            long_lat = long_data[0]
            degradation = (long_lat - short_lat) / short_lat * 100
            degradation_data.append((algo, display, short_lat, long_lat, degradation))
            print(f"  {display}: 2步={short_lat:.0f}ms → 5步={long_lat:.0f}ms, 退化={degradation:+.1f}%")

    # 按退化率排序（退化越小越好）
    degradation_data.sort(key=lambda x: x[4])
    print("\n=== Ranking by Long-term Planning (Lower Degradation = Better) ===")
    for i, (algo, display, short, long, deg) in enumerate(degradation_data, 1):
        print(f"  {i}. {display}: {deg:+.1f}%")

    # ==================== 绘图 ====================
    fig, ax = plt.subplots(figsize=(9, 7))

    x = np.arange(len(categories))

    # 筛选有完整数据的算法
    complete_algos = []
    for algo in ALGORITHMS:
        display = DISPLAY_NAMES[algo]
        has_all = all(
            len(df[(df['Algorithm'] == display) & (df['Steps'] == s)]) > 0
            for s in [2, 3, 5]
        )
        if has_all:
            complete_algos.append(algo)

    # 按 5 步性能排序
    long_scores = {}
    for algo in complete_algos:
        display = DISPLAY_NAMES[algo]
        long_scores[algo] = df[(df['Algorithm'] == display) & (df['Steps'] == 5)]['Latency'].values[0]

    algo_order = sorted(complete_algos, key=lambda x: long_scores[x])

    width = 0.08
    n_algos = len(algo_order)

    for i, algo in enumerate(algo_order):
        display = DISPLAY_NAMES[algo]
        algo_df = df[df['Algorithm'] == display]

        latencies = []
        for cat in categories:
            lat_val = algo_df[algo_df['Category'] == cat]['Latency'].values
            latencies.append(lat_val[0] if len(lat_val) > 0 else 0)

        offset = (i - n_algos/2 + 0.5) * width

        # 深色配色 + 防色盲 hatch
        bars = ax.bar(x + offset, latencies, width,
                     label=display, color=COLORS[algo],
                     edgecolor='black', linewidth=1.2, hatch=HATCHES[algo])


    ax.set_xlabel('Workflow Length')
    ax.set_ylabel('Average Latency (ms)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)

    # 图例放在图内左上角，确保 STAR_PPO 第一位
    handles, labels = ax.get_legend_handles_labels()
    # 找到 STAR-PPO 的索引并移动到最前
    if 'STAR-PPO' in labels:
        idx = labels.index('STAR-PPO')
        handles.insert(0, handles.pop(idx))
        labels.insert(0, labels.pop(idx))

    # 单列，放在左上角
    ax.legend(handles, labels, loc='upper left', fontsize=16, ncol=1,
              frameon=True, framealpha=0.9)

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # 移除左上角标注

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, 'Workflow_Length_Analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")

    # ==================== 额外：折线图版本 ====================
    fig2, ax2 = plt.subplots(figsize=(9, 7))

    steps_x = [2, 3, 5]

    for algo in algo_order:
        display = DISPLAY_NAMES[algo]
        algo_df = df[df['Algorithm'] == display]

        latencies = []
        for s in steps_x:
            lat_val = algo_df[algo_df['Steps'] == s]['Latency'].values
            latencies.append(lat_val[0] if len(lat_val) > 0 else 0)

        # STAR-PPO 稍微突出但不过分
        if algo == 'STAR_PPO':
            ax2.plot(steps_x, latencies, 'o-', label=display, color=COLORS[algo],
                    linewidth=2.5, markersize=8)
        else:
            ax2.plot(steps_x, latencies, 'o-', label=display, color=COLORS[algo],
                    linewidth=1.5, markersize=6, alpha=0.8)

    ax2.set_xlabel('Workflow Length')
    ax2.set_ylabel('Average Latency (ms)')
    ax2.set_xticks(steps_x)
    ax2.legend(title='Algorithm', bbox_to_anchor=(1.02, 1), loc='upper left',
               fontsize=18, title_fontsize=20)
    ax2.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()

    output_path2 = os.path.join(OUTPUT_DIR, 'Workflow_Length_Analysis_Line.png')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"Line plot saved to {output_path2}")


if __name__ == '__main__':
    main()
