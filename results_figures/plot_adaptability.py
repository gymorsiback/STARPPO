#!/usr/bin/env python3
"""
适应性测试曲线图：10 个算法，线性坐标，STAR-PPO 置顶加粗
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import SLA_MS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 22,
    'axes.labelsize': 24,
    'axes.titlesize': 26,
    'axes.linewidth': 2.0,
    'axes.edgecolor': 'black',
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 17,
    'mathtext.fontset': 'stix',
})

RESULTS_DIR = os.path.join(SCRIPT_DIR, 'adaptability_results')
OUTPUT_DIR = SCRIPT_DIR

# 全部 10 个算法（绘制顺序：非 STAR 先画，STAR 最后置顶）
ALL_ALGORITHMS = [
    'Random', 'Greedy', 'PPO_GNN', 'PFAPPO', 'Stark',
    'Trans', 'A3C', 'PPO_CN', 'PPO', 'STAR_PPO',
]

COLORS = {
    'STAR_PPO': '#d62728',
    'A3C': '#8c564b',
    'PPO': '#1f77b4',
    'PPO_CN': '#ff7f0e',
    'PFAPPO': '#17becf',
    'PPO_GNN': '#2ca02c',
    'Trans': '#9467bd',
    'Stark': '#e377c2',
    'Greedy': '#7f7f7f',
    'Random': '#bcbd22',
}

DISPLAY_NAMES = {
    'STAR_PPO': 'STAR-PPO',
    'A3C': 'A3C',
    'PPO': 'PPO-Std',
    'PPO_CN': 'PPO-CN',
    'PFAPPO': 'PF-PPO',
    'PPO_GNN': 'GA-PPO',
    'Trans': 'Equity-Trans',
    'Stark': 'STARK',
    'Greedy': 'Greedy',
    'Random': 'Random',
}


def smooth(data, window=8):
    data = np.array(data, dtype=float)
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    smoothed = np.convolve(data, kernel, mode='same')
    half = window // 2
    for i in range(half):
        smoothed[i] = np.mean(data[: i + half + 1])
        smoothed[-(i + 1)] = np.mean(data[-(i + half + 1) :])
    return smoothed


def load_all():
    all_data = {}
    meta = None
    for algo in ALL_ALGORITHMS:
        path = os.path.join(RESULTS_DIR, f'{algo}_adaptability.npz')
        if os.path.exists(path):
            data = np.load(path)
            all_data[algo] = data['episode_latencies']
            if meta is None:
                meta = {
                    'normal_episodes': int(data['normal_episodes']),
                    'total_episodes': int(data['total_episodes']),
                }
    return all_data, meta


def main():
    all_data, meta = load_all()
    if not all_data or 'STAR_PPO' not in all_data:
        print('Error: 缺少 STAR_PPO 适应性数据')
        return

    normal_end = meta['normal_episodes']
    total = meta['total_episodes']

    # 绘制顺序：按故障期均值从低到高，STAR_PPO 强制最后（置顶）
    failure_avg = {a: np.mean(all_data[a][normal_end:]) for a in all_data}
    draw_order = sorted(all_data.keys(), key=lambda a: failure_avg[a], reverse=True)
    if 'STAR_PPO' in draw_order:
        draw_order.remove('STAR_PPO')
        draw_order.append('STAR_PPO')

    fig, ax = plt.subplots(figsize=(14, 7.5))
    y_vals = []

    for algo in draw_order:
        latencies = all_data[algo]
        x = np.arange(len(latencies))
        win = 5 if algo == 'STAR_PPO' else 8
        smoothed = smooth(latencies, window=win)
        y_vals.extend(smoothed.tolist())

        is_ours = algo == 'STAR_PPO'
        ax.plot(
            x, smoothed,
            color=COLORS[algo],
            linewidth=2.6 if is_ours else 1.4,
            alpha=1.0 if is_ours else 0.9,
            label=DISPLAY_NAMES.get(algo, algo),
            zorder=8 if is_ours else 3,
            linestyle='-' if is_ours else '-',
        )
        if is_ours:
            ax.scatter(
                [x[-1]], [smoothed[-1]],
                color=COLORS['STAR_PPO'], s=90, marker='*',
                zorder=9, edgecolors='white', linewidth=0.8,
            )

    # 用平滑曲线分位数定上限，避免个别尖峰把 Y 轴拉得过高
    y_lo = max(300, min(y_vals) * 0.92)
    y_hi = float(np.percentile(y_vals, 99) * 1.06)
    y_hi = max(y_hi, max(y_vals) * 1.03)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(0, total)
    ax.axvspan(normal_end, total, color='#ffcccc', alpha=0.35, zorder=0)
    ax.axvline(x=normal_end, color='black', linestyle='--', linewidth=2, zorder=2)
    shock_label = 'Traffic Spike'
    shock_title = 'Adaptability to Traffic Spike'
    star_path = os.path.join(RESULTS_DIR, 'STAR_PPO_adaptability.npz')
    if os.path.exists(star_path):
        meta_npz = np.load(star_path, allow_pickle=True)
        if 'shock_mode' in meta_npz:
            sm = str(meta_npz['shock_mode'])
            if 'region' in sm:
                shock_label = 'Region Outage'
                shock_title = 'Adaptability to Regional Server Outage'

    ax.text(
        normal_end + 3, y_hi * 0.92,
        shock_label,
        fontsize=15, fontweight='bold', va='top',
    )

    ax.set_ylabel('Latency (ms)')
    ax.set_xlabel('Episode')
    ax.grid(True, alpha=0.35)

    handles, labels = ax.get_legend_handles_labels()
    if 'STAR-PPO' in labels:
        i = labels.index('STAR-PPO')
        handles = [handles[i]] + handles[:i] + handles[i + 1 :]
        labels = [labels[i]] + labels[:i] + labels[i + 1 :]
    ax.legend(handles, labels, loc='upper left', fontsize=16, frameon=True, ncol=2)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'Adaptability_Test.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'图表已保存: {out}')

    star = all_data['STAR_PPO']
    print('\n扰动后相对正常期变化 (末 20 ep 均值 vs 正常期):')
    for algo in sorted(all_data.keys()):
        lat = all_data[algo]
        n = lat[:normal_end].mean()
        shock_mean = lat[normal_end:].mean()
        recover = lat[-20:].mean() if len(lat) >= 20 else shock_mean
        ch = (shock_mean - n) / n * 100
        rec = (recover - n) / n * 100
        dup = ' [与STAR完全相同]' if algo != 'STAR_PPO' and np.array_equal(lat, star) else ''
        print(
            f"  {DISPLAY_NAMES.get(algo, algo):<18} {n:7.0f} -> {shock_mean:7.0f} ms ({ch:+6.1f}%)"
            f"  恢复段={recover:.0f} ({rec:+5.1f}%){dup}"
        )


if __name__ == '__main__':
    main()
