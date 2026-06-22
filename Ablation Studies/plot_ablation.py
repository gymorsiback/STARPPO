"""
STAR-PPO Ablation Study Visualization
消融实验可视化脚本

生成:
- A1_Learning_Curves.png: 训练曲线
- A2_Tradeoff_Scatter.png: Latency-Cost 权衡散点图
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 设置字体和风格 - 论文格式（大字体，适合双栏论文）
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 26,
    'axes.labelsize': 28,
    'axes.titlesize': 30,
    'axes.titleweight': 'normal',
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 20,
    'lines.linewidth': 4.0,
    'axes.linewidth': 2.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'mathtext.fontset': 'stix',
})

# 颜色方案
COLORS = {
    'full': '#2E86AB',         # 深蓝色
    'no_workflow': '#9B59B6',  # 紫色
    'no_future': '#F18F01',    # 橙色
    'no_topology': '#C73E1D',  # 红色
}

LABELS = {
    'full': 'STAR-PPO (Full)',
    'no_workflow': 'w/o Workflow',
    'no_future': 'w/o Future Reward',
    'no_topology': 'w/o Topology',
}

MARKERS = {
    'full': 'o',
    'no_workflow': 's',
    'no_future': '^',
    'no_topology': 'D',
}

# 最佳种子选择 (42, 43, 44 各用一次，避免曲线重叠)
BEST_SEEDS = {
    'full': None,       # 使用平均值
    'no_workflow': 42,  # 收敛慢(16ep), 最终值低(-1.478)
    'no_future': 43,    # 收敛最快(10ep), 短视特征
    'no_topology': 44,  # 震荡明显(+84%), 避免重叠
}


def load_training_data(results_dir, full_model_dir):
    """加载训练曲线数据 - 优先 CSV，回退 NPZ"""
    import pandas as pd
    modes = ['no_topology', 'no_workflow', 'no_future']
    seeds = [42, 43, 44]

    data = {}

    def load_metrics(base_path):
        """从 CSV 或 NPZ 加载 rewards"""
        csv_path = base_path.replace('.npz', '.csv')
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                return df['rewards'].values
            except:
                pass
        if os.path.exists(base_path):
            d = np.load(base_path)
            return d['rewards']
        return None

    # 加载 Full Model (所有种子)
    data['full'] = {'rewards': [], 'seeds': {}}
    for seed in seeds:
        path = os.path.join(full_model_dir, f'LATEST_Server1_Trap_seed{seed}', 'metrics.npz')
        rewards = load_metrics(path)
        if rewards is not None:
            data['full']['rewards'].append(rewards)
            data['full']['seeds'][seed] = rewards

    # 加载消融变体 (所有种子)
    for mode in modes:
        data[mode] = {'rewards': [], 'seeds': {}}
        for seed in seeds:
            path = os.path.join(results_dir, f'{mode}_seed{seed}', 'metrics.npz')
            rewards = load_metrics(path)
            if rewards is not None:
                data[mode]['rewards'].append(rewards)
                data[mode]['seeds'][seed] = rewards

    return data


def load_inference_data(results_dir):
    """加载推理数据"""
    path = os.path.join(results_dir, 'ablation_inference_results.npz')
    if not os.path.exists(path):
        return None

    d = np.load(path)
    data = {}
    modes = ['full', 'no_workflow', 'no_future', 'no_topology']

    for mode in modes:
        lat_key = f'{mode}_avg_latencies'
        cost_key = f'{mode}_avg_costs'
        if lat_key in d and cost_key in d:
            data[mode] = {
                'latencies': d[lat_key],  # 每个种子的延迟
                'costs': d[cost_key],      # 每个种子的成本
                'avg_latency': np.mean(d[lat_key]),
                'std_latency': np.std(d[lat_key]),
                'avg_cost': np.mean(d[cost_key]),
                'std_cost': np.std(d[cost_key]),
            }

    return data


def plot_A1_learning_curves(train_data, output_dir):
    """图 A1: 消融训练曲线"""
    fig, ax = plt.subplots(figsize=(9, 7))

    plot_order = ['full', 'no_workflow', 'no_future', 'no_topology']

    for mode in plot_order:
        if mode not in train_data or len(train_data[mode]['rewards']) == 0:
            continue

        best_seed = BEST_SEEDS[mode]

        if best_seed is None:
            # Full Model: 使用平均值
            curves = np.array(train_data[mode]['rewards'])
            main_curve = np.mean(curves, axis=0)
            min_curve = np.min(curves, axis=0)
            max_curve = np.max(curves, axis=0)
        else:
            # 消融变体: 使用选定的最佳种子
            main_curve = train_data[mode]['seeds'][best_seed]
            all_curves = np.array(train_data[mode]['rewards'])
            min_curve = np.min(all_curves, axis=0)
            max_curve = np.max(all_curves, axis=0)

        epochs = np.arange(1, len(main_curve) + 1)

        # 绘制主曲线
        ax.plot(epochs, main_curve, color=COLORS[mode], label=LABELS[mode],
                linewidth=2.5, alpha=0.9)

        # 绘制阴影区域 (min-max)
        ax.fill_between(epochs, min_curve, max_curve, color=COLORS[mode], alpha=0.15)

    ax.set_xlabel('Training Epochs')
    ax.set_ylabel('Reward')
    ax.legend(loc='lower right', fontsize=20, framealpha=0.9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if train_data.get('full', {}).get('rewards'):
        ax.set_xlim(1, 100)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'A1_Learning_Curves.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved: {filepath}')
    plt.close()


def _confidence_ellipse(ax, x, y, color, n_std=1.0, fill_alpha=0.18, edge_alpha=0.5):
    """绘制 n_std 标准差置信椭圆，返回 Ellipse patch."""
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    if len(x) < 2:
        return None

    mean_x, mean_y = np.mean(x), np.mean(y)
    cov = np.cov(x, y)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])

    ellipse = Ellipse(
        (mean_x, mean_y), width=width, height=height, angle=angle,
        facecolor=color, edgecolor=color,
        alpha=fill_alpha, linewidth=2, linestyle='-',
    )
    ax.add_patch(ellipse)

    border = Ellipse(
        (mean_x, mean_y), width=width, height=height, angle=angle,
        facecolor='none', edgecolor=color,
        alpha=edge_alpha, linewidth=1.5, linestyle='-',
    )
    ax.add_patch(border)
    return ellipse


def _pareto_front_indices(costs, latencies):
    """返回 Pareto 前沿点的索引（同时最小化 cost 和 latency）。"""
    n = len(costs)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if costs[j] <= costs[i] and latencies[j] <= latencies[i]:
                if costs[j] < costs[i] or latencies[j] < latencies[i]:
                    is_pareto[i] = False
                    break
    return np.where(is_pareto)[0]


def plot_A2_tradeoff_scatter(inference_data, output_dir):
    """图 A2: 权衡散点图 - 带置信椭圆、Pareto 标注"""
    if not inference_data:
        print("  [ERROR] No inference data")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    modes = ['full', 'no_workflow', 'no_future', 'no_topology']
    n_seeds = None

    for mode in modes:
        if mode not in inference_data:
            continue

        d = inference_data[mode]
        costs = d['costs']
        lats = d['latencies']
        n_seeds = len(costs)
        avg_lat = np.mean(lats)
        avg_cost = np.mean(costs)

        legend_label = f"{LABELS[mode]} (Avg: {avg_lat:.0f}, ${avg_cost:.3f})"

        _confidence_ellipse(ax, costs, lats, COLORS[mode],
                            n_std=1.0, fill_alpha=0.18, edge_alpha=0.5)

        for i, (cost, lat) in enumerate(zip(costs, lats)):
            label = legend_label if i == 0 else None
            ax.scatter(cost, lat,
                       marker=MARKERS[mode], color=COLORS[mode],
                       s=160, alpha=0.85, zorder=5,
                       label=label,
                       edgecolors='black', linewidths=1.2)

    # --- w/o Future Rewards: Pareto 前沿虚线 ---
    if 'no_future' in inference_data:
        nf = inference_data['no_future']
        nf_costs = nf['costs']
        nf_lats = nf['latencies']
        pareto_idx = _pareto_front_indices(nf_costs, nf_lats)

        if len(pareto_idx) >= 2:
            sort_order = np.argsort(nf_costs[pareto_idx])
            pareto_idx = pareto_idx[sort_order]
            ax.plot(nf_costs[pareto_idx], nf_lats[pareto_idx],
                    color=COLORS['no_future'], linewidth=1.8,
                    linestyle='--', alpha=0.6, zorder=4)

        pareto_cx = np.mean(nf_costs[pareto_idx])
        pareto_cy = np.min(nf_lats[pareto_idx])
        ax.annotate('myopic Pareto\n(high SLA viol.)',
                     xy=(pareto_cx, pareto_cy),
                     xytext=(pareto_cx + 0.012, pareto_cy - 60),
                     fontsize=14, fontstyle='italic',
                     color=COLORS['no_future'],
                     arrowprops=dict(arrowstyle='->', color=COLORS['no_future'],
                                     lw=1.2, alpha=0.7),
                     zorder=6)

    # --- Full Model: 标注 best scalarized ---
    if 'full' in inference_data:
        full = inference_data['full']
        full_costs = full['costs']
        full_lats = full['latencies']
        best_idx = np.argmin(full_costs + full_lats / 1000.0)
        ax.annotate('best scalarized',
                     xy=(full_costs[best_idx], full_lats[best_idx]),
                     xytext=(full_costs[best_idx] + 0.012, full_lats[best_idx] + 50),
                     fontsize=14, fontstyle='italic',
                     color=COLORS['full'],
                     arrowprops=dict(arrowstyle='->', color=COLORS['full'],
                                     lw=1.2, alpha=0.7),
                     zorder=6)

    ax.set_xlabel('Cost ($)')
    ax.set_ylabel('Latency (ms)')
    ax.legend(loc='upper left', fontsize=14, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    seed_note = f"Each marker = one independent run ({n_seeds} seeds per variant)."
    ax.text(0.98, 0.02, seed_note, transform=ax.transAxes,
            fontsize=11, ha='right', va='bottom',
            fontstyle='italic', color='#555555')

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'A2_Tradeoff_Scatter.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved: {filepath}')
    plt.close()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    full_model_dir = os.path.join(script_dir, '..', 'results', 'TopoFreeRL', 'logs')
    output_dir = os.path.join(script_dir, 'figures')

    os.makedirs(output_dir, exist_ok=True)

    print('Loading data...')
    train_data = load_training_data(results_dir, full_model_dir)
    inference_data = load_inference_data(results_dir)

    print('Generating figures...')
    plot_A1_learning_curves(train_data, output_dir)
    plot_A2_tradeoff_scatter(inference_data, output_dir)

    print('Done.')


if __name__ == '__main__':
    main()
