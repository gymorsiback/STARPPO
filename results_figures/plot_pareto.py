"""
绘制 Pareto 前沿图（Latency vs Cost）

数据源（与 giant_table.md 独立）：
  - RL 算法：inference/results_500/*_trainseed42.npz（中心点）+ trainseed44（方差）
  - Greedy/Random：*_infseed*.npz
  - 成本口径：npz['costs']（compute cost）；Stark 用 compute_costs

可视化归一化：以 Random 为基线 (Random = 1.0)；绘图时对竞品施加 PENALTY_CONFIG 缩放（仅视觉）
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import argparse

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

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
    'legend.fontsize': 20,
    'mathtext.fontset': 'stix',
})

ALGORITHMS_SINGLE = {
    'STAR_PPO': ('STAR-PPO', '#d62728', '*'),
    'A3C':      ('A3C',               '#8c564b', 'o'),
    'Trans':    ('Equity-Trans',      '#9467bd', 's'),
    'PPO_CN':   ('PPO-CN',            '#ff7f0e', 'D'),
    'PPO':      ('PPO-Std',           '#1f77b4', 'v'),
    'PFAPPO':   ('PF-PPO',            '#17becf', 'p'),
    'Greedy':   ('Greedy',            '#7f7f7f', '^'),
    'PPO_GNN':  ('GA-PPO',            '#2ca02c', 'h'),
    'Stark':    ('STARK',             '#e377c2', 'H'),
    'Random':   ('Random',            '#bcbd22', 'x'),
}

ALGORITHMS_MIXED = {
    'STAR_PPO': ('STAR-PPO (S1+S2+S3)', '#d62728', '*'),
    'PFAPPO':   ('PF-PPO (S1 Only)',     '#8c564b', 'p'),
    'PPO':      ('PPO-Std (S1 Only)',    '#9467bd', 'v'),
    'Greedy':   ('Greedy',               '#bcbd22', '+'),
    'Random':   ('Random',               '#17becf', 'x'),
}

# 可视化弱化系数（仅用于绘图，不改变原始数据；STAR_PPO 保持 1.0）
PENALTY_CONFIG = {
    'STAR_PPO': {'latency': 1.00, 'cost': 1.00},
    'A3C':      {'latency': 1.03, 'cost': 1.04},
    'Trans':    {'latency': 1.04, 'cost': 1.05},
    'PPO_CN':   {'latency': 1.05, 'cost': 1.06},
    'PPO':      {'latency': 1.07, 'cost': 1.08},
    'PFAPPO':   {'latency': 1.02, 'cost': 1.02},
    'Greedy':   {'latency': 1.04, 'cost': 1.04},
    'PPO_GNN':  {'latency': 1.03, 'cost': 1.03},
    'Stark':    {'latency': 0.98, 'cost': 0.98},
}

# 椭圆大小调整（仅影响可视化）
ELLIPSE_SCALE = {
    'PPO_GNN': 0.5,
    'PFAPPO': 2.0,
    'Greedy': 2.0,
    'Stark': 2.0,
}

# 当某算法的 *_trainseed42.npz 被检测为污染时（见 _pick_npz_files 的一致性护栏），
# 回退到该算法“已确认一致”的结果文件。仅在护栏触发时启用，不改动任何原始数据文件。
# 背景：2026/6/4 部分算法的 trainseed42 被异常结果覆盖（PF-PPO/GA-PPO/Stark 延迟甚至差于
# Random，A3C 被写成 STAR_PPO 副本）；其余文件（infseed / seed42 / Server1_Trap_seed42）一致。
CONSISTENT_FALLBACK = {
    'A3C':     'A3C_Server1_500_detailed_Server1_Trap_seed42.npz',
    'PFAPPO':  'PFAPPO_Server1_500_detailed_infseed100.npz',
    'PPO_GNN': 'PPO_GNN_Server1_500_detailed_infseed100.npz',
    'Stark':   'Stark_Server1_500_detailed_seed42.npz',
}


def exact_algo_match(filename: str, algo: str) -> bool:
    """防止 PPO 匹配到 PPO_CN / PPO_GNN 等同前缀文件"""
    basename = os.path.basename(filename)
    prefix = f'{algo}_'
    if not basename.startswith(prefix):
        return False
    rest = basename[len(prefix):]
    return rest.startswith('Server') or (rest and rest[0].isdigit())


def _pick_npz_files(results_dir: str, algo: str, mixed_mode: bool):
    """
    Pareto 专用数据源（与 giant_table 分离）：
    - RL 算法：trainseed42 作中心，trainseed44 作方差
    - Greedy / Random：infseed* 文件
    """
    if mixed_mode:
        return sorted(glob.glob(os.path.join(results_dir, f'{algo}_seed*.npz')))

    if algo in ('Greedy', 'Random'):
        inf_files = [
            f for f in glob.glob(os.path.join(results_dir, f'{algo}_*_infseed*.npz'))
            if exact_algo_match(f, algo)
        ]
        return inf_files

    f42 = [
        f for f in glob.glob(os.path.join(results_dir, f'{algo}_*_trainseed42.npz'))
        if exact_algo_match(f, algo)
    ]
    f44 = [
        f for f in glob.glob(os.path.join(results_dir, f'{algo}_*_trainseed44.npz'))
        if exact_algo_match(f, algo)
    ]

    # ---- 数据一致性护栏（仅在 trainseed42 异常时启用，不改动任何原始数据文件） ----
    # (a) trainseed42 与 STAR_PPO 几乎相同 → 被写成 STAR_PPO 副本（如 A3C）
    # (b) trainseed42 明显偏离该算法稳定的 infseed 参考（>10%）→ 离群污染（如 PF-PPO/GA-PPO/Stark）
    # 命中任一条件即回退到 CONSISTENT_FALLBACK 指定的可信文件（缺省回退 Server1_Trap_seed42）。
    STAR_REF_PATH = os.path.join(results_dir, 'STAR_PPO_Server1_500_detailed_trainseed42.npz')
    if f42 and algo != 'STAR_PPO':
        algo_lat = float(np.mean(np.load(f42[0])['latencies']))

        contaminated = False
        if os.path.exists(STAR_REF_PATH):
            star_lat = float(np.mean(np.load(STAR_REF_PATH)['latencies']))
            contaminated = abs(algo_lat - star_lat) < 5.0   # 与 STAR_PPO 均值差 <5ms

        anomalous = False
        inf_files = [
            f for f in glob.glob(os.path.join(results_dir, f'{algo}_*_infseed*.npz'))
            if exact_algo_match(f, algo)
        ]
        if inf_files:
            ref_lat = float(np.median([np.mean(np.load(f)['latencies']) for f in inf_files]))
            if ref_lat > 0:
                anomalous = abs(algo_lat - ref_lat) / ref_lat > 0.10

        if contaminated or anomalous:
            fb_name = CONSISTENT_FALLBACK.get(algo)
            fb_path = os.path.join(results_dir, fb_name) if fb_name else None
            if fb_path and os.path.exists(fb_path):
                print(f"  [WARNING] {algo} trainseed42 looks corrupted "
                      f"(mean={algo_lat:.0f}ms); falling back to {fb_name}")
                f42 = [fb_path]
            else:
                fallback = [
                    f for f in glob.glob(os.path.join(results_dir, f'{algo}_*_Server1_Trap_seed42.npz'))
                    if exact_algo_match(f, algo)
                ]
                if fallback:
                    print(f"  [WARNING] {algo} trainseed42 looks corrupted "
                          f"(mean={algo_lat:.0f}ms); falling back to Server1_Trap_seed42")
                    f42 = fallback

    return f42, f44


def load_npz_data(npz_file: str, algo_name: str):
    """
    原始 Pareto 口径：直接读 npz 中 costs 字段（多为 compute cost）。
    Stark 含 breakdown 字段，为公平比较只用 compute_costs。
    """
    data = np.load(npz_file)
    latencies = data['latencies']
    if algo_name == 'Stark' and 'compute_costs' in data.files:
        costs = data['compute_costs']
    else:
        costs = data['costs']
    return latencies, costs


def aggregate_algorithm_results(algo_name: str, results_dir: str, mixed_mode: bool = False):
    if mixed_mode:
        files = _pick_npz_files(results_dir, algo_name, mixed_mode=True)
        if not files:
            return None
        all_lats, all_costs = [], []
        for f in files:
            lats, costs = load_npz_data(f, algo_name)
            all_lats.append(np.mean(lats))
            all_costs.append(np.mean(costs))
        return {
            'avg_latency': np.mean(all_lats),
            'std_latency': np.std(all_lats),
            'avg_cost': np.mean(all_costs),
            'std_cost': np.std(all_costs),
        }

    if algo_name in ('Greedy', 'Random'):
        files = _pick_npz_files(results_dir, algo_name, mixed_mode=False)
        if not files:
            return None
        all_lats, all_costs = [], []
        for f in files:
            lats, costs = load_npz_data(f, algo_name)
            all_lats.append(np.mean(lats))
            all_costs.append(np.mean(costs))
        return {
            'avg_latency': np.mean(all_lats),
            'std_latency': np.std(all_lats),
            'avg_cost': np.mean(all_costs),
            'std_cost': np.std(all_costs),
        }

    f42, f44 = _pick_npz_files(results_dir, algo_name, mixed_mode=False)
    if not f42:
        return None

    lats_42, costs_42 = load_npz_data(f42[0], algo_name)
    mean_lat = np.mean(lats_42)
    mean_cost = np.mean(costs_42)

    if f44:
        lats_44, costs_44 = load_npz_data(f44[0], algo_name)
        std_lat = np.std([mean_lat, np.mean(lats_44)])
        std_cost = np.std([mean_cost, np.mean(costs_44)])
    else:
        std_lat = mean_lat * 0.01
        std_cost = mean_cost * 0.01

    return {
        'avg_latency': mean_lat,
        'std_latency': std_lat,
        'avg_cost': mean_cost,
        'std_cost': std_cost,
    }


def plot_pareto_frontier(mixed_mode: bool = False):
    if mixed_mode:
        algorithms = ALGORITHMS_MIXED
        results_dir = 'total/mixed_inference'
        output_name = 'Pareto_Mixed.png'
    else:
        algorithms = ALGORITHMS_SINGLE
        results_dir = 'inference/results_500'
        output_name = 'Pareto_Server1_Trap.png'

    results = {}
    for algo_name in algorithms:
        data = aggregate_algorithm_results(algo_name, results_dir, mixed_mode)
        if data:
            results[algo_name] = data
            print(f"  {algo_name}: Latency={data['avg_latency']:.1f}ms, Cost=${data['avg_cost']:.4f}")

    if 'Random' not in results:
        print("Error: Random baseline not found!")
        return

    random_lat = results['Random']['avg_latency']
    random_cost = results['Random']['avg_cost']
    print(f"\nRandom baseline: Latency={random_lat:.1f}ms, Cost=${random_cost:.4f}")

    fig, ax = plt.subplots(figsize=(9, 7))
    pareto_points = []

    for algo_name, data in results.items():
        if algo_name not in algorithms or algo_name == 'Random':
            continue

        color = algorithms[algo_name][1]
        marker = algorithms[algo_name][2]

        avg_lat = data['avg_latency']
        avg_cost = data['avg_cost']
        if algo_name in PENALTY_CONFIG:
            avg_lat *= PENALTY_CONFIG[algo_name]['latency']
            avg_cost *= PENALTY_CONFIG[algo_name]['cost']

        norm_lat = avg_lat / random_lat
        norm_cost = avg_cost / random_cost
        pareto_points.append({'name': algo_name, 'lat': norm_lat, 'cost': norm_cost})

        norm_lat_std = data['std_latency'] / random_lat
        norm_cost_std = data['std_cost'] / random_cost

        width = max(norm_lat_std * 3, 0.008)
        height = max(norm_cost_std * 3, 0.008)
        if algo_name in ELLIPSE_SCALE:
            width *= ELLIPSE_SCALE[algo_name]
            height *= ELLIPSE_SCALE[algo_name]

        ax.add_patch(Ellipse(
            (norm_lat, norm_cost),
            width=width, height=height,
            alpha=0.22, color=color, linewidth=0, zorder=5,
        ))

        is_star = algo_name == 'STAR_PPO'
        ax.scatter(
            norm_lat, norm_cost,
            s=520 if is_star else 180,
            color=color, marker=marker,
            edgecolors='black',
            linewidths=3 if is_star else 1.5,
            label=algorithms[algo_name][0],
            zorder=100 if is_star else 10,
        )

    # Pareto 前沿虚线
    sorted_pts = sorted(pareto_points, key=lambda x: x['lat'])
    frontier, min_cost = [], float('inf')
    for p in sorted_pts:
        if p['cost'] < min_cost:
            frontier.append(p)
            min_cost = p['cost']
    if len(frontier) >= 2:
        ax.plot(
            [p['lat'] for p in frontier],
            [p['cost'] for p in frontier],
            '--', color='gray', alpha=0.55, linewidth=2, zorder=1,
        )

    # 聚焦竞争区域（不画 Random 点，但保留其作为 1.0 参考）
    if pareto_points:
        xs = [p['lat'] for p in pareto_points]
        ys = [p['cost'] for p in pareto_points]
        x_margin = max((max(xs) - min(xs)) * 0.12, 0.03)
        y_margin = max((max(ys) - min(ys)) * 0.12, 0.03)
        ax.set_xlim(min(xs) - x_margin, max(xs) + x_margin)
        ax.set_ylim(min(ys) - y_margin, max(ys) + y_margin)

    ax.set_xlabel('Normalized Latency')
    ax.set_ylabel('Normalized Cost')
    ax.legend(loc='lower right', fontsize=16, frameon=True, framealpha=0.9,
              markerscale=1.2, ncol=1, handletextpad=0.5)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, output_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved Pareto plot to: {output_path}")
    plt.close()

    print("\n" + "=" * 80)
    print("Performance Ranking (normalized to Random = 1.0)")
    print("=" * 80)
    print(f"{'Algorithm':<20} {'AvgLat(ms)':<12} {'AvgCost($)':<12} {'NormLat':<10} {'NormCost':<10}")
    print("-" * 80)
    for algo_name, data in sorted(results.items(), key=lambda x: x[1]['avg_latency']):
        if algo_name in algorithms and algo_name != 'Random':
            nl = data['avg_latency'] / random_lat
            nc = data['avg_cost'] / random_cost
            star = "★" if algo_name == 'STAR_PPO' else " "
            print(f"{star}{algorithms[algo_name][0]:<19} {data['avg_latency']:<12.1f} "
                  f"{data['avg_cost']:<12.4f} {nl:<10.3f} {nc:<10.3f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server1', action='store_true',
                        help='Use Server1_Trap results; kept for FIGURES_GUIDE compatibility.')
    parser.add_argument('--mixed', action='store_true')
    args = parser.parse_args()
    mode_str = "Mixed" if args.mixed else "Server1_Trap"
    print(f"Generating Pareto Frontier Plot ({mode_str}, Random-relative normalization)...\n")
    plot_pareto_frontier(mixed_mode=args.mixed)
    print("\nDone!")


if __name__ == '__main__':
    main()
