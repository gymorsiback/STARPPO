"""
成本分解堆叠柱状图
使用混合推理结果展示 ComputeCost、NetworkCost、CommunicationCost
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import glob

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 配置matplotlib样式 - 论文格式
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 24,
    'axes.labelsize': 26,
    'axes.titlesize': 28,
    'axes.titleweight': 'normal',  # 标题不加粗
    'axes.linewidth': 2.0,
    'axes.edgecolor': 'black',
    'xtick.labelsize': 20,
    'ytick.labelsize': 22,
    'legend.fontsize': 20,
    'ytick.color': 'black',
    'xtick.color': 'black',
    'axes.labelcolor': 'black',
    'mathtext.fontset': 'stix',
})

# 算法配置
ALGORITHMS = {
    'STAR_PPO': {'display': 'STAR-PPO', 'color': '#d62728'},
    'PFAPPO': {'display': 'PF-PPO', 'color': '#17becf'},
    'PPO': {'display': 'PPO-Std', 'color': '#1f77b4'},
    'PPO_CN': {'display': 'PPO-CN', 'color': '#ff7f0e'},
    'PPO_GNN': {'display': 'GA-PPO', 'color': '#2ca02c'},
    'Trans': {'display': 'Trans', 'color': '#9467bd'},
    'A3C': {'display': 'A3C', 'color': '#8c564b'},
    'Stark': {'display': 'STARK', 'color': '#e377c2'},
    'Greedy': {'display': 'Greedy', 'color': '#7f7f7f'},
    'Random': {'display': 'Random', 'color': '#bcbd22'},
}

def exact_algo_match_plot(filename, algo):
    """防止 PPO 匹配到 PPO_CN 等前缀相同的算法"""
    basename = os.path.basename(filename)
    prefix = f'{algo}_'
    if not basename.startswith(prefix):
        return False
    # 检查：PPO_ 后的内容不能是 CN_, GNN_, 等（即不能是另一个算法的首字母段）
    rest = basename[len(prefix):]
    for other in ['CN_', 'GNN_', 'CN.', 'GNN.']:
        if algo == 'PPO' and rest.startswith(other):
            return False
    return True


def load_cost_data(results_dir='total/mixed_inference', use_server1=False):
    """加载所有算法的成本数据

    对于 Server1_Trap:
    - Compute Cost: 从原始 _Server1_Trap_seed42.npz 读取（和 giant_table 一致）
    - Network/Comm Cost: 从 _cost_breakdown_seed42.npz 读取
    """
    data = {}

    if use_server1:
        results_dir = 'inference/results_500'

    for algo_name in ALGORITHMS.keys():
        if use_server1:
            # 原始文件：优先 trainseed42，其次 Server1_Trap_seed42，其次 cost_breakdown_seed42
            # 优先 Server1_Trap_seed42（保证每个算法独立跑的结果），
            # 其次 trainseed42（历史上部分算法存在数据污染风险），
            # 最后 cost_breakdown_seed42
            candidates = (
                glob.glob(os.path.join(results_dir, f'{algo_name}_*_Server1_Trap_seed42.npz')) +
                glob.glob(os.path.join(results_dir, f'{algo_name}_*_trainseed42.npz')) +
                glob.glob(os.path.join(results_dir, f'{algo_name}_*_cost_breakdown_seed42.npz'))
            )
            orig_files = [f for f in candidates if 'cost_breakdown_final' not in f and exact_algo_match_plot(f, algo_name)]
            # 成本分解文件（Network/Comm Cost）
            breakdown_files = [f for f in glob.glob(os.path.join(results_dir, f'{algo_name}_*_cost_breakdown_final.npz'))
                               if exact_algo_match_plot(f, algo_name)]

            compute_cost = 0.0
            network_cost = 0.0
            comm_cost = 0.0

            # 从原始文件读取成本（优先使用细分字段，兼容新旧 npz 格式）
            if orig_files:
                npz = np.load(orig_files[0])
                if 'compute_costs' in npz:
                    compute_cost = np.mean(npz['compute_costs'])
                    # 若新版 npz 已包含 network/comm，直接使用
                    if 'network_costs' in npz:
                        network_cost = np.mean(npz['network_costs'])
                    if 'communication_costs' in npz:
                        comm_cost = np.mean(npz['communication_costs'])
                else:
                    # 旧版 npz：'costs' 是 compute_costs
                    compute_cost = np.mean(npz['costs'])

            # 从 cost_breakdown 文件读取 Network/Comm Cost
            if breakdown_files:
                npz = np.load(breakdown_files[0])
                network_cost = np.mean(npz['network_costs'])
                comm_cost = np.mean(npz['communication_costs'])

            if compute_cost > 0:
                data[algo_name] = {
                    'compute': compute_cost,
                    'network': network_cost,
                    'communication': comm_cost,
                    'total': compute_cost + network_cost + comm_cost,
                }
        else:
            files = glob.glob(os.path.join(results_dir, f'{algo_name}_seed*.npz'))
            if not files:
                print(f"Warning: No files found for {algo_name}")
                continue

            all_costs = []
            all_compute = []
            all_network = []
            all_communication = []

            for f in files:
                npz = np.load(f)
                if 'costs' in npz:
                    all_costs.extend(npz['costs'])
                if 'compute_costs' in npz:
                    all_compute.extend(npz['compute_costs'])
                if 'network_costs' in npz:
                    all_network.extend(npz['network_costs'])
                if 'communication_costs' in npz:
                    all_communication.extend(npz['communication_costs'])

            if all_costs:
                data[algo_name] = {
                    'total': np.mean(all_costs),
                    'compute': np.mean(all_compute) if all_compute else np.mean(all_costs),
                    'network': np.mean(all_network) if all_network else 0.0,
                    'communication': np.mean(all_communication) if all_communication else 0.0,
                }

    return data

def plot_cost_breakdown(data, output_path=None, use_server1=False):
    """生成成本分解堆叠柱状图"""

    # 按总成本排序
    sorted_algos = sorted(data.keys(), key=lambda x: data[x]['total'])

    # 准备数据
    labels = [ALGORITHMS[a]['display'] for a in sorted_algos]
    compute_costs = [data[a]['compute'] for a in sorted_algos]
    network_costs = [data[a]['network'] for a in sorted_algos]
    communication_costs = [data[a]['communication'] for a in sorted_algos]
    total_costs = [data[a]['total'] for a in sorted_algos]

    # 检查是否有三部分成本
    has_breakdown = any(network_costs) and any(communication_costs)

    # 创建图表 - 长方形，适合柱状图
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(labels))
    width = 0.6

    if has_breakdown:
        # 堆叠柱状图 - 三种对比色：浅蓝、浅绿、灰棕
        bars1 = ax.bar(x, compute_costs, width, label='Compute Cost', color='#a6cee3', edgecolor='black', linewidth=1.0, hatch='/')
        bars2 = ax.bar(x, network_costs, width, bottom=compute_costs, label='Network Cost', color='#b2df8a', edgecolor='black', linewidth=1.0, hatch='...')
        bottom2 = [c + n for c, n in zip(compute_costs, network_costs)]
        bars3 = ax.bar(x, communication_costs, width, bottom=bottom2, label='Communication Cost', color='#a89078', edgecolor='black', linewidth=1.0, hatch='xx')

        # 添加总成本标签
        for i, total in enumerate(total_costs):
            ax.text(i, total + 0.003, f'${total:.4f}', ha='center', va='bottom',
                    fontsize=16)

        ax.legend(loc='upper left', fontsize=20)
        title = 'Cost Breakdown: Compute vs Network vs Communication'
    else:
        # 普通柱状图
        colors = [ALGORITHMS[a]['color'] for a in sorted_algos]
        bars = ax.bar(x, total_costs, width, color=colors, edgecolor='white', linewidth=0.5)

        for i, cost in enumerate(total_costs):
            ax.text(i, cost + 0.005, f'${cost:.4f}', ha='center', va='bottom',
                    fontsize=16)

        title = 'Cost Comparison'

    # 设置标签和标题
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Average Cost per Request')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=14)  # 横着写，字体稍小避免重叠

    # 网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # 确保边框统一
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_edgecolor('black')

    # 设置 Y 轴范围
    max_cost = max(total_costs)
    ax.set_ylim(0, max_cost * 1.25)

    plt.tight_layout()
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, 'Cost_Breakdown.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存: {output_path}")
    plt.close()

    # 打印统计信息
    print("\n=== 成本分解统计 ===")
    print(f"{'Algorithm':<12} {'Compute':>10} {'Network':>10} {'Comm':>10} {'Total':>10}")
    print("-" * 55)
    for algo in sorted_algos:
        c = data[algo]['compute']
        n = data[algo]['network']
        m = data[algo]['communication']
        t = data[algo]['total']
        print(f"{ALGORITHMS[algo]['display']:<12} ${c:>8.4f} ${n:>8.4f} ${m:>8.4f} ${t:>8.4f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--server1', action='store_true',
                        help='使用 Server1_Trap 推理结果 (inference/results_500/)')
    args = parser.parse_args()

    data = load_cost_data(use_server1=args.server1)
    if data:
        output_name = 'Cost_Breakdown_Server1_Trap.png' if args.server1 else 'Cost_Breakdown.png'
        output_path = os.path.join(OUTPUT_DIR, output_name)
        plot_cost_breakdown(data, output_path=output_path, use_server1=args.server1)
    else:
        print("Error: No cost data found!")
