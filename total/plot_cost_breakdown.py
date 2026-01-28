
import os
import numpy as np
import matplotlib.pyplot as plt
import glob

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 24,
    'axes.labelsize': 26,
    'axes.titlesize': 28,
    'axes.titleweight': 'normal',  
    'xtick.labelsize': 20,
    'ytick.labelsize': 22,
    'legend.fontsize': 20,
    'ytick.color': 'black',
    'xtick.color': 'black',
    'axes.labelcolor': 'black',
    'mathtext.fontset': 'stix',
})

ALGORITHMS = {
    'STAR_PPO': {'display': 'STARPPO', 'color': '#d62728'},
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

def load_cost_data(results_dir='total/mixed_inference', use_server1=False):
    data = {}
    
    if use_server1:
        results_dir = 'inference/results_500'
    
    for algo_name in ALGORITHMS.keys():
        if use_server1:
            orig_files = glob.glob(os.path.join(results_dir, f'{algo_name}_*_Server1_Trap_seed42.npz'))
            breakdown_files = glob.glob(os.path.join(results_dir, f'{algo_name}_*_cost_breakdown_final.npz'))
            
            compute_cost = 0.0
            network_cost = 0.0
            comm_cost = 0.0

            if orig_files:
                npz = np.load(orig_files[0])
                compute_cost = np.mean(npz['costs'])

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

def plot_cost_breakdown(data, output_path='total/Cost_Breakdown.png', use_server1=False):
    sorted_algos = sorted(data.keys(), key=lambda x: data[x]['total'])

    labels = [ALGORITHMS[a]['display'] for a in sorted_algos]
    compute_costs = [data[a]['compute'] for a in sorted_algos]
    network_costs = [data[a]['network'] for a in sorted_algos]
    communication_costs = [data[a]['communication'] for a in sorted_algos]
    total_costs = [data[a]['total'] for a in sorted_algos]
 
    has_breakdown = any(network_costs) and any(communication_costs)
  
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(labels))
    width = 0.6
    
    if has_breakdown:
        bars1 = ax.bar(x, compute_costs, width, label='Compute Cost', color='#a6cee3', edgecolor='black', linewidth=1.0, hatch='/')
        bars2 = ax.bar(x, network_costs, width, bottom=compute_costs, label='Network Cost', color='#b2df8a', edgecolor='black', linewidth=1.0, hatch='...')
        bottom2 = [c + n for c, n in zip(compute_costs, network_costs)]
        bars3 = ax.bar(x, communication_costs, width, bottom=bottom2, label='Communication Cost', color='#a89078', edgecolor='black', linewidth=1.0, hatch='xx')
 
        for i, total in enumerate(total_costs):
            ax.text(i, total + 0.003, f'${total:.4f}', ha='center', va='bottom', 
                    fontsize=16)
        
        ax.legend(loc='upper left', fontsize=20)
        title = 'Cost Breakdown: Compute vs Network vs Communication'
    else:
        colors = [ALGORITHMS[a]['color'] for a in sorted_algos]
        bars = ax.bar(x, total_costs, width, color=colors, edgecolor='white', linewidth=0.5)
        
        for i, cost in enumerate(total_costs):
            ax.text(i, cost + 0.005, f'${cost:.4f}', ha='center', va='bottom', 
                    fontsize=16)
        
        title = 'Cost Comparison'

    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Average Cost per Request')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=16)  
  
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
 
    max_cost = max(total_costs)
    ax.set_ylim(0, max_cost * 1.25)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存: {output_path}")
    plt.close()
 
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
        output_path = 'total/Cost_Breakdown_Server1_Trap.png' if args.server1 else 'total/Cost_Breakdown.png'
        plot_cost_breakdown(data, output_path=output_path, use_server1=args.server1)
    else:
        print("Error: No cost data found!")
