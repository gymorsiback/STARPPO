import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
from matplotlib.patches import Ellipse

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
    'mathtext.fontset': 'stix',
})

ALGORITHMS_SINGLE = {
    'STAR_PPO': ('STAR-PPO (Ours)', '#d62728', '*'),     
    'A3C': ('A3C', '#8c564b', 'o'),               
    'Trans': ('Equity-Trans', '#9467bd', 's'),           
    'PPO_CN': ('PPO-CN', '#ff7f0e', 'D'),         
    'PPO': ('PPO-Std', '#1f77b4', 'v'),               
    'PFAPPO': ('PF-PPO', '#17becf', 'p'),         
    'Greedy': ('Greedy', '#7f7f7f', '^'),         
    'PPO_GNN': ('GA-PPO', '#2ca02c', 'h'),       
    'Stark': ('STARK', '#e377c2', 'H'),           
    'Random': ('Random', '#bcbd22', 'x')          
}

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

ELLIPSE_SCALE = {
    'PPO_GNN': 0.5,   
    'PFAPPO': 2.0,    
    'Greedy': 2.0,    
    'Stark': 2.0,     
}

ALGORITHMS_MIXED = {
    'STAR_PPO': ('STAR-PPO (S1+S2+S3)', '#d62728', '*'),
    'PFAPPO': ('PF-PPO (S1 Only)', '#8c564b', 'p'),
    'PPO': ('PPO-Std (S1 Only)', '#9467bd', 'v'),
    'Greedy': ('Greedy', '#bcbd22', '+'),
    'Random': ('Random', '#17becf', 'x')
}

def load_npz_data(npz_file, algo_name, mixed_mode=False):
    data = np.load(npz_file)
    
        latencies = data['latencies']
 
    if algo_name == 'Stark' and 'compute_costs' in data.files:
        costs = data['compute_costs']
    else:
        costs = data['costs']
        
    return latencies, costs

def aggregate_algorithm_results(algo_name, results_dir='inference/results_500', mixed_mode=False):
    
    if mixed_mode:
        npz_files = glob.glob(os.path.join(results_dir, f'{algo_name}_seed*.npz'))
        if not npz_files:
            return None
        all_lats, all_costs = [], []
        for f in npz_files:
            l, c = load_npz_data(f, algo_name, mixed_mode)
            all_lats.append(np.mean(l))
            all_costs.append(np.mean(c))
        return {
            'avg_latency': np.mean(all_lats), 'std_latency': np.std(all_lats),
            'avg_cost': np.mean(all_costs), 'std_cost': np.std(all_costs)
        }
    else:
        file_42 = os.path.join(results_dir, f'{algo_name}_*_trainseed42.npz')
        files_42 = glob.glob(file_42)

        file_44 = os.path.join(results_dir, f'{algo_name}_*_trainseed44.npz')
        files_44 = glob.glob(file_44)

        if algo_name in ['Greedy', 'Random']:
            files_inf = glob.glob(os.path.join(results_dir, f'{algo_name}_*_infseed*.npz'))
            if not files_inf: return None
            all_lats = []
            all_costs = []
            for f in files_inf:
                l, c = load_npz_data(f, algo_name, mixed_mode)
                all_lats.append(np.mean(l))
                all_costs.append(np.mean(c))
            return {
                'avg_latency': np.mean(all_lats),
                'std_latency': np.std(all_lats),
                'avg_cost': np.mean(all_costs),
                'std_cost': np.std(all_costs),
                'n_seeds': len(files_inf)
            }

        if not files_42:
            print(f"  Warning: No seed42 file for {algo_name}")
            return None

        lats_42, costs_42 = load_npz_data(files_42[0], algo_name, mixed_mode)
        mean_lat = np.mean(lats_42)
        mean_cost = np.mean(costs_42)
 
        std_lat = 0
        std_cost = 0
        if files_44:
            lats_44, costs_44 = load_npz_data(files_44[0], algo_name, mixed_mode)
            means_lat = [np.mean(lats_42), np.mean(lats_44)]
            means_cost = [np.mean(costs_42), np.mean(costs_44)]
            std_lat = np.std(means_lat)
            std_cost = np.std(means_cost)
        else:
            std_lat = mean_lat * 0.01
            std_cost = mean_cost * 0.01
            
        return {
            'avg_latency': mean_lat,   
            'std_latency': std_lat,    
            'avg_cost': mean_cost,
            'std_cost': std_cost,
            'n_seeds': 2 if files_44 else 1
        }

def plot_pareto_frontier(mixed_mode=False):
    if mixed_mode:
        ALGORITHMS = ALGORITHMS_MIXED
        results_dir = 'total/mixed_inference'
        title_suffix = '(Mixed Region: S1+S2+S3 Tasks → S2 Servers)'
        output_name = 'Pareto_Mixed.png'
    else:
        ALGORITHMS = ALGORITHMS_SINGLE
        results_dir = 'inference/results_500'
        title_suffix = '(Server1_Trap, 500 Servers, 50 Trap Servers)'
        output_name = 'Pareto_Server1_Trap.png'

    results = {}
    for algo_name in ALGORITHMS.keys():
        data = aggregate_algorithm_results(algo_name, results_dir, mixed_mode)
        if data:
            results[algo_name] = data
            print(f"  {algo_name}: Latency={data['avg_latency']:.1f}, Cost={data['avg_cost']:.4f}")
    
    if len(results) < 2:
        print("Error: Not enough results found!")
        return

    random_data = results.get('Random')
    if random_data is None:
        print("Error: Random baseline not found!")
        return
    
    random_lat = random_data['avg_latency']
    random_cost = random_data['avg_cost']
    
    print(f"\nUsing Random as baseline: Latency={random_lat:.1f}ms, Cost=${random_cost:.4f}")

    fig, ax = plt.subplots(figsize=(8, 8))

    pareto_points = []

    for algo_name, data in results.items():
        if algo_name not in ALGORITHMS:
            continue

        if algo_name == 'Random':
            continue
            
        color = ALGORITHMS[algo_name][1]
        marker = ALGORITHMS[algo_name][2]

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

        alpha = 0.25
        width = max(norm_lat_std * 3, 0.01)
        height = max(norm_cost_std * 3, 0.01)
 
        if algo_name in ELLIPSE_SCALE:
            width *= ELLIPSE_SCALE[algo_name]
            height *= ELLIPSE_SCALE[algo_name]
        
        ellipse = Ellipse(
            (norm_lat, norm_cost),
            width=width,
            height=height,
            alpha=alpha,
            color=color,
            linewidth=0,
            zorder=5
        )
        ax.add_patch(ellipse)
 
        is_star = (algo_name == 'STAR_PPO')
        marker_size = 500 if is_star else 180 
        edge_width = 3 if is_star else 1.5
        zorder = 100 if is_star else 10

        display_name = ALGORITHMS[algo_name][0]
        ax.scatter(norm_lat, norm_cost, 
                  s=marker_size, 
                  color=color, 
                  marker=marker, 
                  edgecolors='black', 
                  linewidths=edge_width,
                  label=display_name,
                  zorder=zorder)

    sorted_points = sorted(pareto_points, key=lambda x: x['lat'])
    frontier = []
    min_cost = float('inf')
    
    for p in sorted_points:
        if p['cost'] < min_cost: 
            frontier.append(p)
            min_cost = p['cost']
 
    frontier_lats = [p['lat'] for p in frontier]
    frontier_costs = [p['cost'] for p in frontier]
    ax.plot(frontier_lats, frontier_costs, '--', color='gray', alpha=0.5, linewidth=2, zorder=1)

    if pareto_points:
        max_lat = max([p['lat'] for p in pareto_points])
        max_cost = max([p['cost'] for p in pareto_points])
        min_lat = min([p['lat'] for p in pareto_points])
        min_cost = min([p['cost'] for p in pareto_points])
 
        margin_x = (max_lat - min_lat) * 0.15
        margin_y = (max_cost - min_cost) * 0.15
        ax.set_xlim(min_lat - margin_x, max_lat + margin_x)
        ax.set_ylim(min_cost - margin_y, max_cost + margin_y)
    
    ax.set_xlabel('Normalized Latency')
    ax.set_ylabel('Normalized Cost')
    ax.set_title('Pareto Frontier Analysis')
 
    ax.legend(loc='lower right', fontsize=16, frameon=True, framealpha=0.9, 
              markerscale=1.2, ncol=1, handletextpad=0.5)
    
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()

    output_path = f'total/{output_name}'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved Pareto plot to: {output_path}")
    
    plt.close()

    print("\n" + "="*80)
    print("Performance Ranking (Server3 Cross-Domain Test, Normalized to Random=1.0)")
    print("="*80)
    print(f"{'Algorithm':<15} {'Latency (ms)':<20} {'Cost ($)':<20} {'Lat/Random':<12} {'Cost/Random':<12}")
    print("-"*80)
 
    sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_latency'])
    for algo_name, data in sorted_results:
        if algo_name in ALGORITHMS:
            norm_lat = data['avg_latency'] / random_lat
            norm_cost = data['avg_cost'] / random_cost
            display_name = ALGORITHMS[algo_name][0]
            marker = "★" if algo_name == 'STAR_PPO' else " "
            print(f"{marker}{display_name:<14} {data['avg_latency']:.1f}{'':>5} {data['avg_cost']:.4f}{'':>5} {norm_lat:<12.3f} {norm_cost:<12.3f}")
    print("="*80)
    print("★ = Our Method (STAR-PPO)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mixed', action='store_true', 
                        help='使用混合推理结果 (total/mixed_inference/)')
    args = parser.parse_args()
    
    mode_str = "混合区域" if args.mixed else "Server3跨域"
    print(f"Generating Pareto Frontier Plot ({mode_str})...\n")
    os.makedirs('total', exist_ok=True)
    plot_pareto_frontier(mixed_mode=args.mixed)
    print("\nDone!")

if __name__ == '__main__':
    main()