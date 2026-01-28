
import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
    'ytick.color': 'black',
    'xtick.color': 'black',
    'axes.labelcolor': 'black',
    'mathtext.fontset': 'stix',
})

ALGORITHMS_SINGLE = {
    'STAR_PPO': ('STAR-PPO', '#d62728', '-', 5.5),  
    'PFAPPO': ('PF-PPO', '#17becf', '-', 4.0),
    'PPO': ('PPO-Std', '#1f77b4', '--', 4.0),
    'PPO_CN': ('PPO-CN', '#ff7f0e', '-.', 4.0),
    'PPO_GNN': ('GA-PPO', '#2ca02c', ':', 4.0),
    'Trans': ('Equity-Trans', '#9467bd', '--', 4.0),
    'A3C': ('A3C', '#8c564b', '-.', 4.0),
    'Stark': ('STARK', '#e377c2', '-', 4.0),
    'Greedy': ('Greedy', '#7f7f7f', '--', 4.0),
    'Random': ('Random', '#bcbd22', ':', 3.5),
}

ALGORITHMS_MIXED = {
    'PFAPPO': ('PF-PPO', '#E74C3C', '-', 2.5),
    'PPO': ('PPO-Std', '#3498DB', '--', 1.5),
    'PPO_CN': ('PPO-CN', '#2ECC71', '-.', 1.5),
    'PPO_GNN': ('GA-PPO', '#9B59B6', ':', 1.5),
    'Trans': ('Equity-Trans', '#F39C12', '--', 1.5),
    'A3C': ('A3C', '#1ABC9C', '-.', 1.5),
    'Stark': ('STARK', '#E91E63', ':', 1.5),
    'Greedy': ('Greedy', '#7F8C8D', '-', 1.5),
    'Random': ('Random', '#95A5A6', '-', 1.0),
}

def load_npz_data(npz_file, algo_name, mixed_mode=False):
    data = np.load(npz_file)
    
    if not mixed_mode and algo_name == 'PPO_algorithm':
        latencies = data.get('lat_pp', data.get('latencies', []))
    else:
        latencies = data.get('latencies', [])
    
    return latencies

def load_all_data(mixed_mode=False, use_server1=False):
    if mixed_mode:
        results_dir = 'total/mixed_inference'
        ALGORITHMS = ALGORITHMS_MIXED
    elif use_server1:
        results_dir = 'inference/results_500'  
        ALGORITHMS = ALGORITHMS_SINGLE
    else:
        results_dir = 'inference/results'
        ALGORITHMS = ALGORITHMS_SINGLE
    
    all_latencies = {}
    
    for algo_name in ALGORITHMS.keys():
        if mixed_mode:
            npz_files = glob.glob(os.path.join(results_dir, f'{algo_name}_seed*.npz'))
        else:
            all_files = glob.glob(os.path.join(results_dir, f'{algo_name}_*.npz'))
            npz_files = [f for f in all_files if 'detailed' in f and 'cost_breakdown' not in f and 'workflow' not in f]
            
        if not npz_files:
            print(f"Warning: No files found for {algo_name}")
            continue

        lats = []
        for f in npz_files:
            try:
                lats.extend(load_npz_data(f, algo_name, mixed_mode))
            except Exception as e:
                print(f"Error loading {f}: {e}")
                
        if lats:
            all_latencies[algo_name] = np.array(lats)
            print(f"Loaded {algo_name}: {len(lats)} samples")
            
    return all_latencies, ALGORITHMS

def plot_cdf(data, ALGORITHMS, mixed_mode=False, use_server1=False):
    fig, ax = plt.subplots(figsize=(8, 8))
 
    if use_server1:
        plot_order = ['Random', 'Greedy', 'A3C', 'Stark', 'Trans', 
                      'PPO', 'PPO_CN', 'PPO_GNN', 'PFAPPO', 'STAR_PPO']
    elif mixed_mode:
        plot_order = ['Random', 'Greedy', 'A3C', 'Stark', 'Trans', 
                      'PPO', 'PPO_CN', 'PPO_GNN', 'PFAPPO', 'STAR_PPO']
    else:
        plot_order = ['Random', 'Greedy', 'A3C_algorithm', 'Stark_Scheduler', 
                      'Trans', 'PPO_CN', 'PPO_algorithm', 'PPO_GNN', 'PFAPPO', 'STAR_PPO']
    
    for algo_name in plot_order:
        if algo_name not in data or algo_name not in ALGORITHMS:
            continue
            
        latencies = data[algo_name]
        display_name, color, linestyle, linewidth = ALGORITHMS[algo_name]
 
        sorted_data = np.sort(latencies)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)

        if algo_name == 'STAR_PPO':
            ax.plot(sorted_data, yvals, label=display_name, 
                    color=color, linestyle='-', linewidth=3.0, zorder=10)
        else:
            ax.plot(sorted_data, yvals, label=display_name, 
                    color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.8)
 
    ax.axvline(x=3000, color='black', linestyle='--', alpha=0.7, linewidth=3)
    ax.text(3100, 0.15, 'SLA (3000ms)', rotation=90, verticalalignment='bottom', 
            fontsize=24)

    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=2)
    ax.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5, linewidth=2)
    ax.axhline(y=0.99, color='gray', linestyle=':', alpha=0.5, linewidth=2)
    ax.text(150, 0.51, 'P50', fontsize=22, color='gray')
    ax.text(150, 0.91, 'P90', fontsize=22, color='gray')
    ax.text(150, 0.995, 'P99', fontsize=22, color='gray')
 
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Latency CDF Comparison')
  
    if use_server1:
        ax.set_xlim(0, 6000)  
    else:
        ax.set_xlim(0, 4000)
    ax.set_ylim(0, 1.02)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1000 if use_server1 else 500))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
 
    ax.grid(True, alpha=0.3, linestyle='--')

    handles, labels = ax.get_legend_handles_labels()
    star_idx = None
    for i, label in enumerate(labels):
        if 'STAR-PPO' in label:
            star_idx = i
            break
    if star_idx is not None:
        handles = [handles[star_idx]] + handles[:star_idx] + handles[star_idx+1:]
        labels = [labels[star_idx]] + labels[:star_idx] + labels[star_idx+1:]
    ax.legend(handles, labels, loc='lower right', fontsize=18, ncol=1, framealpha=0.8)
    
    plt.tight_layout()

    if use_server1:
        output_name = 'Latency_CDF_Server1_Trap.png'
    elif mixed_mode:
        output_name = 'Latency_CDF_Mixed.png'
    else:
        output_name = 'Latency_CDF_Server3.png'
    output_path = f'total/{output_name}'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved CDF plot to: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mixed', action='store_true', 
                        help='使用混合推理结果 (total/mixed_inference/)')
    parser.add_argument('--server1', action='store_true',
                        help='使用 Server1_Trap 推理结果 (inference/results_500/)')
    args = parser.parse_args()
    
    if args.server1:
        mode_str = "Server1_Trap (500服务器, 10%陷阱)"
    elif args.mixed:
        mode_str = "混合区域"
    else:
        mode_str = "Server3跨域"
    print(f"Generating Latency CDF Plot ({mode_str})...\n")
    
    os.makedirs('total', exist_ok=True)
    
    data, ALGORITHMS = load_all_data(mixed_mode=args.mixed, use_server1=args.server1)
    if not data:
        print("Error: No data loaded!")
        return
        
    plot_cdf(data, ALGORITHMS, mixed_mode=args.mixed, use_server1=args.server1)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
