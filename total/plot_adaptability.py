
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 24,
    'axes.labelsize': 26,
    'axes.titlesize': 28,
    'axes.titleweight': 'normal',
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 20,
    'lines.linewidth': 2.5,
    'ytick.color': 'black',
    'xtick.color': 'black',
    'axes.labelcolor': 'black',
    'mathtext.fontset': 'stix',
})

RESULTS_DIR = 'total/adaptability_results'
OUTPUT_DIR = 'total'

ALGORITHMS = ['STAR_PPO', 'A3C', 'Trans', 'PPO', 'PPO_CN', 'Stark', 'PPO_GNN', 'PFAPPO', 'Greedy', 'Random']

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
    'Random': '#bcbd22'     
}

MARKERS = {
    'STAR_PPO': '*',
    'A3C': 'o',
    'PPO': 's',
    'PPO_CN': '^',
    'PFAPPO': 'D',
    'PPO_GNN': 'v',
    'Trans': '<',
    'Stark': '>',
    'Greedy': 'p',
    'Random': 'h'
}

DISPLAY_NAMES = {
    'STAR_PPO': 'STAR-PPO (Ours)',
    'A3C': 'A3C',
    'PPO': 'PPO-Std',
    'PPO_CN': 'PPO-CN',
    'PFAPPO': 'PF-PPO',
    'PPO_GNN': 'GA-PPO',
    'Trans': 'Equity-Trans',
    'Stark': 'STARK',
    'Greedy': 'Greedy',
    'Random': 'Random'
}


def smooth(data, window=10):
    data = np.array(data, dtype=float)
    kernel = np.ones(window) / window
    smoothed = np.convolve(data, kernel, mode='same')
    for i in range(window // 2):
        smoothed[i] = np.mean(data[:i+window//2+1])
        smoothed[-(i+1)] = np.mean(data[-(i+window//2+1):])
    return smoothed


def main():
    all_data = {}
    meta = None
    
    for algo in ALGORITHMS:
        path = os.path.join(RESULTS_DIR, f'{algo}_adaptability.npz')
        if os.path.exists(path):
            data = np.load(path)
            all_data[algo] = data['episode_latencies']
            if meta is None:
                meta = {
                    'normal_episodes': int(data['normal_episodes']),
                    'total_episodes': int(data['total_episodes']),
                }
    
    if not all_data:
        print("Error: No data found!")
        return
  
    fig, ax = plt.subplots(figsize=(12, 7))
    
    normal_end = meta['normal_episodes']

    algo_failure_avg = {}
    for algo in ALGORITHMS:
        if algo in all_data:
            algo_failure_avg[algo] = np.mean(all_data[algo][normal_end:])
    draw_order = sorted(algo_failure_avg.keys(), key=lambda x: algo_failure_avg[x], reverse=True)
    
    for algo in draw_order:
        if algo not in all_data:
            continue
        
        latencies = all_data[algo]
        x = np.arange(len(latencies))
 
        smoothed = smooth(latencies, window=10)
 
        lw = 3.5 if algo == 'STAR_PPO' else 2.0
        display_name = DISPLAY_NAMES.get(algo, algo)
        
        ax.plot(x, smoothed, color=COLORS[algo], linewidth=lw, label=display_name)
 
        marker_size = 120 if algo == 'STAR_PPO' else 60
        ax.scatter([x[-1]], [smoothed[-1]], color=COLORS[algo], s=marker_size, 
                   marker=MARKERS[algo], zorder=5, edgecolors='white', linewidth=1)
 
    ax.axvline(x=normal_end, color='black', linestyle='--', linewidth=2, zorder=3)

    y_max = ax.get_ylim()[1]
    ax.annotate('Server Failure', 
                xy=(normal_end, y_max * 0.5),
                xytext=(normal_end + 20, y_max * 0.6),
                fontsize=18, color='black', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
 
    ax.set_yscale('log')
 
    y_min = min([np.min(d) for d in all_data.values()]) * 0.8
    y_max = max([np.max(d) for d in all_data.values()]) * 1.2
    ax.set_ylim(y_min, y_max)
 
    ax.set_xlim(0, meta['total_episodes'])
    ax.set_ylabel('Latency (ms)')
    ax.set_xlabel('Episode')
    ax.set_title('Adaptability to Server Failures', pad=15)

    ax.grid(True, alpha=0.3, linestyle='-', which='both')
    ax.grid(True, alpha=0.15, linestyle='-', which='minor')
 
    handles, labels = ax.get_legend_handles_labels()
    if 'STAR-PPO (Ours)' in labels:
        idx = labels.index('STAR-PPO (Ours)')
        handles.insert(0, handles.pop(idx))
        labels.insert(0, labels.pop(idx))
    
    ax.legend(handles, labels, loc='upper left', fontsize=16, frameon=True, 
              fancybox=True, shadow=True, ncol=2)  
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    
    plt.tight_layout()
 
    output_path = os.path.join(OUTPUT_DIR, 'Adaptability_Test.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"图表已保存: {output_path}")

    print("\n" + "="*60)
    print("性能变化统计 (对数坐标显示):")
    print("="*60)
    
    stats = []
    for algo in ALGORITHMS:
        if algo not in all_data:
            continue
        normal_avg = np.mean(all_data[algo][:normal_end])
        failure_avg = np.mean(all_data[algo][normal_end:])
        change = (failure_avg - normal_avg) / normal_avg * 100
        stats.append((algo, normal_avg, failure_avg, change))

    stats.sort(key=lambda x: x[3])
    
    print(f"{'Algorithm':<12} {'Normal (ms)':<12} {'Failure (ms)':<14} {'Change':<12} {'Assessment'}")
    print("-"*60)
    for algo, normal, failure, change in stats:
        if change < 100:
            assessment = "✓ Excellent"
        elif change < 200:
            assessment = "○ Good"
        elif change < 400:
            assessment = "△ Fair"
        else:
            assessment = "✗ Poor"
        print(f"{algo:<12} {normal:>10.0f}   {failure:>12.0f}   {change:>+10.1f}%   {assessment}")
    
    print("="*60)
    print(f"\nBest adaptability: {stats[0][0]} (only {stats[0][3]:+.1f}% increase)")
    print(f"Worst adaptability: {stats[-1][0]} ({stats[-1][3]:+.1f}% increase)")


if __name__ == '__main__':
    main()
