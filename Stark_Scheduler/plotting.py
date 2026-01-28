import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def smooth(data, window):
    if len(data) < window or window < 2: 
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0))
    ma = (cumsum[window:] - cumsum[:-window]) / window
    start_ma = [cumsum[i+1]/(i+1) for i in range(window-1)]
    return np.concatenate((start_ma, ma))


def plot_single_run(log_dir, run_id=None):
    if run_id is None:
        files = [f for f in os.listdir(log_dir) if f.startswith('metrics_') and f.endswith('.npz')]
        if not files:
            print(f"No metrics files found in {log_dir}")
            return
        files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True)
        latest_file = files[0]
        run_id = latest_file.replace('metrics_', '').replace('.npz', '')
        print(f"Plotting latest run: {run_id}")
    
    metrics_path = os.path.join(log_dir, f"metrics_{run_id}.npz")
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found: {metrics_path}")
        return
        
    data = np.load(metrics_path)
    
    if 'latency' not in data or 'cost' not in data:
        print("No latency/cost data found in metrics file.")
        return
    
    latency = data['latency']
    cost = data['cost']
    
    x = np.arange(1, len(latency) + 1)  
    window = max(3, len(latency) // 20)
    l_smooth = smooth(latency, window)
    c_smooth = smooth(cost, window)
 
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color_lat = '#E07070'   
    color_cost = '#4CAF50' 
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Latency (ms)', color=color_lat, fontsize=12, fontweight='bold')
    
    ax1.plot(x, l_smooth, color=color_lat, linewidth=1.2, label='Latency')
    ax1.plot(x, latency, color=color_lat, linewidth=0.5, alpha=0.3)  
    
    ax1.tick_params(axis='y', labelcolor=color_lat)
    ax1.grid(alpha=0.3, linestyle='-')

    l_margin = (latency.max() - latency.min()) * 0.1
    ax1.set_ylim(latency.min() - l_margin, latency.max() + l_margin)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Cost ($)', color=color_cost, fontsize=12, fontweight='bold')
    
    ax2.plot(x, c_smooth, color=color_cost, linewidth=1.2, label='Cost')
    ax2.plot(x, cost, color=color_cost, linewidth=0.5, alpha=0.3)  
    
    ax2.tick_params(axis='y', labelcolor=color_cost)
    
    c_margin = (cost.max() - cost.min()) * 0.1
    ax2.set_ylim(cost.min() - c_margin, cost.max() + c_margin)
    
    ax1.set_title('Performance Metrics Evolution', fontsize=14, fontweight='bold')

    save_path = os.path.join(log_dir, f"plot_{run_id}.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)

    print("\n" + "=" * 50)
    print("Final Training Metrics")
    print("=" * 50)
    print(f"Latency: {latency[-1]:.1f} ms")
    print(f"Cost:    ${cost[-1]:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, default=None, help='Run ID to plot (optional, defaults to latest)')
    parser.add_argument('--log_dir', type=str, default='results/Stark_Scheduler/logs')
    args = parser.parse_args()
    
    plot_single_run(args.log_dir, args.run_id)
