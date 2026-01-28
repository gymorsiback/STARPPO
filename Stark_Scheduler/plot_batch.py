import os
import glob
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


def plot_batch_results(log_dir, n_runs=5):
    metrics_files = glob.glob(os.path.join(log_dir, "metrics_*.npz"))
    
    if not metrics_files:
        print(f"No metrics files found in {log_dir}")
        print("Please run training first: python Stark_Scheduler/run_batch.py --n 5 --epochs 100")
        return
 
    metrics_files.sort(key=os.path.getmtime, reverse=True)
    metrics_files = metrics_files[:n_runs]
    
    print(f"Found {len(metrics_files)} training runs to plot")

    import pandas as pd
    all_latency = []
    all_cost = []
    
    for f in metrics_files:
        csv_path = f.replace('.npz', '.csv')
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if 'latency' in df.columns and 'cost' in df.columns:
                    all_latency.append(df['latency'].values)
                    all_cost.append(df['cost'].values)
                continue
            except Exception as e:
                print(f"Warning: Failed to load CSV {csv_path}: {e}")
 
        data = np.load(f)
        if 'latency' in data and 'cost' in data:
            all_latency.append(data['latency'])
            all_cost.append(data['cost'])
    
    if not all_latency or not all_cost:
        print("No latency/cost data found in metrics files.")
        return
 
    min_len = min(len(l) for l in all_latency)
    all_latency = [l[:min_len] for l in all_latency]
    all_cost = [c[:min_len] for c in all_cost]
 
    latency_matrix = np.array(all_latency)
    cost_matrix = np.array(all_cost)

    cost_threshold = 0.20
    for i in range(cost_matrix.shape[0]):
        mask = cost_matrix[i] > cost_threshold
        if np.any(mask):
            median_cost = np.median(cost_matrix[i][~mask])
            cost_matrix[i][mask] = median_cost
    
    print(f"Data points per curve: {min_len}")

    x = np.arange(1, min_len + 1) 
    l_mean_raw = np.mean(latency_matrix, axis=0)
    c_mean_raw = np.mean(cost_matrix, axis=0)
    
    l_sem = np.std(latency_matrix, axis=0) / np.sqrt(len(latency_matrix))
    c_sem = np.std(cost_matrix, axis=0) / np.sqrt(len(cost_matrix))
 
    window = max(5, min_len // 20)
    l_mean = smooth(l_mean_raw, window)
    c_mean = smooth(c_mean_raw, window)
    l_sem = smooth(l_sem, window) * 0.5
    c_sem = smooth(c_sem, window) * 0.5
 
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color_lat = '#E07070'   
    color_cost = '#4CAF50' 
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Latency (ms)', color=color_lat, fontsize=12, fontweight='bold')
    
    ax1.plot(x, l_mean, color=color_lat, linewidth=1.2, label='Latency')
    ax1.fill_between(x, l_mean - l_sem, l_mean + l_sem, color=color_lat, alpha=0.25)
    
    ax1.tick_params(axis='y', labelcolor=color_lat)
    ax1.grid(alpha=0.3, linestyle='-')
 
    l_margin = (l_mean.max() - l_mean.min()) * 0.1
    ax1.set_ylim(l_mean.min() - l_margin, l_mean.max() + l_margin)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Cost ($)', color=color_cost, fontsize=12, fontweight='bold')
    
    ax2.plot(x, c_mean, color=color_cost, linewidth=1.2, label='Cost')
    ax2.fill_between(x, c_mean - c_sem, c_mean + c_sem, color=color_cost, alpha=0.25)
    
    ax2.tick_params(axis='y', labelcolor=color_cost)
    
    c_margin = (c_mean.max() - c_mean.min()) * 0.1
    ax2.set_ylim(c_mean.min() - c_margin, c_mean.max() + c_margin)
    
    ax1.set_title('Performance Metrics Evolution', fontsize=14, fontweight='bold')
 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(log_dir, f"batch_training_results_{timestamp}.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    plt.close(fig)
  
    print("\n" + "=" * 50)
    print("Summary (Final Epoch)")
    print("=" * 50)
    print(f"Latency: {l_mean[-1]:.1f} ms (±{l_sem[-1]:.1f})")
    print(f"Cost:    ${c_mean[-1]:.4f} (±${c_sem[-1]:.4f})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='results/Stark_Scheduler/logs')
    parser.add_argument('--n', type=int, default=5, help='Number of latest runs to aggregate')
    args = parser.parse_args()
    
    plot_batch_results(args.log_dir, args.n)
