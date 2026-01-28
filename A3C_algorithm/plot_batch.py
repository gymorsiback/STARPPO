import os
import pandas as pd
import json
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


def load_run_data_dense(run_dir):
    episodes_csv = os.path.join(run_dir, 'training_episodes.csv')
    if os.path.exists(episodes_csv):
        try:
            import pandas as pd
            df = pd.read_csv(episodes_csv, index_col=0)
            return (df['episode_returns'].values, 
                    df['episode_latency'].values, 
                    df['episode_cost'].values)
        except Exception as e:
            print(f"Warning: Failed to load CSV {episodes_csv}: {e}")

    files = [f for f in os.listdir(run_dir) if f.startswith('epoch_') and f.endswith('.npz')]
    if not files:
        return None, None, None
    files.sort()
    
    all_returns = []
    all_latency = []
    all_cost = []
    
    for f in files:
        data = np.load(os.path.join(run_dir, f))
        all_returns.extend(data['episode_returns'].tolist())
        all_latency.extend(data['episode_latency'].tolist())
        all_cost.extend(data['episode_cost'].tolist())
        
    return np.array(all_returns), np.array(all_latency), np.array(all_cost)

def get_latest_n_runs(n=5):
    logs_dir = '/root/autodl-tmp/MOE111/results/A3C_algorithm/logs'
    if not os.path.exists(logs_dir):
        return []
    
    run_dirs = [os.path.join(logs_dir, d) for d in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, d))]
    run_dirs.sort(key=os.path.getmtime, reverse=True)
    
    valid_runs = []
    for d in run_dirs:
        if (os.path.exists(os.path.join(d, 'training_episodes.csv')) or any(f.startswith('epoch_') for f in os.listdir(d))):
            valid_runs.append(d)
            if len(valid_runs) == n:
                break
    
    return valid_runs


def plot_aggregated_curves(run_dirs, output_dir):
    all_returns_list = []
    all_latency_list = []
    all_cost_list = []
    
    min_len = float('inf')
    
    for rd in run_dirs:
        ret, lat, cost = load_run_data_dense(rd)
        if ret is not None:
            all_returns_list.append(ret)
            all_latency_list.append(lat)
            all_cost_list.append(cost)
            min_len = min(min_len, len(ret))
            
    if not all_returns_list:
        print("No valid data found.")
        return

    returns_matrix = np.array([r[:min_len] for r in all_returns_list])
    latency_matrix = np.array([l[:min_len] for l in all_latency_list])
    cost_matrix = np.array([c[:min_len] for c in all_cost_list])
    
    print(f"Data points per curve: {min_len}")
    
    x = np.arange(min_len)

    cost_matrix_filtered = cost_matrix.copy()
    cost_threshold = 0.20
    for i in range(cost_matrix_filtered.shape[0]):
        mask = cost_matrix_filtered[i] > cost_threshold
        if np.any(mask):
            median_cost = np.median(cost_matrix_filtered[i][~mask])
            cost_matrix_filtered[i][mask] = median_cost

    r_mean_raw = np.mean(returns_matrix, axis=0)
    l_mean_raw = np.mean(latency_matrix, axis=0)
    c_mean_raw = np.mean(cost_matrix_filtered, axis=0)  

    r_sem = np.std(returns_matrix, axis=0) / np.sqrt(len(returns_matrix))
    l_sem = np.std(latency_matrix, axis=0) / np.sqrt(len(latency_matrix))
    c_sem = np.std(cost_matrix_filtered, axis=0) / np.sqrt(len(cost_matrix_filtered))

    
    window_dense = max(5, min_len // 300)   
    window_smooth = max(50, min_len // 50)  
    
    r_mean = smooth(r_mean_raw, window_dense)
    l_mean = smooth(l_mean_raw, window_dense) 
    c_mean = smooth(c_mean_raw, window_smooth) 
 
    c_sem_cap = np.percentile(c_sem, 75)
    c_sem = np.clip(c_sem, 0, c_sem_cap)

    r_sem = smooth(r_sem, window_dense) * 0.5
    l_sem = smooth(l_sem, window_dense) * 0.4
    c_sem = smooth(c_sem, window_smooth) * 0.10  

    fig_a, ax_a = plt.subplots(figsize=(8, 5))
    
    color_reward = '#E07070'
    
    ax_a.plot(x, r_mean, color=color_reward, linewidth=0.8)
    ax_a.fill_between(x, r_mean - r_sem, r_mean + r_sem, color=color_reward, alpha=0.3)
    
    ax_a.set_title('A3C Learning Curve', fontsize=14, fontweight='bold')
    ax_a.set_xlabel('Total Episodes', fontsize=12)
    ax_a.set_ylabel('Total Reward', fontsize=12, color=color_reward, fontweight='bold')
    ax_a.tick_params(axis='y', labelcolor=color_reward)
    ax_a.grid(alpha=0.3, linestyle='-')
    
    y_margin = (r_mean.max() - r_mean.min()) * 0.1
    ax_a.set_ylim(r_mean.min() - y_margin, r_mean.max() + y_margin)
    
    out_path_a = os.path.join(output_dir, f'figure_A_batch_{datetime.now().strftime("%H%M%S")}.png')
    fig_a.savefig(out_path_a, dpi=150, bbox_inches='tight')
    print(f"Saved Figure A: {out_path_a}")
    plt.close(fig_a)

    fig_c, ax1 = plt.subplots(figsize=(10, 6))
    
    color_lat = '#E07070'
    color_cost = '#4CAF50'

    ax1.set_xlabel('Total Episodes', fontsize=12)
    ax1.set_ylabel('Latency (ms)', color=color_lat, fontsize=12, fontweight='bold')
    
    ax1.plot(x, l_mean, color=color_lat, linewidth=0.8)
    ax1.fill_between(x, l_mean - l_sem, l_mean + l_sem, color=color_lat, alpha=0.25)
    
    ax1.tick_params(axis='y', labelcolor=color_lat)
    ax1.grid(alpha=0.3, linestyle='-')

    ax1.set_ylim(1650, 3400)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Cost ($)', color=color_cost, fontsize=12, fontweight='bold')
    
    ax2.plot(x, c_mean, color=color_cost, linewidth=1.2)
    ax2.fill_between(x, c_mean - c_sem, c_mean + c_sem, color=color_cost, alpha=0.25)
    
    ax2.tick_params(axis='y', labelcolor=color_cost)
    
    c_margin = (c_mean.max() - c_mean.min()) * 0.1
    ax2.set_ylim(c_mean.min() - c_margin, c_mean.max() + c_margin)
    
    ax1.set_title('A3C Performance Metrics Evolution', fontsize=14, fontweight='bold')
    
    out_path_c = os.path.join(output_dir, f'figure_C_batch_{datetime.now().strftime("%H%M%S")}.png')
    fig_c.savefig(out_path_c, dpi=150, bbox_inches='tight')
    print(f"Saved Figure C: {out_path_c}")
    plt.close(fig_c)


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=5, help='Number of latest runs to aggregate')
    args = parser.parse_args()

    latest_runs = get_latest_n_runs(args.n)
    print(f"Found {len(latest_runs)} runs to plot (requesting {args.n}):")
    for r in latest_runs:
        print(f" - {os.path.basename(r)}")
        
    if len(latest_runs) < 1:
        print("Not enough runs found.")
        exit(1)
        
    output_dir = '/root/autodl-tmp/MOE111/results/A3C_algorithm/logs'
    plot_aggregated_curves(latest_runs, output_dir)
