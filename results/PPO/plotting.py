import os
import json
from datetime import datetime
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_fig(fig: plt.Figure, out_dir: str, base_name: str):
    ensure_dir(out_dir)
    ts = timestamp_tag()
    fname = f"{base_name}_{ts}.png"
    fpath = os.path.join(out_dir, fname)
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    latest = os.path.join(out_dir, f"{base_name}_latest.png")
    try:
        if os.path.exists(latest):
            os.remove(latest)
        import shutil
        shutil.copyfile(fpath, latest)
    except Exception:
        pass
    print(f"Saved figure: {fpath}")



def plot_learning_curve(episode_returns: np.ndarray, out_dir: str, window: int = None):
    if window is None:
        window = max(20, min(100, len(episode_returns) // 10))  
    
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(episode_returns))
  
    y = episode_returns
    if len(y) >= window:
        cumsum = np.cumsum(np.insert(y, 0, 0))
        ma = np.zeros(len(y))
        for i in range(len(y)):
            if i < window:
                ma[i] = cumsum[i+1] / (i+1)
            else:
                ma[i] = (cumsum[i+1] - cumsum[i+1-window]) / window
        ax.plot(x, ma, color='tab:blue', linewidth=2.5, label=f'DWA Reward (MA-{window})')
    else:
        ax.plot(x, y, color='tab:blue', linewidth=2.5, label='DWA Reward')
    
    ax.set_title('Learning Curve: Episode Reward Over Time\n(Goal: Ascending toward 0)', 
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward (Closer to 0 = Better)', fontsize=12)
    ax.grid(alpha=0.3, linestyle='--')
    
    save_fig(fig, out_dir, 'figure_A_learning_curve')
    plt.close(fig)



def plot_subobjectives(episode_latency: np.ndarray, episode_cost: np.ndarray, out_dir: str, window: int = 50):
    fig, ax = plt.subplots(2, 1, figsize=(7,6), sharex=True)
    x = np.arange(len(episode_latency))
    yL = episode_latency
    if len(yL) >= window:
        maL = np.convolve(yL, np.ones(window)/window, mode='valid')
        ax[0].plot(x[window-1:], maL, color='tab:orange', linewidth=2)
    else:
        ax[0].plot(x, yL, color='tab:orange', linewidth=2)
    ax[0].set_ylabel('Average Latency per Episode (ms)')
    ax[0].set_title('Sub-objectives Evolution')
    ax[0].grid(alpha=0.3)
    yC = episode_cost
    if len(yC) >= window:
        maC = np.convolve(yC, np.ones(window)/window, mode='valid')
        ax[1].plot(x[window-1:], maC, color='tab:green', linewidth=2)
    else:
        ax[1].plot(x, yC, color='tab:green', linewidth=2)
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Average Cost per Episode ($)')
    ax[1].grid(alpha=0.3)
    save_fig(fig, out_dir, 'figure_B_subobjectives')
    plt.close(fig)



def plot_pareto(episode_latency: np.ndarray, episode_cost: np.ndarray, episodes_per_epoch: int, out_dir: str):
    if len(episode_latency) == 0:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    lat_norm = (episode_latency - episode_latency.min()) / (episode_latency.max() - episode_latency.min())
    cost_norm = (episode_cost - episode_cost.min()) / (episode_cost.max() - episode_cost.min())

    weighted_score = 0.5 * lat_norm + 0.5 * cost_norm

    window = max(20, len(episode_latency) // 20)
    
    def moving_average(data, w):
        return np.convolve(data, np.ones(w)/w, mode='valid')

    score_ma = moving_average(weighted_score, window)
    x_ma = np.arange(len(score_ma)) + window//2
    
    ax1.plot(x_ma, score_ma, color='purple', linewidth=2.5, label='Weighted Score (Latency+Cost)')

    start_y = score_ma[0]
    end_y = score_ma[-1]
    ax1.annotate(f'Improvement: {(start_y-end_y)/start_y*100:.1f}%', 
                 xy=(len(score_ma), end_y), xytext=(len(score_ma)//2, start_y),
                 arrowprops=dict(arrowstyle='->', color='purple', lw=2),
                 fontsize=12, fontweight='bold', color='purple')
    
    ax1.set_ylabel('Normalized Weighted Score\n(Lower is Better)', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Performance Optimization Trend', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.legend(loc='upper right')

    lat_ma = moving_average(episode_latency, window)
    cost_ma = moving_average(episode_cost, window)
    
    color_lat = 'tab:orange'
    color_cost = 'tab:green'

    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Latency (ms)', color=color_lat, fontsize=12, fontweight='bold')
    l1 = ax2.plot(x_ma, lat_ma, color=color_lat, linewidth=2, label='Latency')
    ax2.tick_params(axis='y', labelcolor=color_lat)
 
    ax2_twin = ax2.twinx()
    ax2_twin.set_ylabel('Cost ($)', color=color_cost, fontsize=12, fontweight='bold')
    l2 = ax2_twin.plot(x_ma, cost_ma, color=color_cost, linewidth=2, label='Cost')
    ax2_twin.tick_params(axis='y', labelcolor=color_cost)

    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc='upper right')
    
    ax2.grid(alpha=0.3)
    ax2.set_title('Individual Metrics Evolution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_fig(fig, out_dir, 'figure_C_pareto')
    plt.close(fig)



def plot_weights(weights_hist: np.ndarray, out_dir: str):
    if weights_hist.size == 0:
        return
    fig, ax = plt.subplots(figsize=(7,4))
    x = np.arange(weights_hist.shape[0])
    ax.plot(x, weights_hist[:,0], label='w_latency', color='tab:orange')
    ax.plot(x, weights_hist[:,1], label='w_cost', color='tab:green')
    if weights_hist.shape[1] > 2:
        ax.plot(x, weights_hist[:,2], label='w_switch', color='tab:red')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Weight value')
    ax.set_ylim(0, 1)
    ax.set_title('DWA Weights Over Time')
    ax.grid(alpha=0.3)
    ax.legend()
    save_fig(fig, out_dir, 'figure_D_weights')
    plt.close(fig)



def plot_cdf(latencies: dict, costs: dict, out_dir: str, title_suffix: str = ''):
    def cdf_data(vals: np.ndarray):
        vals = np.sort(vals)
        y = np.linspace(0, 1, len(vals), endpoint=False)
        return vals, y

    fig, ax = plt.subplots(1,2, figsize=(12,4))
    for k, v in latencies.items():
        v = np.array(v, dtype=float)
        if v.size == 0:
            continue
        x, y = cdf_data(v)
        ax[0].plot(x, y, label=k)
    ax[0].set_xlabel('Latency per Episode (ms)')
    ax[0].set_ylabel('CDF')
    ax[0].set_title('CDF of Latency ' + title_suffix)
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    for k, v in costs.items():
        v = np.array(v, dtype=float)
        if v.size == 0:
            continue
        x, y = cdf_data(v)
        ax[1].plot(x, y, label=k)
    ax[1].set_xlabel('Cost per Episode ($)')
    ax[1].set_ylabel('CDF')
    ax[1].set_title('CDF of Cost ' + title_suffix)
    ax[1].grid(alpha=0.3)
    ax[1].legend()

    save_fig(fig, out_dir, 'figure_E_cdf')
    plt.close(fig)


def plots_from_run_dir(run_dir: str):
    episodes_csv = os.path.join(run_dir, 'training_episodes.csv')
    epochs_csv = os.path.join(run_dir, 'training_epochs.csv')
    
    if os.path.exists(episodes_csv):
        import pandas as pd
        df_ep = pd.read_csv(episodes_csv, index_col=0)
        episode_returns = df_ep['episode_returns'].values
        episode_latency = df_ep['episode_latency'].values
        episode_cost = df_ep['episode_cost'].values
 
        if os.path.exists(epochs_csv):
            df_epochs = pd.read_csv(epochs_csv, index_col=0)
            w_lat = df_epochs['w_latency'].values
            w_cost = df_epochs['w_cost'].values
            w_switch = df_epochs['w_switch'].values if 'w_switch' in df_epochs.columns else np.zeros_like(w_lat)
            weights_hist = np.column_stack([w_lat, w_cost, w_switch])
        else:
            weights_hist = np.array([])
    else:
    files = [f for f in os.listdir(run_dir) if f.startswith('epoch_') and f.endswith('.npz')]
    if not files:
            print('No training data files in', run_dir)
        return
    files.sort()
    last = os.path.join(run_dir, files[-1])
    data = np.load(last)
    episode_returns = data['episode_returns']
    episode_latency = data['episode_latency']
    episode_cost = data['episode_cost']
    weights_hist = data['weights_hist']
    
    meta_path = os.path.join(run_dir, 'meta.json')
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path, 'r'))
    episodes_per_epoch = int(meta.get('episodes_per_epoch', 200))
    else:
        episodes_per_epoch = 200

    out_dir = os.path.dirname(run_dir)
    plot_learning_curve(episode_returns, out_dir)
    plot_subobjectives(episode_latency, episode_cost, out_dir)
    plot_pareto(episode_latency, episode_cost, episodes_per_epoch, out_dir)
    plot_weights(weights_hist, out_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate plots from PPO training results.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Use the latest run automatically
  python plotting.py
  
  # Specify a run directory
  python plotting.py --run_dir /path/to/results/PPO/logs/ppo_dwa_20251203_183846_bc52b3
  
  # Specify by run ID (searches in logs directory)
  python plotting.py --run_id ppo_dwa_20251203_183846_bc52b3
        '''
    )
    parser.add_argument('--run_dir', type=str, default=None, help='Path to results/PPO/logs/<run_id>')
    parser.add_argument('--run_id', type=str, default=None, help='Run ID (searches in logs directory)')
    args = parser.parse_args()
    
    run_dir = args.run_dir
    
    if run_dir is None and args.run_id is not None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        run_dir = os.path.join(script_dir, 'logs', args.run_id)
    
    if run_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(script_dir, 'logs')
        if not os.path.exists(logs_dir):
            print(f"Error: logs directory not found at {logs_dir}")
            exit(1)
        
        run_dirs = [d for d in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, d))]
        if not run_dirs:
            print(f"Error: No run directories found in {logs_dir}")
            exit(1)
  
        run_dirs_full = [os.path.join(logs_dir, d) for d in run_dirs]
        run_dir = max(run_dirs_full, key=os.path.getmtime)
        print(f"[INFO] Auto-detected latest run: {os.path.basename(run_dir)}")
    
    if not os.path.exists(run_dir):
        print(f"Error: run_dir not found: {run_dir}")
        exit(1)
    
    print(f"[INFO] Plotting from: {run_dir}")
    plots_from_run_dir(run_dir)
    print(f"[INFO] Plotting complete!")

