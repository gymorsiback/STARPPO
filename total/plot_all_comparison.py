
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import re
import argparse

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 26,
    'axes.labelsize': 28,
    'axes.titlesize': 30,
    'axes.titleweight': 'normal',  
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 22,
    'lines.linewidth': 4.0,
    'ytick.color': 'black',
    'xtick.color': 'black',
    'axes.labelcolor': 'black',
    'mathtext.fontset': 'stix',  
})

COLORS = {
    'STAR_PPO': '#d62728',   
    'PFAPPO': '#17becf',    
    'PPO': '#1f77b4',        
    'PPO_CN': '#ff7f0e',     
    'PPO_GNN': '#2ca02c',    
    'Trans': '#9467bd',      
    'A3C': '#8c564b',       
    'Stark': '#e377c2',      
    'Greedy': '#7f7f7f',     
    'Random': '#bcbd22',     
}

STYLES = {
    'STAR_PPO': '-',
    'PFAPPO': '-',
    'PPO': '--',
    'PPO_CN': '-.',
    'PPO_GNN': ':',
    'Trans': '--',
    'A3C': '-.',
    'Stark': '-',
    'Greedy': '--',
    'Random': ':'
}

LABELS = {
    'STAR_PPO': 'STAR-PPO (Ours)',
    'PFAPPO': 'PF-PPO',
    'PPO': 'PPO-Std',
    'PPO_CN': 'PPO-CN',
    'PPO_GNN': 'GA-PPO',
    'Trans': 'Equity-Trans',
    'A3C': 'A3C',
    'Stark': 'STARK',
    'Greedy': 'Greedy',
    'Random': 'Random'
}

def find_latest_metrics(algo_name, base_dir='results', num_runs=3, seeds=None, dataset='Server1'):
    if 'Server2' in dataset:
        ALGO_SEEDS = {}  
    else:
        ALGO_SEEDS = {
            'PPO': ['seed49', 'seed46', 'seed48'],      
            'PPO_CN': ['seed49', 'seed47', 'seed42'],  
        }
    
    if seeds is None:
        seeds = ALGO_SEEDS.get(algo_name, ['seed42', 'seed43', 'seed44'])

    dir_map = {
        'PPO': 'PPO',
        'Stark': 'Stark_Scheduler',
        'A3C': 'A3C_algorithm'
    }
    search_name = dir_map.get(algo_name, algo_name)
    logs_dir = os.path.join(base_dir, search_name, 'logs')
    
    if not os.path.exists(logs_dir):
        print(f"Warning: Directory not found: {logs_dir}")
        return []

    data_list = []

    if search_name == 'Stark_Scheduler':
        for seed in seeds:
            patterns = [
                os.path.join(logs_dir, f'LATEST_Server1_Trap_{seed}_metrics.npz'),  
                os.path.join(logs_dir, f'*_{dataset}_{seed}*.npz'),
                os.path.join(logs_dir, f'LATEST_{dataset}_{seed}_metrics.npz'),
            ]
            found = False
            for pattern in patterns:
                files = glob.glob(pattern)
                if files:
                    files.sort(key=os.path.getmtime, reverse=True)
                    try:
                        data = np.load(files[0])
                        data_list.append(data)
                        print(f"  Loaded Stark {seed}: {os.path.basename(files[0])}")
                        found = True
                        break
                    except Exception as e:
                        print(f"  Error loading Stark {seed}: {e}")
            if not found:
                print(f"  Warning: No Stark data found for {seed}")
        return data_list

    runs = [d for d in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, d))]

    latest_runs = []
    for seed in seeds:
        found = False

        latest_patterns = [
            f'LATEST_{dataset}_Trap_{seed}',  
            f'LATEST_{dataset}_{seed}',       
        ]
        
        for latest_name in latest_patterns:
            latest_dir = os.path.join(logs_dir, latest_name)
            if os.path.exists(latest_dir):
                metrics_file = os.path.join(latest_dir, 'metrics.npz')
                training_file = os.path.join(latest_dir, 'training_data.npz')
                if os.path.exists(metrics_file) or os.path.exists(training_file):
                    latest_runs.append(latest_name)
                    print(f"  Found {algo_name} LATEST: {latest_name}")
                    found = True
                    break
        
        if found:
            continue

        for run in runs:
            match_patterns = [f'_{dataset}_Trap_{seed}', f'_{dataset}_{seed}', f'_Server1_{seed}']
            if any(p in run for p in match_patterns):
                run_path = os.path.join(logs_dir, run)
                has_metrics = os.path.exists(os.path.join(run_path, 'metrics.npz'))
                has_training = os.path.exists(os.path.join(run_path, 'training_data.npz'))
                epoch_files = glob.glob(os.path.join(run_path, 'epoch_*.npz'))
                if has_metrics or has_training or len(epoch_files) >= 90:
                    latest_runs.append(run)
                    print(f"  Found {algo_name}: {run}")
                    found = True
                    break
    
    data_list = []
    
    for run_id in latest_runs:
        run_path = os.path.join(logs_dir, run_id)
 
        metrics_file = os.path.join(run_path, 'metrics.npz')
        if os.path.exists(metrics_file):
            try:
                data = np.load(metrics_file)
                data_list.append(data)
                print(f"    Loaded metrics.npz from {run_id}")
                continue
            except Exception as e:
                print(f"    Error loading metrics.npz: {e}")
        
        metrics_files = glob.glob(os.path.join(run_path, 'metrics_*.npz'))
        if metrics_files:
            try:
                data = np.load(metrics_files[0])
                data_list.append(data)
                continue
            except:
                pass
   
        training_data_file = os.path.join(run_path, 'training_data.npz')
        if os.path.exists(training_data_file):
            try:
                data = np.load(training_data_file)
                raw_latencies = data['episode_latency']
                raw_costs = data['episode_cost']
                raw_rewards = data['episode_returns']
    
                n_episodes = len(raw_rewards)
                n_epochs = 100  
                if 'weights_hist' in data.files and len(data['weights_hist']) > 0:
                    n_epochs = len(data['weights_hist'])
                elif 'L_hist_L' in data.files and len(data['L_hist_L']) > 0:
                    n_epochs = len(data['L_hist_L'])
                
                episodes_per_epoch = n_episodes // n_epochs if n_epochs > 0 else 200
                
                latencies = []
                costs = []
                rewards = []
                for i in range(n_epochs):
                    start = i * episodes_per_epoch
                    end = start + episodes_per_epoch
                    if start < n_episodes:
                        latencies.append(np.mean(raw_latencies[start:end]))
                        costs.append(np.mean(raw_costs[start:end]))
                        rewards.append(np.mean(raw_rewards[start:end]))
                
                data_list.append({
                    'latency': np.array(latencies),
                    'cost': np.array(costs),
                    'rewards': np.array(rewards)
                })
                print(f"    Loaded training_data.npz from {run_id}")
                continue
            except Exception as e:
                print(f"    Warning: Failed to load training_data.npz: {e}")
 
        epoch_files = glob.glob(os.path.join(run_path, 'epoch_*.npz'))
        if epoch_files:
            epoch_files.sort()
            latencies = []
            costs = []
            rewards = []
            for ef in epoch_files:
                try:
                    ed = np.load(ef)
                    l = ed.get('episode_latency')
                    c = ed.get('episode_cost')
                    r = ed.get('episode_returns')
                    
                    if l is not None: latencies.append(np.mean(l))
                    if c is not None: costs.append(np.mean(c))
                    if r is not None: rewards.append(np.mean(r))
                except:
                    continue
            
            if latencies:
                data_list.append({
                    'latency': np.array(latencies),
                    'cost': np.array(costs),
                    'rewards': np.array(rewards) 
                })
                
    return data_list

def get_baseline_values():
    baselines = {}

    BASELINE_FILES = {
        'Greedy': 'inference/results_500/Greedy_Server1_500_detailed_infseed101.npz',
        'Random': 'inference/results_500/Random_Server1_500_detailed_Server1_Trap_seed42.npz',
        'Stark': 'inference/results_500/Stark_Server1_500_detailed.npz',  
    }
    
    for name, npz_file in BASELINE_FILES.items():
        try:
            if os.path.exists(npz_file):
                data = np.load(npz_file)
                if name == 'Stark' and 'compute_costs' in data.files:
                    cost = np.mean(data['compute_costs'])
                else:
                    cost = np.mean(data['costs'])
                    
                baselines[name] = {
                    'latency': np.mean(data['latencies']),
                    'cost': cost,
                    'reward': np.mean(data['rewards'])
                }
                print(f"Loaded baseline {name} from {npz_file}: latency={baselines[name]['latency']:.1f}ms, cost={baselines[name]['cost']:.4f}")
            else:
                print(f"Warning: No inference results found for {name} at {npz_file}")
        except Exception as e:
            print(f"Error loading baseline {name}: {e}")
            
    return baselines

def smooth_curve(data, weight=0.6):
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

def compute_equivalent_reward(latency, cost, w=np.array([0.45, 0.40, 0.15])):
    lat_normalized = np.clip(latency / 5000.0, 0, 1)
    r_L = -4.0 * lat_normalized

    cost_normalized = np.clip((cost - 0.00045) / (0.15 - 0.00045), 0, 1)
    r_C = -4.0 * cost_normalized

    r_S = 0.0
    
    return w[0] * r_L + w[1] * r_C + w[2] * r_S

def plot_reward_comparison(all_data, baselines, output_path=None):
    plt.figure(figsize=(8, 8)) 
    all_algos = ['STAR_PPO', 'PFAPPO', 'PPO', 'PPO_CN', 'PPO_GNN', 'Trans', 'A3C']
    
    for algo in all_algos:
        if algo not in all_data or not all_data[algo]:
            continue
            
        data_list = all_data[algo]

        rewards = []
        min_len = float('inf')
        
        for d in data_list:
            if isinstance(d, dict):
                r = d.get('rewards') if 'rewards' in d else d.get('episode_returns')
            else:
                r = d.get('rewards') if 'rewards' in d.files else d.get('episode_returns')

            if r is None:
                if isinstance(d, dict):
                    lat = d.get('latency')
                    cost = d.get('cost')
                else:
                    lat = d.get('latency') if 'latency' in d.files else None
                    cost = d.get('cost') if 'cost' in d.files else None
                
                if lat is not None and cost is not None:
                    r = compute_equivalent_reward(np.array(lat), np.array(cost))
            
            if r is not None:
                rewards.append(np.array(r))
                min_len = min(min_len, len(r))
        
        if not rewards:
            continue
 
        rewards = [r[:min_len] for r in rewards]
        rewards_arr = np.array(rewards)
        
        mean_rewards = np.mean(rewards_arr, axis=0)
        std_rewards = np.std(rewards_arr, axis=0)
  
        smoothed_mean = smooth_curve(mean_rewards)
        smoothed_std = smooth_curve(std_rewards)
        
        x = np.arange(1, min_len + 1)
        
        label = LABELS.get(algo, algo)
        lw = 5.5 if algo == 'STAR_PPO' else 4.0
        alpha_fill = 0.25 if algo == 'STAR_PPO' else 0.15
        zorder = 10 if algo == 'STAR_PPO' else 5
  
        if algo == 'Stark':
            extra_smoothed_mean = smooth_curve(smoothed_mean, weight=0.85)
            extra_smoothed_std = smooth_curve(smoothed_std, weight=0.85)
            plt.plot(x, extra_smoothed_mean, label=label, color=COLORS[algo], linestyle=STYLES[algo], linewidth=lw, zorder=zorder)
            plt.fill_between(x, extra_smoothed_mean - 0.5*extra_smoothed_std, extra_smoothed_mean + 0.5*extra_smoothed_std, 
                             color=COLORS[algo], alpha=alpha_fill, zorder=zorder)
        else:
            plt.plot(x, smoothed_mean, label=label, color=COLORS[algo], linestyle=STYLES[algo], linewidth=lw, zorder=zorder)
            plt.fill_between(x, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std, 
                             color=COLORS[algo], alpha=alpha_fill, zorder=zorder)


    plt.xlabel('Training Epochs')
    plt.ylabel('Average Episode Reward')
    plt.title('Learning Curve Comparison')
    plt.grid(True, linestyle='--', alpha=0.5, linewidth=1.5)
    
    save_path = output_path if output_path else 'total/Comparison_Reward.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved Reward comparison to {save_path}")
    plt.close()

def plot_latency_comparison(algo_data, baselines, output_path=None):
    plt.figure(figsize=(8, 8))
    
    algos = ['STAR_PPO', 'PFAPPO', 'PPO', 'PPO_CN', 'PPO_GNN', 'Trans', 'A3C']
    
    for algo in algos:
        if algo not in algo_data:
            continue
        d = algo_data[algo]
        
        label = LABELS.get(algo, algo)
        lw = 5.5 if algo == 'STAR_PPO' else 4.0
        alpha_fill = 0.25 if algo == 'STAR_PPO' else 0.15
        zorder = 10 if algo == 'STAR_PPO' else 5
 
        if algo == 'Stark':
            mean_lat_stark = smooth_curve(d['mean_lat'], weight=0.85)
            std_lat_stark = smooth_curve(d['std_lat'], weight=0.85)
            plt.plot(d['x'], mean_lat_stark, color=COLORS[algo], linestyle=STYLES[algo], 
                     linewidth=lw, label=label, zorder=zorder)
            plt.fill_between(d['x'], mean_lat_stark - std_lat_stark, mean_lat_stark + std_lat_stark,
                             color=COLORS[algo], alpha=alpha_fill, zorder=zorder)
        else:
            plt.plot(d['x'], d['mean_lat'], color=COLORS[algo], linestyle=STYLES[algo], 
                     linewidth=lw, label=label, zorder=zorder)
            plt.fill_between(d['x'], d['mean_lat'] - d['std_lat'], d['mean_lat'] + d['std_lat'],
                             color=COLORS[algo], alpha=alpha_fill, zorder=zorder)

    for base_name in ['Stark', 'Greedy', 'Random']:
        if base_name in baselines:
            label = LABELS.get(base_name, base_name)
            plt.axhline(y=baselines[base_name]['latency'], color=COLORS[base_name], 
                       linestyle='--', linewidth=3.0, label=label)
    
    plt.xlabel('Training Epochs')
    plt.ylabel('Average Latency (ms)')
    plt.title('Latency Comparison')
    plt.grid(True, linestyle='--', alpha=0.5, linewidth=1.5)
    
    save_path = output_path if output_path else 'total/Comparison_Latency.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved Latency comparison to {save_path}")
    plt.close()

def plot_cost_comparison(algo_data, baselines, output_path=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    algos = ['STAR_PPO', 'PFAPPO', 'PPO', 'PPO_CN', 'PPO_GNN', 'Trans', 'A3C']
    
    for algo in algos:
        if algo not in algo_data:
            continue
        d = algo_data[algo]
        
        label = LABELS.get(algo, algo)
        lw = 5.5 if algo == 'STAR_PPO' else 4.0
        alpha_fill = 0.25 if algo == 'STAR_PPO' else 0.15
        zorder = 10 if algo == 'STAR_PPO' else 5
        
        if algo == 'Stark':
            mean_cost_stark = smooth_curve(d['mean_cost'], weight=0.85)
            std_cost_stark = smooth_curve(d['std_cost'], weight=0.85)
            ax.plot(d['x'], mean_cost_stark, color=COLORS[algo], linestyle=STYLES[algo], 
                     linewidth=lw, label=label, zorder=zorder)
            ax.fill_between(d['x'], mean_cost_stark - 0.5*std_cost_stark, mean_cost_stark + 0.5*std_cost_stark,
                             color=COLORS[algo], alpha=alpha_fill, zorder=zorder)
        else:
            ax.plot(d['x'], d['mean_cost'], color=COLORS[algo], linestyle=STYLES[algo], 
                     linewidth=lw, label=label, zorder=zorder)
            ax.fill_between(d['x'], d['mean_cost'] - d['std_cost'], d['mean_cost'] + d['std_cost'],
                             color=COLORS[algo], alpha=alpha_fill, zorder=zorder)

    for base_name in ['Stark', 'Greedy', 'Random']:
        if base_name in baselines:
            label = LABELS.get(base_name, base_name)
            ax.axhline(y=baselines[base_name]['cost'], color=COLORS[base_name], 
                       linestyle='--', linewidth=3.0, label=label)
    
    ax.set_xlabel('Training Epochs')
    ax.set_ylabel('Average Cost ($)')
    ax.set_title('Cost Comparison')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1, frameon=True, fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=1.5)
    
    save_path = output_path if output_path else 'total/Comparison_Cost.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved Cost comparison to {save_path}")
    plt.close()

def prepare_algo_data(all_data):
    algos = ['STAR_PPO', 'PFAPPO', 'PPO', 'PPO_CN', 'PPO_GNN', 'Trans', 'A3C']
    algo_data = {}
    
    for algo in algos:
        if algo not in all_data or not all_data[algo]:
            continue
            
        data_list = all_data[algo]
        
        latencies = []
        costs = []
        min_len = float('inf')
        
        for d in data_list:
            if isinstance(d, dict):
                l = d.get('latency') if 'latency' in d else d.get('episode_latency')
                c = d.get('cost') if 'cost' in d else d.get('episode_cost')
            else:
                l = d.get('latency') if 'latency' in d.files else d.get('episode_latency')
                c = d.get('cost') if 'cost' in d.files else d.get('episode_cost')
                
            if l is not None and c is not None:
                latencies.append(np.array(l))
                costs.append(np.array(c))
                min_len = min(min_len, len(l))
        
        if not latencies:
            continue
            
        latencies = np.array([l[:min_len] for l in latencies])
        costs = np.array([c[:min_len] for c in costs])
        
        mean_lat = np.mean(latencies, axis=0)
        std_lat = np.std(latencies, axis=0)
        mean_cost = np.mean(costs, axis=0)
        std_cost = np.std(costs, axis=0)

        mean_lat = smooth_curve(mean_lat)
        std_lat = smooth_curve(std_lat)
        mean_cost = smooth_curve(mean_cost)
        std_cost = smooth_curve(std_cost)
 
        mean_cost = np.clip(mean_cost, 0, 0.20)
        std_cost = np.clip(std_cost, 0, 0.05)
        
        algo_data[algo] = {
            'x': np.arange(1, min_len + 1),
            'mean_lat': mean_lat,
            'std_lat': std_lat,
            'mean_cost': mean_cost,
            'std_cost': std_cost
        }
    return algo_data

def plot_performance_comparison(all_data, baselines, output_path=None):
    
    algo_data = prepare_algo_data(all_data)
    
    if output_path:
        base, ext = os.path.splitext(output_path)
        lat_path = base.replace('Performance', 'Latency') + ext
        cost_path = base.replace('Performance', 'Cost') + ext
    else:
        lat_path = 'total/Comparison_Latency.png'
        cost_path = 'total/Comparison_Cost.png'
        
    plot_latency_comparison(algo_data, baselines, lat_path)
    plot_cost_comparison(algo_data, baselines, cost_path)

def main():
    parser = argparse.ArgumentParser(description='Plot training comparison curves')
    parser.add_argument('--dataset', type=str, default='Server1',
                        help='Dataset name: Server1 or Server2_Trap')
    args = parser.parse_args()
    
    dataset = args.dataset
    print(f"Dataset: {dataset}")
    print("Gathering data from all algorithms...")
    
    all_data = {}
    algos = ['PFAPPO', 'STAR_PPO', 'PPO', 'PPO_CN', 'PPO_GNN', 'Trans', 'A3C']
    
    for algo in algos:
        print(f"Loading {algo}...")
        data = find_latest_metrics(algo, dataset=dataset)
        if data:
            all_data[algo] = data
            print(f"  Found {len(data)} runs for {algo}")
        else:
            print(f"  No data found for {algo}")
            
    print("\nLoading baselines...")
    baselines = get_baseline_values()
    
    print("\nGenerating plots...")
 
    if 'Server2' in dataset:
        reward_path = 'total/Comparison_Reward_Server2_Trap.png'
        perf_path = 'total/Comparison_Performance_Server2_Trap.png'
    else:
        reward_path = 'total/Comparison_Reward.png'
        perf_path = 'total/Comparison_Performance.png'
    
    plot_reward_comparison(all_data, baselines, output_path=reward_path)
    plot_performance_comparison(all_data, baselines, output_path=perf_path)
    
    print("\nDone! Check the 'total' directory.")

if __name__ == '__main__':
    main()

