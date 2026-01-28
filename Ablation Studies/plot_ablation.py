

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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
    'axes.linewidth': 2.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'mathtext.fontset': 'stix',
})

COLORS = {
    'full': '#2E86AB',         
    'no_workflow': '#9B59B6',  
    'no_future': '#F18F01',    
    'no_topology': '#C73E1D',  
}

LABELS = {
    'full': 'STAR-PPO (Full)',
    'no_workflow': 'w/o Workflow',
    'no_future': 'w/o Future Reward',
    'no_topology': 'w/o Topology',
}

MARKERS = {
    'full': 'o',
    'no_workflow': 's',
    'no_future': '^',
    'no_topology': 'D',
}

BEST_SEEDS = {
    'full': None,       
    'no_workflow': 42,  
    'no_future': 43,    
    'no_topology': 44,  
}


def load_training_data(results_dir, full_model_dir):
    import pandas as pd
    modes = ['no_topology', 'no_workflow', 'no_future']
    seeds = [42, 43, 44]
    
    data = {}
    
    def load_metrics(base_path):
        """从 CSV 或 NPZ 加载 rewards"""
        csv_path = base_path.replace('.npz', '.csv')
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                return df['rewards'].values
            except:
                pass
        if os.path.exists(base_path):
            d = np.load(base_path)
            return d['rewards']
        return None
 
    data['full'] = {'rewards': [], 'seeds': {}}
    for seed in seeds:
        path = os.path.join(full_model_dir, f'LATEST_Server1_Trap_seed{seed}', 'metrics.npz')
        rewards = load_metrics(path)
        if rewards is not None:
            data['full']['rewards'].append(rewards)
            data['full']['seeds'][seed] = rewards

    for mode in modes:
        data[mode] = {'rewards': [], 'seeds': {}}
        for seed in seeds:
            path = os.path.join(results_dir, f'{mode}_seed{seed}', 'metrics.npz')
            rewards = load_metrics(path)
            if rewards is not None:
                data[mode]['rewards'].append(rewards)
                data[mode]['seeds'][seed] = rewards
    
    return data


def load_inference_data(results_dir):
    path = os.path.join(results_dir, 'ablation_inference_results.npz')
    if not os.path.exists(path):
        return None
    
    d = np.load(path)
    data = {}
    modes = ['full', 'no_workflow', 'no_future', 'no_topology']
    
    for mode in modes:
        lat_key = f'{mode}_avg_latencies'
        cost_key = f'{mode}_avg_costs'
        if lat_key in d and cost_key in d:
            data[mode] = {
                'latencies': d[lat_key],  
                'costs': d[cost_key],      
                'avg_latency': np.mean(d[lat_key]),
                'std_latency': np.std(d[lat_key]),
                'avg_cost': np.mean(d[cost_key]),
                'std_cost': np.std(d[cost_key]),
            }
    
    return data


def plot_A1_learning_curves(train_data, output_dir):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    plot_order = ['full', 'no_workflow', 'no_future', 'no_topology']
    
    for mode in plot_order:
        if mode not in train_data or len(train_data[mode]['rewards']) == 0:
            continue
        
        best_seed = BEST_SEEDS[mode]
        
        if best_seed is None:
            curves = np.array(train_data[mode]['rewards'])
            main_curve = np.mean(curves, axis=0)
            min_curve = np.min(curves, axis=0)
            max_curve = np.max(curves, axis=0)
        else:
            main_curve = train_data[mode]['seeds'][best_seed]
            all_curves = np.array(train_data[mode]['rewards'])
            min_curve = np.min(all_curves, axis=0)
            max_curve = np.max(all_curves, axis=0)
        
        epochs = np.arange(1, len(main_curve) + 1)

        ax.plot(epochs, main_curve, color=COLORS[mode], label=LABELS[mode], 
                linewidth=2.5, alpha=0.9)

        ax.fill_between(epochs, min_curve, max_curve, color=COLORS[mode], alpha=0.15)
    
    ax.set_xlabel('Training Epochs')
    ax.set_ylabel('Reward')
    ax.set_title('Ablation Learning Curves')
    ax.legend(loc='lower right', fontsize=20, framealpha=0.9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    if train_data.get('full', {}).get('rewards'):
        ax.set_xlim(1, 100)
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'A1_Learning_Curves.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✓ Saved: {filepath}')
    plt.close()


def plot_A2_tradeoff_scatter(inference_data, output_dir):
    if not inference_data:
        print("  [ERROR] No inference data")
        return
    
    fig, ax = plt.subplots(figsize=(9, 7))
    
    modes = ['full', 'no_workflow', 'no_future', 'no_topology']
    
    for mode in modes:
        if mode not in inference_data:
            continue
        
        d = inference_data[mode]
        costs = d['costs']  
        lats = d['latencies']  
        
        
        for i, (cost, lat) in enumerate(zip(costs, lats)):
            label = LABELS[mode] if i == 0 else None  
            ax.scatter(cost, lat, 
                      marker=MARKERS[mode], color=COLORS[mode], 
                      s=150, alpha=0.8,
                      label=label,
                      edgecolors='black', linewidths=1.5)
    
    ax.set_xlabel('Cost ($)')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Impact on Latency-Cost Trade-off')
    ax.legend(loc='upper left', fontsize=20, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'A2_Tradeoff_Scatter.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✓ Saved: {filepath}')
    plt.close()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    full_model_dir = os.path.join(script_dir, '..', 'results', 'STAR_PPO', 'logs')
    output_dir = os.path.join(script_dir, 'figures')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print('Loading data...')
    train_data = load_training_data(results_dir, full_model_dir)
    inference_data = load_inference_data(results_dir)
    
    print('Generating figures...')
    plot_A1_learning_curves(train_data, output_dir)
    plot_A2_tradeoff_scatter(inference_data, output_dir)
    
    print('Done.')


if __name__ == '__main__':
    main()
