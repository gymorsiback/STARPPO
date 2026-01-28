import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot_single_run(log_dir, run_id=None):
    if run_id is None:
        files = [f for f in os.listdir(log_dir) if f.startswith('metrics_') and f.endswith('.npz')]
        if not files:
            print(f"No metrics files found in {log_dir}")
            return
        files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)))
        latest_file = files[-1]
        run_id = latest_file.replace('metrics_', '').replace('.npz', '')
        print(f"Plotting latest run: {run_id}")
 
    csv_path = os.path.join(log_dir, f"metrics_{run_id}.csv")
    npz_path = os.path.join(log_dir, f"metrics_{run_id}.npz")
    
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        loss = df['loss'].values
        accuracy = df['accuracy'].values
    elif os.path.exists(npz_path):
        data = np.load(npz_path)
    loss = data['loss']
    accuracy = data['accuracy']
    else:
        print(f"Metrics file not found: {csv_path} or {npz_path}")
        return
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(loss, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy, color='orange', label='Validation Accuracy')
    plt.title('Accuracy (vs Expert)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    save_path = os.path.join(log_dir, f"plot_{run_id}.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, default=None, help='Run ID to plot (optional, defaults to latest)')
    args = parser.parse_args()
    
    log_dir = os.path.join('results', 'Stark_Scheduler', 'logs')
    plot_single_run(log_dir, args.run_id)

