import os
import argparse
import subprocess
import time

def run_batch_training():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--n', type=int, default=5, help='Number of runs')
    args = parser.parse_args()

    start_seed = 200
    seeds = [start_seed + i for i in range(args.n)]
    
    epochs = args.epochs
    episodes = args.episodes

    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, 'train.py')
    
    print(f"Starting batch training for PPO Algorithm...")
    print(f"Runs: {args.n}, Epochs: {epochs}, Episodes: {episodes}")
    print(f"Seeds: {seeds}")
    
    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{args.n}] Starting Run with Seed {seed}...")
        start_time = time.time()
        
        cmd = [
            'python', script_path,
            '--epochs', str(epochs),
            '--episodes', str(episodes),
            '--seed', str(seed)
        ]
        
        try:
            subprocess.run(cmd, check=True)
            duration = time.time() - start_time
            print(f"[{i+1}/{args.n}] Run finished in {duration:.1f}s")
        except subprocess.CalledProcessError as e:
            print(f"[{i+1}/{args.n}] Run failed with error: {e}")
            continue
            
    print(f"\nAll {args.n} runs completed.")

if __name__ == '__main__':
    run_batch_training()

