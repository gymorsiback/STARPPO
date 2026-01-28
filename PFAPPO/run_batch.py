import os
import subprocess
import time
import argparse

def run_batch_training():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=5, help='Number of runs')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs per run')
    parser.add_argument('--episodes', type=int, default=200, help='Episodes per epoch')
    args = parser.parse_args()

    seeds = [100 + i for i in range(args.n)]

    script_path = os.path.join(os.path.dirname(__file__), 'train.py')
    
    print(f"Starting batch training for {len(seeds)} seeds: {seeds}")
    
    for seed in seeds:
        print(f"\n{'='*40}")
        print(f"Running Seed: {seed}")
        print(f"{'='*40}")

        cmd = [
            'python', script_path,
            '--epochs', str(args.epochs),
            '--episodes', str(args.episodes),
            '--seed', str(seed)
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"Seed {seed} completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error running seed {seed}: {e}")
            
    print("\nAll training runs completed.")

if __name__ == '__main__':
    run_batch_training()
