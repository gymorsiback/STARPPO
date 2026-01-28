import argparse
import subprocess
import time
import sys
import os

def run_batch():
    parser = argparse.ArgumentParser(description='Run batch training for A3C')
    parser.add_argument('--n', type=int, default=5, help='Number of runs')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs per run')
    parser.add_argument('--episodes', type=int, default=200, help='Episodes per epoch')
    args = parser.parse_args()

    seeds = [300 + i for i in range(args.n)]
    
    script_path = os.path.join(os.path.dirname(__file__), 'train.py')
    
    print(f"Starting Batch Training for A3C_algorithm: {args.n} runs, {args.epochs} epochs each")
    
    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{args.n}] Starting Run with Seed {seed}...")
        start_time = time.time()
        
        cmd = [
            'python', script_path,
            '--epochs', str(args.epochs),
            '--episodes', str(args.episodes),
            '--seed', str(seed)
        ]
        
        try:
            subprocess.run(cmd, check=True)
            duration = time.time() - start_time
            print(f"[{i+1}/{args.n}] Run finished in {duration:.1f}s")
        except subprocess.CalledProcessError as e:
            print(f"[{i+1}/{args.n}] Run failed with error: {e}")
            continue
        except KeyboardInterrupt:
            print("\nBatch training interrupted.")
            sys.exit(1)

    print("\nBatch training completed.")

if __name__ == "__main__":
    run_batch()

