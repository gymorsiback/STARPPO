import argparse
import subprocess
import sys
import time
import os
from concurrent.futures import ThreadPoolExecutor

def run_experiment(seed, gpu_id, n_epochs):
    print(f"Starting experiment with seed {seed} on GPU {gpu_id}")
    cmd = [
        sys.executable, "PPO_CN/train.py",
        "--epochs", str(n_epochs),
        "--episodes", "200",
        "--device", "cuda" if gpu_id is not None else "cpu",
        "--seed", str(seed),
        "--beta", "1.0",    # Pink Noise
        "--scale", "0.03"   # Standard colored noise scale
    ]

    log_dir = os.path.join("results", "PPO_CN", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"PPO_CN_seed_{seed}.log")

    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"Finished experiment with seed {seed}, log saved to {log_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Number of runs")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs per run")
    parser.add_argument("--start_seed", type=int, default=500, help="Starting seed")
    args = parser.parse_args()

    seeds = [args.start_seed + i for i in range(args.n)]

    # Run sequentially to avoid GPU OOM if limited memory, or parallel if possible
    # Given the MLP model is small, we might be able to run parallel.
    # But let's stick to sequential or semi-parallel to be safe.
    # PPO_GNN ran sequentially. PPO ran sequentially.

    for seed in seeds:
        run_experiment(seed, 0, args.epochs)

    # After all runs, plot
    print("All runs completed. Generating plots...")
    subprocess.run([sys.executable, "PPO_CN/plot_batch.py"])

if __name__ == "__main__":
    main()

