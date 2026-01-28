import argparse
import subprocess
import sys
import time
import os

def run_experiment(seed, gpu_id, n_epochs):
    print(f"Starting experiment with seed {seed} on GPU {gpu_id}")
    cmd = [
        sys.executable, "Trans/train.py",
        "--epochs", str(n_epochs),
        "--episodes", "200",
        "--device", "cuda" if gpu_id is not None else "cpu",
        "--seed", str(seed),
        "--seq_len", "5"
    ]
    
    log_dir = os.path.join("results", "Trans", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"Trans_seed_{seed}.log")
    
    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"Finished experiment with seed {seed}, log saved to {log_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Number of runs")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs per run")
    parser.add_argument("--start_seed", type=int, default=600, help="Starting seed")
    args = parser.parse_args()

    seeds = [args.start_seed + i for i in range(args.n)]
    
    for seed in seeds:
        run_experiment(seed, 0, args.epochs)

    print("All runs completed. Generating plots...")
    subprocess.run([sys.executable, "Trans/plot_batch.py"])

if __name__ == "__main__":
    main()


