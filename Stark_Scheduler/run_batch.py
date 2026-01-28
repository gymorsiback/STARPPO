import argparse
import subprocess
import os
import time

def run_batch_training(n_runs=5, epochs=50):
    print("=" * 70)
    print(f"Stark-Scheduler Batch Training")
    print(f"Total Runs: {n_runs}, Epochs per Run: {epochs}")
    print("=" * 70)
 
    log_dir = "results/Stark_Scheduler/logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    total_start = time.time()
    
    for i in range(n_runs):
        run_start = time.time()
        
        print(f"\n{'#' * 70}")
        print(f"# Starting Run {i+1}/{n_runs}")
        print(f"{'#' * 70}\n")
        
        cmd = [
            "python", "Stark_Scheduler/train.py",
            "--epochs", str(epochs),
            "--data_path", "data",
            "--run_idx", str(i + 1),
            "--total_runs", str(n_runs)
        ]
  
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        
        run_time = time.time() - run_start
        
        if process.returncode == 0:
            print(f"\n✓ Run {i+1}/{n_runs} completed in {run_time:.1f}s")
        else:
            print(f"\n✗ Run {i+1}/{n_runs} failed with return code {process.returncode}")
    
    total_time = time.time() - total_start
    
    print("\n" + "=" * 70)
    print(f"Batch Training Complete!")
    print(f"Total Runs: {n_runs}, Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 70)
    print(f"\nResults saved to:")
    print(f"  - Models: results/Stark_Scheduler/models/")
    print(f"  - Logs:   results/Stark_Scheduler/logs/")
    print(f"\nTo plot results, run:")
    print(f"  python Stark_Scheduler/plot_batch.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=5, help='Number of runs')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs per run')
    args = parser.parse_args()
    
    run_batch_training(args.n, args.epochs)
