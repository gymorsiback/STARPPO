import os
import sys
import numpy as np
import time
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env import WorkflowDataset, WorkflowMoEEnv

def run_inference(
    data_root='/root/autodl-tmp/MOE111/data',
    episodes=100,
    test_region='Server2',
    split='train'  
):
    ds = WorkflowDataset(data_root, split=split, regions=[test_region])
    env = WorkflowMoEEnv(ds)
 
    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    
    latencies = []
    costs = []
    rewards = []
    switches_list = []
    inference_times = []
    
    print(f"Running Random inference on {episodes} episodes in {test_region}...")
    
    for i in range(episodes):
        if i % 10 == 0:
            print(f"Episode {i}/{episodes}")
        
        task = ds.tasks[i % len(ds.tasks)]
        env.reset(task)
        
        ep_lat = 0
        ep_cost = 0
        ep_reward = 0
        ep_inference_time = 0
        done = False
        
        while not done:
            t0 = time.time()

            candidates = env.available_actions()
            if candidates:
                action = random.choice(candidates)
            else:
                action = random.randint(0, len(env.actions) - 1)
            
            ep_inference_time += (time.time() - t0) * 1000
            
            _, (rL, rC, rS), done, info = env.step(action)
            
            r_scalar = w[0]*rL + w[1]*rC + w[2]*rS
            
            ep_lat += info['latency_ms']
            ep_cost += info['cost']
            ep_reward += r_scalar
            
        latencies.append(ep_lat)
        costs.append(ep_cost)
        rewards.append(ep_reward)
        switches_list.append(env.ep_switches)
        inference_times.append(ep_inference_time)
 
    print("\n" + "="*50)
    print("Random Inference Results:")
    print("="*50)
    print(f"Episodes: {episodes}")
    print(f"Average Latency: {np.mean(latencies):.2f} ms")
    print(f"Average Cost:    ${np.mean(costs):.4f}")
    print(f"Average Reward:  {np.mean(rewards):.4f}")
    print("="*50)

    output_dir = 'inference/results'
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, f'Random_{test_region}_{split}_detailed.npz')
    np.savez(npz_path, 
             latencies=np.array(latencies),
             costs=np.array(costs),
             rewards=np.array(rewards),
             switches=np.array(switches_list),
             inference_times=np.array(inference_times))
    print(f"Results saved to: {npz_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train', help='train or test')
    parser.add_argument('--region', type=str, default='Server2')
    parser.add_argument('--episodes', type=int, default=200)
    args = parser.parse_args()
    run_inference(test_region=args.region, episodes=args.episodes, split=args.split)
