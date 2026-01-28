import os
import sys
import argparse
import numpy as np
import torch
import time
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env import WorkflowDataset, WorkflowMoEEnv
from Stark_Scheduler.model import StarkScheduler
from Stark_Scheduler.dataset import OnlineExpertDataset

def run_inference(args):
    if hasattr(args, 'regions') and args.regions:
        test_region = args.regions[0]
    else:
        test_region = 'Server2'
        
    print(f"Testing on Region: {test_region}")
 
    ds_test = WorkflowDataset(args.data_path, split='test', regions=[test_region])

    num_servers_trained = 500 

    env = WorkflowMoEEnv(ds_test)
 
    env.reset(ds_test.tasks[0])
    dummy_task, dummy_servers = OnlineExpertDataset(env).extract_structured_state(env)
    task_dim = dummy_task.shape[0]
    server_dim = dummy_servers.shape[1]
    
    model = StarkScheduler(
        task_dim=task_dim,
        server_dim=server_dim,
        num_servers=num_servers_trained,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2
    ).to(args.device)

    print(f"Loading model from {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(state_dict)
    model.eval()

    latencies = []
    costs = []
    rewards = []
    switches_list = []
    inference_times = []
    violations = 0
    total_episodes = 0
    
    episodes_to_run = args.episodes
    
    print(f"Running inference for {episodes_to_run} episodes...")
    
    for ep in range(min(episodes_to_run, len(ds_test.tasks))):
        task = ds_test.tasks[ep]
        state = env.reset(task)
        done = False
        ep_latency = 0
        ep_cost = 0
        ep_reward = 0
        ep_switches = 0
        ep_inference_time = 0
        
        while not done:
            t0 = time.time()
            
            task_feat, server_feats = OnlineExpertDataset(env).extract_structured_state(env)
 
            server_feats = server_feats[:num_servers_trained]
            
            task_tensor = torch.FloatTensor(task_feat).unsqueeze(0).to(args.device) 
            server_tensor = torch.FloatTensor(server_feats).unsqueeze(0).to(args.device) 
            
            with torch.no_grad():
                logits = model(task_tensor, server_tensor)
                server_action = torch.argmax(logits, dim=1).item()  
            
            t1 = time.time()
            ep_inference_time += (t1 - t0) * 1000 

            server_ids = sorted(list(env.servers.keys()))
            target_server_id = server_ids[server_action] if server_action < len(server_ids) else server_ids[0]

            req_type = env.cur_task['RequiredModelTypes'][env.step_idx]
            action = None
            for idx, mi in enumerate(env.ds.model_instances):
                if mi.server_id == target_server_id and mi.model_type == req_type:
                    action = idx
                    break

            if action is None:
                for idx, mi in enumerate(env.ds.model_instances):
                    if mi.model_type == req_type and mi.server_id in server_ids[:num_servers_trained]:
                        action = idx
                        break

            if action is None:
                avail = env.available_actions()
                action = avail[0] if avail else 0
            
            next_state, (rL, rC, rS), done, info = env.step(action)
            
            ep_latency += info.get('latency_ms', info.get('latency', 0))
            ep_cost += info.get('cost', 0)
            w = [0.45, 0.40, 0.15]
            ep_reward += w[0]*rL + w[1]*rC + w[2]*rS
            
            if info.get('switched', False):
                ep_switches += 1
            
            if info.get('sla_violated', False):
                violations += 1
                
        latencies.append(ep_latency)
        costs.append(ep_cost)
        rewards.append(ep_reward)
        switches_list.append(ep_switches)
        steps_in_episode = len(task.get('RequiredModelTypes', [1]))
        inference_times.append(ep_inference_time / steps_in_episode if steps_in_episode > 0 else 0) 
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{episodes_to_run} - Latency: {ep_latency:.2f}, Reward: {ep_reward:.2f}")
 
    avg_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    avg_cost = np.mean(costs)
    std_cost = np.std(costs)
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    total_steps = sum(len(ds_test.tasks[i].get('RequiredModelTypes', [1])) for i in range(min(episodes_to_run, len(ds_test.tasks))))
    violation_rate = (violations / total_steps) * 100 if total_steps > 0 else 0
    
    print("\n" + "="*40)
    print(f"Stark-Scheduler Inference Results ({test_region})")
    print(f"Average Latency: {avg_lat:.2f} ms (std: {std_lat:.2f})")
    print(f"Average Cost: ${avg_cost:.4f} (std: {std_cost:.4f})")
    print(f"Average Reward: {avg_reward:.4f} (std: {std_reward:.4f})")
    print(f"Violation Rate: {violation_rate:.2f}%")
    print("="*40 + "\n")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_id = "unknown"
    match = re.search(r'([a-f0-9]{6})', args.model_path)
    if match:
        model_id = match.group(1)
        
    save_dir = 'inference/results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    filename = f"Stark_stark_{timestamp}_{model_id}_inference.npz"
    save_path = os.path.join(save_dir, filename)
    
    np.savez(
        save_path,
        latencies=np.array(latencies),
        costs=np.array(costs),
        rewards=np.array(rewards),
        switches=np.array(switches_list),
        inference_times=np.array(inference_times),
        avg_latency=avg_lat,
        std_latency=std_lat,
        avg_cost=avg_cost,
        std_cost=std_cost,
        avg_reward=avg_reward,
        std_reward=std_reward,
        violation_rate=violation_rate
    )
    print(f"Detailed results saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, default='data/alibaba_data.csv')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--regions', nargs='+', help='Regions to test on (e.g. Server2, Server3)')
    
    args = parser.parse_args()
    run_inference(args)


    save_path = os.path.join(save_dir, filename)
    
    np.savez(
        save_path,
        latencies=np.array(latencies),
        costs=np.array(costs),
        rewards=np.array(rewards),
        switches=np.array(switches_list),
        inference_times=np.array(inference_times),
        avg_latency=avg_lat,
        std_latency=std_lat,
        avg_cost=avg_cost,
        std_cost=std_cost,
        avg_reward=avg_reward,
        std_reward=std_reward,
        violation_rate=violation_rate
    )
    print(f"Detailed results saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, default='data/alibaba_data.csv')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--regions', nargs='+', help='Regions to test on (e.g. Server2, Server3)')
    
    args = parser.parse_args()
    run_inference(args)


    save_path = os.path.join(save_dir, filename)
    
    np.savez(
        save_path,
        latencies=np.array(latencies),
        costs=np.array(costs),
        rewards=np.array(rewards),
        switches=np.array(switches_list),
        inference_times=np.array(inference_times),
        avg_latency=avg_lat,
        std_latency=std_lat,
        avg_cost=avg_cost,
        std_cost=std_cost,
        avg_reward=avg_reward,
        std_reward=std_reward,
        violation_rate=violation_rate
    )
    print(f"Detailed results saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, default='data/alibaba_data.csv')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--regions', nargs='+', help='Regions to test on (e.g. Server2, Server3)')
    
    args = parser.parse_args()
    run_inference(args)

