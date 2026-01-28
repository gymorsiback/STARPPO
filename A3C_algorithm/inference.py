import os
import sys
import argparse
import numpy as np
import torch
import time
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env import WorkflowDataset, WorkflowMoEEnv
from A3C_algorithm.model import ActorCritic

def build_server_model_mapping(ds, env):
    server_ids = sorted(list(env.servers.keys()))
    server_id_to_idx = {sid: i for i, sid in enumerate(server_ids)}
    
    mapping = {i: {} for i in range(len(server_ids))}
    
    for mi in ds.model_instances:
        server_idx = server_id_to_idx.get(mi.server_id)
        if server_idx is not None:
            if mi.model_type not in mapping[server_idx]:
                mapping[server_idx][mi.model_type] = []
            mapping[server_idx][mi.model_type].append(mi.idx)
    
    return mapping

def map_server_action_to_instance(server_idx, required_model_type, mapping, ds, fallback_action=0):
    if server_idx in mapping:
        instances = mapping[server_idx].get(required_model_type, [])
        if instances:
            return instances[0]
    
    for mi in ds.model_instances:
        if mi.model_type == required_model_type:
            return mi.idx
    
    return fallback_action

def build_state_vector(state_dict, dwa_weights):
    if isinstance(dwa_weights, torch.Tensor):
        w = dwa_weights.tolist()
    else:
        w = dwa_weights
        
    return np.array([
        state_dict['step_norm'],
        state_dict['task_lon'],
        state_dict['task_lat'],
        float(state_dict['prev_region_id']),
        w[0],
        w[1],
        w[2]
    ], dtype=np.float32)

def run_inference(args):
    if hasattr(args, 'regions') and args.regions:
        test_region = args.regions[0]
    else:
        test_region = 'Server2'
        
    print(f"Testing on Region: {test_region}")
    
    ds = WorkflowDataset(args.data_path, split='test', regions=[test_region])
    env = WorkflowMoEEnv(ds)

    original_num_servers = 500
    if len(env.servers) != original_num_servers:
        print(f"[Cross-Domain] Filtering {test_region} ({len(env.servers)}) to first {original_num_servers} servers.")
        filtered_server_ids = sorted(list(env.servers.keys()))[:original_num_servers]

        temp_ds = WorkflowDataset(args.data_path, split='test', regions=[test_region])
        temp_ds.servers = {sid: env.servers[sid] for sid in filtered_server_ids}
        temp_ds.model_instances = [mi for mi in env.ds.model_instances if mi.server_id in filtered_server_ids]
        temp_ds.num_actions = len(temp_ds.model_instances)
        
        env = WorkflowMoEEnv(temp_ds)
        
    num_servers = len(env.servers) 
    server_model_mapping = build_server_model_mapping(env.ds, env)

    model = ActorCritic(state_dim=7, num_servers=num_servers).to(args.device)
    print(f"Loading model from {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(state_dict)
    model.eval()

    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    
    latencies = []
    costs = []
    rewards = []
    switches_list = []
    inference_times = []
    violations = 0
    episodes_to_run = args.episodes
    
    print(f"Running inference for {episodes_to_run} episodes...")
    
    for ep in range(min(episodes_to_run, len(ds.tasks))):
        task = ds.tasks[ep]
        state_dict = env.reset(task)
        done = False
        ep_latency = 0
        ep_cost = 0
        ep_reward = 0
        ep_switches = 0
        ep_inference_time = 0
        
        while not done:
            t0 = time.time()
            
            s_vec = build_state_vector(state_dict, w)
            s_tensor = torch.FloatTensor(s_vec).unsqueeze(0).to(args.device)
            
            with torch.no_grad():
                logits, _ = model(s_tensor)
                action = torch.argmax(logits).item()
                
            t1 = time.time()
            ep_inference_time += (t1 - t0) * 1000
            
            _, _, req_type = env.cur_steps[env.step_idx]
            if req_type is None:
                req_type = env.cur_task['RequiredModelTypes'][env.step_idx]
                
            real_action = map_server_action_to_instance(
                action, str(req_type), server_model_mapping, env.ds
            )
            
            next_state_dict, (rL, rC, rS), done, info = env.step(real_action)
            
            ep_latency += info.get('latency_ms', info.get('latency', 0))
            ep_cost += info.get('cost', 0)
            ep_reward += (w[0] * rL + w[1] * rC + w[2] * rS)
            
            if info.get('switched', False):
                ep_switches += 1
            if info.get('sla_violated', False):
                violations += 1
                
            state_dict = next_state_dict
            
        latencies.append(ep_latency)
        costs.append(ep_cost)
        rewards.append(ep_reward)
        switches_list.append(ep_switches)
        steps_in_episode = len(task.get('RequiredModelTypes', [1]))
        inference_times.append(ep_inference_time / steps_in_episode if steps_in_episode > 0 else 0)
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{episodes_to_run} - Latency: {ep_latency:.2f}, Cost: {ep_cost:.4f}")

    avg_lat = np.mean(latencies)
    avg_cost = np.mean(costs)
    avg_reward = np.mean(rewards)
    total_steps = sum(len(ds.tasks[i].get('RequiredModelTypes', [1])) for i in range(min(episodes_to_run, len(ds.tasks))))
    violation_rate = (violations / total_steps) * 100 if total_steps > 0 else 0
    
    print("\n" + "="*40)
    print(f"A3C Inference Results ({test_region})")
    print(f"Average Latency: {avg_lat:.2f} ms")
    print(f"Average Cost: ${avg_cost:.4f}")
    print(f"Average Reward: {avg_reward:.4f}")
    print(f"Violation Rate: {violation_rate:.2f}%")
    print("="*40 + "\n")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_id = "unknown"
    match = re.search(r'([a-f0-9]{6})', args.model_path)
    if match: model_id = match.group(1)
    
    save_dir = 'inference/results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    filename = f"A3C_a3c_{timestamp}_{model_id}_inference.npz"
    save_path = os.path.join(save_dir, filename)
    
    np.savez(
        save_path,
        latencies=np.array(latencies),
        costs=np.array(costs),
        rewards=np.array(rewards),
        switches=np.array(switches_list),
        inference_times=np.array(inference_times),
        avg_latency=avg_lat,
        avg_cost=avg_cost,
        avg_reward=avg_reward,
        violation_rate=violation_rate
    )
    print(f"Detailed results saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='data/alibaba_data.csv')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu') 
    parser.add_argument('--regions', nargs='+')
    args = parser.parse_args()
    
    run_inference(args)









