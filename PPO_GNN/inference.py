import os
import sys
import numpy as np
import torch
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env import WorkflowDataset, WorkflowMoEEnv

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from agent import PPOAgent
from train import (
    get_static_node_features,
    get_dynamic_node_features,
    build_state_vector,
    map_server_action_to_instance,
    build_server_model_mapping
)

def run_inference(
    data_root='/root/autodl-tmp/MOE111/data',
    model_path=None,
    device='cpu',
    episodes=100,
    test_region='Server3'  
):
    ds = WorkflowDataset(data_root, split='test', regions=[test_region])
    env = WorkflowMoEEnv(ds, device=device)
 
    MODEL_NUM_SERVERS = 500

    all_server_ids = sorted(list(env.servers.keys()))
    actual_num_servers = len(all_server_ids)
 
    if actual_num_servers > MODEL_NUM_SERVERS:
        candidate_server_ids = all_server_ids[:MODEL_NUM_SERVERS]
        print(f"[跨域适配] {test_region}有{actual_num_servers}个服务器，只使用前{MODEL_NUM_SERVERS}个作为候选")
    elif actual_num_servers < MODEL_NUM_SERVERS:
        candidate_server_ids = all_server_ids
        print(f"[跨域适配] {test_region}只有{actual_num_servers}个服务器")
    else:
        candidate_server_ids = all_server_ids
        print(f"[同域测试] {test_region}有{actual_num_servers}个服务器")

    server_ids = candidate_server_ids
    num_servers = len(server_ids)

    agent = PPOAgent(device=device, node_feat_dim=3, global_feat_dim=7, hidden_dim=128)
    if model_path:
        print(f"Loading model from {model_path}")
        agent.net.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Warning: No model path provided, using random agent")
    
    agent.net.eval()
 
    server_model_mapping = build_server_model_mapping(ds, env)

    from torch_geometric.data import Data
    coords = np.array([[env.servers[sid].lon, env.servers[sid].lat] for sid in server_ids])
    K = 10
    edge_indices = []
    for i in range(num_servers):
        dists = np.linalg.norm(coords - coords[i], axis=1)
        nearest_k = np.argsort(dists)[1:K+1]
        for j in nearest_k:
            edge_indices.append([i, j])
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous().to(device)

    edge_attr_list = []
    for i, j in edge_indices:
        dist = np.linalg.norm(coords[i] - coords[j])
        edge_attr_list.append([dist / 20000.0])  
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32).to(device)

    w = np.array([1/3, 1/3, 1/3], dtype=np.float32)
    
    latencies = []
    costs = []
    rewards = []
    switches_list = []
    inference_times = []
    
    print(f"Running inference for {episodes} episodes in {test_region}...")
    
    for i in range(min(episodes, len(ds.tasks))):
        if i % 10 == 0:
            print(f"Episode {i}/{episodes}")
        
        task = ds.tasks[i]
        state_dict = env.reset(task)
        
        ep_lat = 0
        ep_cost = 0
        ep_reward = 0
        done = False
        ep_inference_time = 0
 
        _caps = np.array([env.servers[sid].normalized_compute for sid in server_ids], dtype=np.float32)
        _server_min_costs = []
        for sid in server_ids:
            models_on_server = [mi for mi in env.ds.model_instances if mi.server_id == sid]
            if models_on_server:
                min_cost = min([mi.cost_per_token for mi in models_on_server])
            else:
                min_cost = 0.060
            _server_min_costs.append(min_cost)
        _costs_arr = np.array(_server_min_costs, dtype=np.float32)
        _cost_advantage = 1.0 - np.clip((_costs_arr - 0.0015) / (0.060 - 0.0015), 0, 1.0)
        static_feats = torch.tensor(np.stack([_caps, _cost_advantage], axis=1), dtype=torch.float32).to(device)
        
        while not done:
            current_time = env.current_time_ms
            busy_times = np.array([max(0.0, env.busy_until[sid] - current_time) for sid in server_ids], dtype=np.float32)
            norm_queues = np.clip(busy_times / 5000.0, 0.0, 1.0)
            dynamic_feats = torch.tensor(norm_queues.reshape(-1, 1), dtype=torch.float32).to(device)
            node_feats = torch.cat([static_feats, dynamic_feats], dim=1)  
            
            global_feat = torch.FloatTensor(build_state_vector(state_dict, w)).unsqueeze(0).to(device)  
            
            graph_data = Data(
                x=node_feats,
                edge_index=edge_index,
                edge_attr=edge_attr,
                global_feat=global_feat,
                candidate_mask=torch.ones(num_servers, dtype=torch.bool, device=device)
            )

            import time
            t0 = time.time()
            with torch.no_grad():
                server_idx, _, _ = agent.act(graph_data, deterministic=True)
            ep_inference_time += (time.time() - t0) * 1000  
            required_model_type = state_dict.get('required_model_type', 'gpt-4o-mini')
            model_instance_id = map_server_action_to_instance(
                server_idx, required_model_type, server_model_mapping, ds
            )
            
            if model_instance_id is None:
                break
            
            next_state_dict, (rL, rC, rS), done, info = env.step(model_instance_id)
            
            ep_lat += info['latency_ms']
            ep_cost += info['cost']
            ep_reward += w[0]*rL + w[1]*rC + w[2]*rS
            
            state_dict = next_state_dict
            
        latencies.append(ep_lat)
        costs.append(ep_cost)
        rewards.append(ep_reward)
        switches_list.append(env.ep_switches)
        inference_times.append(ep_inference_time)
    
    avg_lat = np.mean(latencies)
    avg_cost = np.mean(costs)

    print("\n" + "="*50)
    print("Inference Results:")
    print("="*50)
    print(f"Episodes: {len(latencies)}")
    print(f"Average Latency: {np.mean(latencies):.2f} ms (std: {np.std(latencies):.2f})")
    print(f"Average Cost:    ${np.mean(costs):.4f} (std: ${np.std(costs):.4f})")
    print(f"Average Reward:  {np.mean(rewards):.4f} (std: {np.std(rewards):.4f})")
    print(f"Average Switches: {np.mean(switches_list):.2f}")
    print(f"Avg Inference Time: {np.mean(inference_times):.3f} ms")
    print("="*50)

    if model_path:
        output_dir = 'inference/results'
        os.makedirs(output_dir, exist_ok=True)
        model_basename = os.path.basename(model_path).replace('.pt', '')
        npz_path = os.path.join(output_dir, f'PPO_GNN_{model_basename}_detailed.npz')
        np.savez(npz_path, 
                 latencies=np.array(latencies),
                 costs=np.array(costs),
                 rewards=np.array(rewards),
                 switches=np.array(switches_list),
                 inference_times=np.array(inference_times))
        print(f"Detailed results saved to: {npz_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help='Path to trained model')
    parser.add_argument('--model_path', type=str, default=None, help='Alias for --model')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    model_path = args.model or args.model_path
    
    run_inference(model_path=model_path, device=args.device, episodes=args.episodes)

