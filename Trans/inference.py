
import os
import sys
import numpy as np
import torch
from collections import deque

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from env import WorkflowDataset, WorkflowMoEEnv

sys.path.insert(0, os.path.dirname(__file__))
from agent import TransPPOAgent
from train import StateBuffer, build_state_vector

def run_inference(
    data_root='/root/autodl-tmp/MOE111/data',
    model_path=None,
    device='cpu',
    episodes=100,
    test_region='Server3'  
):
    ds = WorkflowDataset(data_root, split='test', regions=[test_region])
    env = WorkflowMoEEnv(ds)

    MODEL_NUM_SERVERS = 500

    actual_num_servers = len(env.servers)
    server_ids = sorted(list(env.servers.keys()))

    if actual_num_servers > MODEL_NUM_SERVERS:
        candidate_server_ids = server_ids[:MODEL_NUM_SERVERS]
        print(f"[跨域适配] {test_region}有{actual_num_servers}个服务器，只使用前{MODEL_NUM_SERVERS}个作为候选")
    elif actual_num_servers < MODEL_NUM_SERVERS:
        candidate_server_ids = server_ids
        print(f"[跨域适配] {test_region}只有{actual_num_servers}个服务器")
    else:
        candidate_server_ids = server_ids
        print(f"[同域测试] {test_region}有{actual_num_servers}个服务器")

    agent = TransPPOAgent(
        state_dim=7, 
        num_servers=MODEL_NUM_SERVERS, 
        lr=3e-4,
        device=device
    )

    if model_path and os.path.exists(model_path):
        agent.model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from: {model_path}")
    else:
        print("No model loaded. Using random policy.")
    
    agent.model.eval()

    def map_action_to_instance(action_idx, candidate_sids, env):
        if action_idx >= len(candidate_sids):
            action_idx = len(candidate_sids) - 1
        target_server_id = candidate_sids[action_idx]
        for idx, mi in enumerate(env.ds.model_instances):
            if mi.server_id == target_server_id:
                return idx
        return 0

    w = np.array([1/3, 1/3, 1/3], dtype=np.float32)

    latencies = []
    costs = []
    rewards = []
    switches_list = []
    inference_times = []
    
    import time

    seq_len = 5
    state_buf = StateBuffer(seq_len=seq_len, state_dim=7)
    
    print(f"Running inference on {episodes} episodes in {test_region}...")
    
    for i in range(min(episodes, len(ds.tasks))):
        if i % 10 == 0:
            print(f"Episode {i}/{episodes}")
        
        task = ds.tasks[i]
        state_dict = env.reset(task)
        state_buf.reset()
        
        ep_lat = 0
        ep_cost = 0
        ep_reward = 0
        ep_inference_time = 0
        done = False
        
        while not done:
            s_vec = build_state_vector(state_dict, w)
            state_buf.append(s_vec)
            s_seq = state_buf.get_sequence()  
            
            s_tensor = torch.FloatTensor(s_seq).unsqueeze(0).to(device)  
            t0 = time.time()
            with torch.no_grad():
                logits, _ = agent.model(s_tensor)
                valid_logits = logits[0, :len(candidate_server_ids)]
                action = torch.argmax(valid_logits).item()
            ep_inference_time += (time.time() - t0) * 1000
            
            env_action = map_action_to_instance(action, candidate_server_ids, env)
            next_state_dict, (rL, rC, rS), done, info = env.step(env_action)
            
            r_scalar = w[0]*rL + w[1]*rC + w[2]*rS
            
            ep_lat += info['latency_ms']
            ep_cost += info['cost']
            ep_reward += r_scalar
            
            state_dict = next_state_dict
        
        latencies.append(ep_lat)
        costs.append(ep_cost)
        rewards.append(ep_reward)
        switches_list.append(env.ep_switches)
        inference_times.append(ep_inference_time)

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
        npz_path = os.path.join(output_dir, f'Trans_{model_basename}_detailed.npz')
        np.savez(npz_path, 
                 latencies=np.array(latencies),
                 costs=np.array(costs),
                 rewards=np.array(rewards),
                 switches=np.array(switches_list),
                 inference_times=np.array(inference_times))
        print(f"Detailed results saved to: {npz_path}")
    
    return {
        'latencies': latencies,
        'costs': costs,
        'rewards': rewards
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
    if torch.cuda.is_available() and args.device == 'cpu':
        args.device = 'cuda'
    
    run_inference(model_path=args.model, episodes=args.episodes, device=args.device)
