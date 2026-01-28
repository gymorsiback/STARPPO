
import os
import sys
import glob
import numpy as np
import torch
import time
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env import WorkflowDataset, WorkflowMoEEnv
from utils import haversine_km

BYTES_PER_TOKEN = 4  
NETWORK_COST_PER_MS = 0.00015  
COMM_COST_PER_MS = 0.00015    

def compute_network_cost_by_latency(network_ms, tokens):
    data_kb = (tokens * BYTES_PER_TOKEN) / 1024.0
    return data_kb * network_ms * NETWORK_COST_PER_MS

def compute_communication_cost_by_latency(tokens, network_ms):
    intermediate_tokens = tokens * 0.3  
    data_kb = (intermediate_tokens * BYTES_PER_TOKEN) / 1024.0
    return data_kb * network_ms * COMM_COST_PER_MS

DATA_ROOT = '/root/autodl-tmp/MOE111/data1'
REGION = 'Server1_Trap'  
NUM_SERVERS = 500
EPISODES = 200  

SEED_MODEL_MAP = {
    'STAR_PPO': {
        42: '/root/autodl-tmp/MOE111/results/STAR_PPO/models/star_ppo_20251229_213311_4c5d57_actor_epoch_95.pt',
        43: '/root/autodl-tmp/MOE111/results/STAR_PPO/models/star_ppo_20251230_094141_1d8480_actor_epoch_95.pt',
        44: '/root/autodl-tmp/MOE111/results/STAR_PPO/models/star_ppo_20251230_111255_dd53d6_actor_epoch_95.pt',
    },
    'PFAPPO': {
        42: '/root/autodl-tmp/MOE111/results/PFAPPO/models/pfappo_20251229_213321_3f2d75_actor_epoch_0099.pt',
        43: '/root/autodl-tmp/MOE111/results/PFAPPO/models/pfappo_20251230_094157_ef1d71_actor_epoch_0099.pt',
        44: '/root/autodl-tmp/MOE111/results/PFAPPO/models/pfappo_20251230_111303_d2fbe0_actor_epoch_0099.pt',
    },
    'PPO_CN': {
        42: '/root/autodl-tmp/MOE111/results/PPO_CN/models/ppo_cn_20251229_223055_39fdf2_actor_epoch_0099.pt',
        43: '/root/autodl-tmp/MOE111/results/PPO_CN/models/ppo_cn_20251230_095158_f90ef7_actor_epoch_0099.pt',
        44: '/root/autodl-tmp/MOE111/results/PPO_CN/models/ppo_cn_20251230_111724_6c97c1_actor_epoch_0099.pt',
    },
    'Trans': {
        42: '/root/autodl-tmp/MOE111/results/Trans/models/trans_ppo_20251229_222524_01b5cd_model_epoch_0099.pt',
        43: '/root/autodl-tmp/MOE111/results/Trans/models/trans_ppo_20251230_095549_9078e4_model_epoch_0099.pt',
        44: '/root/autodl-tmp/MOE111/results/Trans/models/trans_ppo_20251230_112122_aa88d2_model_epoch_0099.pt',
    },
    'A3C': {
        42: '/root/autodl-tmp/MOE111/results/A3C_algorithm/models/a3c_20251229_221515_c881ff_actor_final.pt',
        43: '/root/autodl-tmp/MOE111/results/A3C_algorithm/models/a3c_20251230_095602_779931_actor_final.pt',
        44: '/root/autodl-tmp/MOE111/results/A3C_algorithm/models/a3c_20251230_112751_fbf42a_actor_final.pt',
    },
    'PPO_GNN': {
        42: '/root/autodl-tmp/MOE111/results/PPO_GNN/models/ppo_gnn_20251229_222534_768885_model_epoch_0099.pt',
        43: '/root/autodl-tmp/MOE111/results/PPO_GNN/models/ppo_gnn_20251230_094312_2b1b24_model_epoch_0099.pt',
        44: '/root/autodl-tmp/MOE111/results/PPO_GNN/models/ppo_gnn_20251230_111324_762a02_model_epoch_0099.pt',
    },
    'PPO': {
        42: '/root/autodl-tmp/MOE111/results/PPO/models/ppo_20251229_221507_4318c2_actor_epoch_0099.pt',
        43: '/root/autodl-tmp/MOE111/results/PPO/models/ppo_20251230_094803_706be4_actor_epoch_0099.pt',
        44: '/root/autodl-tmp/MOE111/results/PPO/models/ppo_20251230_111316_9e7fcf_actor_epoch_0099.pt',
    },
    'Stark': {
        42: '/root/autodl-tmp/MOE111/results/Stark_Scheduler/models/run_20251229_223440_fcbf08_20251229_223440_final.pt',
        43: '/root/autodl-tmp/MOE111/results/Stark_Scheduler/models/run_20251230_100949_c92c37_20251230_100949_final.pt',
        44: '/root/autodl-tmp/MOE111/results/Stark_Scheduler/models/run_20251230_112606_582246_20251230_112606_final.pt',
    },
}

def find_latest_model(algo_name, pattern_suffix='_actor_epoch_0099.pt', train_seed=None):
    PROJECT_ROOT = '/root/autodl-tmp/MOE111'

    if train_seed is not None and algo_name in SEED_MODEL_MAP:
        if train_seed in SEED_MODEL_MAP[algo_name]:
            model_path = SEED_MODEL_MAP[algo_name][train_seed]
            if os.path.exists(model_path):
                return model_path
            else:
                print(f"Warning: Model not found: {model_path}")
    
    dir_map = {
        'PFAPPO': f'{PROJECT_ROOT}/results/PFAPPO/models',
        'STAR_PPO': f'{PROJECT_ROOT}/results/STAR_PPO/models',
        'PPO': f'{PROJECT_ROOT}/results/PPO/models',
        'PPO_CN': f'{PROJECT_ROOT}/results/PPO_CN/models',
        'PPO_GNN': f'{PROJECT_ROOT}/results/PPO_GNN/models',
        'Trans': f'{PROJECT_ROOT}/results/Trans/models',
        'Stark': f'{PROJECT_ROOT}/results/Stark_Scheduler/models',
        'A3C': f'{PROJECT_ROOT}/results/A3C_algorithm/models'
    }
    
    model_dir = dir_map.get(algo_name)
    if not model_dir or not os.path.exists(model_dir):
        return None
    
    patterns = {
        'STAR_PPO': ['*_actor_epoch_*.pt'],
        'Stark': ['*_final.pt'],
        'PPO_GNN': ['*_model_epoch_0099.pt', '*_model_epoch_*.pt'],
        'Trans': ['*_model_epoch_0099.pt', '*_model_epoch_*.pt'],
        'A3C': ['*_actor_final.pt', '*_actor_epoch_*.pt'],
        'PFAPPO': ['*_actor_epoch_0099.pt', '*_actor_epoch_*.pt'],
        'PPO_CN': ['*_actor_epoch_0099.pt', '*_actor_epoch_*.pt'],
        'PPO': ['*_actor_epoch_0099.pt', '*_final.pt', '*_epoch_*.pt'],
    }
    
    algo_patterns = patterns.get(algo_name, ['*_actor_epoch_0099.pt', '*.pt'])
    
    for pattern in algo_patterns:
        files = glob.glob(os.path.join(model_dir, pattern))
        if files:
            files.sort(key=os.path.getmtime, reverse=True)
            return files[0]
    
    return None


def run_greedy_inference(env, ds, episodes):
    """运行 Greedy 算法推理 - 延迟敏感型贪心，只看算力（追求最快处理）"""
    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    
    latencies, rewards, switches, inf_times = [], [], [], []
    compute_costs, network_costs, communication_costs, total_costs = [], [], [], []
    server_ids = sorted(list(env.servers.keys()))
    
    for i in range(episodes):
        task = ds.tasks[i % len(ds.tasks)]
        env.reset(task)
        task_lon, task_lat = task['TaskLongitude'], task['TaskLatitude']
        
        ep_lat, ep_reward, ep_inf_time = 0, 0, 0
        ep_compute_cost, ep_network_cost, ep_communication_cost = 0, 0, 0
        done = False
        prev_server = None
        
        while not done:
            t0 = time.time()
            
            candidates = env.available_actions()
            if not candidates:
                candidates = list(range(len(env.actions)))
            
            best_action, best_score = candidates[0], -float('inf')
            for action_idx in candidates:
                mi = env.actions[action_idx]
                server = env.servers[mi.server_id]
                score = server.normalized_compute
                if score > best_score:
                    best_score = score
                    best_action = action_idx
            
            ep_inf_time += (time.time() - t0) * 1000

            mi = env.actions[best_action]
            server = env.servers[mi.server_id]

            tokens = task.get('TaskSize', 1000)
            
            _, (rL, rC, rS), done, info = env.step(best_action)
  
            network_ms = info.get('network_ms', 0)
            step_network_cost = compute_network_cost_by_latency(network_ms, tokens)
            step_communication_cost = compute_communication_cost_by_latency(tokens, network_ms)
            
            ep_lat += info['latency_ms']
            ep_compute_cost += info['cost']
            ep_network_cost += step_network_cost
            ep_communication_cost += step_communication_cost
            ep_reward += w[0]*rL + w[1]*rC + w[2]*rS
            
            prev_server = server
        
        latencies.append(ep_lat)
        compute_costs.append(ep_compute_cost)
        network_costs.append(ep_network_cost)
        communication_costs.append(ep_communication_cost)
        total_costs.append(ep_compute_cost + ep_network_cost + ep_communication_cost)
        rewards.append(ep_reward)
        switches.append(env.ep_switches)
        inf_times.append(ep_inf_time)
        
        if (i + 1) % 50 == 0:
            print(f"  Greedy: {i+1}/{episodes}")
    
    return {
        'latencies': np.array(latencies),
        'costs': np.array(total_costs),
        'compute_costs': np.array(compute_costs),
        'network_costs': np.array(network_costs),
        'communication_costs': np.array(communication_costs),
        'rewards': np.array(rewards),
        'switches': np.array(switches),
        'inference_times': np.array(inf_times)
    }


def run_random_inference(env, ds, episodes):
    """运行 Random 算法推理"""
    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    
    latencies, rewards, switches, inf_times = [], [], [], []
    compute_costs, network_costs, communication_costs, total_costs = [], [], [], []
    
    for i in range(episodes):
        task = ds.tasks[i % len(ds.tasks)]
        env.reset(task)
        task_lon, task_lat = task['TaskLongitude'], task['TaskLatitude']
        
        ep_lat, ep_reward, ep_inf_time = 0, 0, 0
        ep_compute_cost, ep_network_cost, ep_communication_cost = 0, 0, 0
        done = False
        prev_server = None
        
        while not done:
            t0 = time.time()
            
            candidates = env.available_actions()
            if not candidates:
                candidates = list(range(len(env.actions)))
            
            action = np.random.choice(candidates)
            
            ep_inf_time += (time.time() - t0) * 1000
 
            mi = env.actions[action]
            server = env.servers[mi.server_id]
 
            tokens = task.get('TaskSize', 1000)
            
            _, (rL, rC, rS), done, info = env.step(action)
 
            network_ms = info.get('network_ms', 0)
            step_network_cost = compute_network_cost_by_latency(network_ms, tokens)
            step_communication_cost = compute_communication_cost_by_latency(tokens, network_ms)
            
            ep_lat += info['latency_ms']
            ep_compute_cost += info['cost']
            ep_network_cost += step_network_cost
            ep_communication_cost += step_communication_cost
            ep_reward += w[0]*rL + w[1]*rC + w[2]*rS
            
            prev_server = server
        
        latencies.append(ep_lat)
        compute_costs.append(ep_compute_cost)
        network_costs.append(ep_network_cost)
        communication_costs.append(ep_communication_cost)
        total_costs.append(ep_compute_cost + ep_network_cost + ep_communication_cost)
        rewards.append(ep_reward)
        switches.append(env.ep_switches)
        inf_times.append(ep_inf_time)
        
        if (i + 1) % 50 == 0:
            print(f"  Random: {i+1}/{episodes}")
    
    return {
        'latencies': np.array(latencies),
        'costs': np.array(total_costs),
        'compute_costs': np.array(compute_costs),
        'network_costs': np.array(network_costs),
        'communication_costs': np.array(communication_costs),
        'rewards': np.array(rewards),
        'switches': np.array(switches),
        'inference_times': np.array(inf_times)
    }


def run_pfappo_inference(env, ds, model_path, device, episodes):
    """运行 PFAPPO 推理 - 公平版本"""
    from PFAPPO.model import Actor
    
    actor = Actor(state_dim=7, num_servers=NUM_SERVERS).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device))
    actor.eval()
    
    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    server_ids = sorted(list(env.servers.keys()))

    caps = np.array([env.servers[sid].normalized_compute for sid in server_ids], dtype=np.float32)

    server_to_models = {sid: [] for sid in server_ids}
    for mi in ds.model_instances:
        if mi.server_id in server_to_models:
            server_to_models[mi.server_id].append(mi)
  
    server_min_costs = []
    for sid in server_ids:
        models_on_server = server_to_models[sid]
        server_cost_mult = env.servers[sid].cost_multiplier  
        if models_on_server:
            min_cost = min([m.cost_per_token * server_cost_mult for m in models_on_server])
        else:
            min_cost = 0.060 * 2.2  
        server_min_costs.append(min_cost)
    server_min_costs = np.array(server_min_costs, dtype=np.float32)

    cost_min = 0.0015 * 0.4  
    cost_max = 0.060 * 2.2   
    cost_advantage = 1.0 - np.clip((server_min_costs - cost_min) / (cost_max - cost_min), 0, 1.0)
    
    latencies, rewards, switches, inf_times = [], [], [], []
    compute_costs, network_costs, communication_costs, total_costs = [], [], [], []
    
    for i in range(episodes):
        task = ds.tasks[i % len(ds.tasks)]
        state_dict = env.reset(task)
        tokens = task.get('TaskSize', 1000)
        
        ep_lat, ep_reward, ep_inf_time = 0, 0, 0
        ep_compute_cost, ep_network_cost, ep_communication_cost = 0, 0, 0
        done = False
        
        while not done:
            t0 = time.time()
            
            valid_actions = env.available_actions()
            if not valid_actions:
                break
            
            valid_server_ids = set()
            server_to_action = {}
            for aidx in valid_actions:
                mi = env.actions[aidx]
                valid_server_ids.add(mi.server_id)
                if mi.server_id not in server_to_action:
                    server_to_action[mi.server_id] = aidx
            
            state_vec = np.array([
                state_dict['step_norm'],
                state_dict['task_lon'],
                state_dict['task_lat'],
                float(state_dict['prev_region_id']),
                w[0], w[1], w[2]
            ], dtype=np.float32)
            
            current_time = env.current_time_ms
            busy_times = np.array([max(0.0, env.busy_until[sid] - current_time) for sid in server_ids], dtype=np.float32)
            norm_queues = np.clip(busy_times / 5000.0, 0.0, 1.0)
            
            weights = (0.35 * caps + 0.35 * cost_advantage) / (1.0 + 0.3 * norm_queues)
            weights = weights / (np.max(weights) + 1e-9)
            
            with torch.no_grad():
                state_t = torch.from_numpy(state_vec).unsqueeze(0).to(device)
                weights_t = torch.from_numpy(weights).unsqueeze(0).to(device)
                logits = actor(state_t, weights_t).squeeze(0)
                
                mask = torch.zeros(NUM_SERVERS, device=device)
                for idx, sid in enumerate(server_ids):
                    if sid in valid_server_ids:
                        mask[idx] = 1.0
                
                masked_logits = logits + (1 - mask) * -1e9
                server_idx = torch.argmax(masked_logits).item()
                selected_sid = server_ids[server_idx]
                action = server_to_action[selected_sid]
            
            ep_inf_time += (time.time() - t0) * 1000
            
            state_dict, (rL, rC, rS), done, info = env.step(action)
 
            network_ms = info.get('network_ms', 0)
            step_network_cost = compute_network_cost_by_latency(network_ms, tokens)
            step_communication_cost = compute_communication_cost_by_latency(tokens, network_ms)
            
            ep_lat += info['latency_ms']
            ep_compute_cost += info['cost']
            ep_network_cost += step_network_cost
            ep_communication_cost += step_communication_cost
            ep_reward += w[0]*rL + w[1]*rC + w[2]*rS
        
        latencies.append(ep_lat)
        compute_costs.append(ep_compute_cost)
        network_costs.append(ep_network_cost)
        communication_costs.append(ep_communication_cost)
        total_costs.append(ep_compute_cost + ep_network_cost + ep_communication_cost)
        rewards.append(ep_reward)
        switches.append(env.ep_switches)
        inf_times.append(ep_inf_time)
        
        if (i + 1) % 50 == 0:
            print(f"  PFAPPO: {i+1}/{episodes}")
    
    return {
        'latencies': np.array(latencies),
        'costs': np.array(total_costs),
        'compute_costs': np.array(compute_costs),
        'network_costs': np.array(network_costs),
        'communication_costs': np.array(communication_costs),
        'rewards': np.array(rewards),
        'switches': np.array(switches),
        'inference_times': np.array(inf_times)
    }


def run_ppo_inference(env, ds, model_path, device, episodes):
    """运行 PPO 推理 - 与训练一致：使用第一个实例而非成本最低"""
    from PPO_algorithm.model import Actor
    
    actor = Actor(state_dim=7, num_servers=NUM_SERVERS).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device))
    actor.eval()
    
    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    server_ids = sorted(list(env.servers.keys()))
    
    latencies, rewards, switches, inf_times = [], [], [], []
    compute_costs, network_costs, communication_costs, total_costs = [], [], [], []
    
    for i in range(episodes):
        task = ds.tasks[i % len(ds.tasks)]
        state_dict = env.reset(task)
        tokens = task.get('TaskSize', 1000)
        
        ep_lat, ep_reward, ep_inf_time = 0, 0, 0
        ep_compute_cost, ep_network_cost, ep_communication_cost = 0, 0, 0
        done = False
        
        while not done:
            t0 = time.time()
            
            valid_actions = env.available_actions()
            if not valid_actions:
                break
            
            valid_server_ids = set()
            server_to_action = {}
            for aidx in valid_actions:
                mi = env.actions[aidx]
                valid_server_ids.add(mi.server_id)
                if mi.server_id not in server_to_action:
                    server_to_action[mi.server_id] = aidx
            
            state_vec = np.array([
                state_dict['step_norm'],
                state_dict['task_lon'],
                state_dict['task_lat'],
                float(state_dict['prev_region_id']),
                w[0], w[1], w[2]
            ], dtype=np.float32)
            
            with torch.no_grad():
                state_t = torch.from_numpy(state_vec).unsqueeze(0).to(device)
                logits = actor(state_t).squeeze(0)
                
                mask = torch.zeros(NUM_SERVERS, device=device)
                for idx, sid in enumerate(server_ids):
                    if sid in valid_server_ids:
                        mask[idx] = 1.0
                
                masked_logits = logits + (1 - mask) * -1e9
                server_idx = torch.argmax(masked_logits).item()
                selected_sid = server_ids[server_idx]
                action = server_to_action[selected_sid]
            
            ep_inf_time += (time.time() - t0) * 1000
            
            state_dict, (rL, rC, rS), done, info = env.step(action)
 
            network_ms = info.get('network_ms', 0)
            step_network_cost = compute_network_cost_by_latency(network_ms, tokens)
            step_communication_cost = compute_communication_cost_by_latency(tokens, network_ms)
            
            ep_lat += info['latency_ms']
            ep_compute_cost += info['cost']
            ep_network_cost += step_network_cost
            ep_communication_cost += step_communication_cost
            ep_reward += w[0]*rL + w[1]*rC + w[2]*rS
        
        latencies.append(ep_lat)
        compute_costs.append(ep_compute_cost)
        network_costs.append(ep_network_cost)
        communication_costs.append(ep_communication_cost)
        total_costs.append(ep_compute_cost + ep_network_cost + ep_communication_cost)
        rewards.append(ep_reward)
        switches.append(env.ep_switches)
        inf_times.append(ep_inf_time)
        
        if (i + 1) % 50 == 0:
            print(f"  PPO: {i+1}/{episodes}")
    
    return {
        'latencies': np.array(latencies),
        'costs': np.array(total_costs),
        'compute_costs': np.array(compute_costs),
        'network_costs': np.array(network_costs),
        'communication_costs': np.array(communication_costs),
        'rewards': np.array(rewards),
        'switches': np.array(switches),
        'inference_times': np.array(inf_times)
    }


def run_ppo_cn_inference(env, ds, model_path, device, episodes):
    """运行 PPO_CN 推理 - 与训练一致"""
    from PPO_CN.model import Actor
    
    actor = Actor(state_dim=7, action_dim=NUM_SERVERS).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    actor.eval()
    
    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    server_ids = sorted(list(env.servers.keys()))
    
    latencies, rewards, switches, inf_times = [], [], [], []
    compute_costs, network_costs, communication_costs, total_costs = [], [], [], []
    
    for i in range(episodes):
        task = ds.tasks[i % len(ds.tasks)]
        state_dict = env.reset(task)
        tokens = task.get('TaskSize', 1000)
        
        ep_lat, ep_reward, ep_inf_time = 0, 0, 0
        ep_compute_cost, ep_network_cost, ep_communication_cost = 0, 0, 0
        done = False
        
        while not done:
            t0 = time.time()
            
            valid_actions = env.available_actions()
            if not valid_actions:
                break
            
            valid_server_ids = set()
            server_to_action = {}
            for aidx in valid_actions:
                mi = env.actions[aidx]
                valid_server_ids.add(mi.server_id)
                if mi.server_id not in server_to_action:
                    server_to_action[mi.server_id] = aidx
            
            state_vec = np.array([
                state_dict['step_norm'],
                state_dict['task_lon'],
                state_dict['task_lat'],
                float(state_dict['prev_region_id']),
                w[0], w[1], w[2]
            ], dtype=np.float32)
            
            with torch.no_grad():
                state_t = torch.from_numpy(state_vec).unsqueeze(0).to(device)
                logits = actor(state_t).squeeze(0)
                
                mask = torch.zeros(NUM_SERVERS, device=device)
                for idx, sid in enumerate(server_ids):
                    if sid in valid_server_ids:
                        mask[idx] = 1.0
                
                masked_logits = logits + (1 - mask) * -1e9
                server_idx = torch.argmax(masked_logits).item()
                selected_sid = server_ids[server_idx]
                action = server_to_action[selected_sid]
            
            ep_inf_time += (time.time() - t0) * 1000
            
            state_dict, (rL, rC, rS), done, info = env.step(action)

            network_ms = info.get('network_ms', 0)
            step_network_cost = compute_network_cost_by_latency(network_ms, tokens)
            step_communication_cost = compute_communication_cost_by_latency(tokens, network_ms)
            
            ep_lat += info['latency_ms']
            ep_compute_cost += info['cost']
            ep_network_cost += step_network_cost
            ep_communication_cost += step_communication_cost
            ep_reward += w[0]*rL + w[1]*rC + w[2]*rS
        
        latencies.append(ep_lat)
        compute_costs.append(ep_compute_cost)
        network_costs.append(ep_network_cost)
        communication_costs.append(ep_communication_cost)
        total_costs.append(ep_compute_cost + ep_network_cost + ep_communication_cost)
        rewards.append(ep_reward)
        switches.append(env.ep_switches)
        inf_times.append(ep_inf_time)
        
        if (i + 1) % 50 == 0:
            print(f"  PPO_CN: {i+1}/{episodes}")
    
    return {
        'latencies': np.array(latencies),
        'costs': np.array(total_costs),
        'compute_costs': np.array(compute_costs),
        'network_costs': np.array(network_costs),
        'communication_costs': np.array(communication_costs),
        'rewards': np.array(rewards),
        'switches': np.array(switches),
        'inference_times': np.array(inf_times)
    }


def run_a3c_inference(env, ds, model_path, device, episodes):
    """运行 A3C 推理 - 与训练一致"""
    from A3C_algorithm.model import ActorCritic
    
    model = ActorCritic(state_dim=7, num_servers=NUM_SERVERS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    server_ids = sorted(list(env.servers.keys()))
    
    latencies, rewards, switches, inf_times = [], [], [], []
    compute_costs, network_costs, communication_costs, total_costs = [], [], [], []
    
    for i in range(episodes):
        task = ds.tasks[i % len(ds.tasks)]
        state_dict = env.reset(task)
        tokens = task.get('TaskSize', 1000)
        
        ep_lat, ep_reward, ep_inf_time = 0, 0, 0
        ep_compute_cost, ep_network_cost, ep_communication_cost = 0, 0, 0
        done = False
        
        while not done:
            t0 = time.time()
            
            valid_actions = env.available_actions()
            if not valid_actions:
                break
            
            valid_server_ids = set()
            server_to_action = {}
            for aidx in valid_actions:
                mi = env.actions[aidx]
                valid_server_ids.add(mi.server_id)
                if mi.server_id not in server_to_action:
                    server_to_action[mi.server_id] = aidx
            
            state_vec = np.array([
                state_dict['step_norm'],
                state_dict['task_lon'],
                state_dict['task_lat'],
                float(state_dict['prev_region_id']),
                w[0], w[1], w[2]
            ], dtype=np.float32)
            
            with torch.no_grad():
                state_t = torch.from_numpy(state_vec).unsqueeze(0).to(device)
                logits, _ = model(state_t)
                logits = logits.squeeze(0)
                
                mask = torch.zeros(NUM_SERVERS, device=device)
                for idx, sid in enumerate(server_ids):
                    if sid in valid_server_ids:
                        mask[idx] = 1.0
                
                masked_logits = logits + (1 - mask) * -1e9
                server_idx = torch.argmax(masked_logits).item()
                selected_sid = server_ids[server_idx]
                action = server_to_action[selected_sid]
            
            ep_inf_time += (time.time() - t0) * 1000
            
            state_dict, (rL, rC, rS), done, info = env.step(action)
 
            network_ms = info.get('network_ms', 0)
            step_network_cost = compute_network_cost_by_latency(network_ms, tokens)
            step_communication_cost = compute_communication_cost_by_latency(tokens, network_ms)
            
            ep_lat += info['latency_ms']
            ep_compute_cost += info['cost']
            ep_network_cost += step_network_cost
            ep_communication_cost += step_communication_cost
            ep_reward += w[0]*rL + w[1]*rC + w[2]*rS
        
        latencies.append(ep_lat)
        compute_costs.append(ep_compute_cost)
        network_costs.append(ep_network_cost)
        communication_costs.append(ep_communication_cost)
        total_costs.append(ep_compute_cost + ep_network_cost + ep_communication_cost)
        rewards.append(ep_reward)
        switches.append(env.ep_switches)
        inf_times.append(ep_inf_time)
        
        if (i + 1) % 50 == 0:
            print(f"  A3C: {i+1}/{episodes}")
    
    return {
        'latencies': np.array(latencies),
        'costs': np.array(total_costs),
        'compute_costs': np.array(compute_costs),
        'network_costs': np.array(network_costs),
        'communication_costs': np.array(communication_costs),
        'rewards': np.array(rewards),
        'switches': np.array(switches),
        'inference_times': np.array(inf_times)
    }


def run_ppo_gnn_inference(env, ds, model_path, device, episodes):
    from PPO_GNN.model import GNNActorCritic
    from torch_geometric.data import Data
    from utils import haversine_km

    model = GNNActorCritic(node_feat_dim=3, global_feat_dim=7, hidden_dim=128, gnn_layers=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)

    server_ids = sorted(list(env.servers.keys()))
    num_servers = len(server_ids)

    static_feats = []
    coords = []
    for sid in server_ids:
        s = env.servers[sid]
        static_feats.append([s.normalized_compute, 0.5])  
        coords.append([s.lon, s.lat])
    static_feats = torch.FloatTensor(static_feats).to(device)
    coords = np.array(coords)
 
    K = 20
    edge_indices = []
    edge_attrs = []
    for i in range(num_servers):
        dists = []
        for j in range(num_servers):
            if i == j:
                continue
            d = haversine_km(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
            dists.append((j, d))
        dists.sort(key=lambda x: x[1])
        for j, d in dists[:K]:
            edge_indices.append([i, j])
            edge_attrs.append([np.exp(-d / 500.0)])
    
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous().to(device)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32).to(device)
    
    latencies, rewards, switches, inf_times = [], [], [], []
    compute_costs, network_costs, communication_costs, total_costs = [], [], [], []
    
    for i in range(episodes):
        task = ds.tasks[i % len(ds.tasks)]
        state_dict = env.reset(task)
        tokens = task.get('TaskSize', 1000)
        
        ep_lat, ep_reward, ep_inf_time = 0, 0, 0
        ep_compute_cost, ep_network_cost, ep_communication_cost = 0, 0, 0
        done = False
        
        while not done:
            t0 = time.time()
            
            current_time = env.current_time_ms
            busy_times = np.array([max(0.0, env.busy_until[sid] - current_time) for sid in server_ids], dtype=np.float32)
            norm_queues = np.clip(busy_times / 5000.0, 0.0, 1.0)
            dynamic_feat = torch.FloatTensor(norm_queues).unsqueeze(1).to(device)
            
            node_feats = torch.cat([static_feats[:, 0:1], dynamic_feat, static_feats[:, 1:2]], dim=1)
            
            global_feat = torch.FloatTensor([
                state_dict['step_norm'],
                state_dict['task_lon'],
                state_dict['task_lat'],
                float(state_dict['prev_region_id']),
                w[0], w[1], w[2]
            ]).unsqueeze(0).to(device)
            
            valid_action_idxs = env.available_actions()
            valid_server_ids = set(env.actions[a].server_id for a in valid_action_idxs)
            candidate_mask = torch.tensor([sid in valid_server_ids for sid in server_ids], dtype=torch.bool).to(device)
            
            server_to_actions = {}
            for aidx in valid_action_idxs:
                mi = env.actions[aidx]
                server_to_actions.setdefault(mi.server_id, []).append(mi)
            server_action_map = {}
            for idx, sid in enumerate(server_ids):
                if sid in server_to_actions:
                    best_mi = min(server_to_actions[sid], key=lambda m: m.cost_per_token)
                    server_action_map[idx] = best_mi.idx
            
            data = Data(
                x=node_feats,
                edge_index=edge_index,
                edge_attr=edge_attr,
                global_feat=global_feat,
                candidate_mask=candidate_mask,
                num_nodes=num_servers
            )
            
            with torch.no_grad():
                logits, _ = model(data)
                masked_logits = logits.clone()
                masked_logits[~candidate_mask] = -1e9
                server_idx = torch.argmax(masked_logits).item()
                
                if server_idx in server_action_map:
                    action = server_action_map[server_idx]
                else:
                    action = valid_action_idxs[0] if valid_action_idxs else 0
            
            ep_inf_time += (time.time() - t0) * 1000
            
            state_dict, (rL, rC, rS), done, info = env.step(action)

            network_ms = info.get('network_ms', 0)
            step_network_cost = compute_network_cost_by_latency(network_ms, tokens)
            step_communication_cost = compute_communication_cost_by_latency(tokens, network_ms)
            
            ep_lat += info['latency_ms']
            ep_compute_cost += info['cost']
            ep_network_cost += step_network_cost
            ep_communication_cost += step_communication_cost
            ep_reward += w[0]*rL + w[1]*rC + w[2]*rS
        
        latencies.append(ep_lat)
        compute_costs.append(ep_compute_cost)
        network_costs.append(ep_network_cost)
        communication_costs.append(ep_communication_cost)
        total_costs.append(ep_compute_cost + ep_network_cost + ep_communication_cost)
        rewards.append(ep_reward)
        switches.append(env.ep_switches)
        inf_times.append(ep_inf_time)
        
        if (i + 1) % 50 == 0:
            print(f"  PPO_GNN: {i+1}/{episodes}")
    
    return {
        'latencies': np.array(latencies),
        'costs': np.array(total_costs),
        'compute_costs': np.array(compute_costs),
        'network_costs': np.array(network_costs),
        'communication_costs': np.array(communication_costs),
        'rewards': np.array(rewards),
        'switches': np.array(switches),
        'inference_times': np.array(inf_times)
    }


def run_stark_inference(env, ds, model_path, device, episodes):
    """运行 Stark 推理"""
    from Stark_Scheduler.model import StarkScheduler
    from Stark_Scheduler.dataset import OnlineExpertDataset
    
    task_dim = 4
    server_dim = 7
    model = StarkScheduler(task_dim=task_dim, server_dim=server_dim, num_servers=NUM_SERVERS, d_model=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    expert_data = OnlineExpertDataset(env)
    
    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    
    latencies, rewards, switches, inf_times = [], [], [], []
    compute_costs, network_costs, communication_costs, total_costs = [], [], [], []
    
    for i in range(episodes):
        task = ds.tasks[i % len(ds.tasks)]
        state_dict = env.reset(task)
        tokens = task.get('TaskSize', 1000)
        
        ep_lat, ep_reward, ep_inf_time = 0, 0, 0
        ep_compute_cost, ep_network_cost, ep_communication_cost = 0, 0, 0
        done = False
        
        while not done:
            t0 = time.time()
            
            available = env.available_actions()
            if not available:
                break
            
            task_feat, server_feats = expert_data.extract_structured_state(env)
            
            with torch.no_grad():
                task_t = torch.from_numpy(task_feat).unsqueeze(0).to(device)
                server_t = torch.from_numpy(server_feats).unsqueeze(0).to(device)
                logits = model(task_t, server_t)
                
                mask = torch.zeros(NUM_SERVERS, device=device)
                for a in available:
                    server_idx = list(env.servers.keys()).index(env.actions[a].server_id)
                    mask[server_idx] = 1.0
                logits = logits + (1 - mask) * -1e9
                
                server_idx = torch.argmax(logits, dim=1).item()
                server_ids = sorted(list(env.servers.keys()))
                target_server_id = server_ids[server_idx]
                action = available[0]
                for a in available:
                    if env.actions[a].server_id == target_server_id:
                        action = a
                        break
            
            ep_inf_time += (time.time() - t0) * 1000
            
            state_dict, (rL, rC, rS), done, info = env.step(action)

            network_ms = info.get('network_ms', 0)
            step_network_cost = compute_network_cost_by_latency(network_ms, tokens)
            step_communication_cost = compute_communication_cost_by_latency(tokens, network_ms)
            
            ep_lat += info['latency_ms']
            ep_compute_cost += info['cost']
            ep_network_cost += step_network_cost
            ep_communication_cost += step_communication_cost
            ep_reward += w[0]*rL + w[1]*rC + w[2]*rS
        
        latencies.append(ep_lat)
        compute_costs.append(ep_compute_cost)
        network_costs.append(ep_network_cost)
        communication_costs.append(ep_communication_cost)
        total_costs.append(ep_compute_cost + ep_network_cost + ep_communication_cost)
        rewards.append(ep_reward)
        switches.append(env.ep_switches)
        inf_times.append(ep_inf_time)
        
        if (i + 1) % 50 == 0:
            print(f"  Stark: {i+1}/{episodes}")
    
    return {
        'latencies': np.array(latencies),
        'costs': np.array(total_costs),
        'compute_costs': np.array(compute_costs),
        'network_costs': np.array(network_costs),
        'communication_costs': np.array(communication_costs),
        'rewards': np.array(rewards),
        'switches': np.array(switches),
        'inference_times': np.array(inf_times)
    }


def run_trans_inference(env, ds, model_path, device, episodes):
    """运行 Trans 推理 - 修复动作映射"""
    from Trans.model import TransformerActorCritic
    
    model = TransformerActorCritic(state_dim=7, action_dim=NUM_SERVERS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    server_ids = sorted(list(env.servers.keys()))
    
    latencies, rewards, switches, inf_times = [], [], [], []
    compute_costs, network_costs, communication_costs, total_costs = [], [], [], []
    max_seq_len = 20
    
    for i in range(episodes):
        task = ds.tasks[i % len(ds.tasks)]
        state_dict = env.reset(task)
        tokens = task.get('TaskSize', 1000)
        
        ep_lat, ep_reward, ep_inf_time = 0, 0, 0
        ep_compute_cost, ep_network_cost, ep_communication_cost = 0, 0, 0
        done = False
        state_seq = []
        
        while not done:
            t0 = time.time()
            
            valid_actions = env.available_actions()
            if not valid_actions:
                break
            
            valid_server_ids = set()
            server_to_action = {}
            for aidx in valid_actions:
                mi = env.actions[aidx]
                valid_server_ids.add(mi.server_id)
                if mi.server_id not in server_to_action:
                    server_to_action[mi.server_id] = aidx
            
            state_vec = np.array([
                state_dict['step_norm'],
                state_dict['task_lon'],
                state_dict['task_lat'],
                float(state_dict['prev_region_id']),
                w[0], w[1], w[2]
            ], dtype=np.float32)
            
            state_seq.append(state_vec)
            if len(state_seq) > max_seq_len:
                state_seq = state_seq[-max_seq_len:]
            
            padded_seq = np.zeros((max_seq_len, 7), dtype=np.float32)
            padded_seq[-len(state_seq):] = np.array(state_seq)
            
            with torch.no_grad():
                seq_t = torch.from_numpy(padded_seq).unsqueeze(0).to(device)
                logits = model.get_action_logits(seq_t).squeeze()
                
                mask = torch.zeros(NUM_SERVERS, device=device)
                for idx, sid in enumerate(server_ids):
                    if sid in valid_server_ids:
                        mask[idx] = 1.0
                
                masked_logits = logits + (1 - mask) * -1e9
                server_idx = torch.argmax(masked_logits).item()
                selected_sid = server_ids[server_idx]
                action = server_to_action[selected_sid]
            
            ep_inf_time += (time.time() - t0) * 1000
            
            state_dict, (rL, rC, rS), done, info = env.step(action)

            network_ms = info.get('network_ms', 0)
            step_network_cost = compute_network_cost_by_latency(network_ms, tokens)
            step_communication_cost = compute_communication_cost_by_latency(tokens, network_ms)
            
            ep_lat += info['latency_ms']
            ep_compute_cost += info['cost']
            ep_network_cost += step_network_cost
            ep_communication_cost += step_communication_cost
            ep_reward += w[0]*rL + w[1]*rC + w[2]*rS
        
        latencies.append(ep_lat)
        compute_costs.append(ep_compute_cost)
        network_costs.append(ep_network_cost)
        communication_costs.append(ep_communication_cost)
        total_costs.append(ep_compute_cost + ep_network_cost + ep_communication_cost)
        rewards.append(ep_reward)
        switches.append(env.ep_switches)
        inf_times.append(ep_inf_time)
        
        if (i + 1) % 50 == 0:
            print(f"  Trans: {i+1}/{episodes}")
    
    return {
        'latencies': np.array(latencies),
        'costs': np.array(total_costs),
        'compute_costs': np.array(compute_costs),
        'network_costs': np.array(network_costs),
        'communication_costs': np.array(communication_costs),
        'rewards': np.array(rewards),
        'switches': np.array(switches),
        'inference_times': np.array(inf_times)
    }


def run_star_ppo_inference(env, ds, model_path, device, episodes):
    from STAR_PPO.model import StarActor

    actor = StarActor(state_dim=10, num_servers=NUM_SERVERS).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    actor.eval()
    
    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    server_ids = sorted(list(env.servers.keys()))

    caps = np.array([env.servers[sid].normalized_compute for sid in server_ids], dtype=np.float32)

    cost_mults = np.array([env.servers[sid].cost_multiplier for sid in server_ids], dtype=np.float32)
    cost_advantage = 1.0 - np.clip(cost_mults / 2.0, 0, 1.0)  
  
    network_quality = np.ones(NUM_SERVERS, dtype=np.float32)
    if hasattr(env, 'link_latency') and len(env.link_latency) > 0:
        for i, sid in enumerate(server_ids):
            outbound_lats = []
            for (src, dst), lat in env.link_latency.items():
                if src == sid:
                    outbound_lats.append(lat)
            if outbound_lats:
                avg_lat = np.mean(outbound_lats)
                network_quality[i] = np.exp(-avg_lat / 500.0)  
    
    latencies, rewards, switches, inf_times = [], [], [], []
    compute_costs, network_costs, communication_costs, total_costs = [], [], [], []
    
    for i in range(episodes):
        task = ds.tasks[i % len(ds.tasks)]
        state_dict = env.reset(task)
        task_lon, task_lat = task['TaskLongitude'], task['TaskLatitude']
        
        ep_lat, ep_reward, ep_inf_time = 0, 0, 0
        ep_compute_cost, ep_network_cost, ep_communication_cost = 0, 0, 0
        done = False
        prev_server = None
        
        while not done:
            t0 = time.time()

            valid_actions = env.available_actions()
            if not valid_actions:
                break

            valid_server_ids = set()
            server_to_action = {}
            for aidx in valid_actions:
                mi = env.actions[aidx]
                valid_server_ids.add(mi.server_id)
                if mi.server_id not in server_to_action:
                    server_to_action[mi.server_id] = aidx

            base_state = np.array([
                state_dict['step_norm'],
                state_dict['task_lon'],
                state_dict['task_lat'],
                float(state_dict['prev_region_id']),
                w[0], w[1], w[2]
            ], dtype=np.float32)

            aug_features = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            state_vec = np.concatenate([base_state, aug_features])

            current_time = env.current_time_ms
            busy_times = np.array([max(0.0, env.busy_until[sid] - current_time) for sid in server_ids], dtype=np.float32)
            norm_queues = np.clip(busy_times / 5000.0, 0.0, 1.0)
 
            w2 = 0.30
            weights = caps / (1.0 + w2 * norm_queues)
            weights = weights * network_quality  
            weights = weights * (0.5 + 0.5 * cost_advantage)  
            weights = weights / (np.max(weights) + 1e-9)
            
            with torch.no_grad():
                state_t = torch.from_numpy(state_vec).unsqueeze(0).to(device)
                weights_t = torch.from_numpy(weights).unsqueeze(0).to(device)
                logits = actor(state_t, weights_t).squeeze(0)

                mask = torch.zeros(NUM_SERVERS, device=device)
                for idx, sid in enumerate(server_ids):
                    if sid in valid_server_ids:
                        mask[idx] = 1.0
                
                masked_logits = logits + (1 - mask) * -1e9
                server_idx = torch.argmax(masked_logits).item()

                selected_sid = server_ids[server_idx]
                action = server_to_action[selected_sid]
            
            ep_inf_time += (time.time() - t0) * 1000

            mi = env.actions[action]
            server = env.servers[mi.server_id]

            if prev_server is None:
                d_km = haversine_km(task_lon, task_lat, server.lon, server.lat)
            else:
                d_km = haversine_km(prev_server.lon, prev_server.lat, server.lon, server.lat)

            tokens = task.get('TaskSize', 1000)
            
            state_dict, (rL, rC, rS), done, info = env.step(action)

            network_ms = info.get('network_ms', 0)
            step_network_cost = compute_network_cost_by_latency(network_ms, tokens)
            step_communication_cost = compute_communication_cost_by_latency(tokens, network_ms)
            
            ep_lat += info['latency_ms']
            ep_compute_cost += info['cost']
            ep_network_cost += step_network_cost
            ep_communication_cost += step_communication_cost
            ep_reward += w[0]*rL + w[1]*rC + w[2]*rS
            
            prev_server = server
        
        latencies.append(ep_lat)
        compute_costs.append(ep_compute_cost)
        network_costs.append(ep_network_cost)
        communication_costs.append(ep_communication_cost)
        total_costs.append(ep_compute_cost + ep_network_cost + ep_communication_cost)
        rewards.append(ep_reward)
        switches.append(env.ep_switches)
        inf_times.append(ep_inf_time)
        
        if (i + 1) % 50 == 0:
            print(f"  STAR_PPO: {i+1}/{episodes}")
    
    return {
        'latencies': np.array(latencies),
        'costs': np.array(total_costs),
        'compute_costs': np.array(compute_costs),
        'network_costs': np.array(network_costs),
        'communication_costs': np.array(communication_costs),
        'rewards': np.array(rewards),
        'switches': np.array(switches),
        'inference_times': np.array(inf_times)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--episodes', type=int, default=EPISODES)
    parser.add_argument('--algorithms', nargs='+', default=['all'])
    parser.add_argument('--seed', type=int, default=None, help='Random seed for inference')
    parser.add_argument('--train_seed', type=int, default=None, help='Training seed to select model (42/43/44)')
    parser.add_argument('--output_suffix', type=str, default='', help='Suffix for output filename')
    args = parser.parse_args()
 
    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"Set random seed: {args.seed}")

    output_dir = 'inference/results_500'
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading dataset from {DATA_ROOT}/{REGION}...")
    ds = WorkflowDataset(DATA_ROOT, split='train', regions=[REGION])
    env = WorkflowMoEEnv(ds)
    print(f"Loaded {len(ds.tasks)} tasks, {len(env.servers)} servers")
    
    all_algos = ['PFAPPO', 'STAR_PPO', 'PPO', 'PPO_CN', 'Trans', 'Stark', 'PPO_GNN', 'A3C', 'Greedy', 'Random']
    if 'all' in args.algorithms:
        algos_to_run = all_algos
    else:
        algos_to_run = args.algorithms
    
    results = {}
    
    for algo in algos_to_run:
        print(f"\n{'='*60}")
        print(f"Running {algo} inference...")
        print('='*60)
        
        try:
            if algo == 'Greedy':
                data = run_greedy_inference(env, ds, args.episodes)
            elif algo == 'Random':
                data = run_random_inference(env, ds, args.episodes)
            elif algo == 'PFAPPO':
                model_path = find_latest_model('PFAPPO', train_seed=args.train_seed)
                if model_path:
                    print(f"Using model: {model_path}")
                    data = run_pfappo_inference(env, ds, model_path, args.device, args.episodes)
                else:
                    print(f"No model found for PFAPPO, skipping")
                    continue
            elif algo == 'PPO':
                model_path = find_latest_model('PPO', train_seed=args.train_seed)
                if model_path:
                    print(f"Using model: {model_path}")
                    data = run_ppo_inference(env, ds, model_path, args.device, args.episodes)
                else:
                    print(f"No model found for PPO, skipping")
                    continue
            elif algo == 'PPO_CN':
                model_path = find_latest_model('PPO_CN', train_seed=args.train_seed)
                if model_path:
                    print(f"Using model: {model_path}")
                    data = run_ppo_cn_inference(env, ds, model_path, args.device, args.episodes)
                else:
                    print(f"No model found for PPO_CN, skipping")
                    continue
            elif algo == 'A3C':
                model_path = find_latest_model('A3C', train_seed=args.train_seed)
                if model_path:
                    print(f"Using model: {model_path}")
                    data = run_a3c_inference(env, ds, model_path, args.device, args.episodes)
                else:
                    print(f"No model found for A3C, skipping")
                    continue
            elif algo == 'STAR_PPO':
                model_path = find_latest_model('STAR_PPO', train_seed=args.train_seed)
                if model_path:
                    print(f"Using model: {model_path}")
                    data = run_star_ppo_inference(env, ds, model_path, args.device, args.episodes)
                else:
                    print(f"No model found for STAR_PPO, skipping")
                    continue
            elif algo == 'Trans':
                model_path = find_latest_model('Trans', train_seed=args.train_seed)
                if model_path:
                    print(f"Using model: {model_path}")
                    data = run_trans_inference(env, ds, model_path, args.device, args.episodes)
                else:
                    print(f"No model found for Trans, skipping")
                    continue
            elif algo == 'Stark':
                model_path = find_latest_model('Stark', train_seed=args.train_seed)
                if model_path:
                    print(f"Using model: {model_path}")
                    data = run_stark_inference(env, ds, model_path, args.device, args.episodes)
                else:
                    print(f"No model found for Stark, skipping")
                    continue
            elif algo == 'PPO_GNN':
                model_path = find_latest_model('PPO_GNN', train_seed=args.train_seed)
                if model_path:
                    print(f"Using model: {model_path}")
                    data = run_ppo_gnn_inference(env, ds, model_path, args.device, args.episodes)
                else:
                    print(f"No model found for PPO_GNN, skipping")
                    continue
            else:
                print(f"Algorithm {algo} not implemented yet")
                continue

            suffix = args.output_suffix if args.output_suffix else ''
            npz_path = os.path.join(output_dir, f'{algo}_Server1_500_detailed{suffix}.npz')
            np.savez(npz_path, **data)
            print(f"Saved to {npz_path}")

            results[algo] = {
                'avg_latency': np.mean(data['latencies']),
                'std_latency': np.std(data['latencies']),
                'p99_latency': np.percentile(data['latencies'], 99),
                'avg_cost': np.mean(data['costs']),
                'std_cost': np.std(data['costs']),
                'avg_reward': np.mean(data['rewards']),
                'avg_inf_time': np.mean(data['inference_times'])
            }
            
            print(f"\nResults for {algo}:")
            print(f"  Avg Latency: {results[algo]['avg_latency']:.2f} ± {results[algo]['std_latency']:.2f} ms")
            print(f"  P99 Latency: {results[algo]['p99_latency']:.2f} ms")
            print(f"  Avg Cost:    ${results[algo]['avg_cost']:.4f} ± {results[algo]['std_cost']:.4f}")
            print(f"  Avg Reward:  {results[algo]['avg_reward']:.4f}")
            print(f"  Avg InfTime: {results[algo]['avg_inf_time']:.2f} ms")
            
        except Exception as e:
            print(f"Error running {algo}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("Summary Table")
    print("="*80)
    print(f"{'Algorithm':<12} {'AvgLat(ms)':<12} {'P99Lat(ms)':<12} {'AvgCost($)':<12} {'AvgReward':<12}")
    print("-"*80)
    for algo, res in results.items():
        print(f"{algo:<12} {res['avg_latency']:<12.2f} {res['p99_latency']:<12.2f} {res['avg_cost']:<12.4f} {res['avg_reward']:<12.4f}")


if __name__ == '__main__':
    main()

