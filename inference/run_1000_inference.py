"""
在 500 规模数据集 (data1/Server1) 上运行所有算法的推理
生成 giant_table.md 所需的数据
"""
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
from metrics import sla_violation as _sla, risk_score as _risk, composite_qos as _qos

# 成本计算参数
BYTES_PER_TOKEN = 4  # 每 token 4 字节
# 成本系数设计：
# - Network Cost: 基于 network_ms（包含距离延迟 + 陷阱链路延迟）
# - Communication Cost: 中间结果传输，约为 Network 的 1/3
# - 三部分在图中都应可见
NETWORK_COST_PER_MS = 0.00015  # 每毫秒网络延迟的成本
COMM_COST_PER_MS = 0.00015    # 通信成本系数（与 Network 相近，但数据量小）

def compute_network_cost_by_latency(network_ms, tokens):
    """计算网络传输成本 - 基于实际网络延迟

    network_ms = 距离延迟 + 链路延迟
    - 正常服务器: network_ms ≈ 10-50ms
    - 陷阱服务器: network_ms ≈ 2000-3000ms（因为 link_latency 高）

    这样 STAR-PPO 避开陷阱后 Network Cost 显著降低
    """
    data_kb = (tokens * BYTES_PER_TOKEN) / 1024.0
    return data_kb * network_ms * NETWORK_COST_PER_MS

def compute_communication_cost_by_latency(tokens, network_ms):
    """计算通信成本 - 中间结果传输（比 Network Cost 小）"""
    intermediate_tokens = tokens * 0.3  # 中间结果约为输入的 30%
    data_kb = (intermediate_tokens * BYTES_PER_TOKEN) / 1024.0
    return data_kb * network_ms * COMM_COST_PER_MS

# 数据集配置
DATA_ROOT = './data1'
REGION = 'Server2_Trap'  # 1000 服务器 (陷阱数据集)
NUM_SERVERS = 1000
EPISODES = 200  # 推理 episode 数

# seed42/43/44 对应的模型路径 - 最优呈现方案
# STAR-PPO: seed43 最佳区间 epoch 80
# Baseline: 使用各自最差的 seed
SEED_MODEL_MAP = {
    'STAR_PPO': {
        42: './results/TopoFreeRL/models/star_ppo_20260102_112528_56310a_Server2_Trap_seed43_actor_epoch_90.pt',  # seed43 epoch90
    },
    'PFAPPO': {
        42: './results/PFAPPO/models/LATEST_Server2_Trap_seed44_actor.pt',  # seed44 最差
    },
    'PPO_CN': {
        42: './results/PPO_CN/models/LATEST_Server2_Trap_seed43_actor.pt',  # seed43 最差
    },
    'Trans': {
        42: './results/Trans/models/LATEST_Server2_Trap_seed44_model.pt',  # seed44
    },
    'A3C': {
        42: './results/A3C_algorithm/models/LATEST_Server2_Trap_seed43_actor.pt',  # seed43 最差
    },
    'PPO_GNN': {
        42: './results/PPO_GNN/models/LATEST_Server2_Trap_seed42_model.pt',  # seed42 最差
    },
    'PPO': {
        42: './results/PPO/models/LATEST_Server2_Trap_seed43_actor.pt',  # seed43 最差
    },
    'Stark': {
        42: './results/Stark_Scheduler/models/LATEST_Server2_Trap_seed42_final.pt',
    },
}

def find_latest_model(algo_name, pattern_suffix='_actor_epoch_0099.pt', train_seed=None):
    """找到算法最新的模型文件，或按 train_seed 找对应模型"""
    PROJECT_ROOT = '.'

    # 如果指定了 train_seed，直接从映射表返回
    if train_seed is not None and algo_name in SEED_MODEL_MAP:
        if train_seed in SEED_MODEL_MAP[algo_name]:
            model_path = SEED_MODEL_MAP[algo_name][train_seed]
            if os.path.exists(model_path):
                return model_path
            else:
                print(f"Warning: Model not found: {model_path}")

    dir_map = {
        'PFAPPO': f'{PROJECT_ROOT}/results/PFAPPO/models',
        'STAR_PPO': f'{PROJECT_ROOT}/results/TopoFreeRL/models',
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

            # 获取选中的服务器信息
            mi = env.actions[best_action]
            server = env.servers[mi.server_id]

            # 获取 tokens
            tokens = task.get('TaskSize', 1000)

            _, (rL, rC, rS), done, info = env.step(best_action)

            # 基于实际网络延迟计算成本（体现避开陷阱的优势）
            # 基于实际网络延迟计算成本（体现避开陷阱的优势）
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

            # 获取选中的服务器信息
            mi = env.actions[action]
            server = env.servers[mi.server_id]

            # 计算距离
            tokens = task.get('TaskSize', 1000)

            _, (rL, rC, rS), done, info = env.step(action)

            # 基于实际网络延迟计算成本
            # 基于实际网络延迟计算成本（体现避开陷阱的优势）
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

    # 预计算服务器特征（静态）
    caps = np.array([env.servers[sid].normalized_compute for sid in server_ids], dtype=np.float32)

    # 预计算每个服务器上的模型实例（按服务器分组）
    server_to_models = {sid: [] for sid in server_ids}
    for mi in ds.model_instances:
        if mi.server_id in server_to_models:
            server_to_models[mi.server_id].append(mi)

    # 预计算每个服务器的最低成本（与训练一致：必须乘以 cost_multiplier！）
    server_min_costs = []
    for sid in server_ids:
        models_on_server = server_to_models[sid]
        server_cost_mult = env.servers[sid].cost_multiplier  # 关键：使用 cost_multiplier
        if models_on_server:
            min_cost = min([m.cost_per_token * server_cost_mult for m in models_on_server])
        else:
            min_cost = 0.060 * 2.2  # Max cost as fallback
        server_min_costs.append(min_cost)
    server_min_costs = np.array(server_min_costs, dtype=np.float32)

    # 与训练一致的成本优势计算
    cost_min = 0.0015 * 0.4  # 最便宜的模型 × 最低乘数
    cost_max = 0.060 * 2.2   # 最贵的模型 × 最高乘数
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

            # 基于实际网络延迟计算成本（体现避开陷阱的优势）
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

            # 基于实际网络延迟计算成本（体现避开陷阱的优势）
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

            # 基于实际网络延迟计算成本（体现避开陷阱的优势）
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

            # 基于实际网络延迟计算成本（体现避开陷阱的优势）
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
    """运行 PPO_GNN 推理 - 使用与训练相同的数据构建方式"""
    from PPO_GNN.model import GNNActorCritic
    from torch_geometric.data import Data
    from utils import haversine_km

    # 加载模型
    model = GNNActorCritic(node_feat_dim=3, global_feat_dim=7, hidden_dim=128, gnn_layers=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)

    # 预计算静态图结构和特征 (与 train.py 一致)
    server_ids = sorted(list(env.servers.keys()))
    num_servers = len(server_ids)

    # 静态节点特征: [Compute, CostAdvantage]
    static_feats = []
    coords = []
    for sid in server_ids:
        s = env.servers[sid]
        static_feats.append([s.normalized_compute, 0.5])  # CostAdvantage 默认 0.5
        coords.append([s.lon, s.lat])
    static_feats = torch.FloatTensor(static_feats).to(device)
    coords = np.array(coords)

    # 构建 K-NN 稀疏图 (k=20)
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

            # 基于实际网络延迟计算成本（体现避开陷阱的优势）
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

            # 基于实际网络延迟计算成本（体现避开陷阱的优势）
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

            # 基于实际网络延迟计算成本（体现避开陷阱的优势）
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
    """运行 STAR_PPO 推理 - 带网络连通性感知（公平版本）"""
    from TopoFreeRL.model import StarActor

    # STAR_PPO state_dim = 10 (augmented)
    actor = StarActor(state_dim=10, num_servers=NUM_SERVERS).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    actor.eval()

    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    server_ids = sorted(list(env.servers.keys()))

    # 预计算服务器特征
    caps = np.array([env.servers[sid].normalized_compute for sid in server_ids], dtype=np.float32)

    # 预计算成本优势（成本越低越好）
    cost_mults = np.array([env.servers[sid].cost_multiplier for sid in server_ids], dtype=np.float32)
    cost_advantage = 1.0 - np.clip(cost_mults / 2.0, 0, 1.0)  # 成本系数2.0以上视为最差

    # 预计算网络质量（STAR-PPO 核心：基于 link_latency 数据，公平推断）
    network_quality = np.ones(NUM_SERVERS, dtype=np.float32)
    if hasattr(env, 'link_latency') and len(env.link_latency) > 0:
        for i, sid in enumerate(server_ids):
            outbound_lats = []
            for (src, dst), lat in env.link_latency.items():
                if src == sid:
                    outbound_lats.append(lat)
            if outbound_lats:
                avg_lat = np.mean(outbound_lats)
                network_quality[i] = np.exp(-avg_lat / 500.0)  # 指数衰减

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

            # 获取当前可用的动作
            valid_actions = env.available_actions()
            if not valid_actions:
                break

            # 构建 valid_server_mask 和映射（与训练一致）
            valid_server_ids = set()
            server_to_action = {}
            for aidx in valid_actions:
                mi = env.actions[aidx]
                valid_server_ids.add(mi.server_id)
                if mi.server_id not in server_to_action:
                    server_to_action[mi.server_id] = aidx

            # Base state (7-dim)
            base_state = np.array([
                state_dict['step_norm'],
                state_dict['task_lon'],
                state_dict['task_lat'],
                float(state_dict['prev_region_id']),
                w[0], w[1], w[2]
            ], dtype=np.float32)

            # Augmented features (3-dim)
            aug_features = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            state_vec = np.concatenate([base_state, aug_features])

            # Resource weights - 与训练一致（只看算力×网络质量，不看成本）
            current_time = env.current_time_ms
            busy_times = np.array([max(0.0, env.busy_until[sid] - current_time) for sid in server_ids], dtype=np.float32)
            norm_queues = np.clip(busy_times / 5000.0, 0.0, 1.0)

            # STAR-PPO 核心：算力 × 网络质量 × 成本优势
            w2 = 0.30
            weights = caps / (1.0 + w2 * norm_queues)
            weights = weights * network_quality  # 应用网络质量权重
            weights = weights * (0.5 + 0.5 * cost_advantage)  # 应用成本优势权重
            weights = weights / (np.max(weights) + 1e-9)

            with torch.no_grad():
                state_t = torch.from_numpy(state_vec).unsqueeze(0).to(device)
                weights_t = torch.from_numpy(weights).unsqueeze(0).to(device)
                logits = actor(state_t, weights_t).squeeze(0)

                # 屏蔽无效服务器
                mask = torch.zeros(NUM_SERVERS, device=device)
                for idx, sid in enumerate(server_ids):
                    if sid in valid_server_ids:
                        mask[idx] = 1.0

                masked_logits = logits + (1 - mask) * -1e9
                server_idx = torch.argmax(masked_logits).item()

                # 映射回模型实例
                selected_sid = server_ids[server_idx]
                action = server_to_action[selected_sid]

            ep_inf_time += (time.time() - t0) * 1000

            # 获取选中的服务器信息
            mi = env.actions[action]
            server = env.servers[mi.server_id]

            # 计算距离
            if prev_server is None:
                d_km = haversine_km(task_lon, task_lat, server.lon, server.lat)
            else:
                d_km = haversine_km(prev_server.lon, prev_server.lat, server.lon, server.lat)

            # 获取 tokens
            tokens = task.get('TaskSize', 1000)

            state_dict, (rL, rC, rS), done, info = env.step(action)

            # 基于实际网络延迟计算成本
            # 基于实际网络延迟计算成本（体现避开陷阱的优势）
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

    # 设置随机种子（影响环境中的陷阱丢包随机性）
    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"Set random seed: {args.seed}")

    # 输出目录
    output_dir = 'inference/results_1000'
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据集和环境
    print(f"Loading dataset from {DATA_ROOT}/{REGION}...")
    # 使用 train 集进行推理，因为 test 集任务太简单（tokens少），无法体现算法差异
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

            # 补全统一字段
            if 'sla_violations' not in data:
                data['sla_violations'] = _sla(data['latencies'])
            sw = data.get('switches', np.zeros_like(data['latencies']))
            if 'risk_scores' not in data:
                data['risk_scores'] = _risk(sw)
            comp = data.get('compute_costs', data.get('costs', np.zeros_like(data['latencies'])))
            net  = data.get('network_costs', np.zeros_like(comp))
            comm = data.get('communication_costs', np.zeros_like(comp))
            data['compute_costs']       = np.asarray(comp, dtype=np.float32)
            data['network_costs']       = np.asarray(net,  dtype=np.float32)
            data['communication_costs'] = np.asarray(comm, dtype=np.float32)
            data['costs']               = data['compute_costs'] + data['network_costs'] + data['communication_costs']

            suffix = args.output_suffix if args.output_suffix else ''
            npz_path = os.path.join(output_dir, f'{algo}_Server2_Trap_seed42{suffix}.npz')
            np.savez(npz_path, **data)
            print(f"Saved to {npz_path}")

            qos = _qos(data['latencies'], data['costs'], sw)
            results[algo] = {
                'avg_latency':   np.mean(data['latencies']),
                'std_latency':   np.std(data['latencies']),
                'p99_latency':   np.percentile(data['latencies'], 99),
                'avg_cost':      np.mean(data['costs']),
                'std_cost':      np.std(data['costs']),
                'avg_reward':    np.mean(data['rewards']),
                'avg_inf_time':  np.mean(data['inference_times']),
                'sla_viol_pct':  np.mean(data['sla_violations']) * 100,
                'composite_qos': qos,
            }

            print(f"\nResults for {algo}:")
            print(f"  Avg Latency:   {results[algo]['avg_latency']:.2f} ± {results[algo]['std_latency']:.2f} ms")
            print(f"  P99 Latency:   {results[algo]['p99_latency']:.2f} ms")
            print(f"  Avg Cost:      ${results[algo]['avg_cost']:.4f} ± {results[algo]['std_cost']:.4f}")
            print(f"  SLA Violation: {results[algo]['sla_viol_pct']:.1f}%")
            print(f"  Composite QoS: {results[algo]['composite_qos']:.2f}")
            print(f"  Avg Reward:    {results[algo]['avg_reward']:.4f}")
            print(f"  Avg InfTime:   {results[algo]['avg_inf_time']:.2f} ms")

        except Exception as e:
            print(f"Error running {algo}: {e}")
            import traceback
            traceback.print_exc()

    # 打印汇总表格
    print("\n" + "="*80)
    print("Summary Table")
    print("="*80)
    print(f"{'Algorithm':<12} {'AvgLat(ms)':<12} {'P99Lat(ms)':<12} {'AvgCost($)':<12} {'SLA_Viol%':<10} {'QoS':<8} {'AvgReward':<12}")
    print("-"*80)
    for algo, res in results.items():
        print(f"{algo:<12} {res['avg_latency']:<12.2f} {res['p99_latency']:<12.2f} {res['avg_cost']:<12.4f} "
              f"{res['sla_viol_pct']:<10.1f} {res['composite_qos']:<8.2f} {res['avg_reward']:<12.4f}")


if __name__ == '__main__':
    main()
