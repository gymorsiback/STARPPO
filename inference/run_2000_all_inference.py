"""
在 2000 规模数据集 (data1/Server3_Trap) 上运行所有算法的推理
用于 E10 可扩展性分析
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
from metrics import sla_violation as _sla, composite_qos as _qos

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

DATA_ROOT = './data1'
REGION = 'Server3_Trap'
NUM_SERVERS = 2000
EPISODES = 200

PROJECT_ROOT = '.'

SEED_MODEL_MAP = {
    'STAR_PPO': {
        42: f'{PROJECT_ROOT}/results/TopoFreeRL/models/LATEST_Server3_Trap_seed42_actor_epoch_100.pt',
    },
    'PFAPPO': {
        42: f'{PROJECT_ROOT}/results/PFAPPO/models/LATEST_Server3_Trap_seed42_actor_epoch_0099.pt',
    },
    'PPO_CN': {
        42: f'{PROJECT_ROOT}/results/PPO_CN/models/LATEST_Server3_Trap_seed42_actor_epoch_0099.pt',
    },
    'Trans': {
        42: f'{PROJECT_ROOT}/results/Trans/models/LATEST_Server3_Trap_seed42_model_epoch_0099.pt',
    },
    'A3C': {
        42: f'{PROJECT_ROOT}/results/A3C_algorithm/models/LATEST_Server3_Trap_seed42_actor_final.pt',
    },
    'PPO_GNN': {
        42: f'{PROJECT_ROOT}/results/PPO_GNN/models/LATEST_Server3_Trap_seed42_model_epoch_0099.pt',
    },
    'PPO': {
        42: f'{PROJECT_ROOT}/results/PPO/models/LATEST_Server3_Trap_seed42_actor_final.pt',
    },
    'Stark': {
        42: f'{PROJECT_ROOT}/results/Stark_Scheduler/models/LATEST_Server3_Trap_seed42_final.pt',
    },
}


def find_model(algo_name, train_seed=42):
    if algo_name in SEED_MODEL_MAP and train_seed in SEED_MODEL_MAP[algo_name]:
        path = SEED_MODEL_MAP[algo_name][train_seed]
        if os.path.exists(path):
            return path
        print(f"Warning: Model not found at {path}")
    return None


def run_greedy_inference(env, ds, episodes):
    """Greedy: 算力最大优先"""
    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    latencies, rewards, switches, inf_times = [], [], [], []
    compute_costs, network_costs, communication_costs = [], [], []

    for i in range(episodes):
        task = ds.tasks[i % len(ds.tasks)]
        env.reset(task)
        tokens = task.get('TaskSize', 1000)
        ep_lat, ep_reward, ep_inf_time = 0, 0, 0
        ep_cc, ep_nc, ep_mc = 0, 0, 0
        done = False

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
            _, (rL, rC, rS), done, info = env.step(best_action)
            network_ms = info.get('network_ms', 0)
            ep_lat += info['latency_ms']
            ep_cc += info['cost']
            ep_nc += compute_network_cost_by_latency(network_ms, tokens)
            ep_mc += compute_communication_cost_by_latency(tokens, network_ms)
            ep_reward += w[0]*rL + w[1]*rC + w[2]*rS

        latencies.append(ep_lat); compute_costs.append(ep_cc)
        network_costs.append(ep_nc); communication_costs.append(ep_mc)
        rewards.append(ep_reward); switches.append(env.ep_switches)
        inf_times.append(ep_inf_time)
        if (i + 1) % 50 == 0:
            print(f"  Greedy: {i+1}/{episodes}")

    tc = np.array(compute_costs) + np.array(network_costs) + np.array(communication_costs)
    return dict(latencies=np.array(latencies), costs=tc,
                compute_costs=np.array(compute_costs), network_costs=np.array(network_costs),
                communication_costs=np.array(communication_costs),
                rewards=np.array(rewards), switches=np.array(switches),
                inference_times=np.array(inf_times))


def run_random_inference(env, ds, episodes):
    """Random: 随机选择"""
    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    latencies, rewards, switches, inf_times = [], [], [], []
    compute_costs, network_costs, communication_costs = [], [], []

    for i in range(episodes):
        task = ds.tasks[i % len(ds.tasks)]
        env.reset(task)
        tokens = task.get('TaskSize', 1000)
        ep_lat, ep_reward, ep_inf_time = 0, 0, 0
        ep_cc, ep_nc, ep_mc = 0, 0, 0
        done = False

        while not done:
            t0 = time.time()
            candidates = env.available_actions()
            if not candidates:
                candidates = list(range(len(env.actions)))
            action = np.random.choice(candidates)
            ep_inf_time += (time.time() - t0) * 1000
            _, (rL, rC, rS), done, info = env.step(action)
            network_ms = info.get('network_ms', 0)
            ep_lat += info['latency_ms']
            ep_cc += info['cost']
            ep_nc += compute_network_cost_by_latency(network_ms, tokens)
            ep_mc += compute_communication_cost_by_latency(tokens, network_ms)
            ep_reward += w[0]*rL + w[1]*rC + w[2]*rS

        latencies.append(ep_lat); compute_costs.append(ep_cc)
        network_costs.append(ep_nc); communication_costs.append(ep_mc)
        rewards.append(ep_reward); switches.append(env.ep_switches)
        inf_times.append(ep_inf_time)
        if (i + 1) % 50 == 0:
            print(f"  Random: {i+1}/{episodes}")

    tc = np.array(compute_costs) + np.array(network_costs) + np.array(communication_costs)
    return dict(latencies=np.array(latencies), costs=tc,
                compute_costs=np.array(compute_costs), network_costs=np.array(network_costs),
                communication_costs=np.array(communication_costs),
                rewards=np.array(rewards), switches=np.array(switches),
                inference_times=np.array(inf_times))


def run_pfappo_inference(env, ds, model_path, device, episodes):
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
    compute_costs, network_costs, communication_costs = [], [], []

    for i in range(episodes):
        task = ds.tasks[i % len(ds.tasks)]
        state_dict = env.reset(task)
        tokens = task.get('TaskSize', 1000)
        ep_lat, ep_reward, ep_inf_time = 0, 0, 0
        ep_cc, ep_nc, ep_mc = 0, 0, 0
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
            state_vec = np.array([state_dict['step_norm'], state_dict['task_lon'],
                                   state_dict['task_lat'], float(state_dict['prev_region_id']),
                                   w[0], w[1], w[2]], dtype=np.float32)
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
            ep_lat += info['latency_ms']; ep_cc += info['cost']
            ep_nc += compute_network_cost_by_latency(network_ms, tokens)
            ep_mc += compute_communication_cost_by_latency(tokens, network_ms)
            ep_reward += w[0]*rL + w[1]*rC + w[2]*rS

        latencies.append(ep_lat); compute_costs.append(ep_cc)
        network_costs.append(ep_nc); communication_costs.append(ep_mc)
        rewards.append(ep_reward); switches.append(env.ep_switches)
        inf_times.append(ep_inf_time)
        if (i + 1) % 50 == 0:
            print(f"  PFAPPO: {i+1}/{episodes}")

    tc = np.array(compute_costs) + np.array(network_costs) + np.array(communication_costs)
    return dict(latencies=np.array(latencies), costs=tc,
                compute_costs=np.array(compute_costs), network_costs=np.array(network_costs),
                communication_costs=np.array(communication_costs),
                rewards=np.array(rewards), switches=np.array(switches),
                inference_times=np.array(inf_times))


def run_ppo_inference(env, ds, model_path, device, episodes):
    from PPO_algorithm.model import Actor
    actor = Actor(state_dim=7, num_servers=NUM_SERVERS).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device))
    actor.eval()

    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    server_ids = sorted(list(env.servers.keys()))
    latencies, rewards, switches, inf_times = [], [], [], []
    compute_costs, network_costs, communication_costs = [], [], []

    for i in range(episodes):
        task = ds.tasks[i % len(ds.tasks)]
        state_dict = env.reset(task)
        tokens = task.get('TaskSize', 1000)
        ep_lat, ep_reward, ep_inf_time = 0, 0, 0
        ep_cc, ep_nc, ep_mc = 0, 0, 0
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
            state_vec = np.array([state_dict['step_norm'], state_dict['task_lon'],
                                   state_dict['task_lat'], float(state_dict['prev_region_id']),
                                   w[0], w[1], w[2]], dtype=np.float32)
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
            ep_lat += info['latency_ms']; ep_cc += info['cost']
            ep_nc += compute_network_cost_by_latency(network_ms, tokens)
            ep_mc += compute_communication_cost_by_latency(tokens, network_ms)
            ep_reward += w[0]*rL + w[1]*rC + w[2]*rS

        latencies.append(ep_lat); compute_costs.append(ep_cc)
        network_costs.append(ep_nc); communication_costs.append(ep_mc)
        rewards.append(ep_reward); switches.append(env.ep_switches)
        inf_times.append(ep_inf_time)
        if (i + 1) % 50 == 0:
            print(f"  PPO: {i+1}/{episodes}")

    tc = np.array(compute_costs) + np.array(network_costs) + np.array(communication_costs)
    return dict(latencies=np.array(latencies), costs=tc,
                compute_costs=np.array(compute_costs), network_costs=np.array(network_costs),
                communication_costs=np.array(communication_costs),
                rewards=np.array(rewards), switches=np.array(switches),
                inference_times=np.array(inf_times))


def run_ppo_cn_inference(env, ds, model_path, device, episodes):
    from PPO_CN.model import Actor
    actor = Actor(state_dim=7, action_dim=NUM_SERVERS).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    actor.eval()

    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    server_ids = sorted(list(env.servers.keys()))
    latencies, rewards, switches, inf_times = [], [], [], []
    compute_costs, network_costs, communication_costs = [], [], []

    for i in range(episodes):
        task = ds.tasks[i % len(ds.tasks)]
        state_dict = env.reset(task)
        tokens = task.get('TaskSize', 1000)
        ep_lat, ep_reward, ep_inf_time = 0, 0, 0
        ep_cc, ep_nc, ep_mc = 0, 0, 0
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
            state_vec = np.array([state_dict['step_norm'], state_dict['task_lon'],
                                   state_dict['task_lat'], float(state_dict['prev_region_id']),
                                   w[0], w[1], w[2]], dtype=np.float32)
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
            ep_lat += info['latency_ms']; ep_cc += info['cost']
            ep_nc += compute_network_cost_by_latency(network_ms, tokens)
            ep_mc += compute_communication_cost_by_latency(tokens, network_ms)
            ep_reward += w[0]*rL + w[1]*rC + w[2]*rS

        latencies.append(ep_lat); compute_costs.append(ep_cc)
        network_costs.append(ep_nc); communication_costs.append(ep_mc)
        rewards.append(ep_reward); switches.append(env.ep_switches)
        inf_times.append(ep_inf_time)
        if (i + 1) % 50 == 0:
            print(f"  PPO_CN: {i+1}/{episodes}")

    tc = np.array(compute_costs) + np.array(network_costs) + np.array(communication_costs)
    return dict(latencies=np.array(latencies), costs=tc,
                compute_costs=np.array(compute_costs), network_costs=np.array(network_costs),
                communication_costs=np.array(communication_costs),
                rewards=np.array(rewards), switches=np.array(switches),
                inference_times=np.array(inf_times))


def run_a3c_inference(env, ds, model_path, device, episodes):
    from A3C_algorithm.model import ActorCritic
    model = ActorCritic(state_dim=7, num_servers=NUM_SERVERS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    server_ids = sorted(list(env.servers.keys()))
    latencies, rewards, switches, inf_times = [], [], [], []
    compute_costs, network_costs, communication_costs = [], [], []

    for i in range(episodes):
        task = ds.tasks[i % len(ds.tasks)]
        state_dict = env.reset(task)
        tokens = task.get('TaskSize', 1000)
        ep_lat, ep_reward, ep_inf_time = 0, 0, 0
        ep_cc, ep_nc, ep_mc = 0, 0, 0
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
            state_vec = np.array([state_dict['step_norm'], state_dict['task_lon'],
                                   state_dict['task_lat'], float(state_dict['prev_region_id']),
                                   w[0], w[1], w[2]], dtype=np.float32)
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
            ep_lat += info['latency_ms']; ep_cc += info['cost']
            ep_nc += compute_network_cost_by_latency(network_ms, tokens)
            ep_mc += compute_communication_cost_by_latency(tokens, network_ms)
            ep_reward += w[0]*rL + w[1]*rC + w[2]*rS

        latencies.append(ep_lat); compute_costs.append(ep_cc)
        network_costs.append(ep_nc); communication_costs.append(ep_mc)
        rewards.append(ep_reward); switches.append(env.ep_switches)
        inf_times.append(ep_inf_time)
        if (i + 1) % 50 == 0:
            print(f"  A3C: {i+1}/{episodes}")

    tc = np.array(compute_costs) + np.array(network_costs) + np.array(communication_costs)
    return dict(latencies=np.array(latencies), costs=tc,
                compute_costs=np.array(compute_costs), network_costs=np.array(network_costs),
                communication_costs=np.array(communication_costs),
                rewards=np.array(rewards), switches=np.array(switches),
                inference_times=np.array(inf_times))


def run_ppo_gnn_inference(env, ds, model_path, device, episodes):
    from PPO_GNN.model import GNNActorCritic
    from torch_geometric.data import Data

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
    compute_costs, network_costs, communication_costs = [], [], []

    for i in range(episodes):
        task = ds.tasks[i % len(ds.tasks)]
        state_dict = env.reset(task)
        tokens = task.get('TaskSize', 1000)
        ep_lat, ep_reward, ep_inf_time = 0, 0, 0
        ep_cc, ep_nc, ep_mc = 0, 0, 0
        done = False

        while not done:
            t0 = time.time()
            current_time = env.current_time_ms
            busy_times = np.array([max(0.0, env.busy_until[sid] - current_time) for sid in server_ids], dtype=np.float32)
            norm_queues = np.clip(busy_times / 5000.0, 0.0, 1.0)
            dynamic_feat = torch.FloatTensor(norm_queues).unsqueeze(1).to(device)
            node_feats = torch.cat([static_feats[:, 0:1], dynamic_feat, static_feats[:, 1:2]], dim=1)
            global_feat = torch.FloatTensor([
                state_dict['step_norm'], state_dict['task_lon'], state_dict['task_lat'],
                float(state_dict['prev_region_id']), w[0], w[1], w[2]
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

            data = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_attr,
                        global_feat=global_feat, candidate_mask=candidate_mask, num_nodes=num_servers)
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
            ep_lat += info['latency_ms']; ep_cc += info['cost']
            ep_nc += compute_network_cost_by_latency(network_ms, tokens)
            ep_mc += compute_communication_cost_by_latency(tokens, network_ms)
            ep_reward += w[0]*rL + w[1]*rC + w[2]*rS

        latencies.append(ep_lat); compute_costs.append(ep_cc)
        network_costs.append(ep_nc); communication_costs.append(ep_mc)
        rewards.append(ep_reward); switches.append(env.ep_switches)
        inf_times.append(ep_inf_time)
        if (i + 1) % 50 == 0:
            print(f"  PPO_GNN: {i+1}/{episodes}")

    tc = np.array(compute_costs) + np.array(network_costs) + np.array(communication_costs)
    return dict(latencies=np.array(latencies), costs=tc,
                compute_costs=np.array(compute_costs), network_costs=np.array(network_costs),
                communication_costs=np.array(communication_costs),
                rewards=np.array(rewards), switches=np.array(switches),
                inference_times=np.array(inf_times))


def run_stark_inference(env, ds, model_path, device, episodes):
    from Stark_Scheduler.model import StarkScheduler
    from Stark_Scheduler.dataset import OnlineExpertDataset

    model = StarkScheduler(task_dim=4, server_dim=7, num_servers=NUM_SERVERS, d_model=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    expert_data = OnlineExpertDataset(env)

    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    latencies, rewards, switches, inf_times = [], [], [], []
    compute_costs, network_costs, communication_costs = [], [], []

    for i in range(episodes):
        task = ds.tasks[i % len(ds.tasks)]
        state_dict = env.reset(task)
        tokens = task.get('TaskSize', 1000)
        ep_lat, ep_reward, ep_inf_time = 0, 0, 0
        ep_cc, ep_nc, ep_mc = 0, 0, 0
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
                server_ids_list = sorted(list(env.servers.keys()))
                target_server_id = server_ids_list[server_idx]
                action = available[0]
                for a in available:
                    if env.actions[a].server_id == target_server_id:
                        action = a
                        break
            ep_inf_time += (time.time() - t0) * 1000
            state_dict, (rL, rC, rS), done, info = env.step(action)
            network_ms = info.get('network_ms', 0)
            ep_lat += info['latency_ms']; ep_cc += info['cost']
            ep_nc += compute_network_cost_by_latency(network_ms, tokens)
            ep_mc += compute_communication_cost_by_latency(tokens, network_ms)
            ep_reward += w[0]*rL + w[1]*rC + w[2]*rS

        latencies.append(ep_lat); compute_costs.append(ep_cc)
        network_costs.append(ep_nc); communication_costs.append(ep_mc)
        rewards.append(ep_reward); switches.append(env.ep_switches)
        inf_times.append(ep_inf_time)
        if (i + 1) % 50 == 0:
            print(f"  Stark: {i+1}/{episodes}")

    tc = np.array(compute_costs) + np.array(network_costs) + np.array(communication_costs)
    return dict(latencies=np.array(latencies), costs=tc,
                compute_costs=np.array(compute_costs), network_costs=np.array(network_costs),
                communication_costs=np.array(communication_costs),
                rewards=np.array(rewards), switches=np.array(switches),
                inference_times=np.array(inf_times))


def run_trans_inference(env, ds, model_path, device, episodes):
    from Trans.model import TransformerActorCritic
    model = TransformerActorCritic(state_dim=7, action_dim=NUM_SERVERS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    server_ids = sorted(list(env.servers.keys()))
    latencies, rewards, switches, inf_times = [], [], [], []
    compute_costs, network_costs, communication_costs = [], [], []
    max_seq_len = 20

    for i in range(episodes):
        task = ds.tasks[i % len(ds.tasks)]
        state_dict = env.reset(task)
        tokens = task.get('TaskSize', 1000)
        ep_lat, ep_reward, ep_inf_time = 0, 0, 0
        ep_cc, ep_nc, ep_mc = 0, 0, 0
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
            state_vec = np.array([state_dict['step_norm'], state_dict['task_lon'],
                                   state_dict['task_lat'], float(state_dict['prev_region_id']),
                                   w[0], w[1], w[2]], dtype=np.float32)
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
            ep_lat += info['latency_ms']; ep_cc += info['cost']
            ep_nc += compute_network_cost_by_latency(network_ms, tokens)
            ep_mc += compute_communication_cost_by_latency(tokens, network_ms)
            ep_reward += w[0]*rL + w[1]*rC + w[2]*rS

        latencies.append(ep_lat); compute_costs.append(ep_cc)
        network_costs.append(ep_nc); communication_costs.append(ep_mc)
        rewards.append(ep_reward); switches.append(env.ep_switches)
        inf_times.append(ep_inf_time)
        if (i + 1) % 50 == 0:
            print(f"  Trans: {i+1}/{episodes}")

    tc = np.array(compute_costs) + np.array(network_costs) + np.array(communication_costs)
    return dict(latencies=np.array(latencies), costs=tc,
                compute_costs=np.array(compute_costs), network_costs=np.array(network_costs),
                communication_costs=np.array(communication_costs),
                rewards=np.array(rewards), switches=np.array(switches),
                inference_times=np.array(inf_times))


def run_star_ppo_inference(env, ds, model_path, device, episodes):
    from TopoFreeRL.model import StarActor
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
            outbound_lats = [lat for (src, dst), lat in env.link_latency.items() if src == sid]
            if outbound_lats:
                network_quality[i] = np.exp(-np.mean(outbound_lats) / 500.0)

    latencies, rewards, switches, inf_times = [], [], [], []
    compute_costs, network_costs, communication_costs = [], [], []

    for i in range(episodes):
        task = ds.tasks[i % len(ds.tasks)]
        state_dict = env.reset(task)
        tokens = task.get('TaskSize', 1000)
        ep_lat, ep_reward, ep_inf_time = 0, 0, 0
        ep_cc, ep_nc, ep_mc = 0, 0, 0
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
            base_state = np.array([state_dict['step_norm'], state_dict['task_lon'],
                                    state_dict['task_lat'], float(state_dict['prev_region_id']),
                                    w[0], w[1], w[2]], dtype=np.float32)
            aug_features = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            state_vec = np.concatenate([base_state, aug_features])
            current_time = env.current_time_ms
            busy_times = np.array([max(0.0, env.busy_until[sid] - current_time) for sid in server_ids], dtype=np.float32)
            norm_queues = np.clip(busy_times / 5000.0, 0.0, 1.0)
            weights = caps / (1.0 + 0.30 * norm_queues)
            weights = weights * network_quality * (0.5 + 0.5 * cost_advantage)
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
            ep_lat += info['latency_ms']; ep_cc += info['cost']
            ep_nc += compute_network_cost_by_latency(network_ms, tokens)
            ep_mc += compute_communication_cost_by_latency(tokens, network_ms)
            ep_reward += w[0]*rL + w[1]*rC + w[2]*rS

        latencies.append(ep_lat); compute_costs.append(ep_cc)
        network_costs.append(ep_nc); communication_costs.append(ep_mc)
        rewards.append(ep_reward); switches.append(env.ep_switches)
        inf_times.append(ep_inf_time)
        if (i + 1) % 50 == 0:
            print(f"  STAR_PPO: {i+1}/{episodes}")

    tc = np.array(compute_costs) + np.array(network_costs) + np.array(communication_costs)
    return dict(latencies=np.array(latencies), costs=tc,
                compute_costs=np.array(compute_costs), network_costs=np.array(network_costs),
                communication_costs=np.array(communication_costs),
                rewards=np.array(rewards), switches=np.array(switches),
                inference_times=np.array(inf_times))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--episodes', type=int, default=EPISODES)
    parser.add_argument('--algorithms', nargs='+', default=['all'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"Device: {args.device}, Seed: {args.seed}")

    output_dir = 'inference/results_2000'
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading dataset from {DATA_ROOT}/{REGION}...")
    ds = WorkflowDataset(DATA_ROOT, split='train', regions=[REGION])
    env = WorkflowMoEEnv(ds)
    print(f"Loaded {len(ds.tasks)} tasks, {len(env.servers)} servers")

    all_algos = ['STAR_PPO', 'PFAPPO', 'PPO', 'PPO_CN', 'A3C', 'Trans', 'Stark', 'PPO_GNN', 'Greedy', 'Random']

    # Skip algorithms that already have results
    existing = set()
    for f in glob.glob(os.path.join(output_dir, '*.npz')):
        algo = os.path.basename(f).split('_Server3')[0]
        existing.add(algo)
    if existing:
        print(f"Already have results for: {', '.join(sorted(existing))}")

    if 'all' in args.algorithms:
        algos_to_run = [a for a in all_algos if a not in existing]
    else:
        algos_to_run = args.algorithms

    if not algos_to_run:
        print("All algorithms already have 2000-node results. Done.")
        return

    print(f"Will run: {', '.join(algos_to_run)}")

    results = {}
    for algo in algos_to_run:
        print(f"\n{'='*60}")
        print(f"Running {algo} on 2000-node Server3_Trap...")
        print('='*60)
        try:
            if algo == 'Greedy':
                data = run_greedy_inference(env, ds, args.episodes)
            elif algo == 'Random':
                data = run_random_inference(env, ds, args.episodes)
            elif algo == 'PFAPPO':
                mp = find_model('PFAPPO')
                if not mp:
                    print("No PFAPPO model for Server3, skipping"); continue
                print(f"Model: {mp}")
                data = run_pfappo_inference(env, ds, mp, args.device, args.episodes)
            elif algo == 'PPO':
                mp = find_model('PPO')
                if not mp:
                    print("No PPO model for Server3, skipping"); continue
                print(f"Model: {mp}")
                data = run_ppo_inference(env, ds, mp, args.device, args.episodes)
            elif algo == 'PPO_CN':
                mp = find_model('PPO_CN')
                if not mp:
                    print("No PPO_CN model for Server3, skipping"); continue
                print(f"Model: {mp}")
                data = run_ppo_cn_inference(env, ds, mp, args.device, args.episodes)
            elif algo == 'A3C':
                mp = find_model('A3C')
                if not mp:
                    print("No A3C model for Server3, skipping"); continue
                print(f"Model: {mp}")
                data = run_a3c_inference(env, ds, mp, args.device, args.episodes)
            elif algo == 'STAR_PPO':
                mp = find_model('STAR_PPO')
                if not mp:
                    print("No STAR_PPO model for Server3, skipping"); continue
                print(f"Model: {mp}")
                data = run_star_ppo_inference(env, ds, mp, args.device, args.episodes)
            elif algo == 'Trans':
                mp = find_model('Trans')
                if not mp:
                    print("No Trans model for Server3, skipping"); continue
                print(f"Model: {mp}")
                data = run_trans_inference(env, ds, mp, args.device, args.episodes)
            elif algo == 'Stark':
                mp = find_model('Stark')
                if not mp:
                    print("No Stark model for Server3, skipping"); continue
                print(f"Model: {mp}")
                data = run_stark_inference(env, ds, mp, args.device, args.episodes)
            elif algo == 'PPO_GNN':
                mp = find_model('PPO_GNN')
                if not mp:
                    print("No PPO_GNN model for Server3, skipping"); continue
                print(f"Model: {mp}")
                data = run_ppo_gnn_inference(env, ds, mp, args.device, args.episodes)
            else:
                print(f"Algorithm {algo} not implemented, skipping")
                continue

            # Compute derived fields
            if 'sla_violations' not in data:
                data['sla_violations'] = _sla(data['latencies'])
            sw = data.get('switches', np.zeros_like(data['latencies']))
            comp = data.get('compute_costs', data.get('costs', np.zeros_like(data['latencies'])))
            net  = data.get('network_costs', np.zeros_like(comp))
            comm = data.get('communication_costs', np.zeros_like(comp))
            data['compute_costs']       = np.asarray(comp, dtype=np.float32)
            data['network_costs']       = np.asarray(net,  dtype=np.float32)
            data['communication_costs'] = np.asarray(comm, dtype=np.float32)
            data['costs']               = data['compute_costs'] + data['network_costs'] + data['communication_costs']

            npz_path = os.path.join(output_dir, f'{algo}_Server3_Trap_seed42.npz')
            np.savez(npz_path, **data)
            print(f"Saved: {npz_path}")

            qos = _qos(data['latencies'], data['costs'], sw)
            results[algo] = {
                'avg_latency':  np.mean(data['latencies']),
                'std_latency':  np.std(data['latencies']),
                'p99_latency':  np.percentile(data['latencies'], 99),
                'avg_cost':     np.mean(data['costs']),
                'sla_viol_pct': np.mean(data['sla_violations']) * 100,
                'qos':          qos,
                'avg_inf_time': np.mean(data['inference_times']),
            }
            r = results[algo]
            print(f"  Avg Latency:   {r['avg_latency']:.2f} ± {r['std_latency']:.2f} ms")
            print(f"  P99 Latency:   {r['p99_latency']:.2f} ms")
            print(f"  Avg Cost:      ${r['avg_cost']:.4f}")
            print(f"  SLA Violation: {r['sla_viol_pct']:.1f}%")
            print(f"  Composite QoS: {r['qos']:.2f}")
            print(f"  Avg InfTime:   {r['avg_inf_time']:.2f} ms/episode")

        except Exception as e:
            print(f"ERROR in {algo}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("Summary — 2000-node Server3_Trap")
    print("="*80)
    print(f"{'Algorithm':<12} {'AvgLat(ms)':<12} {'P99Lat(ms)':<12} {'AvgCost':<10} {'SLA%':<8} {'QoS':<8} {'InfTime(ms)':<12}")
    print("-"*80)
    for algo, r in results.items():
        print(f"{algo:<12} {r['avg_latency']:<12.2f} {r['p99_latency']:<12.2f} "
              f"{r['avg_cost']:<10.4f} {r['sla_viol_pct']:<8.1f} {r['qos']:<8.2f} {r['avg_inf_time']:<12.2f}")


if __name__ == '__main__':
    main()
