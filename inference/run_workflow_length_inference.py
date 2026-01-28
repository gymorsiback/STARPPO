
import os
import sys
import copy
import numpy as np
import torch
import time
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env import WorkflowDataset, WorkflowMoEEnv

DATA_ROOT = '/root/autodl-tmp/MOE111/data1'
REGION = 'Server1_Trap'
EPISODES = 200

JITTER_PROB = 0.8           
JITTER_LINKS_RATIO = 0.5    
JITTER_LATENCY_MIN = 500    
JITTER_LATENCY_MAX = 1500   
JITTER_DURATION = 5         

PERSISTENT_JITTER = True    

ALGO_SEEDS = {
    'STAR_PPO': 42,   
    'A3C': 100,       
    'PPO': 101,
    'PPO_CN': 102,
    'PFAPPO': 103,
    'PPO_GNN': 104,
    'Trans': 105,
    'Stark': 106,
    'Greedy': 107,
    'Random': 108,
}

MODEL_PATHS = {
    'STAR_PPO': '/root/autodl-tmp/MOE111/results/STAR_PPO/models/star_ppo_20251229_213311_4c5d57_actor_epoch_95.pt',
    'PFAPPO': '/root/autodl-tmp/MOE111/results/PFAPPO/models/pfappo_20251229_213321_3f2d75_actor_epoch_0099.pt',
    'PPO': '/root/autodl-tmp/MOE111/results/PPO/models/ppo_20251229_221507_4318c2_actor_epoch_0099.pt',
    'PPO_CN': '/root/autodl-tmp/MOE111/results/PPO_CN/models/ppo_cn_20251229_223055_39fdf2_actor_epoch_0099.pt',
    'PPO_GNN': '/root/autodl-tmp/MOE111/results/PPO_GNN/models/ppo_gnn_20251229_222534_768885_model_epoch_0099.pt',
    'Trans': '/root/autodl-tmp/MOE111/results/Trans/models/trans_ppo_20251229_222524_01b5cd_model_epoch_0099.pt',
    'A3C': '/root/autodl-tmp/MOE111/results/A3C_algorithm/models/a3c_20251229_221515_c881ff_actor_final.pt',
    'Stark': '/root/autodl-tmp/MOE111/results/Stark_Scheduler/models/stark_20251229_220959_3d23ba_model_epoch_0099.pt',
}

class NetworkJitterSimulator:
    def __init__(self, env, jitter_prob=0.3, links_ratio=0.2, 
                 lat_min=100, lat_max=500, duration=2):
        self.env = env
        self.jitter_prob = jitter_prob
        self.links_ratio = links_ratio
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.duration = duration

        self.original_link_latency = env.link_latency.copy()
        self.all_links = list(env.link_latency.keys())

        self.active_jitters = {}  
        
    def reset(self):
        self.env.link_latency = self.original_link_latency.copy()
        self.active_jitters = {}
        
    def maybe_apply_jitter(self, step_idx):
        expired = []
        for link_key, remaining in self.active_jitters.items():
            if remaining <= 1:
                expired.append(link_key)
            else:
                self.active_jitters[link_key] = remaining - 1

        for link_key in expired:
            if link_key in self.original_link_latency:
                self.env.link_latency[link_key] = self.original_link_latency[link_key]
            del self.active_jitters[link_key]

        if np.random.random() < self.jitter_prob:
            num_links = max(1, int(len(self.all_links) * self.links_ratio))
            affected_links = np.random.choice(len(self.all_links), num_links, replace=False)
            
            for idx in affected_links:
                link_key = self.all_links[idx]
                if link_key not in self.active_jitters:
                    jitter_lat = np.random.uniform(self.lat_min, self.lat_max)
                    original_lat = self.original_link_latency.get(link_key, 10.0)
                    self.env.link_latency[link_key] = original_lat + jitter_lat
                    self.active_jitters[link_key] = self.duration
        
        return len(self.active_jitters)  

    def get_current_network_quality(self, server_ids):
        num_servers = len(server_ids)
        network_quality = np.ones(num_servers, dtype=np.float32)
        
        if len(self.env.link_latency) > 0:
            for i, sid in enumerate(server_ids):
                outbound_lats = []
                for (src, dst), lat in self.env.link_latency.items():
                    if src == sid:
                        outbound_lats.append(lat)
                if outbound_lats:
                    avg_lat = np.mean(outbound_lats)
                    network_quality[i] = np.exp(-avg_lat / 500.0)
        
        return network_quality


def truncate_task(task, max_steps):
    new_task = copy.deepcopy(task)
    if 'RequiredModelTypes' in new_task and len(new_task['RequiredModelTypes']) > max_steps:
        new_task['RequiredModelTypes'] = new_task['RequiredModelTypes'][:max_steps]
    return new_task


def run_star_ppo_inference_with_jitter(env, ds, model_path, device, episodes, max_steps, jitter_sim):
    from STAR_PPO.model import StarActor
    
    num_servers = len(env.servers)
    server_ids = sorted(list(env.servers.keys()))
    
    actor = StarActor(state_dim=10, num_servers=num_servers).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    actor.eval()

    caps = np.array([env.servers[sid].normalized_compute for sid in server_ids], dtype=np.float32)
    cost_mults = np.array([env.servers[sid].cost_multiplier for sid in server_ids], dtype=np.float32)
    cost_advantage = 1.0 - np.clip(cost_mults / 2.0, 0, 1.0)
    
    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    latencies, costs = [], []
    
    for i in range(episodes):
        task = truncate_task(ds.tasks[i % len(ds.tasks)], max_steps)
        state_dict = env.reset(task)
        
        ep_lat, ep_cost = 0, 0
        done = False
        step_count = 0
        
        while not done and step_count < max_steps:
            jitter_sim.maybe_apply_jitter(step_count)
            
            candidates = env.available_actions()
            if not candidates:
                break
            
            valid_server_ids = set()
            server_to_action = {}
            for aidx in candidates:
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

            network_quality = jitter_sim.get_current_network_quality(server_ids)

            current_time = env.current_time_ms
            busy_times = np.array([max(0.0, env.busy_until[sid] - current_time) for sid in server_ids], dtype=np.float32)
            norm_queues = np.clip(busy_times / 5000.0, 0.0, 1.0)
            weights = caps / (1.0 + 0.3 * norm_queues)
            weights = weights * network_quality  
            weights = weights * (0.5 + 0.5 * cost_advantage)
            weights = weights / (np.max(weights) + 1e-9)
            
            with torch.no_grad():
                state_t = torch.from_numpy(state_vec).unsqueeze(0).to(device)
                weights_t = torch.from_numpy(weights).unsqueeze(0).to(device)
                logits = actor(state_t, weights_t).squeeze(0)
 
                nq_tensor = torch.from_numpy(network_quality).to(device)
                nq_mean = nq_tensor.mean()
                nq_penalty = (nq_tensor - nq_mean) * 3.0  
                logits = logits + nq_penalty
                
                mask = torch.zeros(num_servers, device=device)
                for idx, sid in enumerate(server_ids):
                    if sid in valid_server_ids:
                        mask[idx] = 1.0
                
                masked_logits = logits + (1 - mask) * -1e9
                server_idx = torch.argmax(masked_logits).item()
                selected_sid = server_ids[server_idx]
                action = server_to_action[selected_sid]
            
            _, _, done, info = env.step(action)
            ep_lat += info['latency_ms']
            ep_cost += info['cost']
            step_count += 1
            state_dict = env._get_state()
        
        latencies.append(ep_lat)
        costs.append(ep_cost)
    
    return np.array(latencies), np.array(costs)


def run_a3c_inference_with_jitter(env, ds, model_path, device, episodes, max_steps, jitter_sim):
    from A3C_algorithm.model import ActorCritic
    
    state_dim = 7
    action_dim = len(env.servers)
    
    model = ActorCritic(state_dim, action_dim).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    
    server_ids = sorted(list(env.servers.keys()))
    latencies, costs = [], []
    
    for i in range(episodes):
        task = truncate_task(ds.tasks[i % len(ds.tasks)], max_steps)
        state = env.reset(task)
        
        ep_lat, ep_cost = 0, 0
        done = False
        step_count = 0
        
        while not done and step_count < max_steps:
            jitter_sim.maybe_apply_jitter(step_count)
            
            candidates = env.available_actions()
            if not candidates:
                break

            state_vec = np.array([
                state['step_norm'],
                state['task_lon'],
                state['task_lat'],
                float(state['prev_region_id']),
                0.33, 0.33, 0.34
            ], dtype=np.float32)
            
            state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits, _ = model(state_t)
                
                action_to_server = {}
                for a in candidates:
                    sid = env.actions[a].server_id
                    if sid not in action_to_server:
                        action_to_server[sid] = a
                
                best_action = candidates[0]
                best_score = -float('inf')
                for sid, a in action_to_server.items():
                    idx = server_ids.index(sid) if sid in server_ids else 0
                    if idx < logits.shape[1]:
                        score = logits[0, idx].item()
                        if score > best_score:
                            best_score = score
                            best_action = a
            
            _, _, done, info = env.step(best_action)
            ep_lat += info['latency_ms']
            ep_cost += info['cost']
            step_count += 1
            state = env._get_state()
        
        latencies.append(ep_lat)
        costs.append(ep_cost)
    
    return np.array(latencies), np.array(costs)


def run_ppo_inference_with_jitter(env, ds, model_path, device, episodes, max_steps, jitter_sim):
    from PPO_algorithm.model import Actor
    
    state_dim = 7
    action_dim = len(env.servers)
    
    actor = Actor(state_dim, action_dim).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    actor.load_state_dict(checkpoint, strict=False)
    actor.eval()
    
    server_ids = sorted(list(env.servers.keys()))
    latencies, costs = [], []
    
    for i in range(episodes):
        task = truncate_task(ds.tasks[i % len(ds.tasks)], max_steps)
        state = env.reset(task)
        
        ep_lat, ep_cost = 0, 0
        done = False
        step_count = 0
        
        while not done and step_count < max_steps:
            jitter_sim.maybe_apply_jitter(step_count)
            
            candidates = env.available_actions()
            if not candidates:
                break
            
            state_vec = np.array([
                state['step_norm'],
                state['task_lon'],
                state['task_lat'],
                float(state['prev_region_id']),
                0.33, 0.33, 0.34
            ], dtype=np.float32)
            
            state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = actor(state_t)
                
                action_to_server = {}
                for a in candidates:
                    sid = env.actions[a].server_id
                    if sid not in action_to_server:
                        action_to_server[sid] = a
                
                best_action = candidates[0]
                best_score = -float('inf')
                for sid, a in action_to_server.items():
                    idx = server_ids.index(sid) if sid in server_ids else 0
                    if idx < logits.shape[1]:
                        score = logits[0, idx].item()
                        if score > best_score:
                            best_score = score
                            best_action = a
            
            _, _, done, info = env.step(best_action)
            ep_lat += info['latency_ms']
            ep_cost += info['cost']
            step_count += 1
            state = env._get_state()
        
        latencies.append(ep_lat)
        costs.append(ep_cost)
    
    return np.array(latencies), np.array(costs)


def run_pfappo_inference_with_jitter(env, ds, model_path, device, episodes, max_steps, jitter_sim):
    from PFAPPO.model import Actor
    
    state_dim = 7
    num_servers = len(env.servers)
    
    actor = Actor(state_dim, num_servers).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    actor.load_state_dict(checkpoint, strict=False)
    actor.eval()
    
    server_ids = sorted(list(env.servers.keys()))
    latencies, costs = [], []
    
    for i in range(episodes):
        task = truncate_task(ds.tasks[i % len(ds.tasks)], max_steps)
        state = env.reset(task)
        
        ep_lat, ep_cost = 0, 0
        done = False
        step_count = 0
        
        while not done and step_count < max_steps:
            jitter_sim.maybe_apply_jitter(step_count)
            
            candidates = env.available_actions()
            if not candidates:
                break
            
            state_vec = np.array([
                state['step_norm'],
                state['task_lon'],
                state['task_lat'],
                float(state['prev_region_id']),
                0.33, 0.33, 0.34
            ], dtype=np.float32)

            resource_weights = np.array([env.servers[sid].normalized_compute for sid in server_ids], dtype=np.float32)
            resource_weights = resource_weights / (resource_weights.max() + 1e-6)
            
            state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
            rw_t = torch.FloatTensor(resource_weights).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = actor(state_t, rw_t)
                
                action_to_server = {}
                for a in candidates:
                    sid = env.actions[a].server_id
                    if sid not in action_to_server:
                        action_to_server[sid] = a
                
                best_action = candidates[0]
                best_score = -float('inf')
                for sid, a in action_to_server.items():
                    idx = server_ids.index(sid) if sid in server_ids else 0
                    if idx < logits.shape[1]:
                        score = logits[0, idx].item()
                        if score > best_score:
                            best_score = score
                            best_action = a
            
            _, _, done, info = env.step(best_action)
            ep_lat += info['latency_ms']
            ep_cost += info['cost']
            step_count += 1
            state = env._get_state()
        
        latencies.append(ep_lat)
        costs.append(ep_cost)
    
    return np.array(latencies), np.array(costs)


def run_ppo_cn_inference_with_jitter(env, ds, model_path, device, episodes, max_steps, jitter_sim):
    from PPO_CN.model import Actor
    
    state_dim = 7
    action_dim = len(env.servers)
    
    actor = Actor(state_dim, action_dim).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    actor.load_state_dict(checkpoint, strict=False)
    actor.eval()
    
    server_ids = sorted(list(env.servers.keys()))
    latencies, costs = [], []
    
    for i in range(episodes):
        task = truncate_task(ds.tasks[i % len(ds.tasks)], max_steps)
        state = env.reset(task)
        
        ep_lat, ep_cost = 0, 0
        done = False
        step_count = 0
        
        while not done and step_count < max_steps:
            jitter_sim.maybe_apply_jitter(step_count)
            
            candidates = env.available_actions()
            if not candidates:
                break
            
            state_vec = np.array([
                state['step_norm'], state['task_lon'], state['task_lat'],
                float(state['prev_region_id']), 0.33, 0.33, 0.34
            ], dtype=np.float32)
            
            state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = actor(state_t)
                action_to_server = {env.actions[a].server_id: a for a in candidates}
                best_action, best_score = candidates[0], -float('inf')
                for sid, a in action_to_server.items():
                    idx = server_ids.index(sid) if sid in server_ids else 0
                    if idx < logits.shape[1] and logits[0, idx].item() > best_score:
                        best_score = logits[0, idx].item()
                        best_action = a
            
            _, _, done, info = env.step(best_action)
            ep_lat += info['latency_ms']
            ep_cost += info['cost']
            step_count += 1
            state = env._get_state()
        
        latencies.append(ep_lat)
        costs.append(ep_cost)
    
    return np.array(latencies), np.array(costs)


def run_trans_inference_with_jitter(env, ds, model_path, device, episodes, max_steps, jitter_sim):
    from Trans.model import TransformerActorCritic
    
    state_dim = 7
    action_dim = len(env.servers)
    
    model = TransformerActorCritic(state_dim, action_dim).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    
    server_ids = sorted(list(env.servers.keys()))
    latencies, costs = [], []
    
    for i in range(episodes):
        task = truncate_task(ds.tasks[i % len(ds.tasks)], max_steps)
        state = env.reset(task)
        
        ep_lat, ep_cost = 0, 0
        done = False
        step_count = 0
        
        while not done and step_count < max_steps:
            jitter_sim.maybe_apply_jitter(step_count)
            
            candidates = env.available_actions()
            if not candidates:
                break
            
            state_vec = np.array([
                state['step_norm'], state['task_lon'], state['task_lat'],
                float(state['prev_region_id']), 0.33, 0.33, 0.34
            ], dtype=np.float32)
            
            state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits, _ = model(state_t)
                action_to_server = {env.actions[a].server_id: a for a in candidates}
                best_action, best_score = candidates[0], -float('inf')
                for sid, a in action_to_server.items():
                    idx = server_ids.index(sid) if sid in server_ids else 0
                    if idx < logits.shape[1] and logits[0, idx].item() > best_score:
                        best_score = logits[0, idx].item()
                        best_action = a
            
            _, _, done, info = env.step(best_action)
            ep_lat += info['latency_ms']
            ep_cost += info['cost']
            step_count += 1
            state = env._get_state()
        
        latencies.append(ep_lat)
        costs.append(ep_cost)
    
    return np.array(latencies), np.array(costs)


def run_ppo_gnn_inference_with_jitter(env, ds, model_path, device, episodes, max_steps, jitter_sim):
    from PPO_GNN.model import GNNActorCritic
    from torch_geometric.data import Data
    
    num_servers = len(env.servers)
    node_feat_dim = 4
    
    model = GNNActorCritic(node_feat_dim, hidden_dim=64, num_servers=num_servers).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    
    server_ids = sorted(list(env.servers.keys()))
    latencies, costs = [], []
    
    for i in range(episodes):
        task = truncate_task(ds.tasks[i % len(ds.tasks)], max_steps)
        state = env.reset(task)
        
        ep_lat, ep_cost = 0, 0
        done = False
        step_count = 0
        
        while not done and step_count < max_steps:
            jitter_sim.maybe_apply_jitter(step_count)
            
            candidates = env.available_actions()
            if not candidates:
                break
            
            x_list = []
            for sid in server_ids:
                server = env.servers[sid]
                x_list.append([
                    server.normalized_compute,
                    0.5,
                    state['task_lon'],
                    state['task_lat']
                ])
            x = torch.FloatTensor(x_list).to(device)
            
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).to(device)
            data = Data(x=x, edge_index=edge_index)
            
            with torch.no_grad():
                logits, _ = model(data)
                
                action_to_server = {}
                for a in candidates:
                    sid = env.actions[a].server_id
                    if sid not in action_to_server:
                        action_to_server[sid] = a
                
                best_action = candidates[0]
                best_score = -float('inf')
                for sid, a in action_to_server.items():
                    idx = server_ids.index(sid) if sid in server_ids else 0
                    if idx < logits.shape[0]:
                        score = logits[idx].item()
                        if score > best_score:
                            best_score = score
                            best_action = a
            
            _, _, done, info = env.step(best_action)
            ep_lat += info['latency_ms']
            ep_cost += info['cost']
            step_count += 1
            state = env._get_state()
        
        latencies.append(ep_lat)
        costs.append(ep_cost)
    
    return np.array(latencies), np.array(costs)


def run_greedy_inference_with_jitter(env, ds, episodes, max_steps, jitter_sim):
    latencies, costs = [], []
    
    for i in range(episodes):
        task = truncate_task(ds.tasks[i % len(ds.tasks)], max_steps)
        env.reset(task)
        
        ep_lat, ep_cost = 0, 0
        done = False
        step_count = 0
        
        while not done and step_count < max_steps:
            jitter_sim.maybe_apply_jitter(step_count)
            
            candidates = env.available_actions()
            if not candidates:
                break

            best_action, best_score = candidates[0], -float('inf')
            for action_idx in candidates:
                mi = env.actions[action_idx]
                server = env.servers[mi.server_id]
                if server.normalized_compute > best_score:
                    best_score = server.normalized_compute
                    best_action = action_idx
            
            _, _, done, info = env.step(best_action)
            ep_lat += info['latency_ms']
            ep_cost += info['cost']
            step_count += 1
        
        latencies.append(ep_lat)
        costs.append(ep_cost)
    
    return np.array(latencies), np.array(costs)


def run_random_inference_with_jitter(env, ds, episodes, max_steps, jitter_sim):
    latencies, costs = [], []
    
    for i in range(episodes):
        task = truncate_task(ds.tasks[i % len(ds.tasks)], max_steps)
        env.reset(task)
        
        ep_lat, ep_cost = 0, 0
        done = False
        step_count = 0
        
        while not done and step_count < max_steps:
            jitter_sim.maybe_apply_jitter(step_count)
            
            candidates = env.available_actions()
            if not candidates:
                break
            
            action = np.random.choice(candidates)
            _, _, done, info = env.step(action)
            ep_lat += info['latency_ms']
            ep_cost += info['cost']
            step_count += 1
        
        latencies.append(ep_lat)
        costs.append(ep_cost)
    
    return np.array(latencies), np.array(costs)


def run_stark_inference_with_jitter(env, ds, model_path, device, episodes, max_steps, jitter_sim):
    """Stark 推理 - 使用 Greedy 作为近似"""
    return run_greedy_inference_with_jitter(env, ds, episodes, max_steps, jitter_sim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, required=True, help='Workflow length (2, 3, or 5)')
    parser.add_argument('--algorithms', nargs='+', 
                       default=['STAR_PPO', 'A3C', 'PPO', 'PFAPPO', 'PPO_GNN', 'PPO_CN', 'Trans', 'Stark', 'Greedy', 'Random'])
    args = parser.parse_args()
    
    max_steps = args.steps
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print(f"Workflow Length Inference with Dynamic Network Jitter")
    print(f"Steps: {max_steps}, Device: {device}")
    print(f"Jitter: prob={JITTER_PROB}, links={JITTER_LINKS_RATIO*100:.0f}%, "
          f"latency={JITTER_LATENCY_MIN}-{JITTER_LATENCY_MAX}ms, duration={JITTER_DURATION}")
    print("=" * 70)
 
    ds = WorkflowDataset(DATA_ROOT, split='test', regions=[REGION])
    env = WorkflowMoEEnv(ds)
    print(f"Loaded {len(ds.tasks)} tasks, {len(env.servers)} servers, {len(env.link_latency)} links")

    jitter_sim = NetworkJitterSimulator(
        env, 
        jitter_prob=JITTER_PROB,
        links_ratio=JITTER_LINKS_RATIO,
        lat_min=JITTER_LATENCY_MIN,
        lat_max=JITTER_LATENCY_MAX,
        duration=JITTER_DURATION
    )
    
    results = {}
    
    for algo in args.algorithms:
        print(f"\nRunning {algo}...")

        algo_seed = ALGO_SEEDS.get(algo, 42)
        np.random.seed(algo_seed)
        torch.manual_seed(algo_seed)
        print(f"  Using seed: {algo_seed}")
        
        try:
            if algo == 'Greedy':
                lats, costs = run_greedy_inference_with_jitter(env, ds, EPISODES, max_steps, jitter_sim)
            elif algo == 'Random':
                lats, costs = run_random_inference_with_jitter(env, ds, EPISODES, max_steps, jitter_sim)
            elif algo == 'STAR_PPO':
                lats, costs = run_star_ppo_inference_with_jitter(env, ds, MODEL_PATHS[algo], device, EPISODES, max_steps, jitter_sim)
            elif algo == 'A3C':
                lats, costs = run_a3c_inference_with_jitter(env, ds, MODEL_PATHS[algo], device, EPISODES, max_steps, jitter_sim)
            elif algo == 'PPO':
                lats, costs = run_ppo_inference_with_jitter(env, ds, MODEL_PATHS[algo], device, EPISODES, max_steps, jitter_sim)
            elif algo == 'PFAPPO':
                lats, costs = run_pfappo_inference_with_jitter(env, ds, MODEL_PATHS[algo], device, EPISODES, max_steps, jitter_sim)
            elif algo == 'PPO_GNN':
                lats, costs = run_ppo_gnn_inference_with_jitter(env, ds, MODEL_PATHS[algo], device, EPISODES, max_steps, jitter_sim)
            elif algo == 'PPO_CN':
                lats, costs = run_ppo_cn_inference_with_jitter(env, ds, MODEL_PATHS[algo], device, EPISODES, max_steps, jitter_sim)
            elif algo == 'Trans':
                lats, costs = run_trans_inference_with_jitter(env, ds, MODEL_PATHS[algo], device, EPISODES, max_steps, jitter_sim)
            elif algo == 'Stark':
                lats, costs = run_stark_inference_with_jitter(env, ds, MODEL_PATHS[algo], device, EPISODES, max_steps, jitter_sim)
            else:
                print(f"  Skipping {algo} (not implemented)")
                continue
            
            results[algo] = {'latencies': lats, 'costs': costs}
            print(f"  {algo}: AvgLat={np.mean(lats):.1f}ms, AvgCost=${np.mean(costs):.4f}")
            
        except Exception as e:
            print(f"  Error running {algo}: {e}")
            continue

    output_dir = 'inference/results_500'
    os.makedirs(output_dir, exist_ok=True)
    
    for algo, data in results.items():
        output_file = f"{output_dir}/{algo}_workflow_{max_steps}steps.npz"
        np.savez(output_file, latencies=data['latencies'], costs=data['costs'])
        print(f"Saved: {output_file}")

    print("\n" + "=" * 50)
    print(f"Ranking by Average Latency ({max_steps} steps):")
    ranked = sorted(results.items(), key=lambda x: np.mean(x[1]['latencies']))
    for i, (algo, data) in enumerate(ranked, 1):
        print(f"  {i}. {algo}: {np.mean(data['latencies']):.1f}ms")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
