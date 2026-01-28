
import os
import sys
import numpy as np
import torch
import time

sys.path.insert(0, '/root/autodl-tmp/MOE111')

from env import WorkflowDataset, WorkflowMoEEnv
from utils import haversine_km

DATA_ROOT = '/root/autodl-tmp/MOE111/data1'
REGION_TARGET = 'Server3_Trap'  
NUM_SERVERS_MODEL = 500  
NUM_SERVERS_ENV = 2000   
NUM_PARTITIONS = 4       
EPISODES = 200
SEED = 42

MODELS_500 = {
    'STAR_PPO': '/root/autodl-tmp/MOE111/results/STAR_PPO/models/star_ppo_20251229_213311_4c5d57_actor_epoch_95.pt',
    'PPO_GNN': '/root/autodl-tmp/MOE111/results/PPO_GNN/models/ppo_gnn_20251229_222534_768885_model_epoch_0099.pt',
    'A3C': '/root/autodl-tmp/MOE111/results/A3C_algorithm/models/a3c_20251229_221515_c881ff_actor_final.pt',
    'PPO': '/root/autodl-tmp/MOE111/results/PPO/models/ppo_20251229_221507_4318c2_actor_epoch_0099.pt',
    'Stark': '/root/autodl-tmp/MOE111/results/Stark_Scheduler/models/LATEST_Server1_Trap_seed42_final.pt',
    'Trans': '/root/autodl-tmp/MOE111/results/Trans/models/trans_ppo_20251229_222524_01b5cd_model_epoch_0099.pt',
}

MODELS_TARGET = {
    'STAR_PPO': '/root/autodl-tmp/MOE111/results/STAR_PPO/models/LATEST_Server3_Trap_seed42_actor_epoch_100.pt',
}

OUTPUT_DIR = '/root/autodl-tmp/MOE111/Generalization Experiments/Zero-Shot Scalability Transfer'


def run_star_ppo_retrained(env, ds, model_path, device, episodes):
    from STAR_PPO.model import StarActor
    
    num_servers = len(env.servers)
    actor = StarActor(state_dim=10, num_servers=num_servers).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    actor.eval()
    
    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    server_ids = sorted(list(env.servers.keys()))
    
    caps = np.array([env.servers[sid].normalized_compute for sid in server_ids], dtype=np.float32)
    cost_mults = np.array([env.servers[sid].cost_multiplier for sid in server_ids], dtype=np.float32)
    cost_advantage = 1.0 - np.clip(cost_mults / 2.0, 0, 1.0)
    
    network_quality = np.ones(num_servers, dtype=np.float32)
    if hasattr(env, 'link_latency') and len(env.link_latency) > 0:
        for i, sid in enumerate(server_ids):
            outbound_lats = [lat for (src, dst), lat in env.link_latency.items() if src == sid]
            if outbound_lats:
                network_quality[i] = np.exp(-np.mean(outbound_lats) / 500.0)
    
    latencies, compute_costs = [], []
    
    for i in range(episodes):
        task = ds.tasks[i % len(ds.tasks)]
        state_dict = env.reset(task)
        ep_lat, ep_cost, done = 0, 0, False
        
        while not done:
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
            
            base_state = np.array([state_dict['step_norm'], state_dict['task_lon'], state_dict['task_lat'],
                                   float(state_dict['prev_region_id']), w[0], w[1], w[2]], dtype=np.float32)
            state_vec = np.concatenate([base_state, np.array([0.5, 0.5, 0.5], dtype=np.float32)])
            
            current_time = env.current_time_ms
            busy_times = np.array([max(0.0, env.busy_until[sid] - current_time) for sid in server_ids], dtype=np.float32)
            norm_queues = np.clip(busy_times / 5000.0, 0.0, 1.0)
            weights = caps / (1.0 + 0.30 * norm_queues) * network_quality * (0.5 + 0.5 * cost_advantage)
            weights = weights / (np.max(weights) + 1e-9)
            
            with torch.no_grad():
                state_t = torch.from_numpy(state_vec).unsqueeze(0).to(device)
                weights_t = torch.from_numpy(weights).unsqueeze(0).to(device)
                logits = actor(state_t, weights_t).squeeze(0)
                
                mask = torch.zeros(num_servers, device=device)
                for idx, sid in enumerate(server_ids):
                    if sid in valid_server_ids:
                        mask[idx] = 1.0
                
                masked_logits = logits + (1 - mask) * -1e9
                server_idx = torch.argmax(masked_logits).item()
                action = server_to_action[server_ids[server_idx]]
            
            state_dict, (rL, rC, rS), done, info = env.step(action)
            ep_lat += info['latency_ms']
            ep_cost += info['cost']
        
        latencies.append(ep_lat)
        compute_costs.append(ep_cost)
        if (i + 1) % 50 == 0:
            print(f'  STAR_PPO (Retrained): {i+1}/{episodes}')
    
    return {'latencies': np.array(latencies), 'costs': np.array(compute_costs)}


def run_star_ppo_partition(env, ds, model_path, device, episodes):
    from STAR_PPO.model import StarActor
    
    actor = StarActor(state_dim=10, num_servers=NUM_SERVERS_MODEL).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    actor.eval()
    
    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    server_ids_full = sorted(list(env.servers.keys()))
    num_servers_full = len(server_ids_full)

    server_coords = [(env.servers[sid].lon, env.servers[sid].lat) for sid in server_ids_full]
 
    lons = [c[0] for c in server_coords]
    lats = [c[1] for c in server_coords]
    lon_mid = (min(lons) + max(lons)) / 2
    lat_mid = (min(lats) + max(lats)) / 2

    partitions = [[] for _ in range(NUM_PARTITIONS)]
    for idx, (lon, lat) in enumerate(server_coords):
        if lon <= lon_mid and lat <= lat_mid:
            partitions[0].append(idx)
        elif lon > lon_mid and lat <= lat_mid:
            partitions[1].append(idx)
        elif lon <= lon_mid and lat > lat_mid:
            partitions[2].append(idx)
        else:
            partitions[3].append(idx)
  
    for i in range(NUM_PARTITIONS):
        if len(partitions[i]) > NUM_SERVERS_MODEL:
            partitions[i] = partitions[i][:NUM_SERVERS_MODEL]
        elif len(partitions[i]) < NUM_SERVERS_MODEL:
            all_used = set()
            for p in partitions:
                all_used.update(p)
            remaining = [idx for idx in range(num_servers_full) if idx not in all_used]
            need = NUM_SERVERS_MODEL - len(partitions[i])
            partitions[i].extend(remaining[:need])
    
    print(f"  Partition sizes: {[len(p) for p in partitions]}")
    
    latencies, compute_costs = [], []
    
    for ep in range(episodes):
        task = ds.tasks[ep % len(ds.tasks)]
        state_dict = env.reset(task)
        task_lon, task_lat = task['TaskLongitude'], task['TaskLatitude']
        ep_lat, ep_cost, done = 0, 0, False
        
        while not done:
            valid_actions = env.available_actions()
            if not valid_actions:
                break

            valid_server_set = set()
            server_to_action = {}
            for aidx in valid_actions:
                mi = env.actions[aidx]
                valid_server_set.add(mi.server_id)
                if mi.server_id not in server_to_action:
                    server_to_action[mi.server_id] = aidx
     
            candidates = []  
            
            for partition_indices in partitions:
                partition_server_ids = [server_ids_full[idx] for idx in partition_indices]
  
                valid_in_partition = [sid for sid in partition_server_ids if sid in valid_server_set]
                if not valid_in_partition:
                    continue

                dist_array = np.array([
                    haversine_km(task_lon, task_lat, env.servers[sid].lon, env.servers[sid].lat)
                    for sid in partition_server_ids
                ], dtype=np.float32)
                max_dist = np.max(dist_array) + 1e-6
                dist_norm = dist_array / max_dist
                
                avg_dist_norm = np.mean(dist_norm)
                base_state = np.array([state_dict['step_norm'], state_dict['task_lon'], state_dict['task_lat'],
                                       float(state_dict['prev_region_id']), w[0], w[1], w[2]], dtype=np.float32)
                state_vec = np.concatenate([base_state, np.array([avg_dist_norm, 0.5, 0.5], dtype=np.float32)])
                
                caps = np.array([env.servers[sid].normalized_compute for sid in partition_server_ids], dtype=np.float32)
                cost_mults = np.array([env.servers[sid].cost_multiplier for sid in partition_server_ids], dtype=np.float32)
                cost_advantage = 1.0 - np.clip(cost_mults / 2.0, 0, 1.0)
                
                distance_quality = 1.0 - dist_norm
                
                network_quality = np.ones(NUM_SERVERS_MODEL, dtype=np.float32)
                if hasattr(env, 'link_latency') and len(env.link_latency) > 0:
                    for idx, sid in enumerate(partition_server_ids):
                        outbound_lats = [lat for (src, dst), lat in env.link_latency.items() if src == sid]
                        if outbound_lats:
                            network_quality[idx] = np.exp(-np.mean(outbound_lats) / 500.0)
                
                combined_quality = network_quality * (0.3 + 0.7 * distance_quality)
                
            current_time = env.current_time_ms
                busy_times = np.array([max(0.0, env.busy_until[sid] - current_time) for sid in partition_server_ids], dtype=np.float32)
                norm_queues = np.clip(busy_times / 5000.0, 0.0, 1.0)
                weights = caps / (1.0 + 0.30 * norm_queues) * combined_quality * (0.5 + 0.5 * cost_advantage)
                weights = weights / (np.max(weights) + 1e-9)
       
                valid_mask = np.zeros(NUM_SERVERS_MODEL, dtype=np.float32)
                partition_to_action = {}
                for idx, sid in enumerate(partition_server_ids):
                    if sid in valid_server_set:
                        valid_mask[idx] = 1.0
                        partition_to_action[idx] = server_to_action[sid]
                
                if np.sum(valid_mask) == 0:
                    continue
                
                with torch.no_grad():
                    state_t = torch.from_numpy(state_vec).unsqueeze(0).to(device)
                    weights_t = torch.from_numpy(weights).unsqueeze(0).to(device)
                    logits = actor(state_t, weights_t).squeeze(0).cpu().numpy()
       
                    masked_logits = logits + (1 - valid_mask) * -1e9
                    best_idx = np.argmax(masked_logits)
                    best_sid = partition_server_ids[best_idx]
       
                    score = weights[best_idx]
                    
                    candidates.append((best_sid, score, partition_to_action[best_idx]))
            
            if not candidates:
                action = np.random.choice(valid_actions)
            else:
                best_candidate = max(candidates, key=lambda x: x[1])
                action = best_candidate[2]
            
            state_dict, (rL, rC, rS), done, info = env.step(action)
            ep_lat += info['latency_ms']
            ep_cost += info['cost']
        
        latencies.append(ep_lat)
        compute_costs.append(ep_cost)
        if (ep + 1) % 50 == 0:
            print(f'  STAR_PPO (Partition): {ep+1}/{episodes}')
    
    mean_lat = np.mean(latencies)
    print(f'  Result: Latency = {mean_lat:.1f} ms')
    
    return {'latencies': np.array(latencies), 'costs': np.array(compute_costs)}


def run_star_ppo_zeroshot(env, ds, model_path, device, episodes):
    """STAR-PPO Zero-Shot: 500模型 + Nearest Subnet -> 2000环境 (对照组)"""
    from STAR_PPO.model import StarActor
    
    actor = StarActor(state_dim=10, num_servers=NUM_SERVERS_MODEL).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    actor.eval()
    
    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    server_ids_full = sorted(list(env.servers.keys()))
    num_servers_full = len(server_ids_full)
    
    latencies, compute_costs = [], []
    
    for i in range(episodes):
        np.random.seed(SEED + i)  
        task = ds.tasks[i % len(ds.tasks)]
        state_dict = env.reset(task)
        task_lon, task_lat = task['TaskLongitude'], task['TaskLatitude']
        ep_lat, ep_cost, done = 0, 0, False
        
        while not done:
            valid_actions = env.available_actions()
            if not valid_actions:
                break

            distances = []
            for idx, sid in enumerate(server_ids_full):
                srv = env.servers[sid]
                d = haversine_km(task_lon, task_lat, srv.lon, srv.lat)
                distances.append((idx, d, sid))
            distances.sort(key=lambda x: x[1])

            sampled_indices = [d[0] for d in distances[:NUM_SERVERS_MODEL]]
            sampled_server_ids = [d[2] for d in distances[:NUM_SERVERS_MODEL]]
            sampled_distances = [d[1] for d in distances[:NUM_SERVERS_MODEL]]

            valid_in_sample = set()
            sample_to_action = {}
            for aidx in valid_actions:
                mi = env.actions[aidx]
                if mi.server_id in sampled_server_ids:
                    sample_idx = sampled_server_ids.index(mi.server_id)
                    valid_in_sample.add(sample_idx)
                    if sample_idx not in sample_to_action:
                        sample_to_action[sample_idx] = aidx
            
            if len(valid_in_sample) == 0:
                action = np.random.choice(valid_actions)
            else:
                dist_array = np.array(sampled_distances, dtype=np.float32)
                max_dist = np.max(dist_array) + 1e-6
                dist_norm = dist_array / max_dist  
       
                avg_dist_norm = np.mean(dist_norm)
                base_state = np.array([state_dict['step_norm'], state_dict['task_lon'], state_dict['task_lat'],
                                       float(state_dict['prev_region_id']), w[0], w[1], w[2]], dtype=np.float32)
                state_vec = np.concatenate([base_state, np.array([avg_dist_norm, 0.5, 0.5], dtype=np.float32)])
                
                caps = np.array([env.servers[sid].normalized_compute for sid in sampled_server_ids], dtype=np.float32)
                cost_mults = np.array([env.servers[sid].cost_multiplier for sid in sampled_server_ids], dtype=np.float32)
                cost_advantage = 1.0 - np.clip(cost_mults / 2.0, 0, 1.0)
    
                distance_quality = 1.0 - dist_norm  
                
                network_quality = np.ones(NUM_SERVERS_MODEL, dtype=np.float32)
                if hasattr(env, 'link_latency') and len(env.link_latency) > 0:
                    for idx, sid in enumerate(sampled_server_ids):
                        outbound_lats = [lat for (src, dst), lat in env.link_latency.items() if src == sid]
                        if outbound_lats:
                            network_quality[idx] = np.exp(-np.mean(outbound_lats) / 500.0)
 
                combined_quality = network_quality * (0.3 + 0.7 * distance_quality)
                
                current_time = env.current_time_ms
                busy_times = np.array([max(0.0, env.busy_until[sid] - current_time) for sid in sampled_server_ids], dtype=np.float32)
                norm_queues = np.clip(busy_times / 5000.0, 0.0, 1.0)
                weights = caps / (1.0 + 0.30 * norm_queues) * combined_quality * (0.5 + 0.5 * cost_advantage)
                weights = weights / (np.max(weights) + 1e-9)
            
            with torch.no_grad():
                    state_t = torch.from_numpy(state_vec).unsqueeze(0).to(device)
                    weights_t = torch.from_numpy(weights).unsqueeze(0).to(device)
                    logits = actor(state_t, weights_t).squeeze(0)
                
                    mask = torch.ones(NUM_SERVERS_MODEL, device=device) * float('-inf')
                    for idx in valid_in_sample:
                        mask[idx] = 0.0
                
                masked_logits = logits + mask
                    sample_idx = torch.argmax(masked_logits).item()
                    action = sample_to_action[sample_idx]
            
            state_dict, (rL, rC, rS), done, info = env.step(action)
            ep_lat += info['latency_ms']
            ep_cost += info['cost']
        
        latencies.append(ep_lat)
        compute_costs.append(ep_cost)
        if (i + 1) % 50 == 0:
            print(f'  STAR_PPO (Zero-Shot): {i+1}/{episodes}')
    
    mean_lat = np.mean(latencies)
    print(f'  Result: Latency = {mean_lat:.1f} ms')
    
    return {'latencies': np.array(latencies), 'costs': np.array(compute_costs)}


def run_baseline_zeroshot(env, ds, model_path, algo_name, device, episodes):
    
    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    server_ids_full = sorted(list(env.servers.keys()))
    num_servers_full = len(server_ids_full)

    if algo_name == 'A3C':
        from A3C_algorithm.model import ActorCritic
        model = ActorCritic(state_dim=7, num_servers=NUM_SERVERS_MODEL).to(device)
        state_dict_loaded = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict_loaded)
        model.eval()
    elif algo_name == 'PPO':
        from PPO_algorithm.model import ActorCritic
        model = ActorCritic(state_dim=7, num_servers=NUM_SERVERS_MODEL).to(device)
        state_dict_loaded = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict_loaded)
        model.eval()
    elif algo_name == 'Trans':
        from Trans.model import TransformerActorCritic
        model = TransformerActorCritic(
            state_dim=7, action_dim=NUM_SERVERS_MODEL, d_model=64
        ).to(device)
        state_dict_loaded = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict_loaded)
        model.eval()
    elif algo_name == 'Stark':
        from Stark_Scheduler.model import StarkScheduler
        model = StarkScheduler(
            task_dim=4, server_dim=7, num_servers=NUM_SERVERS_MODEL,
            d_model=128, nhead=4, num_encoder_layers=2, num_decoder_layers=2
        ).to(device)
        state_dict_loaded = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict_loaded)
        model.eval()
    elif algo_name == 'PPO_GNN':
        from PPO_GNN.model import GNNActorCritic
        model = GNNActorCritic(node_feat_dim=3, global_feat_dim=7, hidden_dim=128, gnn_layers=2).to(device)
        state_dict_loaded = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict_loaded)
        model.eval()
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    latencies, compute_costs = [], []
    
    for i in range(episodes):
        np.random.seed(SEED + i)  
        task = ds.tasks[i % len(ds.tasks)]
        state_dict = env.reset(task)
        task_lon, task_lat = task['TaskLongitude'], task['TaskLatitude']
        ep_lat, ep_cost, done = 0, 0, False
        
        while not done:
            valid_actions = env.available_actions()
            if not valid_actions:
                break
 
            distances = []
            for idx, sid in enumerate(server_ids_full):
                srv = env.servers[sid]
                d = haversine_km(task_lon, task_lat, srv.lon, srv.lat)
                distances.append((idx, d, sid))
            distances.sort(key=lambda x: x[1])
 
            sampled_indices = [d[0] for d in distances[:NUM_SERVERS_MODEL]]
            sampled_server_ids = [d[2] for d in distances[:NUM_SERVERS_MODEL]]
 
            valid_in_sample = set()
            sample_to_action = {}
            for aidx in valid_actions:
                mi = env.actions[aidx]
                if mi.server_id in sampled_server_ids:
                    sample_idx = sampled_server_ids.index(mi.server_id)
                    valid_in_sample.add(sample_idx)
                    if sample_idx not in sample_to_action:
                        sample_to_action[sample_idx] = aidx
            
            if len(valid_in_sample) == 0:
                action = np.random.choice(valid_actions)
            else:
                caps = np.array([env.servers[sid].normalized_compute for sid in sampled_server_ids], dtype=np.float32)
                current_time = env.current_time_ms
                busy_times = np.array([max(0.0, env.busy_until[sid] - current_time) for sid in sampled_server_ids], dtype=np.float32)
                norm_queues = np.clip(busy_times / 5000.0, 0.0, 1.0)
                cost_mults = np.array([env.servers[sid].cost_multiplier for sid in sampled_server_ids], dtype=np.float32)
                cost_adv = 1.0 - np.clip(cost_mults / 2.0, 0, 1.0)
                
                weights = caps / (1.0 + 0.30 * norm_queues)
                weights = weights / (np.max(weights) + 1e-9)
                
                    base_state = np.array([
                        state_dict['step_norm'],
                        state_dict['task_lon'],
                        state_dict['task_lat'],
                        float(state_dict['prev_region_id']),
                        w[0], w[1], w[2]
                    ], dtype=np.float32)
                    
                try:
                    if algo_name in ['A3C', 'PPO']:
                    with torch.no_grad():
                        state_t = torch.from_numpy(base_state).unsqueeze(0).to(device)
                        weights_t = torch.from_numpy(weights).unsqueeze(0).to(device)
                        logits, _ = model(state_t, weights_t)
                        
                            mask = torch.ones(NUM_SERVERS_MODEL, device=device) * float('-inf')
                        for idx in valid_in_sample:
                            mask[idx] = 0.0
                        
                        masked_logits = logits.squeeze(0) + mask
                        sample_idx = torch.argmax(masked_logits).item()
                        action = sample_to_action[sample_idx]
                        
                elif algo_name == 'Trans':
                    with torch.no_grad():
                        state_t = torch.from_numpy(base_state).unsqueeze(0).to(device)
                        weights_t = torch.from_numpy(weights).unsqueeze(0).to(device)
                        logits, _ = model(state_t, weights_t)
                        
                            mask = torch.ones(NUM_SERVERS_MODEL, device=device) * float('-inf')
                        for idx in valid_in_sample:
                            mask[idx] = 0.0
                        
                        masked_logits = logits.squeeze(0) + mask
                        sample_idx = torch.argmax(masked_logits).item()
                        action = sample_to_action[sample_idx]
                        
                elif algo_name == 'Stark':
                    server_feats = np.stack([caps, norm_queues, cost_adv], axis=1)
                    
                    with torch.no_grad():
                            task_t = torch.from_numpy(base_state).unsqueeze(0).to(device)
                        server_t = torch.from_numpy(server_feats).unsqueeze(0).to(device)
                        logits = model(task_t, server_t)
                        
                            mask = torch.ones(NUM_SERVERS_MODEL, device=device) * float('-inf')
                        for idx in valid_in_sample:
                            mask[idx] = 0.0
                        
                        masked_logits = logits.squeeze(0) + mask
                        sample_idx = torch.argmax(masked_logits).item()
                        action = sample_to_action[sample_idx]
                            
                    elif algo_name == 'PPO_GNN':
                        from torch_geometric.data import Data
                        
                        node_feats = np.stack([caps, norm_queues, cost_adv], axis=1)
   
                        edge_index = []
                        edge_attr = []
                        for idx in range(min(NUM_SERVERS_MODEL, 50)):
                            for jdx in range(min(NUM_SERVERS_MODEL, 50)):
                                if idx != jdx:
                                    edge_index.append([idx, jdx])
                                    edge_attr.append([0.5])
                        
                        if len(edge_index) == 0:
                            edge_index = [[0, 1], [1, 0]]
                            edge_attr = [[0.5], [0.5]]
                        
                        graph_data = Data(
                            x=torch.FloatTensor(node_feats).to(device),
                            edge_index=torch.LongTensor(edge_index).t().contiguous().to(device),
                            edge_attr=torch.FloatTensor(edge_attr).to(device),
                            global_feat=torch.FloatTensor(base_state).unsqueeze(0).to(device)
                        )
                        graph_data.num_graphs = 1
                        
                        with torch.no_grad():
                            logits, _ = model(graph_data)
                            
                            mask = torch.ones(NUM_SERVERS_MODEL, device=device) * float('-inf')
                            for idx in valid_in_sample:
                                mask[idx] = 0.0
                            
                            masked_logits = logits + mask
                            sample_idx = torch.argmax(masked_logits).item()
                            action = sample_to_action[sample_idx]
                            
                except Exception as e:
                    action = np.random.choice(valid_actions)
            
            state_dict, (rL, rC, rS), done, info = env.step(action)
            ep_lat += info['latency_ms']
            ep_cost += info['cost']
        
        latencies.append(ep_lat)
        compute_costs.append(ep_cost)
        if (i + 1) % 50 == 0:
            print(f'  {algo_name} (Zero-Shot): {i+1}/{episodes}')
    
    mean_lat = np.mean(latencies)
    print(f'  Result: Latency = {mean_lat:.1f} ms')
    
    return {'latencies': np.array(latencies), 'costs': np.array(compute_costs)}


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    np.random.seed(SEED)
    torch.manual_seed(SEED)
 
    print(f"\nLoading target environment ({REGION_TARGET}, {NUM_SERVERS_ENV} nodes)...")
    ds = WorkflowDataset(DATA_ROOT, split='train', regions=[REGION_TARGET])
    env = WorkflowMoEEnv(ds)
    print(f"Loaded {len(ds.tasks)} tasks, {len(env.servers)} servers")
    
    results = {}

    print("\n" + "="*60)
    print(f"Running STAR-PPO (Retrained): {REGION_TARGET} model -> {REGION_TARGET} env")
    print("="*60)
    results['STAR_PPO_Retrained'] = run_star_ppo_retrained(
        env, ds, MODELS_TARGET['STAR_PPO'], device, EPISODES
    )
    print(f"  Result: Latency = {np.mean(results['STAR_PPO_Retrained']['latencies']):.1f} ms")

    print("\n" + "="*60)
    print(f"Running STAR-PPO (Partition): 500 model + Partition Inference -> {REGION_TARGET} env")
    print("="*60)
    results['STAR_PPO_Partition'] = run_star_ppo_partition(
        env, ds, MODELS_500['STAR_PPO'], device, EPISODES
    )
    print(f"  Result: Latency = {np.mean(results['STAR_PPO_Partition']['latencies']):.1f} ms")

    print("\n" + "="*60)
    print(f"Running STAR-PPO (Zero-Shot): 500 model + Subnet Sampling -> {REGION_TARGET} env")
    print("="*60)
    results['STAR_PPO_ZeroShot'] = run_star_ppo_zeroshot(
        env, ds, MODELS_500['STAR_PPO'], device, EPISODES
        )
    print(f"  Result: Latency = {np.mean(results['STAR_PPO_ZeroShot']['latencies']):.1f} ms")

    for algo in ['PPO_GNN', 'A3C', 'PPO', 'Trans', 'Stark']:
        print("\n" + "="*60)
        print(f"Running {algo} (Zero-Shot): Server1_Trap model -> {REGION_TARGET} env")
        print("="*60)
        try:
            results[f'{algo}_ZeroShot'] = run_baseline_zeroshot(
                env, ds, MODELS_500[algo], algo, device, EPISODES
            )
            print(f"  Result: Latency = {np.mean(results[f'{algo}_ZeroShot']['latencies']):.1f} ms")
        except Exception as e:
            print(f"  {algo} failed: {e}")
            import traceback
            traceback.print_exc()
            results[f'{algo}_ZeroShot'] = {'latencies': np.array([9999.0] * EPISODES), 'costs': np.array([1.0] * EPISODES)}

    print("\n" + "="*60)
    print("Saving results...")
    print("="*60)
    
    for name, data in results.items():
        npz_path = os.path.join(OUTPUT_DIR, f'{name}.npz')
        np.savez(npz_path, **data)
        print(f"  Saved: {npz_path}")

    print("\n" + "="*60)
    print("Summary (Average Latency)")
    print("="*60)
    for name, data in results.items():
        lat = np.mean(data['latencies'])
        print(f"  {name}: {lat:.1f} ms")

    retrained_lat = np.mean(results['STAR_PPO_Retrained']['latencies'])
    print("\n" + "="*60)
    print(f"Normalized Gap (vs STAR-PPO Retrained = {retrained_lat:.1f} ms)")
    print("="*60)
    for name, data in results.items():
        lat = np.mean(data['latencies'])
        gap = lat / retrained_lat
        print(f"  {name}: {gap:.2f}x")


if __name__ == '__main__':
    main()
