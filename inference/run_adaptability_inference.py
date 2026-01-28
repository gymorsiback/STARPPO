
import os
import sys
import numpy as np
import torch
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env import WorkflowDataset, WorkflowMoEEnv

DATA_ROOT = '/root/autodl-tmp/MOE111/data1'
REGION = 'Server1_Trap'
NUM_SERVERS = 500
NORMAL_EPISODES = 100
FAILURE_EPISODES = 100
TOTAL_EPISODES = NORMAL_EPISODES + FAILURE_EPISODES
SEED = 42
FAILURE_RATIO = 0.3
FAILURE_MULTIPLIER = 8

MODEL_PATHS = {
    'STAR_PPO': '/root/autodl-tmp/MOE111/results/STAR_PPO/models/star_ppo_20251229_213311_4c5d57_actor_epoch_95.pt',
    'PFAPPO': '/root/autodl-tmp/MOE111/results/PFAPPO/models/pfappo_20251229_213321_3f2d75_actor_epoch_0099.pt',
    'PPO': '/root/autodl-tmp/MOE111/results/PPO/models/ppo_20251229_221507_4318c2_actor_epoch_0099.pt',
    'PPO_CN': '/root/autodl-tmp/MOE111/results/PPO_CN/models/ppo_cn_20251229_223055_39fdf2_actor_epoch_0099.pt',
    'PPO_GNN': '/root/autodl-tmp/MOE111/results/PPO_GNN/models/ppo_gnn_20251229_222534_768885_model_epoch_0099.pt',
    'Trans': '/root/autodl-tmp/MOE111/results/Trans/models/trans_ppo_20251229_222524_01b5cd_model_epoch_0099.pt',
    'A3C': '/root/autodl-tmp/MOE111/results/A3C_algorithm/models/a3c_20251229_221515_c881ff_actor_final.pt',
    'Stark': '/root/autodl-tmp/MOE111/results/Stark_Scheduler/models/run_20251229_223440_fcbf08_20251229_223440_final.pt',
}


def trigger_failure(env, ratio=0.3, multiplier=8):
    all_servers = list(env.servers.keys())
    num_failed = int(len(all_servers) * ratio)
    failed_servers = set(random.sample(all_servers, num_failed))
    
    for link_key, latency in env.link_latency.items():
        src, dst = link_key
        if src in failed_servers or dst in failed_servers:
            env.link_latency[link_key] = latency * multiplier
    
    print(f"  ⚡ 故障: {num_failed}台服务器延迟×{multiplier}")
    return failed_servers


def run_star_ppo(env, ds, model_path, device):
    from STAR_PPO.model import StarActor
    
    actor = StarActor(state_dim=10, num_servers=NUM_SERVERS).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    actor.eval()
    
    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    server_ids = sorted(list(env.servers.keys()))
    caps = np.array([env.servers[sid].normalized_compute for sid in server_ids], dtype=np.float32)
    cost_mults = np.array([env.servers[sid].cost_multiplier for sid in server_ids], dtype=np.float32)
    cost_advantage = 1.0 - np.clip(cost_mults / 2.0, 0, 1.0)
    
    episode_latencies = []
    failed_servers = None
    
    for ep in range(TOTAL_EPISODES):
        if ep == NORMAL_EPISODES:
            failed_servers = trigger_failure(env, FAILURE_RATIO, FAILURE_MULTIPLIER)

        network_quality = np.ones(NUM_SERVERS, dtype=np.float32)
        for i, sid in enumerate(server_ids):
            outbound_lats = [lat for (src, dst), lat in env.link_latency.items() if src == sid]
            if outbound_lats:
                avg_lat = np.mean(outbound_lats)
                network_quality[i] = np.exp(-avg_lat / 500.0)
        
        task = ds.tasks[ep % len(ds.tasks)]
        state_dict = env.reset(task)
        
        ep_latency = 0
        done = False
        
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
            
            state_vec = np.array([
                state_dict['step_norm'],
                state_dict['task_lon'],
                state_dict['task_lat'],
                float(state_dict['prev_region_id']),
                w[0], w[1], w[2],
                np.mean(network_quality), 
                np.min(network_quality),
                np.std(network_quality)
            ], dtype=np.float32)
            
            current_time = env.current_time_ms
            busy_times = np.array([max(0.0, env.busy_until[sid] - current_time) for sid in server_ids], dtype=np.float32)
            norm_queues = np.clip(busy_times / 5000.0, 0.0, 1.0)

            weights = (0.3 * caps + 0.3 * cost_advantage + 0.4 * network_quality) / (1.0 + 0.3 * norm_queues)
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
            
            state_dict, _, done, info = env.step(action)
            ep_latency += info['latency_ms']
        
        episode_latencies.append(ep_latency)
        if ep % 50 == 0:
            print(f"    Ep {ep}: {ep_latency:.0f}ms")
    
    return np.array(episode_latencies)


def run_pfappo(env, ds, model_path, device):
    from PFAPPO.model import Actor
    
    actor = Actor(state_dim=7, num_servers=NUM_SERVERS).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
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
    cost_advantage = 1.0 - np.clip((server_min_costs - 0.0006) / (0.132 - 0.0006), 0, 1.0)
    
    episode_latencies = []
    
    for ep in range(TOTAL_EPISODES):
        if ep == NORMAL_EPISODES:
            trigger_failure(env, FAILURE_RATIO, FAILURE_MULTIPLIER)
        
        task = ds.tasks[ep % len(ds.tasks)]
        state_dict = env.reset(task)
        
        ep_latency = 0
        done = False
        
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
            
            state_dict, _, done, info = env.step(action)
            ep_latency += info['latency_ms']
        
        episode_latencies.append(ep_latency)
    
    return np.array(episode_latencies)


def run_ppo(env, ds, model_path, device):
    from PPO_algorithm.model import Actor
    
    actor = Actor(state_dim=7, num_servers=NUM_SERVERS).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    actor.eval()
    
    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    server_ids = sorted(list(env.servers.keys()))
    
    episode_latencies = []
    
    for ep in range(TOTAL_EPISODES):
        if ep == NORMAL_EPISODES:
            trigger_failure(env, FAILURE_RATIO, FAILURE_MULTIPLIER)
        
        task = ds.tasks[ep % len(ds.tasks)]
        state_dict = env.reset(task)
        
        ep_latency = 0
        done = False
        
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
            
            state_dict, _, done, info = env.step(action)
            ep_latency += info['latency_ms']
        
        episode_latencies.append(ep_latency)
    
    return np.array(episode_latencies)


def run_ppo_cn(env, ds, model_path, device):
    """PPO-CN"""
    from PPO_CN.model import Actor
    
    actor = Actor(state_dim=7, action_dim=NUM_SERVERS).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    actor.eval()
    
    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    server_ids = sorted(list(env.servers.keys()))
    
    episode_latencies = []
    
    for ep in range(TOTAL_EPISODES):
        if ep == NORMAL_EPISODES:
            trigger_failure(env, FAILURE_RATIO, FAILURE_MULTIPLIER)
        
        task = ds.tasks[ep % len(ds.tasks)]
        state_dict = env.reset(task)
        
        ep_latency = 0
        done = False
        
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
            
            state_dict, _, done, info = env.step(action)
            ep_latency += info['latency_ms']
        
        episode_latencies.append(ep_latency)
    
    return np.array(episode_latencies)


def run_a3c(env, ds, model_path, device):
    """A3C"""
    from A3C_algorithm.model import ActorCritic
    
    model = ActorCritic(state_dim=7, num_servers=NUM_SERVERS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    server_ids = sorted(list(env.servers.keys()))
    
    episode_latencies = []
    
    for ep in range(TOTAL_EPISODES):
        if ep == NORMAL_EPISODES:
            trigger_failure(env, FAILURE_RATIO, FAILURE_MULTIPLIER)
        
        task = ds.tasks[ep % len(ds.tasks)]
        state_dict = env.reset(task)
        
        ep_latency = 0
        done = False
        
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
            
            state_dict, _, done, info = env.step(action)
            ep_latency += info['latency_ms']
        
        episode_latencies.append(ep_latency)
    
    return np.array(episode_latencies)


def run_ppo_gnn(env, ds, model_path, device):
    """PPO-GNN"""
    from PPO_GNN.model import GNNActorCritic
    
    model = GNNActorCritic(node_feat_dim=3, global_feat_dim=7, hidden_dim=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    episode_latencies = []
    
    for ep in range(TOTAL_EPISODES):
        if ep == NORMAL_EPISODES:
            trigger_failure(env, FAILURE_RATIO, FAILURE_MULTIPLIER)
        
        task = ds.tasks[ep % len(ds.tasks)]
        env.reset(task)
        
        ep_latency = 0
        done = False
        
        while not done:
            graph_data = env.get_graph_state()
            
            with torch.no_grad():
                logits, _ = model(graph_data)
                action = torch.argmax(logits, dim=1).item()
            
            _, _, done, info = env.step(action)
            ep_latency += info['latency_ms']
        
        episode_latencies.append(ep_latency)
    
    return np.array(episode_latencies)


def run_trans(env, ds, model_path, device):
    """Transformer"""
    from Trans.model import TransformerActorCritic
    
    model = TransformerActorCritic(state_dim=7, action_dim=NUM_SERVERS, d_model=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    server_ids = sorted(list(env.servers.keys()))
    
    episode_latencies = []
    
    for ep in range(TOTAL_EPISODES):
        if ep == NORMAL_EPISODES:
            trigger_failure(env, FAILURE_RATIO, FAILURE_MULTIPLIER)
        
        task = ds.tasks[ep % len(ds.tasks)]
        state_dict = env.reset(task)
        
        ep_latency = 0
        done = False
        
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
            
            state_dict, _, done, info = env.step(action)
            ep_latency += info['latency_ms']
        
        episode_latencies.append(ep_latency)
    
    return np.array(episode_latencies)


def run_stark(env, ds, model_path, device):
    """Stark Scheduler"""
    from Stark_Scheduler.model import StarkScheduler
    
    model = StarkScheduler(
        task_dim=4, server_dim=7, num_servers=NUM_SERVERS,
        d_model=128, nhead=4, num_encoder_layers=2, num_decoder_layers=2
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    server_ids = sorted(list(env.servers.keys()))
    
    episode_latencies = []
    
    for ep in range(TOTAL_EPISODES):
        if ep == NORMAL_EPISODES:
            trigger_failure(env, FAILURE_RATIO, FAILURE_MULTIPLIER)
        
        task = ds.tasks[ep % len(ds.tasks)]
        state_dict = env.reset(task)
        
        ep_latency = 0
        done = False
        
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
            
            task_feat = np.array([
                state_dict['step_norm'],
                state_dict['task_lon'],
                state_dict['task_lat'],
                float(state_dict['prev_region_id'])
            ], dtype=np.float32)
            
            current_time = env.current_time_ms
            server_feats = []
            for sid in server_ids:
                server = env.servers[sid]
                busy_time = max(0.0, env.busy_until[sid] - current_time)
                server_feats.append([
                    server.normalized_compute,
                    server.queue_tokens / 10000.0,
                    server.cost_per_token,
                    server.cost_multiplier,
                    busy_time / 5000.0,
                    server.lat,
                    server.lon
                ])
            server_feats = np.array(server_feats, dtype=np.float32)
            
            with torch.no_grad():
                task_t = torch.from_numpy(task_feat).unsqueeze(0).to(device)
                server_t = torch.from_numpy(server_feats).unsqueeze(0).to(device)
                logits = model(task_t, server_t).squeeze(0)
                
                mask = torch.zeros(NUM_SERVERS, device=device)
                for idx, sid in enumerate(server_ids):
                    if sid in valid_server_ids:
                        mask[idx] = 1.0
                
                masked_logits = logits + (1 - mask) * -1e9
                server_idx = torch.argmax(masked_logits).item()
                selected_sid = server_ids[server_idx]
                action = server_to_action[selected_sid]
            
            state_dict, _, done, info = env.step(action)
            ep_latency += info['latency_ms']
        
        episode_latencies.append(ep_latency)
    
    return np.array(episode_latencies)


def run_greedy(env, ds):
    """Greedy"""
    episode_latencies = []
    
    for ep in range(TOTAL_EPISODES):
        if ep == NORMAL_EPISODES:
            trigger_failure(env, FAILURE_RATIO, FAILURE_MULTIPLIER)
        
        task = ds.tasks[ep % len(ds.tasks)]
        env.reset(task)
        
        ep_latency = 0
        done = False
        
        while not done:
            candidates = env.available_actions()
            if not candidates:
                candidates = list(range(len(env.actions)))
            
            best_action, best_score = candidates[0], -float('inf')
            for idx in candidates:
                mi = env.actions[idx]
                score = env.servers[mi.server_id].normalized_compute
                if score > best_score:
                    best_score = score
                    best_action = idx
            
            _, _, done, info = env.step(best_action)
            ep_latency += info['latency_ms']
        
        episode_latencies.append(ep_latency)
    
    return np.array(episode_latencies)


def run_random(env, ds):
    """Random"""
    episode_latencies = []
    
    for ep in range(TOTAL_EPISODES):
        if ep == NORMAL_EPISODES:
            trigger_failure(env, FAILURE_RATIO, FAILURE_MULTIPLIER)
        
        task = ds.tasks[ep % len(ds.tasks)]
        env.reset(task)
        
        ep_latency = 0
        done = False
        
        while not done:
            candidates = env.available_actions()
            if not candidates:
                candidates = list(range(len(env.actions)))
            action = random.choice(candidates)
            
            _, _, done, info = env.step(action)
            ep_latency += info['latency_ms']
        
        episode_latencies.append(ep_latency)
    
    return np.array(episode_latencies)


def main():
    print("=" * 60)
    print("适应性测试 - 服务器故障恢复能力")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    output_dir = 'total/adaptability_results'
    os.makedirs(output_dir, exist_ok=True)
    
    algorithms = [
        ('STAR_PPO', lambda e, d: run_star_ppo(e, d, MODEL_PATHS['STAR_PPO'], device)),
        ('PFAPPO', lambda e, d: run_pfappo(e, d, MODEL_PATHS['PFAPPO'], device)),
        ('PPO', lambda e, d: run_ppo(e, d, MODEL_PATHS['PPO'], device)),
        ('PPO_CN', lambda e, d: run_ppo_cn(e, d, MODEL_PATHS['PPO_CN'], device)),
        ('A3C', lambda e, d: run_a3c(e, d, MODEL_PATHS['A3C'], device)),
        ('PPO_GNN', lambda e, d: run_ppo_gnn(e, d, MODEL_PATHS['PPO_GNN'], device)),
        ('Trans', lambda e, d: run_trans(e, d, MODEL_PATHS['Trans'], device)),
        ('Stark', lambda e, d: run_stark(e, d, MODEL_PATHS['Stark'], device)),
        ('Greedy', lambda e, d: run_greedy(e, d)),
        ('Random', lambda e, d: run_random(e, d)),
    ]
    
    for algo_name, run_func in algorithms:
        print(f"\n{'='*40}")
        print(f"测试 {algo_name}...")
        
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        
        ds = WorkflowDataset(DATA_ROOT, regions=[REGION])
        env = WorkflowMoEEnv(ds)
        
        try:
            latencies = run_func(env, ds)
            
            normal_avg = np.mean(latencies[:NORMAL_EPISODES])
            failure_avg = np.mean(latencies[NORMAL_EPISODES:])
            increase = (failure_avg - normal_avg) / normal_avg * 100
            
            print(f"  正常期: {normal_avg:.0f}ms, 故障期: {failure_avg:.0f}ms, 增幅: {increase:+.1f}%")
            
            np.savez(f"{output_dir}/{algo_name}_adaptability.npz",
                     episode_latencies=latencies,
                     normal_episodes=NORMAL_EPISODES,
                     total_episodes=TOTAL_EPISODES)
            print(f"  ✓ 已保存")
            
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("完成！")


if __name__ == '__main__':
    main()
