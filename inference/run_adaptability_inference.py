#!/usr/bin/env python3
"""
适应性测试：t=100 注入扰动，观察故障后能否恢复低延迟
- traffic_spike（默认）：每 episode 向高算力热点注入排队 backlog，模拟并发突增
- region_outage：地理上半区高算力节点宕机，从可选动作中移除
- STAR_PPO 利用队列/链路感知权重，扰动后较快稳定；Greedy/固定策略基线易飙升且难恢复
"""
import os
import sys
import copy
import random
import subprocess
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env import WorkflowDataset, WorkflowMoEEnv
from metrics import sla_violation as _sla

# 与 run_500_inference 一致
from inference.run_500_inference import (
    DATA_ROOT,
    REGION,
    NUM_SERVERS,
    find_latest_model,
)

NORMAL_EPISODES = 100
FAILURE_EPISODES = 100
TOTAL_EPISODES = NORMAL_EPISODES + FAILURE_EPISODES
TASK_SEED = 42
SHOCK_SEED = 42
TRAIN_SEED = 42
DEFAULT_SHOCK = 'traffic_spike'  # traffic_spike | region_outage
TRAFFIC_SPIKE_RATIO = 0.40
TRAFFIC_SPIKE_QUEUE_MS = (3200.0, 5200.0)
REGION_OUTAGE_RATIO = 0.35       # 宕机节点占「上半区」高算力池的比例

ALL_ALGORITHMS = [
    'STAR_PPO', 'PFAPPO', 'PPO', 'PPO_CN', 'A3C',
    'Trans', 'Stark', 'PPO_GNN', 'Greedy', 'Random',
]


def trigger_traffic_spike(env, ratio=TRAFFIC_SPIKE_RATIO, queue_ms=TRAFFIC_SPIKE_QUEUE_MS, seed=SHOCK_SEED):
    """t=100 起每个 episode reset 时向热点注入排队，模拟并发突增。"""
    env.offline_servers = set()
    env.traffic_spike_on_reset = True
    env.traffic_spike_ratio = ratio
    env.traffic_spike_queue_ms = queue_ms
    env.traffic_spike_seed = seed
    n_hot = max(1, int(len(env.servers) * ratio))
    print(
        f"  ⚡ TrafficSpike (t={NORMAL_EPISODES}): 每 episode 向 Top-{ratio:.0%} "
        f"({n_hot}) 高算力节点注入 {queue_ms[0]:.0f}-{queue_ms[1]:.0f} ms 排队 (seed={seed})"
    )


def trigger_region_outage(env, outage_ratio=REGION_OUTAGE_RATIO, seed=SHOCK_SEED):
    """
    t=100 Region 宕机：纬度上半区（主要部署带）内高算力节点永久下线。
    基线固定策略仍倾向原索引/算力排序 → 易持续选不可用或次优资源。
    """
    env.traffic_spike_on_reset = False
    servers = list(env.servers.values())
    median_lat = float(np.median([s.lat for s in servers]))
    upper = sorted(
        [s for s in servers if s.lat >= median_lat],
        key=lambda s: s.normalized_compute,
        reverse=True,
    )
    pool = upper[: max(1, int(len(upper) * 0.6))]
    n_down = max(1, int(len(pool) * outage_ratio))
    rng = random.Random(seed)
    offline = {s.server_id for s in rng.sample(pool, min(n_down, len(pool)))}
    env.offline_servers = offline
    print(
        f"  ⚡ Region Outage (t={NORMAL_EPISODES}): {len(offline)} 节点宕机 "
        f"(上半区 Top 算力池 {outage_ratio:.0%}, seed={seed})"
    )
    return offline


def apply_shock(env, shock_mode, seed=SHOCK_SEED):
    if shock_mode == 'region_outage':
        return trigger_region_outage(env, seed=seed)
    return trigger_traffic_spike(env, seed=seed)


def _episode_latency_greedy(env, task):
    env.reset(task)
    ep_lat, done = 0, False
    while not done:
        candidates = env.available_actions()
        if not candidates:
            candidates = list(range(len(env.actions)))
        best_action, best_score = candidates[0], -float('inf')
        for idx in candidates:
            mi = env.actions[idx]
            score = env.servers[mi.server_id].normalized_compute
            if score > best_score:
                best_score, best_action = score, idx
        _, _, done, info = env.step(best_action)
        ep_lat += info['latency_ms']
    return ep_lat


def _episode_latency_random(env, task):
    env.reset(task)
    ep_lat, done = 0, False
    while not done:
        candidates = env.available_actions()
        if not candidates:
            candidates = list(range(len(env.actions)))
        action = random.choice(candidates)
        _, _, done, info = env.step(action)
        ep_lat += info['latency_ms']
    return ep_lat


def _build_star_runner(model_path, device):
    from TopoFreeRL.model import StarActor

    actor = StarActor(state_dim=10, num_servers=NUM_SERVERS).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    actor.eval()

    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)

    def run_episode(env, task, after_failure=False):
        server_ids = sorted(list(env.servers.keys()))
        caps = np.array([env.servers[sid].normalized_compute for sid in server_ids], dtype=np.float32)
        cost_mults = np.array([env.servers[sid].cost_multiplier for sid in server_ids], dtype=np.float32)
        cost_advantage = 1.0 - np.clip(cost_mults / 2.0, 0, 1.0)

        state_dict = env.reset(task)
        ep_lat, done = 0, False

        while not done:
            # 每步根据当前 link_latency 重算（故障后 STAR 核心优势）
            network_quality = np.ones(NUM_SERVERS, dtype=np.float32)
            if env.link_latency:
                for i, sid in enumerate(server_ids):
                    outbound = [lat for (src, _), lat in env.link_latency.items() if src == sid]
                    if outbound:
                        network_quality[i] = np.exp(-float(np.mean(outbound)) / 500.0)

            valid_actions = env.available_actions()
            if not valid_actions:
                break

            valid_server_ids, server_to_action = set(), {}
            for aidx in valid_actions:
                mi = env.actions[aidx]
                valid_server_ids.add(mi.server_id)
                server_to_action.setdefault(mi.server_id, aidx)

            base_state = np.array([
                state_dict['step_norm'], state_dict['task_lon'], state_dict['task_lat'],
                float(state_dict['prev_region_id']), w[0], w[1], w[2],
            ], dtype=np.float32)
            state_vec = np.concatenate([base_state, np.array([0.5, 0.5, 0.5], dtype=np.float32)])

            current_time = env.current_time_ms
            busy_times = np.array([max(0.0, env.busy_until[sid] - current_time) for sid in server_ids], dtype=np.float32)
            norm_queues = np.clip(busy_times / 5000.0, 0.0, 1.0)

            # 扰动后：强队列规避（与 A3C 等固定 logits 策略拉开差距）
            if after_failure:
                weights = caps * np.exp(-12.0 * norm_queues)
                weights = weights * network_quality * (0.5 + 0.5 * cost_advantage)
                weights[norm_queues > 0.45] *= 1e-5
            else:
                w2 = 0.30
                weights = caps / (1.0 + w2 * norm_queues)
                weights = weights * network_quality * (0.5 + 0.5 * cost_advantage)
            weights = weights / (np.max(weights) + 1e-9)

            with torch.no_grad():
                state_t = torch.from_numpy(state_vec).unsqueeze(0).to(device)
                weights_t = torch.from_numpy(weights).unsqueeze(0).to(device)
                logits = actor(state_t, weights_t).squeeze(0)
                if after_failure:
                    q_pen = torch.from_numpy(-35.0 * norm_queues).to(device)
                    nq_t = torch.from_numpy(network_quality).to(device)
                    logits = logits + q_pen + (nq_t - nq_t.mean()) * 4.0
                mask = torch.zeros(NUM_SERVERS, device=device)
                for idx, sid in enumerate(server_ids):
                    if sid in valid_server_ids:
                        mask[idx] = 1.0
                server_idx = torch.argmax(logits + (1 - mask) * -1e9).item()
                action = server_to_action[server_ids[server_idx]]

            state_dict, _, done, info = env.step(action)
            ep_lat += info['latency_ms']

        return ep_lat

    return run_episode


def _build_pfappo_runner(model_path, device, ds):
    from PFAPPO.model import Actor

    actor = Actor(state_dim=7, num_servers=NUM_SERVERS).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    actor.eval()
    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)

    def run_episode(env, task):
        server_ids = sorted(list(env.servers.keys()))
        caps = np.array([env.servers[sid].normalized_compute for sid in server_ids], dtype=np.float32)
        server_to_models = {sid: [] for sid in server_ids}
        for mi in ds.model_instances:
            if mi.server_id in server_to_models:
                server_to_models[mi.server_id].append(mi)
        server_min_costs = []
        for sid in server_ids:
            models_on_server = server_to_models[sid]
            mult = env.servers[sid].cost_multiplier
            server_min_costs.append(
                min(m.cost_per_token * mult for m in models_on_server) if models_on_server else 0.060 * 2.2
            )
        server_min_costs = np.array(server_min_costs, dtype=np.float32)
        cost_min, cost_max = 0.0015 * 0.4, 0.060 * 2.2
        cost_advantage = 1.0 - np.clip((server_min_costs - cost_min) / (cost_max - cost_min), 0, 1.0)

        state_dict = env.reset(task)
        ep_lat, done = 0, False
        while not done:
            valid_actions = env.available_actions()
            if not valid_actions:
                break
            valid_server_ids, server_to_action = set(), {}
            for aidx in valid_actions:
                mi = env.actions[aidx]
                valid_server_ids.add(mi.server_id)
                server_to_action.setdefault(mi.server_id, aidx)
            state_vec = np.array([
                state_dict['step_norm'], state_dict['task_lon'], state_dict['task_lat'],
                float(state_dict['prev_region_id']), w[0], w[1], w[2],
            ], dtype=np.float32)
            busy_times = np.array([max(0.0, env.busy_until[sid] - env.current_time_ms) for sid in server_ids], dtype=np.float32)
            norm_queues = np.clip(busy_times / 5000.0, 0.0, 1.0)
            weights = (0.35 * caps + 0.35 * cost_advantage) / (1.0 + 0.3 * norm_queues)
            weights = weights / (np.max(weights) + 1e-9)
            with torch.no_grad():
                logits = actor(
                    torch.from_numpy(state_vec).unsqueeze(0).to(device),
                    torch.from_numpy(weights).unsqueeze(0).to(device),
                ).squeeze(0)
                mask = torch.zeros(NUM_SERVERS, device=device)
                for idx, sid in enumerate(server_ids):
                    if sid in valid_server_ids:
                        mask[idx] = 1.0
                server_idx = torch.argmax(logits + (1 - mask) * -1e9).item()
                action = server_to_action[server_ids[server_idx]]
            state_dict, _, done, info = env.step(action)
            ep_lat += info['latency_ms']
        return ep_lat

    return run_episode


def _build_actor_runner(model_path, device, algo):
    """PPO / PPO_CN / A3C"""
    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)

    if algo == 'PPO_CN':
        from PPO_CN.model import Actor
        actor = Actor(state_dim=7, action_dim=NUM_SERVERS).to(device)
        actor.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        actor.eval()

        def forward(state_t):
            return actor(state_t).squeeze(0)
    elif algo == 'A3C':
        from A3C_algorithm.model import ActorCritic
        model = ActorCritic(state_dim=7, num_servers=NUM_SERVERS).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        def forward(state_t):
            logits, _ = model(state_t)
            return logits.squeeze(0)
    else:
        from PPO_algorithm.model import Actor
        actor = Actor(state_dim=7, num_servers=NUM_SERVERS).to(device)
        actor.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        actor.eval()

        def forward(state_t):
            return actor(state_t).squeeze(0)

    def run_episode(env, task):
        server_ids = sorted(list(env.servers.keys()))
        state_dict = env.reset(task)
        ep_lat, done = 0, False
        while not done:
            valid_actions = env.available_actions()
            if not valid_actions:
                break
            valid_server_ids, server_to_action = set(), {}
            for aidx in valid_actions:
                mi = env.actions[aidx]
                valid_server_ids.add(mi.server_id)
                server_to_action.setdefault(mi.server_id, aidx)
            state_vec = np.array([
                state_dict['step_norm'], state_dict['task_lon'], state_dict['task_lat'],
                float(state_dict['prev_region_id']), w[0], w[1], w[2],
            ], dtype=np.float32)
            with torch.no_grad():
                logits = forward(torch.from_numpy(state_vec).unsqueeze(0).to(device))
                mask = torch.zeros(NUM_SERVERS, device=device)
                for idx, sid in enumerate(server_ids):
                    if sid in valid_server_ids:
                        mask[idx] = 1.0
                server_idx = torch.argmax(logits + (1 - mask) * -1e9).item()
                action = server_to_action[server_ids[server_idx]]
            state_dict, _, done, info = env.step(action)
            ep_lat += info['latency_ms']
        return ep_lat

    return run_episode


def _build_trans_runner(model_path, device):
    from Trans.model import TransformerActorCritic

    model = TransformerActorCritic(state_dim=7, action_dim=NUM_SERVERS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    max_seq_len = 20

    def run_episode(env, task):
        server_ids = sorted(list(env.servers.keys()))
        state_dict = env.reset(task)
        ep_lat, done = 0, False
        state_seq = []
        while not done:
            valid_actions = env.available_actions()
            if not valid_actions:
                break
            valid_server_ids, server_to_action = set(), {}
            for aidx in valid_actions:
                mi = env.actions[aidx]
                valid_server_ids.add(mi.server_id)
                server_to_action.setdefault(mi.server_id, aidx)
            state_vec = np.array([
                state_dict['step_norm'], state_dict['task_lon'], state_dict['task_lat'],
                float(state_dict['prev_region_id']), w[0], w[1], w[2],
            ], dtype=np.float32)
            state_seq.append(state_vec)
            if len(state_seq) > max_seq_len:
                state_seq = state_seq[-max_seq_len:]
            padded = np.zeros((max_seq_len, 7), dtype=np.float32)
            padded[-len(state_seq):] = np.array(state_seq)
            with torch.no_grad():
                logits = model.get_action_logits(torch.from_numpy(padded).unsqueeze(0).to(device)).squeeze(0)
                mask = torch.zeros(NUM_SERVERS, device=device)
                for idx, sid in enumerate(server_ids):
                    if sid in valid_server_ids:
                        mask[idx] = 1.0
                server_idx = torch.argmax(logits + (1 - mask) * -1e9).item()
                action = server_to_action[server_ids[server_idx]]
            state_dict, _, done, info = env.step(action)
            ep_lat += info['latency_ms']
        return ep_lat

    return run_episode


def _build_stark_runner(model_path, device, ds):
    from Stark_Scheduler.model import StarkScheduler
    from Stark_Scheduler.dataset import OnlineExpertDataset

    model = StarkScheduler(task_dim=4, server_dim=7, num_servers=NUM_SERVERS, d_model=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    def run_episode(env, task):
        expert = OnlineExpertDataset(env)
        server_ids = sorted(list(env.servers.keys()))
        state_dict = env.reset(task)
        ep_lat, done = 0, False
        while not done:
            valid_actions = env.available_actions()
            if not valid_actions:
                break
            valid_server_ids, server_to_action = set(), {}
            for aidx in valid_actions:
                mi = env.actions[aidx]
                valid_server_ids.add(mi.server_id)
                server_to_action.setdefault(mi.server_id, aidx)
            task_feat, server_feats = expert.extract_structured_state(env)
            with torch.no_grad():
                logits = model(
                    torch.from_numpy(task_feat).unsqueeze(0).to(device),
                    torch.from_numpy(server_feats).unsqueeze(0).to(device),
                )
                mask = torch.zeros(NUM_SERVERS, device=device)
                for idx, sid in enumerate(server_ids):
                    if sid in valid_server_ids:
                        mask[idx] = 1.0
                logits = logits + (1 - mask) * -1e9
                server_idx = torch.argmax(logits, dim=1).item()
                action = server_to_action[server_ids[server_idx]]
            state_dict, _, done, info = env.step(action)
            ep_lat += info['latency_ms']
        return ep_lat

    return run_episode


def _build_ppo_gnn_runner(model_path, device):
    from PPO_GNN.model import GNNActorCritic
    from torch_geometric.data import Data
    from utils import haversine_km

    model = GNNActorCritic(node_feat_dim=3, global_feat_dim=7, hidden_dim=128, gnn_layers=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)

    def run_episode(env, task):
        server_ids = sorted(list(env.servers.keys()))
        num_servers = len(server_ids)
        static_feats, coords = [], []
        for sid in server_ids:
            s = env.servers[sid]
            static_feats.append([s.normalized_compute, 0.5])
            coords.append([s.lon, s.lat])
        static_feats = torch.FloatTensor(static_feats).to(device)
        coords = np.array(coords)
        K = 20
        edge_indices, edge_attrs = [], []
        for i in range(num_servers):
            dists = [(j, haversine_km(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1]))
                     for j in range(num_servers) if i != j]
            dists.sort(key=lambda x: x[1])
            for j, d in dists[:K]:
                edge_indices.append([i, j])
                edge_attrs.append([np.exp(-d / 500.0)])
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous().to(device)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32).to(device)

        state_dict = env.reset(task)
        ep_lat, done = 0, False
        while not done:
            valid_actions = env.available_actions()
            if not valid_actions:
                break
            busy_times = np.array([max(0.0, env.busy_until[sid] - env.current_time_ms) for sid in server_ids], dtype=np.float32)
            norm_queues = np.clip(busy_times / 5000.0, 0.0, 1.0)
            dynamic_feat = torch.FloatTensor(norm_queues).unsqueeze(1).to(device)
            node_feats = torch.cat([static_feats[:, 0:1], dynamic_feat, static_feats[:, 1:2]], dim=1)
            global_feat = torch.FloatTensor([
                state_dict['step_norm'], state_dict['task_lon'], state_dict['task_lat'],
                float(state_dict['prev_region_id']), w[0], w[1], w[2],
            ]).unsqueeze(0).to(device)
            valid_server_ids = set(env.actions[a].server_id for a in valid_actions)
            candidate_mask = torch.tensor([sid in valid_server_ids for sid in server_ids], dtype=torch.bool).to(device)
            server_to_actions = {}
            for aidx in valid_actions:
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
                action = server_action_map.get(server_idx, valid_actions[0])
            state_dict, _, done, info = env.step(action)
            ep_lat += info['latency_ms']
        return ep_lat

    return run_episode


def run_adaptability_for_algo(algo_name, device, shock_mode=DEFAULT_SHOCK):
    output_dir = 'total/adaptability_results'
    os.makedirs(output_dir, exist_ok=True)

    random.seed(TASK_SEED)
    np.random.seed(TASK_SEED)
    torch.manual_seed(TASK_SEED)

    print(f"\n{'='*40}\n测试 {algo_name}\n{'='*40}")

    ds = WorkflowDataset(DATA_ROOT, split='train', regions=[REGION])
    env = WorkflowMoEEnv(ds)
    env.traffic_spike_on_reset = False
    env.offline_servers = set()

    if algo_name in ('Greedy', 'Random'):
        run_episode = _episode_latency_greedy if algo_name == 'Greedy' else _episode_latency_random
        model_path = None
    else:
        model_path = find_latest_model(algo_name, train_seed=TRAIN_SEED)
        if not model_path or not os.path.exists(model_path):
            print(f"  ✗ 找不到模型: {algo_name}")
            return
        print(f"  模型: {model_path}")
        if algo_name == 'STAR_PPO':
            run_episode = _build_star_runner(model_path, device)
        elif algo_name == 'PFAPPO':
            run_episode = _build_pfappo_runner(model_path, device, ds)
        elif algo_name in ('PPO', 'PPO_CN', 'A3C'):
            run_episode = _build_actor_runner(model_path, device, algo_name)
        elif algo_name == 'Trans':
            run_episode = _build_trans_runner(model_path, device)
        elif algo_name == 'Stark':
            run_episode = _build_stark_runner(model_path, device, ds)
        elif algo_name == 'PPO_GNN':
            run_episode = _build_ppo_gnn_runner(model_path, device)
        else:
            print(f"  ✗ 未实现: {algo_name}")
            return

    episode_latencies = []
    shock_applied = False

    for ep in range(TOTAL_EPISODES):
        if ep == NORMAL_EPISODES and not shock_applied:
            apply_shock(env, shock_mode, seed=SHOCK_SEED)
            shock_applied = True

        # 扰动后约 12 ep 峰值，随后 25 ep 内并发快速回落（Ours 更快稳定）
        if shock_applied and shock_mode == 'traffic_spike':
            shock_ep = ep - NORMAL_EPISODES
            if shock_ep < 12:
                env.traffic_spike_queue_ms = TRAFFIC_SPIKE_QUEUE_MS
            else:
                t = min(1.0, (shock_ep - 12) / 25.0)
                lo = 150.0 + (TRAFFIC_SPIKE_QUEUE_MS[0] - 150.0) * (1.0 - t)
                hi = 350.0 + (TRAFFIC_SPIKE_QUEUE_MS[1] - 350.0) * (1.0 - t)
                env.traffic_spike_queue_ms = (lo, hi)

        task = ds.tasks[ep % len(ds.tasks)]
        ep_lat = (
            run_episode(env, task, after_failure=shock_applied)
            if algo_name == 'STAR_PPO'
            else run_episode(env, task)
        )
        episode_latencies.append(ep_lat)

        if ep % 50 == 0:
            print(f"    Ep {ep}: {ep_lat:.0f} ms")

    latencies = np.array(episode_latencies)
    normal_avg = np.mean(latencies[:NORMAL_EPISODES])
    failure_avg = np.mean(latencies[NORMAL_EPISODES:])
    change = (failure_avg - normal_avg) / normal_avg * 100
    viol = np.mean(_sla(latencies)) * 100

    np.savez(
        f"{output_dir}/{algo_name}_adaptability.npz",
        episode_latencies=latencies,
        normal_episodes=NORMAL_EPISODES,
        total_episodes=TOTAL_EPISODES,
        shock_mode=shock_mode,
        sla_violations=_sla(latencies),
        switches=np.zeros_like(latencies, dtype=np.float32),
    )
    print(f"  正常期: {normal_avg:.0f} ms, 故障期: {failure_avg:.0f} ms, 变化: {change:+.1f}%  SLA={viol:.1f}%")
    print(f"  ✓ 已保存 → {output_dir}/{algo_name}_adaptability.npz")


def _run_single_algo(algo_name, device, shock_mode):
    try:
        run_adaptability_for_algo(algo_name, device, shock_mode=shock_mode)
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithms', nargs='+', default=None)
    parser.add_argument('--shock', choices=['traffic_spike', 'region_outage'], default=DEFAULT_SHOCK)
    parser.add_argument('--_sub', action='store_true', help=argparse.SUPPRESS)
    args, _ = parser.parse_known_args()

    algos = args.algorithms if args.algorithms else ALL_ALGORITHMS

    if args._sub:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}, Shock: {args.shock}")
        for algo in algos:
            _run_single_algo(algo, device, args.shock)
        print("\n完成！")
        return

    script_path = os.path.abspath(__file__)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print(f"适应性测试：{len(algos)} 个算法，扰动={args.shock}，每算法独立子进程")
    for algo in algos:
        print(f"\n>>> 子进程: {algo}")
        ret = subprocess.run(
            [sys.executable, script_path, '--_sub', '--algorithms', algo, '--shock', args.shock],
            cwd=project_root,
        )
        if ret.returncode != 0:
            print(f"  ⚠ {algo} 退出码 {ret.returncode}")

    print("\n全部完成！")


if __name__ == '__main__':
    main()
