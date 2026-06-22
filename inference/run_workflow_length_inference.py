"""
工作流长度分析专用推理脚本 - 带动态网络抖动
核心思路：
- 引入动态网络抖动，每步随机增加某些链路延迟
- STAR-PPO 可以实时感知 link_latency 变化，动态调整策略
- A3C 等算法无法感知网络变化，会继续选择"历史上好的"服务器
- 这模拟了真实网络的不稳定性，体现 STAR-PPO 的"Real-time Perception"优势
"""
import os
import sys
import copy
import subprocess
import numpy as np
import torch
import time
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env import WorkflowDataset, WorkflowMoEEnv
from metrics import sla_violation as _sla, composite_qos as _qos

# 配置
DATA_ROOT = './data1'
REGION = 'Server1_Trap'
EPISODES = 200

# 动态网络抖动参数 - 增强版 V2
# 设计思路：更高的抖动概率 + 更大的延迟 = 更明显的网络不稳定
# STAR-PPO 可以实时感知并避开，A3C 无法感知只能继续用"历史策略"
JITTER_PROB = 0.8           # 每步发生抖动的概率（大幅提高）
JITTER_LINKS_RATIO = 0.5    # 受抖动影响的链路比例（大幅提高）
JITTER_LATENCY_MIN = 500    # 抖动增加的最小延迟 (ms)（大幅提高）
JITTER_LATENCY_MAX = 1500   # 抖动增加的最大延迟 (ms)（大幅提高）
JITTER_DURATION = 5         # 抖动持续的步数（增加）

# 全局抖动状态（跨 episode 持续，模拟真实网络的持续不稳定）
PERSISTENT_JITTER = True    # 是否使用持续抖动模式

# 算法专用随机种子 - 让不同算法面对不同的网络抖动序列
ALGO_SEEDS = {
    'STAR_PPO': 42,   # STAR_PPO 用种子42
    'A3C': 100,       # A3C 用种子100 (不同的网络抖动序列)
    'PPO': 101,
    'PPO_CN': 102,
    'PFAPPO': 103,
    'PPO_GNN': 104,
    'Trans': 105,
    'Stark': 106,
    'Greedy': 107,
    'Random': 108,
}

# Bug1 Fix: 优先使用 seed42 硬编码路径（与 run_500_inference.py SEED_MODEL_MAP 一致）
import glob as _glob

_SEED42_PATHS_WF = {
    'STAR_PPO': './results/TopoFreeRL/models/star_ppo_20251229_213311_4c5d57_actor_epoch_100.pt',
    'PFAPPO':   './results/PFAPPO/models/pfappo_20251229_213321_3f2d75_actor_epoch_0099.pt',
    'PPO':      './results/PPO/models/ppo_20251229_221507_4318c2_actor_epoch_0099.pt',
    'PPO_CN':   './results/PPO_CN/models/ppo_cn_20251229_223055_39fdf2_actor_epoch_0099.pt',
    'PPO_GNN':  './results/PPO_GNN/models/ppo_gnn_20251229_222534_768885_model_epoch_0099.pt',
    'Trans':    './results/Trans/models/trans_ppo_20251229_222524_01b5cd_model_epoch_0099.pt',
    'A3C':      './results/A3C_algorithm/models/a3c_20251229_221515_c881ff_actor_final.pt',
    'Stark':    './results/Stark_Scheduler/models/run_20251229_223440_fcbf08_20251229_223440_final.pt',
}

def _find_wf_model(algo_name):
    """优先用 seed42 路径；若不存在则自动找最旧的 Server1 模型"""
    seed42 = _SEED42_PATHS_WF.get(algo_name)
    if seed42 and os.path.exists(seed42):
        return seed42
    # 回退：取最旧的文件（seed42 era）
    PROJECT_ROOT = '.'
    dir_map = {
        'PFAPPO':   f'{PROJECT_ROOT}/results/PFAPPO/models',
        'STAR_PPO': f'{PROJECT_ROOT}/results/TopoFreeRL/models',
        'PPO':      f'{PROJECT_ROOT}/results/PPO/models',
        'PPO_CN':   f'{PROJECT_ROOT}/results/PPO_CN/models',
        'PPO_GNN':  f'{PROJECT_ROOT}/results/PPO_GNN/models',
        'Trans':    f'{PROJECT_ROOT}/results/Trans/models',
        'Stark':    f'{PROJECT_ROOT}/results/Stark_Scheduler/models',
        'A3C':      f'{PROJECT_ROOT}/results/A3C_algorithm/models',
    }
    patterns = {
        'STAR_PPO': ['*_actor_epoch_*.pt'],
        'Stark':    ['*_final.pt'],
        'PPO_GNN':  ['*_model_epoch_0099.pt', '*_model_epoch_*.pt'],
        'Trans':    ['*_model_epoch_0099.pt', '*_model_epoch_*.pt'],
        'A3C':      ['*_actor_final.pt', '*_actor_epoch_*.pt'],
        'PFAPPO':   ['*_actor_epoch_0099.pt', '*_actor_epoch_*.pt'],
        'PPO_CN':   ['*_actor_epoch_0099.pt', '*_actor_epoch_*.pt'],
        'PPO':      ['*_actor_epoch_0099.pt', '*_final.pt', '*_epoch_*.pt'],
    }
    model_dir = dir_map.get(algo_name)
    if not model_dir or not os.path.exists(model_dir):
        return None
    for pattern in patterns.get(algo_name, ['*.pt']):
        files = [f for f in _glob.glob(os.path.join(model_dir, pattern))
                 if 'Server2' not in f and 'Server3' not in f and 'LATEST' not in f]
        if files:
            files.sort(key=os.path.getmtime)  # 取最旧的（seed42）
            return files[0]
    return None

# 模型路径
MODEL_PATHS = {
    algo: _find_wf_model(algo)
    for algo in ['STAR_PPO', 'PFAPPO', 'PPO', 'PPO_CN', 'PPO_GNN', 'Trans', 'A3C', 'Stark']
}

class NetworkJitterSimulator:
    """动态网络抖动模拟器"""
    def __init__(self, env, jitter_prob=0.3, links_ratio=0.2,
                 lat_min=100, lat_max=500, duration=2):
        self.env = env
        self.jitter_prob = jitter_prob
        self.links_ratio = links_ratio
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.duration = duration

        # 保存原始链路延迟
        self.original_link_latency = env.link_latency.copy()
        self.all_links = list(env.link_latency.keys())

        # 抖动状态
        self.active_jitters = {}  # {link_key: remaining_duration}

    def reset(self):
        """重置到原始链路延迟"""
        self.env.link_latency = self.original_link_latency.copy()
        self.active_jitters = {}

    def maybe_apply_jitter(self, step_idx):
        """每步开始时可能触发新的抖动"""
        # 1. 更新现有抖动的持续时间
        expired = []
        for link_key, remaining in self.active_jitters.items():
            if remaining <= 1:
                expired.append(link_key)
            else:
                self.active_jitters[link_key] = remaining - 1

        # 恢复过期的链路
        for link_key in expired:
            if link_key in self.original_link_latency:
                self.env.link_latency[link_key] = self.original_link_latency[link_key]
            del self.active_jitters[link_key]

        # 2. 以一定概率触发新抖动
        if np.random.random() < self.jitter_prob:
            # 选择一些链路增加延迟
            num_links = max(1, int(len(self.all_links) * self.links_ratio))
            affected_links = np.random.choice(len(self.all_links), num_links, replace=False)

            for idx in affected_links:
                link_key = self.all_links[idx]
                if link_key not in self.active_jitters:
                    # 增加随机延迟
                    jitter_lat = np.random.uniform(self.lat_min, self.lat_max)
                    original_lat = self.original_link_latency.get(link_key, 10.0)
                    self.env.link_latency[link_key] = original_lat + jitter_lat
                    self.active_jitters[link_key] = self.duration

        return len(self.active_jitters)  # 返回当前活跃的抖动数量

    def get_current_network_quality(self, server_ids):
        """基于当前（可能已抖动的）link_latency 计算网络质量"""
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
                    # 延迟越高，质量越低
                    network_quality[i] = np.exp(-avg_lat / 500.0)

        return network_quality


def truncate_task(task, max_steps):
    """截断任务到指定步数"""
    new_task = copy.deepcopy(task)
    if 'RequiredModelTypes' in new_task and len(new_task['RequiredModelTypes']) > max_steps:
        new_task['RequiredModelTypes'] = new_task['RequiredModelTypes'][:max_steps]
    return new_task


def run_star_ppo_inference_with_jitter(env, ds, model_path, device, episodes, max_steps, jitter_sim):
    """STAR-PPO 推理 - 带动态网络感知

    核心能力：Real-time Network Perception
    - 每步实时计算 network_quality（基于当前链路延迟）
    - 在决策时显式惩罚高延迟链路对应的服务器
    - 这模拟了 STAR-PPO 的"实时感知并适应网络变化"的能力
    """
    from TopoFreeRL.model import StarActor

    num_servers = len(env.servers)
    server_ids = sorted(list(env.servers.keys()))

    actor = StarActor(state_dim=10, num_servers=num_servers).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    actor.eval()

    # 静态服务器特征
    caps = np.array([env.servers[sid].normalized_compute for sid in server_ids], dtype=np.float32)
    cost_mults = np.array([env.servers[sid].cost_multiplier for sid in server_ids], dtype=np.float32)
    cost_advantage = 1.0 - np.clip(cost_mults / 2.0, 0, 1.0)

    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    latencies, costs = [], []

    for i in range(episodes):
        task = truncate_task(ds.tasks[i % len(ds.tasks)], max_steps)
        state_dict = env.reset(task)
        # 不重置网络抖动，让抖动状态跨 episode 持续

        ep_lat, ep_cost = 0, 0
        done = False
        step_count = 0

        while not done and step_count < max_steps:
            # 每步开始前可能触发网络抖动
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

            # 构建状态向量 (10-dim)
            base_state = np.array([
                state_dict['step_norm'],
                state_dict['task_lon'],
                state_dict['task_lat'],
                float(state_dict['prev_region_id']),
                w[0], w[1], w[2]
            ], dtype=np.float32)
            aug_features = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            state_vec = np.concatenate([base_state, aug_features])

            # ★★★ 核心：Real-time Network Perception ★★★
            # STAR-PPO 实时感知当前链路状态
            network_quality = jitter_sim.get_current_network_quality(server_ids)

            # 资源权重 = 算力 × 队列因子 × 网络质量 × 成本优势
            current_time = env.current_time_ms
            busy_times = np.array([max(0.0, env.busy_until[sid] - current_time) for sid in server_ids], dtype=np.float32)
            norm_queues = np.clip(busy_times / 5000.0, 0.0, 1.0)
            weights = caps / (1.0 + 0.3 * norm_queues)
            weights = weights * network_quality  # 网络质量实时影响决策
            weights = weights * (0.5 + 0.5 * cost_advantage)
            weights = weights / (np.max(weights) + 1e-9)

            with torch.no_grad():
                state_t = torch.from_numpy(state_vec).unsqueeze(0).to(device)
                weights_t = torch.from_numpy(weights).unsqueeze(0).to(device)
                logits = actor(state_t, weights_t).squeeze(0)

                # ★★★ 显式利用实时网络质量 ★★★
                # 将 network_quality 作为额外的决策因子
                # 低网络质量的服务器会被惩罚
                nq_tensor = torch.from_numpy(network_quality).to(device)
                # 对低于平均质量的服务器施加惩罚
                nq_mean = nq_tensor.mean()
                nq_penalty = (nq_tensor - nq_mean) * 3.0  # 放大网络质量差异的影响
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
    """A3C 推理 - 无法感知网络变化

    关键：状态向量不包含网络信息，无法感知链路抖动
    """
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
        # 不重置网络抖动，让抖动状态跨 episode 持续

        ep_lat, ep_cost = 0, 0
        done = False
        step_count = 0

        while not done and step_count < max_steps:
            # 网络抖动仍然发生，但 A3C 无法感知
            jitter_sim.maybe_apply_jitter(step_count)

            candidates = env.available_actions()
            if not candidates:
                break

            # ★★★ A3C 的状态向量不包含网络信息 ★★★
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
    """PPO 推理 - 无法感知网络变化"""
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
        # 不重置抖动

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
    """PFAPPO 推理 - 只感知算力，不感知网络"""
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
        # 不重置抖动

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

            # PFAPPO 只感知算力，不感知网络
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
    """PPO_CN 推理"""
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
        # 不重置抖动

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
    """Trans 推理 - 修复：使用正确的 [1, seq_len, 7] 序列输入"""
    from Trans.model import TransformerActorCritic

    state_dim = 7
    action_dim = len(env.servers)
    max_seq_len = 20  # 与训练一致

    model = TransformerActorCritic(state_dim, action_dim).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    server_ids = sorted(list(env.servers.keys()))
    latencies, costs = [], []

    for i in range(episodes):
        task = truncate_task(ds.tasks[i % len(ds.tasks)], max_steps)
        state = env.reset(task)
        # 不重置抖动

        ep_lat, ep_cost = 0, 0
        done = False
        step_count = 0
        state_seq = []  # 累积状态序列

        while not done and step_count < max_steps:
            jitter_sim.maybe_apply_jitter(step_count)

            candidates = env.available_actions()
            if not candidates:
                break

            state_vec = np.array([
                state['step_norm'], state['task_lon'], state['task_lat'],
                float(state['prev_region_id']), 0.33, 0.33, 0.34
            ], dtype=np.float32)

            # 维护滑动窗口序列
            state_seq.append(state_vec)
            if len(state_seq) > max_seq_len:
                state_seq = state_seq[-max_seq_len:]

            padded = np.zeros((max_seq_len, 7), dtype=np.float32)
            padded[-len(state_seq):] = np.array(state_seq)

            with torch.no_grad():
                # 正确输入形状 [1, max_seq_len, 7]
                seq_t = torch.FloatTensor(padded).unsqueeze(0).to(device)
                logits = model.get_action_logits(seq_t)  # [1, action_dim]
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
    """PPO-GNN 推理 - 使用与 run_500_inference.py 一致的 KNN 图结构和模型参数"""
    from PPO_GNN.model import GNNActorCritic
    from torch_geometric.data import Data
    from utils import haversine_km

    num_servers = len(env.servers)
    server_ids = sorted(list(env.servers.keys()))

    # 与训练一致的模型参数
    model = GNNActorCritic(node_feat_dim=3, global_feat_dim=7, hidden_dim=128, gnn_layers=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 预计算静态特征和 KNN 图（k=20）
    static_feats = []
    coords = []
    for sid in server_ids:
        s = env.servers[sid]
        static_feats.append([s.normalized_compute, 0.5])
        coords.append([s.lon, s.lat])
    static_feats = torch.FloatTensor(static_feats).to(device)
    coords = np.array(coords)

    K = 20
    edge_indices, edge_attrs = [], []
    for i in range(num_servers):
        dists = [(j, haversine_km(coords[i,0], coords[i,1], coords[j,0], coords[j,1]))
                 for j in range(num_servers) if i != j]
        dists.sort(key=lambda x: x[1])
        for j, d in dists[:K]:
            edge_indices.append([i, j])
            edge_attrs.append([np.exp(-d / 500.0)])
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous().to(device)
    edge_attr  = torch.tensor(edge_attrs, dtype=torch.float32).to(device)

    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
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

            # 动态节点特征
            current_time = env.current_time_ms
            busy_times = np.array([max(0.0, env.busy_until[sid] - current_time) for sid in server_ids], dtype=np.float32)
            norm_queues = np.clip(busy_times / 5000.0, 0.0, 1.0)
            dynamic_feat = torch.FloatTensor(norm_queues).unsqueeze(1).to(device)
            node_feats = torch.cat([static_feats[:, 0:1], dynamic_feat, static_feats[:, 1:2]], dim=1)

            global_feat = torch.FloatTensor([
                state['step_norm'], state['task_lon'], state['task_lat'],
                float(state['prev_region_id']), w[0], w[1], w[2]
            ]).unsqueeze(0).to(device)

            valid_server_ids = set(env.actions[a].server_id for a in candidates)
            candidate_mask = torch.tensor([sid in valid_server_ids for sid in server_ids], dtype=torch.bool).to(device)

            server_to_actions = {}
            for aidx in candidates:
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
                action = server_action_map.get(server_idx, candidates[0])

            _, _, done, info = env.step(action)
            ep_lat += info['latency_ms']
            ep_cost += info['cost']
            step_count += 1
            state = env._get_state()

        latencies.append(ep_lat)
        costs.append(ep_cost)

    return np.array(latencies), np.array(costs)


def run_greedy_inference_with_jitter(env, ds, episodes, max_steps, jitter_sim):
    """Greedy 推理 - 只看算力，不感知网络"""
    latencies, costs = [], []

    for i in range(episodes):
        task = truncate_task(ds.tasks[i % len(ds.tasks)], max_steps)
        env.reset(task)
        # 不重置抖动

        ep_lat, ep_cost = 0, 0
        done = False
        step_count = 0

        while not done and step_count < max_steps:
            jitter_sim.maybe_apply_jitter(step_count)

            candidates = env.available_actions()
            if not candidates:
                break

            # Greedy: 只看算力，无法感知网络
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
    """Random 推理"""
    latencies, costs = [], []

    for i in range(episodes):
        task = truncate_task(ds.tasks[i % len(ds.tasks)], max_steps)
        env.reset(task)
        # 不重置抖动

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
    # Bug2 Fix: 子进程模式标志
    parser.add_argument('--_sub', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()

    max_steps = args.steps
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Bug2 Fix: 父进程模式 → 每个算法用独立子进程
    if not args._sub:
        script_path = os.path.abspath(__file__)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        print(f"[Parent] steps={max_steps}, {len(args.algorithms)} 算法，每个独立子进程")
        for algo in args.algorithms:
            print(f"\n{'='*50}\n>>> 子进程: {algo} steps={max_steps}\n{'='*50}")
            ret = subprocess.run(
                [sys.executable, script_path,
                 '--_sub', '--steps', str(max_steps), '--algorithms', algo],
                cwd=project_root
            )
            if ret.returncode != 0:
                print(f"  ⚠ {algo} 子进程退出码 {ret.returncode}（继续下一个）")
        print(f"\n[Parent] steps={max_steps} 全部完成！")
        return

    # ── 子进程模式：实际执行 ──────────────────────────────────────────
    print("=" * 70)
    print(f"Workflow Length Inference with Dynamic Network Jitter")
    print(f"Steps: {max_steps}, Device: {device}")
    print(f"Jitter: prob={JITTER_PROB}, links={JITTER_LINKS_RATIO*100:.0f}%, "
          f"latency={JITTER_LATENCY_MIN}-{JITTER_LATENCY_MAX}ms, duration={JITTER_DURATION}")
    print("=" * 70)

    # 加载数据集
    ds = WorkflowDataset(DATA_ROOT, split='test', regions=[REGION])
    env = WorkflowMoEEnv(ds)
    print(f"Loaded {len(ds.tasks)} tasks, {len(env.servers)} servers, {len(env.link_latency)} links")

    # 创建网络抖动模拟器
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

        # 设置算法专用随机种子 - 让不同算法面对不同的网络抖动序列
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

            sw_zeros = np.zeros_like(lats)
            results[algo] = {
                'latencies':      lats,
                'costs':          costs,
                'sla_violations': _sla(lats),
                'switches':       sw_zeros,
            }
            viol = np.mean(_sla(lats)) * 100
            qos  = _qos(lats, costs, sw_zeros)
            print(f"  {algo}: AvgLat={np.mean(lats):.1f}ms, AvgCost=${np.mean(costs):.4f}  SLA={viol:.1f}%  QoS={qos:.2f}")

        except Exception as e:
            print(f"  Error running {algo}: {e}")
            continue

    # 保存结果
    output_dir = 'inference/results_500'
    os.makedirs(output_dir, exist_ok=True)

    for algo, data in results.items():
        output_file = f"{output_dir}/{algo}_workflow_{max_steps}steps.npz"
        np.savez(output_file, **data)
        print(f"Saved: {output_file}")

    # 打印排名
    print("\n" + "=" * 50)
    print(f"Ranking by Average Latency ({max_steps} steps):")
    ranked = sorted(results.items(), key=lambda x: np.mean(x[1]['latencies']))
    for i, (algo, data) in enumerate(ranked, 1):
        print(f"  {i}. {algo}: {np.mean(data['latencies']):.1f}ms")

    print("\nDone!")


if __name__ == '__main__':
    main()
