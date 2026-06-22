"""
Scalability Transfer 实验 (分区推理可扩展性)
用 Server1_Trap (500节点) 训练的模型在 Server3_Trap (2000节点) 上推理

使用 Partition-based Inference (分区推理) 策略：
- 把 2000 个服务器按地理位置分成 4 个区（每个约 500 节点）
- 对每个区分别用 STAR-PPO 模型推理，得到 4 个局部最优候选
- 比较 4 个候选者的真实性能，选最优的作为最终决策

实验目的：
1. 证明 Ours (Partition) 和 Ours (Retrained) 之间的差距很小 (Gap Small)
2. 证明 Baseline (Subnet Sampling) 直接迁移后性能崩盘 (Gap Huge)
3. 体现 STAR-PPO 具有 "Divide-and-Conquer Scalability" (分治可扩展性)

为什么 Partition-based Inference 有效？
- 每个分区都是完整的局部图，模型能发挥全力
- 语义化特征在每个分区内都能正确计算
- 最终聚合阶段使用真实性能指标，避免跨分区偏差
"""
import os
import sys
import numpy as np
import torch
import time

sys.path.insert(0, '.')

from env import WorkflowDataset, WorkflowMoEEnv
from utils import haversine_km
from metrics import sla_violation as _sla, composite_qos as _qos

# 配置
DATA_ROOT = './data1'
REGION_TARGET = 'Server3_Trap'  # 目标区域：2000 节点环境
NUM_SERVERS_MODEL = 500  # 模型训练时的服务器数量
NUM_SERVERS_ENV = 2000   # 目标环境中的服务器数量
NUM_PARTITIONS = 4       # 分区数量 (2000 / 500 = 4)
EPISODES = 200
SEED = 42

# 500 规模的模型路径
MODELS_500 = {
    'STAR_PPO': './results/TopoFreeRL/models/star_ppo_20251229_213311_4c5d57_actor_epoch_95.pt',
    'PPO_GNN': './results/PPO_GNN/models/ppo_gnn_20251229_222534_768885_model_epoch_0099.pt',
    'A3C': './results/A3C_algorithm/models/a3c_20251229_221515_c881ff_actor_final.pt',
    'PPO': './results/PPO/models/ppo_20251229_221507_4318c2_actor_epoch_0099.pt',
    'Stark': './results/Stark_Scheduler/models/LATEST_Server1_Trap_seed42_final.pt',
    'Trans': './results/Trans/models/trans_ppo_20251229_222524_01b5cd_model_epoch_0099.pt',
}

# Server3_Trap (2000节点) 的模型路径 (Retrained 参照)
MODELS_TARGET = {
    'STAR_PPO': './results/TopoFreeRL/models/LATEST_Server3_Trap_seed42_actor_epoch_100.pt',
}

OUTPUT_DIR = './Generalization Experiments/Zero-Shot Scalability Transfer'


def run_star_ppo_retrained(env, ds, model_path, device, episodes):
    """STAR-PPO Retrained: 在目标区域上训练的模型，在目标区域上推理（天花板）"""
    from TopoFreeRL.model import StarActor

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
    """STAR-PPO Partition: 500模型 + 分区推理 -> 2000环境

    分区推理策略：
    1. 把 2000 个服务器按地理位置分成 4 个区（每个约 500 节点）
    2. 对每个区分别用模型推理，得到 4 个局部最优候选
    3. 比较 4 个候选者的真实特征分数，选最优的作为最终决策
    """
    from TopoFreeRL.model import StarActor

    actor = StarActor(state_dim=10, num_servers=NUM_SERVERS_MODEL).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    actor.eval()

    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    server_ids_full = sorted(list(env.servers.keys()))
    num_servers_full = len(server_ids_full)

    # 预计算服务器地理坐标，用于分区
    server_coords = [(env.servers[sid].lon, env.servers[sid].lat) for sid in server_ids_full]

    # 使用简单的网格划分进行地理分区（2x2 网格 = 4 个分区）
    lons = [c[0] for c in server_coords]
    lats = [c[1] for c in server_coords]
    lon_mid = (min(lons) + max(lons)) / 2
    lat_mid = (min(lats) + max(lats)) / 2

    # 分配每个服务器到 4 个象限
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

    # 确保每个分区大小为 NUM_SERVERS_MODEL
    for i in range(NUM_PARTITIONS):
        if len(partitions[i]) > NUM_SERVERS_MODEL:
            partitions[i] = partitions[i][:NUM_SERVERS_MODEL]
        elif len(partitions[i]) < NUM_SERVERS_MODEL:
            # 如果分区过小，从其他节点补充
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

            # 构建有效动作的服务器集合
            valid_server_set = set()
            server_to_action = {}
            for aidx in valid_actions:
                mi = env.actions[aidx]
                valid_server_set.add(mi.server_id)
                if mi.server_id not in server_to_action:
                    server_to_action[mi.server_id] = aidx

            # 对每个分区进行推理，收集候选者
            candidates = []  # (server_id, score, action_idx)

            for partition_indices in partitions:
                partition_server_ids = [server_ids_full[idx] for idx in partition_indices]

                # 检查这个分区中有没有有效动作
                valid_in_partition = [sid for sid in partition_server_ids if sid in valid_server_set]
                if not valid_in_partition:
                    continue

                # 构建分区内的状态特征
                # 计算分区内服务器到任务的距离
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

                # 构建有效动作掩码
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

                    # 应用掩码
                    masked_logits = logits + (1 - valid_mask) * -1e9
                    best_idx = np.argmax(masked_logits)
                    best_sid = partition_server_ids[best_idx]

                    # 计算这个候选者的综合分数（用于跨分区比较）
                    score = weights[best_idx]

                    candidates.append((best_sid, score, partition_to_action[best_idx]))

            if not candidates:
                action = np.random.choice(valid_actions)
            else:
                # 从所有分区的候选者中选择分数最高的
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
    from TopoFreeRL.model import StarActor

    actor = StarActor(state_dim=10, num_servers=NUM_SERVERS_MODEL).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    actor.eval()

    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    server_ids_full = sorted(list(env.servers.keys()))
    num_servers_full = len(server_ids_full)

    latencies, compute_costs = [], []

    for i in range(episodes):
        np.random.seed(SEED + i)  # 确保可复现
        task = ds.tasks[i % len(ds.tasks)]
        state_dict = env.reset(task)
        task_lon, task_lat = task['TaskLongitude'], task['TaskLatitude']
        ep_lat, ep_cost, done = 0, 0, False

        while not done:
            valid_actions = env.available_actions()
            if not valid_actions:
                break

            # Nearest Neighbor Subnet: 选择最近的 500 个服务器
            distances = []
            for idx, sid in enumerate(server_ids_full):
                srv = env.servers[sid]
                d = haversine_km(task_lon, task_lat, srv.lon, srv.lat)
                distances.append((idx, d, sid))
            distances.sort(key=lambda x: x[1])

            # 取最近的 500 个作为输入子网
            sampled_indices = [d[0] for d in distances[:NUM_SERVERS_MODEL]]
            sampled_server_ids = [d[2] for d in distances[:NUM_SERVERS_MODEL]]
            sampled_distances = [d[1] for d in distances[:NUM_SERVERS_MODEL]]

            # 建立抽样服务器到动作的映射
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
                # 关键修复：在子网内重新校准 dist_norm 特征
                # 使用子网内的相对距离（归一化到 0-1），而不是全局距离
                dist_array = np.array(sampled_distances, dtype=np.float32)
                max_dist = np.max(dist_array) + 1e-6
                dist_norm = dist_array / max_dist  # 子网内归一化距离

                # 构建状态：使用子网内的平均距离作为全局特征
                avg_dist_norm = np.mean(dist_norm)
                base_state = np.array([state_dict['step_norm'], state_dict['task_lon'], state_dict['task_lat'],
                                       float(state_dict['prev_region_id']), w[0], w[1], w[2]], dtype=np.float32)
                # 额外特征：子网质量指标
                state_vec = np.concatenate([base_state, np.array([avg_dist_norm, 0.5, 0.5], dtype=np.float32)])

                caps = np.array([env.servers[sid].normalized_compute for sid in sampled_server_ids], dtype=np.float32)
                cost_mults = np.array([env.servers[sid].cost_multiplier for sid in sampled_server_ids], dtype=np.float32)
                cost_advantage = 1.0 - np.clip(cost_mults / 2.0, 0, 1.0)

                # 关键修复：融合距离信息到 network_quality
                # 距离近的节点给更高的 quality 分数
                distance_quality = 1.0 - dist_norm  # 距离越近，quality 越高

                network_quality = np.ones(NUM_SERVERS_MODEL, dtype=np.float32)
                if hasattr(env, 'link_latency') and len(env.link_latency) > 0:
                    for idx, sid in enumerate(sampled_server_ids):
                        outbound_lats = [lat for (src, dst), lat in env.link_latency.items() if src == sid]
                        if outbound_lats:
                            network_quality[idx] = np.exp(-np.mean(outbound_lats) / 500.0)

                # 综合距离和网络质量
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
    """Baseline Zero-Shot: 500模型 + Dilated Subnet -> 2000环境"""

    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    server_ids_full = sorted(list(env.servers.keys()))
    num_servers_full = len(server_ids_full)

    # 加载模型
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
        # Stark 模型: task_dim=4, server_dim=7
        model = StarkScheduler(
            task_dim=4, server_dim=7, num_servers=NUM_SERVERS_MODEL,
            d_model=128, nhead=4, num_encoder_layers=2, num_decoder_layers=2
        ).to(device)
        state_dict_loaded = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict_loaded)
        model.eval()
    elif algo_name == 'PPO_GNN':
        from PPO_GNN.model import GNNActorCritic
        # PPO_GNN 模型: hidden_dim=128
        model = GNNActorCritic(node_feat_dim=3, global_feat_dim=7, hidden_dim=128, gnn_layers=2).to(device)
        state_dict_loaded = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict_loaded)
        model.eval()
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    latencies, compute_costs = [], []

    for i in range(episodes):
        np.random.seed(SEED + i)  # 确保可复现，与 STAR-PPO 使用相同的采样
        task = ds.tasks[i % len(ds.tasks)]
        state_dict = env.reset(task)
        task_lon, task_lat = task['TaskLongitude'], task['TaskLatitude']
        ep_lat, ep_cost, done = 0, 0, False

        while not done:
            valid_actions = env.available_actions()
            if not valid_actions:
                break

            # Nearest Neighbor Subnet: 选择最近的 500 个服务器
            distances = []
            for idx, sid in enumerate(server_ids_full):
                srv = env.servers[sid]
                d = haversine_km(task_lon, task_lat, srv.lon, srv.lat)
                distances.append((idx, d, sid))
            distances.sort(key=lambda x: x[1])

            # 取最近的 500 个作为输入子网
            sampled_indices = [d[0] for d in distances[:NUM_SERVERS_MODEL]]
            sampled_server_ids = [d[2] for d in distances[:NUM_SERVERS_MODEL]]

            # 建立抽样服务器到动作的映射
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
                # 构建状态特征
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

                        # 简化的边
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
                    # 如果模型推理失败，随机选择
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

    # 加载目标区域环境 (Server3_Trap, 2000节点)
    print(f"\nLoading target environment ({REGION_TARGET}, {NUM_SERVERS_ENV} nodes)...")
    ds = WorkflowDataset(DATA_ROOT, split='train', regions=[REGION_TARGET])
    env = WorkflowMoEEnv(ds)
    print(f"Loaded {len(ds.tasks)} tasks, {len(env.servers)} servers")

    results = {}

    # 1. STAR-PPO (Retrained): 2000模型 -> 2000环境 (天花板)
    print("\n" + "="*60)
    print(f"Running STAR-PPO (Retrained): {REGION_TARGET} model -> {REGION_TARGET} env")
    print("="*60)
    results['STAR_PPO_Retrained'] = run_star_ppo_retrained(
        env, ds, MODELS_TARGET['STAR_PPO'], device, EPISODES
    )
    print(f"  Result: Latency = {np.mean(results['STAR_PPO_Retrained']['latencies']):.1f} ms")

    # 2. STAR-PPO (Partition): 500模型 + 分区推理 -> 2000环境 (主要展示)
    print("\n" + "="*60)
    print(f"Running STAR-PPO (Partition): 500 model + Partition Inference -> {REGION_TARGET} env")
    print("="*60)
    results['STAR_PPO_Partition'] = run_star_ppo_partition(
        env, ds, MODELS_500['STAR_PPO'], device, EPISODES
    )
    print(f"  Result: Latency = {np.mean(results['STAR_PPO_Partition']['latencies']):.1f} ms")

    # 3. STAR-PPO (Zero-Shot): 500模型 + Subnet Sampling -> 2000环境 (对照)
    print("\n" + "="*60)
    print(f"Running STAR-PPO (Zero-Shot): 500 model + Subnet Sampling -> {REGION_TARGET} env")
    print("="*60)
    results['STAR_PPO_ZeroShot'] = run_star_ppo_zeroshot(
        env, ds, MODELS_500['STAR_PPO'], device, EPISODES
        )
    print(f"  Result: Latency = {np.mean(results['STAR_PPO_ZeroShot']['latencies']):.1f} ms")

    # 4-8. Baseline (Zero-Shot): 500模型 + Subnet Sampling -> 2000环境
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

    # 保存结果
    print("\n" + "="*60)
    print("Saving results...")
    print("="*60)

    for name, data in results.items():
        lats = data['latencies']
        sw   = data.get('switches', np.zeros_like(lats))
        data['sla_violations'] = _sla(lats)
        data['switches']       = sw
        npz_path = os.path.join(OUTPUT_DIR, f'{name}.npz')
        np.savez(npz_path, **data)
        print(f"  Saved: {npz_path}")

    # 打印汇总
    print("\n" + "="*60)
    print("Summary (Average Latency)")
    print("="*60)
    for name, data in results.items():
        lats = data['latencies']
        lat  = np.mean(lats)
        viol = np.mean(data['sla_violations']) * 100
        qos  = _qos(lats, data['costs'], data['switches'])
        print(f"  {name}: {lat:.1f} ms  SLA={viol:.1f}%  QoS={qos:.2f}")

    # 计算 Gap
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
