"""
在 2000 规模数据集 (data1/Server3_Trap) 上运行 STAR_PPO 的推理
用于可扩展性分析
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

# 成本计算参数
BYTES_PER_TOKEN = 4  # 每 token 4 字节
NETWORK_COST_PER_MS = 0.00015  # 每毫秒网络延迟的成本
COMM_COST_PER_MS = 0.00015    # 通信成本系数

def compute_network_cost_by_latency(network_ms, tokens):
    """计算网络传输成本 - 基于实际网络延迟"""
    data_kb = (tokens * BYTES_PER_TOKEN) / 1024.0
    return data_kb * network_ms * NETWORK_COST_PER_MS

def compute_communication_cost_by_latency(tokens, network_ms):
    """计算通信成本 - 中间结果传输"""
    intermediate_tokens = tokens * 0.3
    data_kb = (intermediate_tokens * BYTES_PER_TOKEN) / 1024.0
    return data_kb * network_ms * COMM_COST_PER_MS

# 数据集配置
DATA_ROOT = './data1'
REGION = 'Server3_Trap'  # 2000 服务器 (陷阱数据集)
NUM_SERVERS = 2000
EPISODES = 200  # 推理 episode 数

# STAR_PPO 模型路径
SEED_MODEL_MAP = {
    'STAR_PPO': {
        42: './results/TopoFreeRL/models/LATEST_Server3_Trap_seed42_actor_epoch_100.pt',
    },
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
        'costs': np.array(compute_costs),  # 只保存 compute_costs 以与其他算法一致
        'compute_costs': np.array(compute_costs),
        'network_costs': np.array(network_costs),
        'communication_costs': np.array(communication_costs),
        'rewards': np.array(rewards),
        'switches': np.array(switches),
        'inference_times': np.array(inf_times)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--episodes', type=int, default=EPISODES)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_seed', type=int, default=42, help='训练时使用的 seed')
    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"Set random seed: {args.seed}")

    # 输出目录
    output_dir = 'inference/results_2000'
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据集和环境
    print(f"Loading dataset from {DATA_ROOT}/{REGION}...")
    # 使用 train 集进行推理
    ds = WorkflowDataset(DATA_ROOT, split='train', regions=[REGION])
    env = WorkflowMoEEnv(ds)

    print(f"Dataset loaded: {len(ds.tasks)} tasks, {NUM_SERVERS} servers")

    # 运行 STAR_PPO 推理
    model_path = SEED_MODEL_MAP['STAR_PPO'][args.train_seed]

    if os.path.exists(model_path):
        print(f"\n{'='*60}")
        print(f"Running STAR_PPO inference")
        print(f"Using model: {model_path}")
        print(f"{'='*60}")

        data = run_star_ppo_inference(env, ds, model_path, args.device, args.episodes)

        # 补全统一字段
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

        npz_path = os.path.join(output_dir, f'STAR_PPO_Server3_Trap_seed{args.train_seed}.npz')
        np.savez(npz_path, **data)
        print(f"Saved to {npz_path}")

        viol = np.mean(data['sla_violations']) * 100
        qos  = _qos(data['latencies'], data['costs'], sw)
        print(f"\n{'='*60}")
        print("STAR_PPO Results Summary (2000 scale):")
        print(f"  Avg Latency:   {np.mean(data['latencies']):.2f} ms")
        print(f"  Std Latency:   {np.std(data['latencies']):.2f} ms")
        print(f"  Avg Cost:      {np.mean(data['costs']):.4f}")
        print(f"  SLA Violation: {viol:.1f}%")
        print(f"  Composite QoS: {qos:.2f}")
        print(f"  Avg Reward:    {np.mean(data['rewards']):.4f}")
        print(f"{'='*60}")
    else:
        print(f"Model not found: {model_path}")


if __name__ == '__main__':
    main()
