import os
import sys
import json
import numpy as np
import torch
import time

# 添加根目录路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env import WorkflowDataset, WorkflowMoEEnv
from TopoFreeRL.env_augmented import AugmentedWorkflowEnv
from TopoFreeRL.agent import StarPPOAgent


def compute_resource_weights(env, dwa_weights=None):
    """
    计算 resource_weights，用于引导模型选择好的服务器。
    STAR-PPO 特有：加入网络连通性感知（通过 link_latency 数据推断）
    与 train.py 保持一致！
    """
    server_ids = sorted(list(env.servers.keys()))
    num_servers = len(server_ids)

    caps = np.array([env.servers[sid].normalized_compute for sid in server_ids], dtype=np.float32)
    norm_caps = caps

    # 计算成本优势（成本越低越好）
    cost_mults = np.array([env.servers[sid].cost_multiplier for sid in server_ids], dtype=np.float32)
    cost_advantage = 1.0 - np.clip(cost_mults / 2.0, 0, 1.0)

    current_time = env.current_time_ms
    busy_times = np.array([max(0.0, env.busy_until[sid] - current_time) for sid in server_ids], dtype=np.float32)
    norm_queues = np.clip(busy_times / 5000.0, 0.0, 1.0)

    # STAR-PPO 核心：算力 × 成本优势，平衡延迟和成本
    w2 = 0.30
    weights = norm_caps / (1.0 + w2 * norm_queues)
    weights = weights * (0.5 + 0.5 * cost_advantage)  # 加入成本因素

    # === STAR-PPO 核心：网络连通性感知 ===
    network_quality = np.ones(num_servers, dtype=np.float32)
    if hasattr(env, 'link_latency') and len(env.link_latency) > 0:
        for i, sid in enumerate(server_ids):
            outbound_lats = []
            for (src, dst), lat in env.link_latency.items():
                if src == sid:
                    outbound_lats.append(lat)
            if outbound_lats:
                avg_lat = np.mean(outbound_lats)
                network_quality[i] = np.exp(-avg_lat / 500.0)

    weights = weights * network_quality

    max_w = np.max(weights)
    if max_w > 1e-9:
        weights = weights / max_w
    else:
        weights = np.ones_like(weights) / num_servers

    return weights


def build_server_model_mapping(ds, env):
    """Build mapping from server index to model instances"""
    server_ids = sorted(list(env.servers.keys()))
    server_id_to_idx = {sid: i for i, sid in enumerate(server_ids)}
    mapping = {i: {} for i in range(len(server_ids))}
    for mi in ds.model_instances:
        server_idx = server_id_to_idx.get(mi.server_id)
        if server_idx is not None:
            if mi.model_type not in mapping[server_idx]:
                mapping[server_idx][mi.model_type] = []
            mapping[server_idx][mi.model_type].append(mi.idx)
    return mapping, server_ids


def map_server_action_to_instance(server_idx, required_model_type, mapping, ds, fallback_action=0):
    """Map server index to model instance index"""
    if server_idx in mapping:
        instances = mapping[server_idx].get(required_model_type, [])
        if instances:
            return instances[0]
    for mi in ds.model_instances:
        if mi.model_type == required_model_type:
            return mi.idx
    return fallback_action


def run_inference(
    data_root='./data',
    model_path=None,
    device='cpu',
    episodes=100,
    test_region='Server3',
    split='test'
):
    """Run inference with STAR-PPO model"""
    # Load Dataset and Augmented Environment
    ds = WorkflowDataset(data_root, split=split, regions=[test_region])
    env = AugmentedWorkflowEnv(ds)

    # 模型训练时的服务器数量（需要与训练时一致）
    MODEL_NUM_SERVERS = 500  # 训练时使用500个服务器

    # 实际环境的服务器数量
    actual_num_servers = len(env.servers)
    server_ids = sorted(list(env.servers.keys()))

    # 跨域适配
    if actual_num_servers > MODEL_NUM_SERVERS:
        candidate_server_ids = server_ids[:MODEL_NUM_SERVERS]
        print(f"[跨域适配] {test_region}有{actual_num_servers}个服务器，只使用前{MODEL_NUM_SERVERS}个作为候选")
    elif actual_num_servers < MODEL_NUM_SERVERS:
        candidate_server_ids = server_ids
        print(f"[跨域适配] {test_region}只有{actual_num_servers}个服务器，模型将只使用前{actual_num_servers}个输出")
    else:
        candidate_server_ids = server_ids
        print(f"[同域测试] {test_region}有{actual_num_servers}个服务器，与训练一致")

    # Build server to instance mapping
    server_model_mapping, _ = build_server_model_mapping(ds, env)

    # Initialize Agent (state_dim=10 for STAR-PPO)
    agent = StarPPOAgent(state_dim=10, num_servers=MODEL_NUM_SERVERS, device=device)

    # Load Model
    if model_path and os.path.exists(model_path):
        agent.actor.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded STAR-PPO model from: {model_path}")
    else:
        print("No model loaded. Using random policy.")

    agent.actor.eval()

    # Fixed DWA weights for inference
    w = np.array([1/3, 1/3, 1/3], dtype=np.float32)

    # Results
    latencies = []
    costs = []
    rewards = []
    switches_list = []
    inference_times = []

    print(f"Running STAR-PPO inference on {episodes} episodes in {test_region}...")

    for i in range(episodes):
        if i % 20 == 0:
            print(f"Episode {i}/{episodes}")

        task = ds.tasks[i % len(ds.tasks)]
        env.reset(task)

        # Get augmented state
        s_vec = env.get_augmented_state(dwa_weights=w)

        ep_lat = 0
        ep_cost = 0
        ep_reward = 0
        ep_inference_time = 0
        done = False

        while not done:
            r_weights = compute_resource_weights(env, dwa_weights=w)

            # Padding if needed
            if len(r_weights) < MODEL_NUM_SERVERS:
                padded_weights = np.zeros(MODEL_NUM_SERVERS, dtype=np.float32)
                padded_weights[:len(r_weights)] = r_weights
                r_weights = padded_weights

            s_tensor = torch.FloatTensor(s_vec).unsqueeze(0).to(device)
            w_tensor = torch.FloatTensor(r_weights).unsqueeze(0).to(device)

            # Time the inference
            t0 = time.time()
            with torch.no_grad():
                logits = agent.actor(s_tensor, w_tensor)

                # STAR-PPO uses resource-aware guidance similar to PFAPPO
                rw_tensor = torch.FloatTensor(r_weights).to(device)
                rw_centered = rw_tensor - rw_tensor.mean()
                rw_std = rw_tensor.std() + 1e-6
                rw_normalized = rw_centered / rw_std

                # guidance_alpha=1.0 for inference
                enhanced_logits = logits.squeeze(0) + 1.0 * rw_normalized * 1.5

                # Only select from candidate servers
                valid_logits = enhanced_logits[:len(candidate_server_ids)]
                action_idx = torch.argmax(valid_logits).item()
            ep_inference_time += (time.time() - t0) * 1000

            # Get required model type
            req_type = env.cur_task['RequiredModelTypes'][env.step_idx]

            # Map to actual model instance
            real_action = map_server_action_to_instance(
                action_idx, str(req_type), server_model_mapping, ds
            )

            # Step environment
            _, (rL, rC, rS), done, info = env.step(real_action)

            # Get next augmented state
            s_vec = env.get_augmented_state(dwa_weights=w)

            r_scalar = w[0]*rL + w[1]*rC + w[2]*rS

            ep_lat += info['latency_ms']
            ep_cost += info['cost']
            ep_reward += r_scalar

        latencies.append(ep_lat)
        costs.append(ep_cost)
        rewards.append(ep_reward)
        switches_list.append(env.ep_switches)
        inference_times.append(ep_inference_time)

    # Statistics
    print("\n" + "="*50)
    print("STAR-PPO Inference Results:")
    print("="*50)
    print(f"Episodes: {episodes}")
    print(f"Average Latency: {np.mean(latencies):.2f} ms (std: {np.std(latencies):.2f})")
    print(f"Average Cost:    ${np.mean(costs):.4f} (std: ${np.std(costs):.4f})")
    print(f"Average Reward:  {np.mean(rewards):.4f} (std: {np.std(rewards):.4f})")
    print(f"Average Switches: {np.mean(switches_list):.2f}")
    print(f"Avg Inference Time: {np.mean(inference_times):.3f} ms")
    print("="*50)

    # Save detailed results
    output_dir = 'inference/results'
    os.makedirs(output_dir, exist_ok=True)

    if model_path:
        model_basename = os.path.basename(model_path).replace('.pt', '')
    else:
        model_basename = 'random'

    npz_path = os.path.join(output_dir, f'TopoFreeRL_{model_basename}_{split}_detailed.npz')
    np.savez(npz_path,
             latencies=np.array(latencies),
             costs=np.array(costs),
             rewards=np.array(rewards),
             switches=np.array(switches_list),
             inference_times=np.array(inference_times))
    print(f"Detailed results saved to: {npz_path}")

    return {
        'latencies': latencies,
        'costs': costs,
        'rewards': rewards,
        'switches': switches_list,
        'inference_times': inference_times
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--region', type=str, default='Server3', help='Test region')
    parser.add_argument('--split', type=str, default='test', help='Data split (train/test)')
    parser.add_argument('--data', type=str, default='./data',
                        help='Data directory')
    args = parser.parse_args()

    run_inference(
        data_root=args.data,
        model_path=args.model,
        episodes=args.episodes,
        device=args.device,
        test_region=args.region,
        split=args.split
    )




