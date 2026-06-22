import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from collections import deque

# 添加根目录路径以导入 env 和 utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env import WorkflowDataset, WorkflowMoEEnv
from utils import ensure_dir, softmax, generate_run_id
from agent import PFAPPOAgent


def build_server_model_mapping(ds, env):
    """
    Build a mapping from (server_index, model_type) -> list of model_instance_idx
    This is the CRITICAL FIX for the action space mismatch bug.

    Agent outputs: server_index (0 to num_servers-1)
    Environment expects: model_instance_idx (0 to num_model_instances-1)

    This function creates the bridge between them.
    """
    server_ids = sorted(list(env.servers.keys()))  # Sorted list of server IDs
    server_id_to_idx = {sid: i for i, sid in enumerate(server_ids)}

    # mapping[server_idx][model_type] = list of model_instance_idx
    mapping = {i: {} for i in range(len(server_ids))}

    for mi in ds.model_instances:
        server_idx = server_id_to_idx.get(mi.server_id)
        if server_idx is not None:
            if mi.model_type not in mapping[server_idx]:
                mapping[server_idx][mi.model_type] = []
            mapping[server_idx][mi.model_type].append(mi.idx)

    return mapping, server_ids


def map_server_action_to_instance(server_idx, required_model_type, mapping, ds, fallback_action=0):
    """
    Map a server index to a specific model instance index.

    Args:
        server_idx: The server index chosen by the agent (0 to num_servers-1)
        required_model_type: The model type required for the current step
        mapping: The (server_idx, model_type) -> [instance_idx] mapping
        ds: Dataset for fallback
        fallback_action: Fallback if no valid instance found

    Returns:
        model_instance_idx: The actual action to pass to env.step()
    """
    if server_idx in mapping:
        instances = mapping[server_idx].get(required_model_type, [])
        if instances:
            # Pick the first (or could be random, or cheapest)
            return instances[0]

    # Fallback: find ANY instance of the required type
    for mi in ds.model_instances:
        if mi.model_type == required_model_type:
            return mi.idx

    # Ultimate fallback
    return fallback_action


def compute_resource_weights(env, dwa_weights=None):
    """
    Calculate Resource-Aware Weights for PFAPPO - DYNAMICALLY ADJUSTED BY DWA
    U_j = (w1 * f_j + w3 * cost_advantage_j) / (1 + w2 * q_j)

    If dwa_weights is provided (latency, cost, switch), we map them:
    - w_latency -> boosts Compute Importance (w1) and Queue Sensitivity (w2)
    - w_cost -> boosts Cost Importance (w3)

    注意：始终返回资源感知权重，渐进式介入在 agent.act() 中通过 guidance_alpha 实现
    """
    if dwa_weights is not None:
        # dwa_weights: [w_L, w_C, w_S]
        # Normalize to ensure reasonable scales
        w_L, w_C = dwa_weights[0], dwa_weights[1]

        # Base values
        w1_base, w2_base, w3_base = 0.35, 0.30, 0.35

        # Dynamic adjustment factors (simple heuristic)
        # If Cost weight is high (e.g., 0.7), w3 should be boosted significantly
        # If Latency weight is high, w1 and w2 should be boosted

        # Scale factors relative to neutral (0.33)
        scale_L = w_L / 0.33
        scale_C = w_C / 0.33

        w1 = w1_base * scale_L
        w2 = w2_base * scale_L
        w3 = w3_base * scale_C
    else:
        w1, w2, w3 = 0.35, 0.30, 0.35

    server_ids = sorted(list(env.servers.keys()))
    num_servers = len(server_ids)

    # 1. Compute Power (f_j)
    caps = np.array([env.servers[sid].normalized_compute for sid in server_ids], dtype=np.float32)
    norm_caps = caps

    # 2. Queue Length / Busy Time (q_j)
    current_time = env.current_time_ms
    busy_times = np.array([max(0.0, env.busy_until[sid] - current_time) for sid in server_ids], dtype=np.float32)
    norm_queues = np.clip(busy_times / 5000.0, 0.0, 1.0)

    # 3. **CRITICAL: Cost Advantage** - Lower cost = Higher advantage
    # 必须考虑 cost_multiplier！这样 Agent 才能看到哪些服务器真正便宜
    server_min_costs = []
    for sid in server_ids:
        models_on_server = [mi for mi in env.ds.model_instances if mi.server_id == sid]
        # 关键：乘以 cost_multiplier 才是真实成本！
        server_cost_mult = env.servers[sid].cost_multiplier
        if models_on_server:
            min_cost = min([mi.cost_per_token * server_cost_mult for mi in models_on_server])
        else:
            min_cost = 0.060 * 2.2  # Max cost as fallback
        server_min_costs.append(min_cost)

    costs = np.array(server_min_costs, dtype=np.float32)
    # Invert and normalize: cheapest (0.0015 * 0.4) -> 1.0, most expensive (0.060 * 2.2) -> 0.0
    cost_min = 0.0015 * 0.4  # 最便宜的模型 × 最低乘数
    cost_max = 0.060 * 2.2   # 最贵的模型 × 最高乘数
    cost_advantage = 1.0 - np.clip((costs - cost_min) / (cost_max - cost_min), 0, 1.0)

    # 4. Combined Resource-Aware Score INCLUDING COST
    # Ensure weights are positive
    w1, w2, w3 = max(0.0, w1), max(0.0, w2), max(0.0, w3)
    weights = (w1 * norm_caps + w3 * cost_advantage) / (1.0 + w2 * norm_queues)

    # 5. Normalize to [0, 1]
    max_w = np.max(weights)
    if max_w > 1e-9:
        weights = weights / max_w
    else:
        weights = np.ones_like(weights) / num_servers

    # 注意：不在这里做混合！
    # 混合逻辑移到 agent.act() 中，通过 guidance_alpha 控制 logits 调整
    # 这样 resource_weights 始终能提供"哪些服务器好"的信息

    return weights

def build_state_vector(state_dict, dwa_weights):
    """
    Construct the 7-dim state vector similar to PPO_algorithm
    """
    return np.array([
        state_dict['step_norm'],
        state_dict['task_lon'],
        state_dict['task_lat'],
        float(state_dict['prev_region_id']),
        dwa_weights[0],
        dwa_weights[1],
        dwa_weights[2]
    ], dtype=np.float32)

def train(
    data_root='./data',
    total_epochs=100,
    episodes_per_epoch=200,
    lr=3e-4,
    batch_size=1024,
    device='cpu',
    seed=42,
    regions=None,
    output_dir=None
):
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Default regions
    if regions is None:
        regions = ['Server2']

    # Init Env
    ds = WorkflowDataset(data_root, split='train', regions=regions)
    env = WorkflowMoEEnv(ds)
    num_servers = len(env.servers)

    # ========================================
    # CRITICAL FIX: Build Server -> Model Instance Mapping
    # ========================================
    # Agent outputs server_idx (0-49), but env.step() expects model_instance_idx (0-449)
    # This mapping bridges the gap by finding the correct model instance on the chosen server
    server_model_mapping, server_ids = build_server_model_mapping(ds, env)
    print(f"Built server-model mapping: {num_servers} servers, {len(ds.model_instances)} model instances")

    # Init Agent
    # State dim is 7 (base features)
    # Agent will handle concatenation of resource weights internally or we pass combined
    # Here we configured Agent to accept separate inputs
    agent = PFAPPOAgent(state_dim=7, num_servers=num_servers, lr=lr, device=device)

    # Init LR Schedulers (Linear Decay to 20%)
    # 更慢的学习率衰减，让收敛曲线更平滑
    lr_lambda = lambda epoch: 1.0 - 0.8 * (epoch / total_epochs)
    actor_scheduler = torch.optim.lr_scheduler.LambdaLR(agent.actor_optimizer, lr_lambda=lr_lambda)
    critic_scheduler = torch.optim.lr_scheduler.LambdaLR(agent.critic_optimizer, lr_lambda=lr_lambda)

    # Directories
    run_id = generate_run_id('pfappo')
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if output_dir is not None:
        results_dir = output_dir
    else:
        results_dir = os.path.join(project_root, 'results', 'PFAPPO')
    run_dir = os.path.join(results_dir, 'logs', run_id)
    models_dir = os.path.join(results_dir, 'models')
    ensure_dir(run_dir)
    ensure_dir(models_dir)

    print(f"Starting PFAPPO Training: {run_id}")
    print(f"Device: {device}, Epochs: {total_epochs}")

    # DWA Init - Balanced strategy for gradual convergence
    # 给 latency 更多权重，确保三个指标都能优化
    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)  # latency slightly higher
    loss_moving_avg = np.zeros(3)
    T = 3.0  # Lower temperature = sharper weight adjustments
    freeze_epoch = int(total_epochs * 0.8)  # Freeze later to allow longer learning
    dwa_start_epoch = 3  # Start DWA early

    # History (across all epochs)
    L_hist = {'L': [], 'C': [], 'S': []}
    weights_hist = []

    for epoch in range(total_epochs):
        # Calculate Entropy Coefficient (Linear Decay)
        # Start: 0.03 (More Exploration) -> End: 0.002 (Exploitation)
        # 更多探索期让 Agent 有时间找到最优服务器
        entropy_decay_ratio = min(1.0, epoch / (total_epochs * 0.9))  # 90% 的 epoch 用于探索衰减
        current_entropy = 0.03 * (1.0 - entropy_decay_ratio) + 0.002 * entropy_decay_ratio

        # ========================================
        # 渐进式介入资源感知权重（Curriculum Learning）
        # ========================================
        # 超强版guidance：从0.7开始，快速增长到1.0
        # 让资源感知引导从一开始就占主导地位
        progress = epoch / total_epochs
        guidance_alpha = 0.7 + 0.3 * progress  # 从0.7增长到1.0，超强引导

        if epoch % 10 == 0:
            print(f"  [Curriculum] guidance_alpha = {guidance_alpha:.3f} (Enhanced)")

        # Per-epoch episode data (reset each epoch)
        episode_returns = []
        episode_latency = []
        episode_cost = []
        # DWA Update - STABILIZED with delayed start and gradual freezing
        if epoch >= dwa_start_epoch and epoch < freeze_epoch:
            # Update Loss Moving Average (slower alpha=0.15 for more stability)
            current_losses = np.array([
                np.mean(ep_L_vals) if 'ep_L_vals' in locals() and len(ep_L_vals) > 0 else 0.0,
                np.mean(ep_C_vals) if 'ep_C_vals' in locals() and len(ep_C_vals) > 0 else 0.0,
                np.mean(ep_S_vals) if 'ep_S_vals' in locals() and len(ep_S_vals) > 0 else 0.0
            ])

            # Initialize or update moving average
            if np.all(loss_moving_avg == 0):
                loss_moving_avg = current_losses + 1e-6  # Avoid zero
            else:
                loss_moving_avg = 0.15 * current_losses + 0.85 * loss_moving_avg

            # Calculate relative change rates
            # Only update if losses are significant
            if np.mean(np.abs(current_losses)) > 1e-5:
                r_n = current_losses / (loss_moving_avg + 1e-7)
                # Very tight clipping to prevent oscillation
                r_n = np.clip(r_n, 0.7, 1.3)

                exp_w = np.exp(r_n / T)
                if not (np.any(np.isnan(exp_w)) or np.any(np.isinf(exp_w))):
                    w_k = len(w) * exp_w / (np.sum(exp_w) + 1e-8)
                    w_new = softmax(w_k)
                    if not np.any(np.isnan(w_new)):
                        # Moderate momentum for smooth but responsive updates
                        w = 0.3 * w_new + 0.7 * w

                        # Ensure minimum weights to prevent any objective from being ignored
                        min_weight = 0.15
                        w = np.clip(w, min_weight, 1.0 - 2*min_weight)
                        w = w / np.sum(w)  # Renormalize
        elif epoch >= freeze_epoch:
            # Freeze weights in later epochs to allow convergence
            pass

        # Track losses for DWA
        ep_L_vals = []
        ep_C_vals = []
        ep_S_vals = []

        # Collecting trajectories
        memory_states = []
        memory_weights = []
        memory_actions = []
        memory_logprobs = []
        memory_rewards = []
        memory_dones = []
        memory_values = []

        epoch_return = 0
        epoch_lat = 0
        epoch_cost = 0

        for ep in range(episodes_per_epoch):
            task = random.choice(ds.tasks)
            state_dict = env.reset(task)

            done = False
            ep_ret = 0
            ep_l = 0
            ep_c = 0

            # Trajectory buffer for GAE
            traj_states = []
            traj_weights = []
            traj_actions = []
            traj_logprobs = []
            traj_rewards = []
            traj_dones = []
            traj_values = []

            while not done:
                # 1. Prepare State
                s_vec = build_state_vector(state_dict, w) # [7]
                # Pass DWA weights to dynamic resource weight calculation
                # resource_weights 始终是资源感知的，guidance_alpha 在 agent.act 中控制
                r_weights = compute_resource_weights(env, dwa_weights=w) # [N]

                # Safety check for NaN
                if np.any(np.isnan(s_vec)) or np.any(np.isnan(r_weights)):
                    print(f"[Error] NaN detected in state/weights at epoch {epoch}, episode {ep}")
                    print(f"  State: {s_vec}")
                    print(f"  DWA Weights: {w}")
                    print(f"  Resource Weights: {r_weights[:5]}...")
                    break

                s_tensor = torch.FloatTensor(s_vec).unsqueeze(0).to(device)
                w_tensor = torch.FloatTensor(r_weights).unsqueeze(0).to(device)

                # 2. Act - Agent outputs SERVER index (0 to num_servers-1)
                # guidance_alpha: 直接影响 logits，引导 Agent 选择好服务器
                # guidance_temperature 设为 2.5，超强资源感知引导
                server_action, log_prob, value = agent.act(s_tensor, w_tensor, guidance_alpha=guidance_alpha, guidance_temperature=2.5)

                # ========================================
                # CRITICAL FIX: Map Server Action to Model Instance
                # ========================================
                # Get required model type for current step
                _, _, req_type = env.cur_steps[env.step_idx]
                if req_type is None:
                    req_type = env.cur_task['RequiredModelTypes'][env.step_idx]

                # Map server_action to model_instance_idx
                action = map_server_action_to_instance(
                    server_action, str(req_type), server_model_mapping, ds
                )

                # 3. Step - Now passing correct model_instance_idx
                next_state_dict, (rL, rC, rS), done, info = env.step(action)

                # Scalar Reward
                r_scalar = w[0]*rL + w[1]*rC + w[2]*rS

                # 4. Store
                # IMPORTANT: Store server_action (what agent chose), not mapped action
                # This ensures PPO update uses correct log probabilities
                traj_states.append(s_vec)
                traj_weights.append(r_weights)
                traj_actions.append(server_action)  # Store server action for PPO
                traj_logprobs.append(log_prob)
                traj_rewards.append(r_scalar)
                traj_dones.append(done)
                traj_values.append(value)

                state_dict = next_state_dict
                ep_ret += r_scalar
                ep_l += info['latency_ms']
                ep_c += info['cost']

                ep_L_vals.append(-rL)
                ep_C_vals.append(-rC)
                ep_S_vals.append(-rS)

            # Record per-episode data
            episode_returns.append(ep_ret)
            episode_latency.append(ep_l)
            episode_cost.append(ep_c)

            # Finish Episode: Compute GAE
            # We need value of next state (which is terminal state = 0)
            next_value = 0
            returns = []
            gae = 0
            gamma = 0.99
            lam = 0.95

            for step in reversed(range(len(traj_rewards))):
                delta = traj_rewards[step] + gamma * next_value * (1 - traj_dones[step]) - traj_values[step]
                gae = delta + gamma * lam * gae
                returns.insert(0, gae + traj_values[step])
                next_value = traj_values[step]

            # Extend epoch memory
            memory_states.extend(traj_states)
            memory_weights.extend(traj_weights)
            memory_actions.extend(traj_actions)
            memory_logprobs.extend(traj_logprobs)
            memory_rewards.extend(returns) # Store returns (target values)
            # Calculate advantages
            memory_values.extend(traj_values)

            epoch_return += ep_ret
            epoch_lat += ep_l
            epoch_cost += ep_c

        # Update DWA stats
        # Note: DWA weights are updated at the START of the loop using loss_moving_avg
        # Here we just log the current epoch's losses
        current_losses = np.array([np.mean(ep_L_vals), np.mean(ep_C_vals), np.mean(ep_S_vals)])

        # If this is the very first epoch, initialize loss_moving_avg for the next loop
        if epoch == 0:
             loss_moving_avg = current_losses

        L_hist['L'].append(current_losses[0])
        L_hist['C'].append(current_losses[1])
        L_hist['S'].append(current_losses[2])
        weights_hist.append(w.copy())

        # PPO Update
        # Convert to tensors
        states_t = torch.FloatTensor(np.array(memory_states))
        weights_t = torch.FloatTensor(np.array(memory_weights))
        actions_t = torch.LongTensor(np.array(memory_actions))
        logprobs_t = torch.FloatTensor(np.array(memory_logprobs))
        returns_t = torch.FloatTensor(np.array(memory_rewards))
        values_t = torch.FloatTensor(np.array(memory_values))
        advantages_t = returns_t - values_t

        # Batch update
        dataset_size = len(states_t)
        indices = np.arange(dataset_size)

        agent_loss = 0
        for _ in range(10): # 10 epochs per update
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                idx = indices[start:end]

                loss, _, _, _ = agent.update_from_batch(
                    states_t[idx], weights_t[idx], actions_t[idx],
                    logprobs_t[idx], returns_t[idx], advantages_t[idx],
                    entropy_coef=current_entropy
                )
                agent_loss += loss

        avg_loss = agent_loss / (10 * (dataset_size // batch_size + 1))

        # Log
        avg_ret = epoch_return / episodes_per_epoch
        avg_lat = epoch_lat / episodes_per_epoch
        avg_cost = epoch_cost / episodes_per_epoch

        # Per-episode data is now appended inside the inner loop

        print(f"Epoch {epoch+1}/{total_epochs} | Ret: {avg_ret:.2f} | Lat: {avg_lat:.1f} | Cost: {avg_cost:.3f} | Loss: {avg_loss:.4f} | W: {w}")

        # 模型检查点（每5个epoch保存一次）
        if (epoch+1) % 5 == 0:
            torch.save(agent.actor.state_dict(), os.path.join(models_dir, f'{run_id}_actor_epoch_{epoch:04d}.pt'))

        # Step LR Schedulers
        actor_scheduler.step()
        critic_scheduler.step()

    # 保存合并的训练数据
    np.savez_compressed(os.path.join(run_dir, 'training_data.npz'),
             episode_returns=np.array(episode_returns),
             episode_latency=np.array(episode_latency),
             episode_cost=np.array(episode_cost),
             weights_hist=np.array(weights_hist),
             L_hist_L=np.array(L_hist['L']),
             L_hist_C=np.array(L_hist['C']),
             L_hist_S=np.array(L_hist['S']))
    print(f"Saved training_data.npz ({len(episode_returns)} episodes)")

    # Save meta
    with open(os.path.join(run_dir, 'meta.json'), 'w') as f:
        json.dump({'run_id': run_id, 'epochs': total_epochs, 'episodes_per_epoch': episodes_per_epoch}, f)

if __name__ == '__main__':
    import argparse
    # Auto-detect device
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--device', type=str, default=default_device)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--regions', type=str, nargs='+', default=['Server2'],
                        help='Regions to train on, e.g., Server1 Server2')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for models and logs')
    parser.add_argument('--data', type=str, default='./data',
                        help='Data directory, e.g., data or data1')
    args = parser.parse_args()

    print(f"Device: {args.device}, Regions: {args.regions}, Data: {args.data}")

    train(total_epochs=args.epochs, episodes_per_epoch=args.episodes,
          device=args.device, seed=args.seed,
          regions=args.regions, output_dir=args.output_dir,
          data_root=args.data)
