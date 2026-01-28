
import os
import sys
import json
import random
import numpy as np
import torch
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env import WorkflowDataset
from utils import ensure_dir, generate_run_id, softmax
from STAR_PPO.env_augmented import AugmentedWorkflowEnv
from STAR_PPO.agent import StarPPOAgent


def build_server_model_mapping(ds, env):
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
    if server_idx in mapping:
        instances = mapping[server_idx].get(required_model_type, [])
        if instances:
            return instances[0]
    for mi in ds.model_instances:
        if mi.model_type == required_model_type:
            return mi.idx
    return fallback_action

def compute_resource_weights(env, dwa_weights=None, disable_network_quality=False):
    if dwa_weights is not None:
        w_L, w_C = dwa_weights[0], dwa_weights[1]
        scale_L = w_L / 0.33
        scale_C = w_C / 0.33
        w1 = 0.35 * scale_L
        w2 = 0.30 * scale_L
        w3 = 0.35 * scale_C
    else:
        w1, w2, w3 = 0.35, 0.30, 0.35

    server_ids = sorted(list(env.servers.keys()))
    num_servers = len(server_ids)
    
    caps = np.array([env.servers[sid].normalized_compute for sid in server_ids], dtype=np.float32)
    norm_caps = caps

    cost_mults = np.array([env.servers[sid].cost_multiplier for sid in server_ids], dtype=np.float32)
    cost_advantage = 1.0 - np.clip(cost_mults / 2.0, 0, 1.0)
    
    current_time = env.current_time_ms
    busy_times = np.array([max(0.0, env.busy_until[sid] - current_time) for sid in server_ids], dtype=np.float32)
    norm_queues = np.clip(busy_times / 5000.0, 0.0, 1.0)
    
    w2_queue = 0.30
    weights = norm_caps / (1.0 + w2_queue * norm_queues)
    weights = weights * (0.5 + 0.5 * cost_advantage)

    if not disable_network_quality:
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


class AblationEnv(AugmentedWorkflowEnv):
    def __init__(self, dataset, ablation_mode='full', **kwargs):
        super().__init__(dataset, **kwargs)
        self.ablation_mode = ablation_mode
        
    def get_augmented_state(self, dwa_weights=None):
        state = super().get_augmented_state(dwa_weights)
   
        if self.ablation_mode == 'no_workflow':
            state[0] = 0.0
            
        elif self.ablation_mode == 'no_topology':
            state[1] = 0.0  
            state[2] = 0.0  
            state[3] = 0.0  
            state[7] = 0.0  
            state[8] = 0.0  
            state[9] = 0.0  
 
        
        return state


def train_ablation(
    ablation_mode='full',
    data_root=None,
    total_epochs=100,
    episodes_per_epoch=200,
    lr=3e-4,
    batch_size=1024,
    device='cpu',
    seed=42,
    regions=None
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if data_root is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        data_root = os.path.join(project_root, 'data')
    
    
    if regions is None:
        regions = ['Server1']
    target_region = regions[0]

    trap_server_ids = set()
    trap_path = os.path.join(data_root, target_region, 'trap_config.json')
    if os.path.exists(trap_path):
        with open(trap_path, 'r') as f:
            trap_cfg = json.load(f)
        trap_server_ids = set(trap_cfg.get('trap_server_ids', []))
        print(f"Loaded {len(trap_server_ids)} trap servers for penalty injection")
    
    print(f"=" * 60)
    print(f"STAR-PPO Ablation Training")
    print(f"Mode: {ablation_mode}")
    print(f"Region: {target_region}")
    print(f"Epochs: {total_epochs}")
    print(f"=" * 60)
    
    ds = WorkflowDataset(data_root, split='train', regions=[target_region])
    env = AblationEnv(ds, ablation_mode=ablation_mode)
    num_servers = len(env.servers)
    
    server_model_mapping, _ = build_server_model_mapping(ds, env)
  
    agent = StarPPOAgent(state_dim=10, num_servers=num_servers, lr=lr, device=device)

    ablation_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(ablation_dir, 'results', f'{ablation_mode}_seed{seed}')
    models_dir = os.path.join(results_dir, 'models')
    ensure_dir(results_dir)
    ensure_dir(models_dir)
 
    if ablation_mode == 'no_future':
        gamma = 0.0  
        print(">>> [ABLATION] Gamma set to 0.0 (Myopic Agent)")
    else:
        gamma = 0.99 
  
    disable_network = (ablation_mode == 'no_topology')
    if disable_network:
        print(">>> [ABLATION] Network Quality disabled in resource_weights")

    lr_lambda = lambda epoch: 1.0 - 0.8 * (epoch / total_epochs)
    actor_scheduler = torch.optim.lr_scheduler.LambdaLR(agent.actor_optimizer, lr_lambda=lr_lambda)
    critic_scheduler = torch.optim.lr_scheduler.LambdaLR(agent.critic_optimizer, lr_lambda=lr_lambda)
 
    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    loss_moving_avg = np.zeros(3) 
    T = 3.0
    freeze_epoch = int(total_epochs * 0.8)
    dwa_start_epoch = 3
    
    L_hist = {'L': [], 'C': [], 'S': []}

    epoch_rewards_history = []
    epoch_latency_history = []
    epoch_cost_history = []
    
    for epoch in range(total_epochs):
        ent_coef = max(0.002, 0.03 - (0.028 * epoch / (total_epochs * 0.9)))
        progress = epoch / total_epochs
        guidance_alpha = 0.6 + 0.6 * progress
   
        if epoch >= dwa_start_epoch and epoch < freeze_epoch:
            current_losses = np.array([
                np.mean(ep_L_vals) if 'ep_L_vals' in locals() and len(ep_L_vals) > 0 else 0.0,
                np.mean(ep_C_vals) if 'ep_C_vals' in locals() and len(ep_C_vals) > 0 else 0.0,
                np.mean(ep_S_vals) if 'ep_S_vals' in locals() and len(ep_S_vals) > 0 else 0.0
            ])
            if np.all(loss_moving_avg == 0):
                loss_moving_avg = current_losses + 1e-6
            else:
                loss_moving_avg = 0.15 * current_losses + 0.85 * loss_moving_avg
            
            if np.mean(np.abs(current_losses)) > 1e-5:
                r_n = current_losses / (loss_moving_avg + 1e-7)
                r_n = np.clip(r_n, 0.7, 1.3)
                exp_w = np.exp(r_n / T)
                w_k = len(w) * exp_w / (np.sum(exp_w) + 1e-8)
                w_new = softmax(w_k)
                w = 0.3 * w_new + 0.7 * w
                w = np.clip(w, 0.15, 0.7)
                w = w / np.sum(w)

        ep_L_vals = []
        ep_C_vals = []
        ep_S_vals = []
        
        memory_states = []
        memory_weights = []
        memory_actions = []
        memory_logprobs = []
        memory_rewards = []
        memory_values = []
        
        ep_returns = []
        ep_lats = []
        ep_costs = []
        
        for ep in range(episodes_per_epoch):
            task = random.choice(ds.tasks)
            env.reset(task)
            s_vec = env.get_augmented_state(dwa_weights=w)
            
            done = False
            
            traj_states = []
            traj_weights = []
            traj_actions = []
            traj_logprobs = []
            traj_rewards = []
            traj_values = []
            
            ep_ret = 0
            ep_l_raw = 0
            ep_c_raw = 0
            
            while not done:
                r_weights = compute_resource_weights(env, dwa_weights=w, 
                                                     disable_network_quality=disable_network)
                
                s_tensor = torch.FloatTensor(s_vec).unsqueeze(0).to(device)
                w_tensor = torch.FloatTensor(r_weights).unsqueeze(0).to(device)
                
                action_idx, log_prob, value = agent.act(s_tensor, w_tensor, guidance_alpha=guidance_alpha)
                
                _, _, req_type = env.cur_steps[env.step_idx]
                if req_type is None:
                     req_type = env.cur_task['RequiredModelTypes'][env.step_idx]
                     
                real_action = map_server_action_to_instance(
                    action_idx, str(req_type), server_model_mapping, ds
                )
                
                _, (rL, rC, rS), done, info = env.step(real_action)
                
                next_s_vec = env.get_augmented_state(dwa_weights=w)
                
                r_scalar = w[0]*rL + w[1]*rC + w[2]*rS
                
                traj_states.append(s_vec)
                traj_weights.append(r_weights)
                traj_actions.append(action_idx)
                traj_logprobs.append(log_prob)
                traj_rewards.append(r_scalar)
                traj_values.append(value)
                
                s_vec = next_s_vec
                ep_ret += r_scalar
                ep_l_raw += info['latency_ms']
                ep_c_raw += info['cost']
                
                ep_L_vals.append(-rL)
                ep_C_vals.append(-rC)
                ep_S_vals.append(-rS)
      
            next_value = 0
            gae = 0
            lam = 0.95
            returns = []
            
            for step in reversed(range(len(traj_rewards))):
                delta = traj_rewards[step] + gamma * next_value - traj_values[step]
                gae = delta + gamma * lam * gae
                returns.insert(0, gae + traj_values[step])
                next_value = traj_values[step]
                
            memory_states.extend(traj_states)
            memory_weights.extend(traj_weights)
            memory_actions.extend(traj_actions)
            memory_logprobs.extend(traj_logprobs)
            memory_rewards.extend(returns)
            memory_values.extend(traj_values)
            
            ep_returns.append(ep_ret)
            ep_lats.append(ep_l_raw)
            ep_costs.append(ep_c_raw)
            
        states_t = torch.FloatTensor(np.array(memory_states))
        weights_t = torch.FloatTensor(np.array(memory_weights))
        actions_t = torch.LongTensor(np.array(memory_actions))
        logprobs_t = torch.FloatTensor(np.array(memory_logprobs))
        returns_t = torch.FloatTensor(np.array(memory_rewards))
        values_t = torch.FloatTensor(np.array(memory_values))
        advantages_t = returns_t - values_t
        
        dataset_size = len(states_t)
        indices = np.arange(dataset_size)
        agent_loss = 0
        
        for _ in range(10):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                idx = indices[start:end]
                
                loss = agent.update_from_batch(
                    states_t[idx], weights_t[idx], actions_t[idx],
                    logprobs_t[idx], returns_t[idx], advantages_t[idx],
                    entropy_coef=ent_coef
                )
                agent_loss += loss
        
        avg_loss = agent_loss / (10 * (dataset_size // batch_size + 1))
        avg_ret = np.mean(ep_returns)
        avg_lat = np.mean(ep_lats)
        avg_cost = np.mean(ep_costs)
        
        epoch_rewards_history.append(avg_ret)
        epoch_latency_history.append(avg_lat)
        epoch_cost_history.append(avg_cost)
        
        L_hist['L'].append(np.mean(ep_L_vals))
        L_hist['C'].append(np.mean(ep_C_vals))
        L_hist['S'].append(np.mean(ep_S_vals))
        
        print(f"[{ablation_mode}] Epoch {epoch+1}/{total_epochs} | "
              f"Ret: {avg_ret:.2f} | Lat: {avg_lat:.1f}ms | Cost: {avg_cost:.3f} | "
              f"Loss: {avg_loss:.4f}")
  
        if (epoch+1) % 5 == 0:
            torch.save(agent.actor.state_dict(), 
                      os.path.join(models_dir, f'{ablation_mode}_actor_epoch_{epoch+1}.pt'))
        
        actor_scheduler.step()
        critic_scheduler.step()
 
    torch.save(agent.actor.state_dict(), 
              os.path.join(models_dir, f'{ablation_mode}_actor_final.pt'))
    
    np.savez(os.path.join(results_dir, 'metrics.npz'),
             rewards=np.array(epoch_rewards_history),
             latency=np.array(epoch_latency_history),
             cost=np.array(epoch_cost_history))
    
    print(f"\n{'='*60}")
    print(f"Training Completed for mode: {ablation_mode}")
    print(f"Final Avg Latency: {epoch_latency_history[-1]:.1f}ms")
    print(f"Final Avg Cost: {epoch_cost_history[-1]:.3f}")
    print(f"Model saved to: {models_dir}")
    print(f"{'='*60}\n")
    
    return {
        'rewards': epoch_rewards_history,
        'latency': epoch_latency_history,
        'cost': epoch_cost_history
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='STAR-PPO Ablation Training')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['full', 'no_workflow', 'no_future', 'no_topology'],
                        help='Ablation mode')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data', type=str, default='/root/autodl-tmp/MOE111/data1',
                        help='Data directory')
    parser.add_argument('--region', type=str, default='Server1_Trap',
                        help='Region to train on (default: Server1_Trap)')
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    
    train_ablation(
        ablation_mode=args.mode,
        data_root=args.data,
        total_epochs=args.epochs,
        episodes_per_epoch=args.episodes,
        device=args.device,
        seed=args.seed,
        regions=[args.region]
    )

