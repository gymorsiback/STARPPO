import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env import WorkflowDataset, WorkflowMoEEnv
from utils import ensure_dir, softmax, generate_run_id
from agent import PFAPPOAgent


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


def compute_resource_weights(env, dwa_weights=None):
    if dwa_weights is not None:
        w_L, w_C = dwa_weights[0], dwa_weights[1]

        w1_base, w2_base, w3_base = 0.35, 0.30, 0.35
    
        scale_L = w_L / 0.33
        scale_C = w_C / 0.33
        
        w1 = w1_base * scale_L
        w2 = w2_base * scale_L
        w3 = w3_base * scale_C
    else:
        w1, w2, w3 = 0.35, 0.30, 0.35

    server_ids = sorted(list(env.servers.keys()))
    num_servers = len(server_ids)

    caps = np.array([env.servers[sid].normalized_compute for sid in server_ids], dtype=np.float32)
    norm_caps = caps
   
    current_time = env.current_time_ms
    busy_times = np.array([max(0.0, env.busy_until[sid] - current_time) for sid in server_ids], dtype=np.float32)
    norm_queues = np.clip(busy_times / 5000.0, 0.0, 1.0)
 
    server_min_costs = []
    for sid in server_ids:
        models_on_server = [mi for mi in env.ds.model_instances if mi.server_id == sid]
        server_cost_mult = env.servers[sid].cost_multiplier
        if models_on_server:
            min_cost = min([mi.cost_per_token * server_cost_mult for mi in models_on_server])
        else:
            min_cost = 0.060 * 2.2  
        server_min_costs.append(min_cost)
    
    costs = np.array(server_min_costs, dtype=np.float32)
    cost_min = 0.0015 * 0.4  
    cost_max = 0.060 * 2.2   
    cost_advantage = 1.0 - np.clip((costs - cost_min) / (cost_max - cost_min), 0, 1.0)
 
    w1, w2, w3 = max(0.0, w1), max(0.0, w2), max(0.0, w3)
    weights = (w1 * norm_caps + w3 * cost_advantage) / (1.0 + w2 * norm_queues)
 
    max_w = np.max(weights)
    if max_w > 1e-9:
        weights = weights / max_w
    else:
        weights = np.ones_like(weights) / num_servers

    
    return weights

def build_state_vector(state_dict, dwa_weights):
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
    data_root='/root/autodl-tmp/MOE111/data',
    total_epochs=100,
    episodes_per_epoch=200,
    lr=3e-4,
    batch_size=1024,
    device='cpu',
    seed=42,
    regions=None,
    output_dir=None
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if regions is None:
        regions = ['Server2']

    ds = WorkflowDataset(data_root, split='train', regions=regions)
    env = WorkflowMoEEnv(ds)
    num_servers = len(env.servers)

    server_model_mapping, server_ids = build_server_model_mapping(ds, env)
    print(f"Built server-model mapping: {num_servers} servers, {len(ds.model_instances)} model instances")
 
    agent = PFAPPOAgent(state_dim=7, num_servers=num_servers, lr=lr, device=device)
 
    lr_lambda = lambda epoch: 1.0 - 0.8 * (epoch / total_epochs)
    actor_scheduler = torch.optim.lr_scheduler.LambdaLR(agent.actor_optimizer, lr_lambda=lr_lambda)
    critic_scheduler = torch.optim.lr_scheduler.LambdaLR(agent.critic_optimizer, lr_lambda=lr_lambda)
 
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

    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)  
    loss_moving_avg = np.zeros(3) 
    T = 3.0  
    freeze_epoch = int(total_epochs * 0.8)  
    dwa_start_epoch = 3  

    L_hist = {'L': [], 'C': [], 'S': []}
    weights_hist = []
    
    for epoch in range(total_epochs):
        entropy_decay_ratio = min(1.0, epoch / (total_epochs * 0.9))  
        current_entropy = 0.03 * (1.0 - entropy_decay_ratio) + 0.002 * entropy_decay_ratio
  
        progress = epoch / total_epochs
        guidance_alpha = 0.7 + 0.3 * progress  
        
        if epoch % 10 == 0:
            print(f"  [Curriculum] guidance_alpha = {guidance_alpha:.3f} (Enhanced)")
  
        episode_returns = []
        episode_latency = []
        episode_cost = []
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
                if not (np.any(np.isnan(exp_w)) or np.any(np.isinf(exp_w))):
                    w_k = len(w) * exp_w / (np.sum(exp_w) + 1e-8)
                    w_new = softmax(w_k)
                    if not np.any(np.isnan(w_new)):
                        w = 0.3 * w_new + 0.7 * w
   
                        min_weight = 0.15
                        w = np.clip(w, min_weight, 1.0 - 2*min_weight)
                        w = w / np.sum(w)  
        elif epoch >= freeze_epoch:
            pass

        ep_L_vals = []
        ep_C_vals = []
        ep_S_vals = []

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

            traj_states = []
            traj_weights = []
            traj_actions = []
            traj_logprobs = []
            traj_rewards = []
            traj_dones = []
            traj_values = []
            
            while not done:
                s_vec = build_state_vector(state_dict, w) 
                r_weights = compute_resource_weights(env, dwa_weights=w) 
                if np.any(np.isnan(s_vec)) or np.any(np.isnan(r_weights)):
                    print(f"[Error] NaN detected in state/weights at epoch {epoch}, episode {ep}")
                    print(f"  State: {s_vec}")
                    print(f"  DWA Weights: {w}")
                    print(f"  Resource Weights: {r_weights[:5]}...")
                    break
                
                s_tensor = torch.FloatTensor(s_vec).unsqueeze(0).to(device)
                w_tensor = torch.FloatTensor(r_weights).unsqueeze(0).to(device)
        
                server_action, log_prob, value = agent.act(s_tensor, w_tensor, guidance_alpha=guidance_alpha, guidance_temperature=2.5)
   
                _, _, req_type = env.cur_steps[env.step_idx]
                if req_type is None:
                    req_type = env.cur_task['RequiredModelTypes'][env.step_idx]

                action = map_server_action_to_instance(
                    server_action, str(req_type), server_model_mapping, ds
                )

                next_state_dict, (rL, rC, rS), done, info = env.step(action)

                r_scalar = w[0]*rL + w[1]*rC + w[2]*rS
      
                traj_states.append(s_vec)
                traj_weights.append(r_weights)
                traj_actions.append(server_action)  
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
 
            episode_returns.append(ep_ret)
            episode_latency.append(ep_l)
            episode_cost.append(ep_c)

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

            memory_states.extend(traj_states)
            memory_weights.extend(traj_weights)
            memory_actions.extend(traj_actions)
            memory_logprobs.extend(traj_logprobs)
            memory_rewards.extend(returns) 
            memory_values.extend(traj_values)
            
            epoch_return += ep_ret
            epoch_lat += ep_l
            epoch_cost += ep_c
  
        current_losses = np.array([np.mean(ep_L_vals), np.mean(ep_C_vals), np.mean(ep_S_vals)])
 
        if epoch == 0:
             loss_moving_avg = current_losses
             
        L_hist['L'].append(current_losses[0])
        L_hist['C'].append(current_losses[1])
        L_hist['S'].append(current_losses[2])
        weights_hist.append(w.copy())
    
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
                
                loss, _, _, _ = agent.update_from_batch(
                    states_t[idx], weights_t[idx], actions_t[idx],
                    logprobs_t[idx], returns_t[idx], advantages_t[idx],
                    entropy_coef=current_entropy
                )
                agent_loss += loss
        
        avg_loss = agent_loss / (10 * (dataset_size // batch_size + 1))

        avg_ret = epoch_return / episodes_per_epoch
        avg_lat = epoch_lat / episodes_per_epoch
        avg_cost = epoch_cost / episodes_per_epoch
 
        
        print(f"Epoch {epoch+1}/{total_epochs} | Ret: {avg_ret:.2f} | Lat: {avg_lat:.1f} | Cost: {avg_cost:.3f} | Loss: {avg_loss:.4f} | W: {w}")

        if (epoch+1) % 5 == 0:
            torch.save(agent.actor.state_dict(), os.path.join(models_dir, f'{run_id}_actor_epoch_{epoch:04d}.pt'))

        actor_scheduler.step()
        critic_scheduler.step()

    np.savez_compressed(os.path.join(run_dir, 'training_data.npz'),
             episode_returns=np.array(episode_returns),
             episode_latency=np.array(episode_latency),
             episode_cost=np.array(episode_cost),
             weights_hist=np.array(weights_hist),
             L_hist_L=np.array(L_hist['L']),
             L_hist_C=np.array(L_hist['C']),
             L_hist_S=np.array(L_hist['S']))
    print(f"Saved training_data.npz ({len(episode_returns)} episodes)")

    with open(os.path.join(run_dir, 'meta.json'), 'w') as f:
        json.dump({'run_id': run_id, 'epochs': total_epochs, 'episodes_per_epoch': episodes_per_epoch}, f)

if __name__ == '__main__':
    import argparse
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
    parser.add_argument('--data', type=str, default='/root/autodl-tmp/MOE111/data',
                        help='Data directory, e.g., data or data1')
    args = parser.parse_args()
    
    print(f"Device: {args.device}, Regions: {args.regions}, Data: {args.data}")
    
    train(total_epochs=args.epochs, episodes_per_epoch=args.episodes, 
          device=args.device, seed=args.seed, 
          regions=args.regions, output_dir=args.output_dir,
          data_root=args.data)
