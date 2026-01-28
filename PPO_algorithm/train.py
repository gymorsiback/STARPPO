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
from agent import PPOAgent


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
    data_root='/root/autodl-tmp/MOE111/data1',
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

    agent = PPOAgent(state_dim=7, num_servers=num_servers, lr=1e-4, device=device)
 
    lr_lambda = lambda epoch: 1.0 - 0.8 * (epoch / total_epochs)
    actor_scheduler = torch.optim.lr_scheduler.LambdaLR(agent.actor_optimizer, lr_lambda=lr_lambda)
    critic_scheduler = torch.optim.lr_scheduler.LambdaLR(agent.critic_optimizer, lr_lambda=lr_lambda)
 
    run_id = generate_run_id('ppo')
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if output_dir is not None:
        results_dir = output_dir
    else:
        results_dir = os.path.join(project_root, 'results', 'PPO')
    run_dir = os.path.join(results_dir, 'logs', run_id)
    models_dir = os.path.join(results_dir, 'models')
    ensure_dir(run_dir)
    ensure_dir(models_dir)

    print(f"Starting PPO Training: {run_id}")
    print(f"Device: {device}, Epochs: {total_epochs}")
    print(f"[PPO] Standard PPO without resource_weights (baseline)")

    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    loss_moving_avg = np.zeros(3) 
    T = 3.0
    freeze_epoch = int(total_epochs * 0.8)
    dwa_start_epoch = 3  
   
    L_hist = {'L': [], 'C': [], 'S': []}
    weights_hist = []
 
    all_episode_returns = []
    all_episode_latency = []
    all_episode_cost = []
    
    for epoch in range(total_epochs):
        entropy_decay_ratio = min(1.0, epoch / (total_epochs * 0.9))
        current_entropy = 0.03 * (1.0 - entropy_decay_ratio) + 0.002 * entropy_decay_ratio

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
        
        ep_L_vals = []
        ep_C_vals = []
        ep_S_vals = []
        
        memory_buffer = []
        
        for ep in range(episodes_per_epoch):
            task = random.choice(ds.tasks)  
            state_dict = env.reset(task)
            done = False
            
            traj_states = []
            traj_actions = []
            traj_log_probs = []
            traj_rewards = []
            traj_dones = []
            traj_values = []
            
            while not done:
                s_vec = build_state_vector(state_dict, w)
                
                if np.any(np.isnan(s_vec)):
                    break
                
                s_tensor = torch.FloatTensor(s_vec).unsqueeze(0).to(device)

                server_action, log_prob, value = agent.act(s_tensor)
                
                _, _, req_type = env.cur_steps[env.step_idx]
                if req_type is None:
                    req_type = env.cur_task['RequiredModelTypes'][env.step_idx]
                
                action = map_server_action_to_instance(
                    server_action, str(req_type), server_model_mapping, ds
                )
                
                next_state_dict, (rL, rC, rS), done, info = env.step(action)
                
                scalar_reward = w[0] * rL + w[1] * rC + w[2] * rS
                
                traj_states.append(s_vec)
                traj_actions.append(server_action)
                traj_log_probs.append(log_prob)
                traj_rewards.append(scalar_reward)
                traj_dones.append(done)
                traj_values.append(value)
                
                state_dict = next_state_dict
                
                ep_L_vals.append(-rL)
                ep_C_vals.append(-rC)
                ep_S_vals.append(-rS)
 
            episode_returns.append(sum(traj_rewards))
            episode_latency.append(sum(env.ep_latency))
            episode_cost.append(sum(env.ep_cost))
  
            returns = []
            gae = 0
            gamma = 0.99
            lam = 0.95
            
            for i in range(len(traj_rewards) - 1, -1, -1):
                next_value = 0 if traj_dones[i] else (traj_values[i+1] if i+1 < len(traj_values) else 0)
                delta = traj_rewards[i] + gamma * next_value - traj_values[i]
                gae = delta + gamma * lam * gae
                returns.insert(0, gae + traj_values[i])
            
            for i in range(len(traj_states)):
                memory_buffer.append({
                    'state': traj_states[i],
                    'action': traj_actions[i],
                    'log_prob': traj_log_probs[i],
                    'return': returns[i],
                    'advantage': returns[i] - traj_values[i]
                })

        if len(memory_buffer) > 0:
            states = torch.FloatTensor(np.array([x['state'] for x in memory_buffer]))
            actions = torch.LongTensor(np.array([x['action'] for x in memory_buffer]))
            old_log_probs = torch.FloatTensor(np.array([x['log_prob'] for x in memory_buffer]))
            returns = torch.FloatTensor(np.array([x['return'] for x in memory_buffer]))
            advantages = torch.FloatTensor(np.array([x['advantage'] for x in memory_buffer]))
            
            dataset_size = len(memory_buffer)
            indices = np.arange(dataset_size)
            
            for _ in range(10):  
                np.random.shuffle(indices)
                for start in range(0, dataset_size, batch_size):
                    end = start + batch_size
                    idx = indices[start:end]
                    
                    agent.update_from_batch(
                        states[idx],
                        actions[idx],
                        old_log_probs[idx],
                        returns[idx],
                        advantages[idx],
                        entropy_coef=current_entropy
                    )
        
        actor_scheduler.step()
        critic_scheduler.step()

        avg_ret = np.mean(episode_returns)
        avg_lat = np.mean(episode_latency)
        avg_cost = np.mean(episode_cost)
        
        print(f"Epoch {epoch+1}/{total_epochs} | Ret: {avg_ret:.2f} | Lat: {avg_lat:.1f}ms | Cost: ${avg_cost:.4f} | DWA: {w} | Ent: {current_entropy:.4f}")
        
        L_hist['L'].append(avg_lat)
        L_hist['C'].append(avg_cost)
        L_hist['S'].append(0.0)
        weights_hist.append(w)

        all_episode_returns.extend(episode_returns)
        all_episode_latency.extend(episode_latency)
        all_episode_cost.extend(episode_cost)
 
        if (epoch + 1) % 5 == 0:
            torch.save(agent.actor.state_dict(), os.path.join(models_dir, f'{run_id}_actor_epoch_{epoch:04d}.pt'))
            torch.save(agent.critic.state_dict(), os.path.join(models_dir, f'{run_id}_critic_epoch_{epoch:04d}.pt'))
 
    np.savez_compressed(os.path.join(run_dir, 'training_data.npz'),
             episode_returns=np.array(all_episode_returns),
             episode_latency=np.array(all_episode_latency),
             episode_cost=np.array(all_episode_cost),
             weights_hist=np.array(weights_hist),
             L_hist_L=np.array(L_hist['L']),
             L_hist_C=np.array(L_hist['C']),
             L_hist_S=np.array(L_hist['S']))
    print(f"Saved training_data.npz ({len(all_episode_returns)} episodes)")

        with open(os.path.join(run_dir, 'meta.json'), 'w') as f:
            json.dump({
            'run_id': run_id,
                'total_epochs': total_epochs,
            'episodes_per_epoch': episodes_per_epoch,
                'dwa_weights': w.tolist()
            }, f)

    torch.save(agent.actor.state_dict(), os.path.join(models_dir, f'{run_id}_actor_final.pt'))
    torch.save(agent.critic.state_dict(), os.path.join(models_dir, f'{run_id}_critic_final.pt'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--regions', type=str, nargs='+', default=['Server2'],
                        help='Regions to train on')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for models and logs')
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    train(total_epochs=args.epochs, episodes_per_epoch=args.episodes, device=device, seed=args.seed, regions=args.regions, output_dir=args.output_dir)
