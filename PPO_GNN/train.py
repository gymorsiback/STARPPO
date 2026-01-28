import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from torch_geometric.data import Data, Batch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env import WorkflowDataset, WorkflowMoEEnv
from utils import ensure_dir, softmax, generate_run_id
from PPO_GNN.agent import PPOAgent


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


_SERVER_MODELS_CACHE = {}

def get_static_node_features(env):
    server_ids = sorted(list(env.servers.keys()))

    global _SERVER_MODELS_CACHE
    if not _SERVER_MODELS_CACHE:
        for sid in server_ids:
            _SERVER_MODELS_CACHE[sid] = [mi for mi in env.ds.model_instances if mi.server_id == sid]

    caps = np.array([env.servers[sid].normalized_compute for sid in server_ids], dtype=np.float32)
    norm_caps = caps 
    server_min_costs = []
    for sid in server_ids:
        models_on_server = _SERVER_MODELS_CACHE[sid]
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
 
    features = np.stack([norm_caps, cost_advantage], axis=1) 
    return torch.FloatTensor(features)

def get_dynamic_node_features(env, server_ids):
    current_time = env.current_time_ms
    busy_times = np.array([max(0.0, env.busy_until[sid] - current_time) for sid in server_ids], dtype=np.float32)
    norm_queues = np.clip(busy_times / 5000.0, 0.0, 1.0)
    return torch.FloatTensor(norm_queues).unsqueeze(1)


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
    batch_size=512,  
    device='cpu',
    seed=300,
    regions=None,
    output_dir=None
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)

    if regions is None:
        regions = ['Server2']

    ds = WorkflowDataset(data_root, split='train', regions=regions)
    env = WorkflowMoEEnv(ds)
    num_servers = len(env.servers)

    server_model_mapping, server_ids = build_server_model_mapping(ds, env)
   
    k_neighbors = min(20, num_servers - 1)  
    
    if num_servers > 100:
        print(f"Building k-NN sparse graph (k={k_neighbors}) for {num_servers} servers...")
        coords = np.array([[env.servers[sid].lon, env.servers[sid].lat] for sid in server_ids])
  
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
   
        src_list = []
        dst_list = []
        for i in range(num_servers):
            nearest = np.argsort(distances[i])[1:k_neighbors+1]
            for j in nearest:
                src_list.append(i)
                dst_list.append(j)
                src_list.append(j)
                dst_list.append(i)

        edges = set(zip(src_list, dst_list))
        src_list = [e[0] for e in edges]
        dst_list = [e[1] for e in edges]
        
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long).to(device)
        print(f"Sparse graph: {len(src_list)} edges (vs {num_servers**2} fully connected)")
    else:
        src_nodes = torch.arange(num_servers).repeat_interleave(num_servers)
        dst_nodes = torch.arange(num_servers).repeat(num_servers)
        edge_index = torch.stack([src_nodes, dst_nodes], dim=0).to(device)
    
    edge_attr = torch.ones((edge_index.size(1), 1), dtype=torch.float32).to(device)

    candidate_mask = torch.ones(num_servers, dtype=torch.bool, device=device)
  
    agent = PPOAgent(
        device=device,
        lr=lr,
        node_feat_dim=3,
        global_feat_dim=7,
        hidden_dim=128  
    )

    lr_lambda = lambda epoch: 1.0 - 0.8 * (epoch / total_epochs)
    optimizer_scheduler = torch.optim.lr_scheduler.LambdaLR(agent.optimizer, lr_lambda=lr_lambda)

    run_id = generate_run_id('ppo_gnn')
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if output_dir is not None:
        results_dir = output_dir
    else:
        results_dir = os.path.join(project_root, 'results', 'PPO_GNN')
    run_dir = os.path.join(results_dir, 'logs', run_id)
    models_dir = os.path.join(results_dir, 'models')
    ensure_dir(run_dir)
    ensure_dir(models_dir)

    print(f"Starting PPO_GNN Training: {run_id}")
    print(f"Device: {device}, Epochs: {total_epochs}, Seed: {seed}")

    w = np.array([0.50, 0.45, 0.05], dtype=np.float32)  
    loss_moving_avg = np.zeros(3) 
    T = 3.0
    freeze_epoch = int(total_epochs * 0.3)  
    dwa_start_epoch = 5  

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
 
            static_feats = get_static_node_features(env).to(device)
            
            traj_data = []
            traj_rewards = []
            traj_dones = []
            traj_values = []
            traj_actions = []
            traj_logprobs = []
            
            while not done:
                dynamic_feat = get_dynamic_node_features(env, server_ids).to(device)
           
                node_feats = torch.cat([static_feats[:, 0:1], dynamic_feat, static_feats[:, 1:2]], dim=1)
                
                global_feat = torch.FloatTensor(build_state_vector(state_dict, w)).unsqueeze(0).to(device)
                
                data = Data(
                    x=node_feats,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    global_feat=global_feat,
                    candidate_mask=candidate_mask,
                    num_nodes=num_servers
                )

                server_action_t, log_prob_t, value_t = agent.act(data)
                
                server_action = server_action_t.item()
                log_prob = log_prob_t.item()
                value = value_t.item()
  
                _, _, req_type = env.cur_steps[env.step_idx]
                if req_type is None:
                    req_type = env.cur_task['RequiredModelTypes'][env.step_idx]
                
                action = map_server_action_to_instance(
                    server_action, str(req_type), server_model_mapping, ds
                )
                
                next_state_dict, (rL, rC, rS), done, info = env.step(action)
                
                scalar_reward = w[0]*rL + w[1]*rC + w[2]*rS
                
                data.action_node_idx = server_action_t.squeeze() 
                data.old_log_prob = log_prob_t
                data.value = value_t
                
                traj_data.append(data)
                traj_rewards.append(scalar_reward)
                traj_dones.append(done)
                traj_values.append(value)
                traj_actions.append(server_action)
                traj_logprobs.append(log_prob)
                
                state_dict = next_state_dict
                
                ep_L_vals.append(-rL)
                ep_C_vals.append(-rC)
                ep_S_vals.append(-rS)
                
            episode_returns.append(sum(traj_rewards))
            episode_latency.append(sum(env.ep_latency))
            episode_cost.append(sum(env.ep_cost))

            next_value = 0
            returns = []
            gae = 0
            gamma = 0.99
            lam = 0.95
            
            for i in reversed(range(len(traj_rewards))):
                delta = traj_rewards[i] + gamma * next_value * (1 - traj_dones[i]) - traj_values[i]
                gae = delta + gamma * lam * gae
                ret = gae + traj_values[i]
                returns.insert(0, ret)
                next_value = traj_values[i]
    
                traj_data[i].ret = torch.tensor(ret, dtype=torch.float32, device=device)
                traj_data[i].advantage = torch.tensor(gae, dtype=torch.float32, device=device)
                traj_data[i].action = torch.tensor(traj_actions[i], dtype=torch.long, device=device)
                traj_data[i].old_log_prob = torch.tensor(traj_logprobs[i], dtype=torch.float32, device=device)

            memory_buffer.extend(traj_data)
   
        update_stats = agent.update(
            memory_buffer, 
            batch_size=batch_size, 
            ent_coef=current_entropy,
            update_iters=10
        )
        
        optimizer_scheduler.step()
        
        avg_ret = np.mean(episode_returns)
        avg_lat = np.mean(episode_latency)
        avg_cost = np.mean(episode_cost)
        
        print(f"Epoch {epoch+1}/{total_epochs} | Ret: {avg_ret:.2f} | Lat: {avg_lat:.1f} | Cost: {avg_cost:.3f} | Loss: {update_stats['loss']:.4f}")
        
        L_hist['L'].append(avg_lat)
        L_hist['C'].append(avg_cost)
        L_hist['S'].append(0.0)
        weights_hist.append(w.copy())
  
        all_episode_returns.extend(episode_returns)
        all_episode_latency.extend(episode_latency)
        all_episode_cost.extend(episode_cost)

        if (epoch+1) % 5 == 0:
             torch.save(agent.net.state_dict(), os.path.join(models_dir, f'{run_id}_model_epoch_{epoch:04d}.pt'))

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
        json.dump({'run_id': run_id, 'epochs': total_epochs, 'episodes_per_epoch': episodes_per_epoch}, f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--seed', type=int, default=300)
    parser.add_argument('--regions', type=str, nargs='+', default=['Server2'],
                        help='Regions to train on')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for models and logs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data', type=str, default='/root/autodl-tmp/MOE111/data',
                        help='Data directory, e.g., data or data1')
    parser.add_argument('--batch_size', type=int, default=512,  
                        help='Batch size for PPO updates (reduce for large graphs)')
    args = parser.parse_args()
    
    train(total_epochs=args.epochs, episodes_per_epoch=args.episodes, device=args.device, seed=args.seed, regions=args.regions, output_dir=args.output_dir, data_root=args.data, batch_size=args.batch_size)
