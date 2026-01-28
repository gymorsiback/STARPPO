import os
import sys
import argparse
import random
import numpy as np
import torch
import time
import re
import multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env import WorkflowDataset, WorkflowMoEEnv
from utils import ensure_dir, softmax, generate_run_id
from A3C_algorithm.model import ActorCritic


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
    if isinstance(dwa_weights, torch.Tensor):
        w = dwa_weights.tolist()
    else:
        w = dwa_weights
        
    return np.array([
        state_dict['step_norm'],
        state_dict['task_lon'],
        state_dict['task_lat'],
        float(state_dict['prev_region_id']),
        w[0],
        w[1],
        w[2]
    ], dtype=np.float32)


class SharedAdam(optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


def worker(rank, global_model, optimizer, global_ep_idx, global_ep_lock, res_queue, 
           total_episodes, data_root, device, dwa_weights_tensor, seed, regions):

    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

    if regions is None:
        regions = ['Server2']

    ds = WorkflowDataset(data_root, split='train', regions=regions)
    env = WorkflowMoEEnv(ds)
    num_servers = len(env.servers)
    server_model_mapping, _ = build_server_model_mapping(ds, env)

    local_model = ActorCritic(state_dim=7, num_servers=num_servers).to(device)
    local_model.train()
    
    gamma = 0.99
    entropy_coef = 0.01
    max_grad_norm = 0.5
    update_freq = 20
    
    while True:
        with global_ep_lock:
            if global_ep_idx.value >= total_episodes:
                break
            global_ep_idx.value += 1

        local_model.load_state_dict(global_model.state_dict())
        
        w = dwa_weights_tensor.clone().cpu().numpy()
        
        task = random.choice(ds.tasks)
        state_dict = env.reset(task)
        done = False
        
        values, log_probs, rewards, entropies = [], [], [], []
        raw_L_list, raw_C_list, raw_S_list = [], [], []
        ep_total_reward = 0
        
        step_count = 0
        
        while not done:
            step_count += 1
            s_vec = build_state_vector(state_dict, w)
            s_tensor = torch.FloatTensor(s_vec).unsqueeze(0).to(device)
            
            logits, value = local_model(s_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            
            server_action = action.item()
            _, _, req_type = env.cur_steps[env.step_idx]
            if req_type is None: req_type = env.cur_task['RequiredModelTypes'][env.step_idx]
            real_action = map_server_action_to_instance(server_action, str(req_type), server_model_mapping, ds)
            
            next_state_dict, (rL, rC, rS), done, info = env.step(real_action)
            
            scalar_reward = w[0] * rL + w[1] * rC + w[2] * rS
            ep_total_reward += scalar_reward
            
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(scalar_reward)
            entropies.append(entropy)
            
            raw_L_list.append(-rL)
            raw_C_list.append(-rC)
            raw_S_list.append(-rS)
            
            state_dict = next_state_dict
            
            if done or len(rewards) >= update_freq:
                R = 0
                if not done:
                    ns_vec = build_state_vector(next_state_dict, w)
                    ns_tensor = torch.FloatTensor(ns_vec).unsqueeze(0).to(device)
                    _, next_val = local_model(ns_tensor)
                    R = next_val.item()
                
                batch_returns = []
                for r in reversed(rewards):
                    R = r + gamma * R
                    batch_returns.insert(0, R)

                batch_returns = torch.FloatTensor(batch_returns).to(device)
                batch_values = torch.cat(values)
                batch_log_probs = torch.cat(log_probs)
                batch_entropies = torch.cat(entropies)
                
                advantage = batch_returns - batch_values
                actor_loss = -(batch_log_probs * advantage.detach()).mean()
                critic_loss = advantage.pow(2).mean()
                total_loss = actor_loss + 0.5 * critic_loss - entropy_coef * batch_entropies.mean()
                
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_grad_norm)
 
                for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
                    if global_param.grad is None:
                        global_param.grad = local_param.grad.cpu()
                    else:
                        global_param.grad = local_param.grad.cpu()
                        
                optimizer.step()
                
                values, log_probs, rewards, entropies = [], [], [], []
  
        ep_latency = sum(env.ep_latency) if hasattr(env, 'ep_latency') else 0
        ep_cost = sum(env.ep_cost) if hasattr(env, 'ep_cost') else 0
                
        res_queue.put({
            'ep_return': ep_total_reward,
            'ep_latency': ep_latency,
            'ep_cost': ep_cost,
            'ep_switch': 0,  
            'mean_L': np.mean(raw_L_list) if raw_L_list else 0,
            'mean_C': np.mean(raw_C_list) if raw_C_list else 0,
            'mean_S': np.mean(raw_S_list) if raw_S_list else 0
        })


def train(data_root, total_epochs, episodes_per_epoch, seed, device_str='cuda', regions=None, output_dir=None):
    if device_str == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        device_str = 'cpu'

    if regions is None:
        regions = ['Server2']
 
    mp.set_start_method('spawn', force=True)
    
    total_episodes = total_epochs * episodes_per_epoch
    num_processes = 8

    ds_dummy = WorkflowDataset(data_root, split='train', regions=regions)
    env_dummy = WorkflowMoEEnv(ds_dummy)
    num_servers = len(env_dummy.servers)
    
    global_model = ActorCritic(state_dim=7, num_servers=num_servers).to('cpu')
    global_model.share_memory()
    
    optimizer = SharedAdam(global_model.parameters(), lr=1e-4)
    
    global_ep_idx = mp.Value('i', 0)
    global_ep_lock = mp.Lock()
    res_queue = mp.Queue()
    dwa_weights_tensor = torch.FloatTensor([0.45, 0.40, 0.15]).share_memory_()
    
    loss_moving_avg = np.zeros(3)
    T = 3.0
    dwa_start_epoch = 3
    freeze_epoch = int(total_epochs * 0.8)

    run_id = generate_run_id('a3c')
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if output_dir is not None:
        results_dir = output_dir
    else:
        results_dir = os.path.join(project_root, 'results', 'A3C_algorithm')
    run_dir = os.path.join(results_dir, 'logs', run_id)
    models_dir = os.path.join(results_dir, 'models')
    ensure_dir(run_dir)
    ensure_dir(models_dir)
    
    print(f"Starting A3C Training: {run_id}")
    print(f"Device for workers: {device_str} | Global Model: CPU Shared")
    
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=worker, args=(
            rank, global_model, optimizer, global_ep_idx, global_ep_lock, res_queue,
            total_episodes, data_root, device_str, dwa_weights_tensor, seed, regions
        ))
        p.start()
        processes.append(p)
        
    ep_rewards = []
    ep_latencies = []
    ep_costs = []

    all_episode_returns = []
    all_episode_latency = []
    all_episode_cost = []
    weights_hist = []
    L_hist = {'L': [], 'C': [], 'S': []}
    
    current_epoch = 0
    episodes_in_epoch = 0
    
    epoch_mean_L = []
    epoch_mean_C = []
    epoch_mean_S = []
    
    while global_ep_idx.value < total_episodes or not res_queue.empty():
        try:
            if not any(p.is_alive() for p in processes) and res_queue.empty():
                break
                
            res = res_queue.get(timeout=1)
            
            ep_rewards.append(res['ep_return'])
            ep_latencies.append(res['ep_latency'])
            ep_costs.append(res['ep_cost'])
            
            epoch_mean_L.append(res['mean_L'])
            epoch_mean_C.append(res['mean_C'])
            epoch_mean_S.append(res['mean_S'])
            
            episodes_in_epoch += 1
            
            if episodes_in_epoch >= episodes_per_epoch:
                avg_ret = np.mean(ep_rewards)
                avg_lat = np.mean(ep_latencies)
                avg_cost = np.mean(ep_costs)
                current_weights = dwa_weights_tensor.clone().cpu().numpy()
                
                print(f"Epoch {current_epoch+1}/{total_epochs} | Ret: {avg_ret:.2f} | Lat: {avg_lat:.1f}ms | Cost: ${avg_cost:.4f} | DWA: {current_weights}")

                all_episode_returns.extend(ep_rewards)
                all_episode_latency.extend(ep_latencies)
                all_episode_cost.extend(ep_costs)
                weights_hist.append(current_weights.copy())
                L_hist['L'].append(avg_lat)
                L_hist['C'].append(avg_cost)
                L_hist['S'].append(0.0)
     
                if current_epoch >= dwa_start_epoch and current_epoch < freeze_epoch:
                    current_losses = np.array([
                        np.mean(epoch_mean_L),
                        np.mean(epoch_mean_C),
                        np.mean(epoch_mean_S)
                    ])
                    
                    if np.all(loss_moving_avg == 0):
                        loss_moving_avg = current_losses + 1e-6
                    else:
                        loss_moving_avg = 0.15 * current_losses + 0.85 * loss_moving_avg
                    
                    if np.mean(np.abs(current_losses)) > 1e-5:
                        r_n = current_losses / (loss_moving_avg + 1e-7)
                        r_n = np.clip(r_n, 0.7, 1.3)
                        exp_w = np.exp(r_n / T)
                        w_k = 3 * exp_w / (np.sum(exp_w) + 1e-8)
                        w_new = softmax(w_k)
                        w_next = 0.3 * w_new + 0.7 * current_weights
                        min_weight = 0.15
                        w_next = np.clip(w_next, min_weight, 1.0 - 2*min_weight)
                        w_next = w_next / np.sum(w_next)
                        dwa_weights_tensor[:] = torch.from_numpy(w_next)

                if (current_epoch + 1) % 5 == 0:
                     torch.save(global_model.state_dict(), os.path.join(models_dir, f'{run_id}_actor_epoch_{current_epoch:04d}.pt'))
                
                ep_rewards = []
                ep_latencies = []
                ep_costs = []
                epoch_mean_L = []
                epoch_mean_C = []
                epoch_mean_S = []
                episodes_in_epoch = 0
                current_epoch += 1
                
                if current_epoch >= total_epochs:
                    break
        except:
            continue
            
    for p in processes:
        p.terminate()
        p.join()

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
        
    torch.save(global_model.state_dict(), os.path.join(models_dir, f'{run_id}_actor_final.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--regions', type=str, nargs='+', default=['Server2'],
                        help='Regions to train on')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for models and logs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data', type=str, default='/root/autodl-tmp/MOE111/data',
                        help='Data directory, e.g., data or data1')
    args = parser.parse_args()
    
    train(args.data, args.epochs, args.episodes, args.seed, args.device,
          regions=args.regions, output_dir=args.output_dir)
