import os
import json
import math
import time
import random
from typing import Dict, List, Tuple, Any

import numpy as np
import torch

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env import WorkflowDataset, WorkflowMoEEnv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent import PPOAgent
from train import build_state_vector

try:
    from results.PPO.plotting import plot_cdf
except Exception:
    plot_cdf = None
    import matplotlib.pyplot as plt
    def _ensure_dir(p: str):
        os.makedirs(p, exist_ok=True)
    def plot_cdf(latencies: dict, costs: dict, out_dir: str, title_suffix: str = ''):
        def cdf_data(vals: np.ndarray):
            vals = np.sort(vals)
            y = np.linspace(0, 1, len(vals), endpoint=False)
            return vals, y
        _ensure_dir(out_dir)
        fig, ax = plt.subplots(1,2, figsize=(12,4))
        for k, v in latencies.items():
            v = np.array(v, dtype=float)
            if v.size == 0:
                continue
            x, y = cdf_data(v)
            ax[0].plot(x, y, label=k)
        ax[0].set_xlabel('Latency per Episode (ms)')
        ax[0].set_ylabel('CDF')
        ax[0].set_title('CDF of Latency ' + title_suffix)
        ax[0].grid(alpha=0.3)
        ax[0].legend()
        for k, v in costs.items():
            v = np.array(v, dtype=float)
            if v.size == 0:
                continue
            x, y = cdf_data(v)
            ax[1].plot(x, y, label=k)
        ax[1].set_xlabel('Cost per Episode ($)')
        ax[1].set_ylabel('CDF')
        ax[1].set_title('CDF of Cost ' + title_suffix)
        ax[1].grid(alpha=0.3)
        ax[1].legend()
        ts = time.strftime('%Y%m%d_%H%M%S')
        path = os.path.join(out_dir, f'figure_E_cdf_{ts}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        latest = os.path.join(out_dir, 'figure_E_cdf_latest.png')
        try:
            if os.path.exists(latest):
                os.remove(latest)
            import shutil
            shutil.copyfile(path, latest)
        except Exception:
            pass
        plt.close(fig)


def load_model(model_path: str, device: str = 'cpu'):
    from model import Actor
 
    actor = Actor(state_dim=7, num_servers=500).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device))
    actor.eval()
  
    class MinimalAgent:
        def __init__(self, actor_net):
            self.actor = actor_net
        
        def act(self, state_vec, action_feats, valid_actions):
            state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(actor_net.device)
            with torch.no_grad():
                logits = self.actor(state_t).squeeze(0) 
                logits_np = logits.cpu().numpy()
                logits_np[~np.isin(np.arange(len(logits_np)), valid_actions)] = -1e9
                action_idx = int(np.argmax(logits_np))
            return action_idx, 0.0, 0.0, None
    
    agent = MinimalAgent(actor)
 
    action_feats = None
    
    return agent, action_feats


def eval_policy(env: WorkflowMoEEnv, ds: WorkflowDataset, agent: PPOAgent, action_feats: np.ndarray,
                tasks: List[Dict[str, Any]], weights: Tuple[float, float, float]) -> Tuple[List[float], List[float]]:
    latencies = []
    costs = []
    for task in tasks:
        state = env.reset(task)
        done = False
        ep_lat = 0.0
        ep_cost = 0.0
        while not done:
            valid = env.available_actions()
            if not valid:
                break
            state_vec = build_state_vector(state, weights)
            aidx, logp, value, probs = agent.act(state_vec, action_feats, np.array(valid, dtype=np.int64))
            next_state, (rL, rC, rS), done, info = env.step(aidx)
            state = next_state
            ep_lat += info['latency_ms']
            ep_cost += info['cost']
        latencies.append(ep_lat)
        costs.append(ep_cost)
    return latencies, costs


def eval_random(env: WorkflowMoEEnv, ds: WorkflowDataset, tasks: List[Dict[str, Any]], seed: int = 1):
    rng = random.Random(seed)
    latencies, costs = [], []
    for task in tasks:
        state = env.reset(task)
        done = False
        ep_lat = 0.0
        ep_cost = 0.0
        while not done:
            valid = env.available_actions()
            if not valid:
                break
            aidx = rng.choice(valid)
            next_state, (_, _, _), done, info = env.step(aidx)
            state = next_state
            ep_lat += info['latency_ms']
            ep_cost += info['cost']
        latencies.append(ep_lat)
        costs.append(ep_cost)
    return latencies, costs


def eval_round_robin(env: WorkflowMoEEnv, ds: WorkflowDataset, tasks: List[Dict[str, Any]]):
    ptr: Dict[str, int] = {}
    latencies, costs = [], []
    model_type_to_idxs = env.model_type_to_action_idxs
    for task in tasks:
        state = env.reset(task)
        done = False
        ep_lat = 0.0
        ep_cost = 0.0
        step_i = 0
        while not done:
            valid = env.available_actions()
            if not valid:
                break
            _, _, req_type = env.cur_steps[env.step_idx]
            req_type = str(req_type)
            pool = model_type_to_idxs.get(req_type, [])
            if not pool:
                break
            p = ptr.get(req_type, 0) % len(pool)
            aidx = pool[p]
            ptr[req_type] = (p + 1) % len(pool)
            next_state, (_, _, _), done, info = env.step(aidx)
            state = next_state
            ep_lat += info['latency_ms']
            ep_cost += info['cost']
            step_i += 1
        latencies.append(ep_lat)
        costs.append(ep_cost)
    return latencies, costs


def eval_greedy_latency(env: WorkflowMoEEnv, ds: WorkflowDataset, tasks: List[Dict[str, Any]]):
    latencies, costs = [], []
    for task in tasks:
        state = env.reset(task)
        done = False
        ep_lat = 0.0
        ep_cost = 0.0
        while not done:
            valids = env.available_actions()
            if not valids:
                break
            best_a = None
            best_lat = float('inf')
            for a in valids:
                est_lat, est_cost, est_sw = env.estimate_step(a)
                if est_lat < best_lat:
                    best_lat = est_lat
                    best_a = a
            next_state, (_, _, _), done, info = env.step(best_a)
            state = next_state
            ep_lat += info['latency_ms']
            ep_cost += info['cost']
        latencies.append(ep_lat)
        costs.append(ep_cost)
    return latencies, costs


def eval_greedy_cost(env: WorkflowMoEEnv, ds: WorkflowDataset, tasks: List[Dict[str, Any]]):
    latencies, costs = [], []
    for task in tasks:
        state = env.reset(task)
        done = False
        ep_lat = 0.0
        ep_cost = 0.0
        while not done:
            valids = env.available_actions()
            if not valids:
                break
            best_a = None
            best_cost = float('inf')
            for a in valids:
                est_lat, est_cost, est_sw = env.estimate_step(a)
                if est_cost < best_cost:
                    best_cost = est_cost
                    best_a = a
            next_state, (_, _, _), done, info = env.step(best_a)
            state = next_state
            ep_lat += info['latency_ms']
            ep_cost += info['cost']
        latencies.append(ep_lat)
        costs.append(ep_cost)
    return latencies, costs


def run_inference(data_root: str,
                  model_path: str,
                  run_dir: str = None,
                  device: str = 'cpu',
                  max_episodes: int = 500,
                  seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ds = WorkflowDataset(data_root, split='test', regions=['Server2'])
    env = WorkflowMoEEnv(ds, device=device)

    agent, action_feats = load_model(model_path, device=device)

    if run_dir is not None and os.path.exists(os.path.join(run_dir, 'meta.json')):
        files = [f for f in os.listdir(run_dir) if f.startswith('epoch_') and f.endswith('.npz')]
        files.sort()
        if files:
            data = np.load(os.path.join(run_dir, files[-1]))
            weights_hist = data['weights_hist']
            w = tuple(map(float, weights_hist[-1]))
        else:
            w = (0.5, 0.5, 0.0)
    else:
        w = (0.5, 0.5, 0.0)

    tasks = [t for t in ds.tasks if len(t['RequiredModelTypes']) >= 2]
    if max_episodes is not None:
        tasks = tasks[:max_episodes]

    lat_pp, cost_pp = eval_policy(env, ds, agent, action_feats, tasks, w)

    env2 = WorkflowMoEEnv(ds, device=device)
    lat_rand, cost_rand = eval_random(env2, ds, tasks)

    env3 = WorkflowMoEEnv(ds, device=device)
    lat_rr, cost_rr = eval_round_robin(env3, ds, tasks)

    env4 = WorkflowMoEEnv(ds, device=device)
    lat_gl, cost_gl = eval_greedy_latency(env4, ds, tasks)

    env5 = WorkflowMoEEnv(ds, device=device)
    lat_gc, cost_gc = eval_greedy_cost(env5, ds, tasks)

    results_root = os.path.join(os.path.dirname(os.path.dirname(data_root)), 'results', 'PPO')
    os.makedirs(results_root, exist_ok=True)
    out_npz = os.path.join(results_root, f"inference_{os.path.basename(model_path).replace('.pt','')}.npz")
    np.savez(out_npz,
             lat_pp=np.array(lat_pp), cost_pp=np.array(cost_pp),
             lat_rand=np.array(lat_rand), cost_rand=np.array(cost_rand),
             lat_rr=np.array(lat_rr), cost_rr=np.array(cost_rr),
             lat_gl=np.array(lat_gl), cost_gl=np.array(cost_gl),
             lat_gc=np.array(lat_gc), cost_gc=np.array(cost_gc))

    if plot_cdf is not None:
        lat_dict = {
            'PPO+DWA': lat_pp,
            'Random': lat_rand,
            'RoundRobin': lat_rr,
            'Greedy-Latency': lat_gl,
            'Greedy-Cost': lat_gc,
        }
        cost_dict = {
            'PPO+DWA': cost_pp,
            'Random': cost_rand,
            'RoundRobin': cost_rr,
            'Greedy-Latency': cost_gl,
            'Greedy-Cost': cost_gc,
        }
        plot_cdf(lat_dict, cost_dict, results_root, title_suffix='(Test Split)')

    print("\n" + "="*50)
    print("Inference Results:")
    print("="*50)
    print(f"Episodes: {len(lat_pp)}")
    print(f"Average Latency: {np.mean(lat_pp):.2f} ms (std: {np.std(lat_pp):.2f})")
    print(f"Average Cost:    ${np.mean(cost_pp):.4f} (std: ${np.std(cost_pp):.4f})")
    print("="*50)

    print('Inference done. Results saved to:', results_root)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/root/autodl-tmp/MOE111/data')
    parser.add_argument('--model_path', type=str, required=False, default=None)
    parser.add_argument('--model', type=str, default=None, help='Alias for --model_path')
    parser.add_argument('--run_dir', type=str, default=None, help='results/PPO/logs/<run_id> for weights history')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--max_episodes', type=int, default=500)
    parser.add_argument('--episodes', type=int, default=None, help='Alias for --max_episodes')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    model_path = args.model or args.model_path
    max_episodes = args.episodes or args.max_episodes
    
    run_inference(args.data_root, model_path, args.run_dir, device=args.device, max_episodes=max_episodes, seed=args.seed)

