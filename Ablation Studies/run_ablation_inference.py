
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env import WorkflowDataset, WorkflowMoEEnv
from STAR_PPO.model import StarActor
from utils import haversine_km


DATA_ROOT = '/root/autodl-tmp/MOE111/data1'
TARGET_REGION = 'Server1_Trap'
EPISODES = 200  
SEEDS = [42, 43, 44]  
NUM_SERVERS = 500


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

def load_full_model_results(ablation_results_path):
    """从已有的消融推理结果加载 Full Model 数据（不重新推理）"""
    print("Loading Full Model results from existing ablation inference file...")
    
    if os.path.exists(ablation_results_path):
        d = np.load(ablation_results_path)
        if 'full_avg_latencies' in d and 'full_avg_costs' in d:
            lats = d['full_avg_latencies']
            costs = d['full_avg_costs']
            print(f"  Found {len(lats)} seeds in file")
            seed_names = [42, 43, 44]
            for i, (lat, cost) in enumerate(zip(lats, costs)):
                if i < len(seed_names):
                    print(f"  Seed {seed_names[i]}: Lat={lat:.1f}ms, Cost={cost:.4f}")
            print(f"  Average: Lat={np.mean(lats):.2f}ms, Cost={np.mean(costs):.4f}")
            return {
                'avg_latencies': lats,
                'avg_costs': costs
            }
    
    print("  [WARNING] Full Model data not found in ablation results file")
    return {
        'avg_latencies': np.array([]),
        'avg_costs': np.array([])
    }


def run_ablation_inference(env, ds, model_path, device, episodes, ablation_mode='full'):
    actor = StarActor(state_dim=10, num_servers=NUM_SERVERS).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    actor.eval()
    
    w = np.array([0.45, 0.40, 0.15], dtype=np.float32)
    server_ids = sorted(list(env.servers.keys()))

    disable_network = (ablation_mode == 'no_topology')
    
    latencies = []
    total_costs = []
    
    for i in range(episodes):
        task = ds.tasks[i % len(ds.tasks)]
        state_dict = env.reset(task)
        
        ep_lat = 0
        ep_cost = 0
        done = False
        
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

            base_state = np.array([
                state_dict['step_norm'],
                state_dict['task_lon'],
                state_dict['task_lat'],
                float(state_dict['prev_region_id']),
                w[0], w[1], w[2]
            ], dtype=np.float32)

            aug_features = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            state_vec = np.concatenate([base_state, aug_features])

            if ablation_mode == 'no_workflow':
                state_vec[0] = 0.0
            elif ablation_mode == 'no_topology':
                state_vec[1] = 0.0  
                state_vec[2] = 0.0  
                state_vec[3] = 0.0  
                state_vec[7] = 0.0  
                state_vec[8] = 0.0  
                state_vec[9] = 0.0  
            weights = compute_resource_weights(env, dwa_weights=w, 
                                              disable_network_quality=disable_network)
            
            with torch.no_grad():
                state_t = torch.from_numpy(state_vec).unsqueeze(0).to(device)
                weights_t = torch.from_numpy(weights).unsqueeze(0).to(device)
                logits = actor(state_t, weights_t).squeeze(0)

                mask = torch.zeros(NUM_SERVERS, device=device)
                for idx, sid in enumerate(server_ids):
                    if sid in valid_server_ids:
                        mask[idx] = 1.0
                
                masked_logits = logits + (1 - mask) * -1e9
                server_idx = torch.argmax(masked_logits).item()
                
                selected_sid = server_ids[server_idx]
                action = server_to_action[selected_sid]
            
            state_dict, (rL, rC, rS), done, info = env.step(action)
            
            ep_lat += info['latency_ms']
            ep_cost += info['cost']
        
        latencies.append(ep_lat)
        total_costs.append(ep_cost)
    
    return {
        'latencies': np.array(latencies),
        'costs': np.array(total_costs),
        'avg_latency': np.mean(latencies),
        'avg_cost': np.mean(total_costs)
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Episodes: {EPISODES}")
    print()
  
    ds = WorkflowDataset(DATA_ROOT, split='train', regions=[TARGET_REGION])
    env = WorkflowMoEEnv(ds)
    
    print(f"Environment: {TARGET_REGION}, Servers: {len(env.servers)}")
    print()
    
    ablation_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(ablation_dir, 'results')
    ablation_results_path = os.path.join(results_dir, 'ablation_inference_results.npz')

    all_results = {}
 
    all_results['full'] = load_full_model_results(ablation_results_path)
    print()

    ablation_modes = ['no_workflow', 'no_future', 'no_topology']
    
    for mode in ablation_modes:
        all_results[mode] = {
            'avg_latencies': [],
            'avg_costs': []
        }
        
        print(f"Running inference for: {mode}")
        
        for seed in SEEDS:
            model_path = os.path.join(results_dir, f'{mode}_seed{seed}', 'models', 
                                     f'{mode}_actor_final.pt')
            
        if not os.path.exists(model_path):
                print(f"  [WARNING] Model not found: {model_path}")
            continue
        
            results = run_ablation_inference(
                env=env,
                ds=ds,
            model_path=model_path,
                device=device,
                episodes=EPISODES,
                ablation_mode=mode
            )
            
            all_results[mode]['avg_latencies'].append(results['avg_latency'])
            all_results[mode]['avg_costs'].append(results['avg_cost'])
            
            print(f"  Seed {seed}: Lat={results['avg_latency']:.1f}ms, Cost={results['avg_cost']:.4f}")
 
        all_results[mode]['avg_latencies'] = np.array(all_results[mode]['avg_latencies'])
        all_results[mode]['avg_costs'] = np.array(all_results[mode]['avg_costs'])
        print()

    save_data = {}
    modes = ['full', 'no_workflow', 'no_future', 'no_topology']
    for mode in modes:
        if len(all_results[mode]['avg_latencies']) > 0:
            save_data[f'{mode}_avg_latencies'] = all_results[mode]['avg_latencies']
            save_data[f'{mode}_avg_costs'] = all_results[mode]['avg_costs']
    
    output_path = os.path.join(results_dir, 'ablation_inference_results.npz')
    np.savez(output_path, **save_data)

    print("=" * 80)
    print("ABLATION INFERENCE RESULTS SUMMARY")
    print("=" * 80)
    
    base_lat = np.mean(all_results['full']['avg_latencies']) if len(all_results['full']['avg_latencies']) > 0 else 0
    base_cost = np.mean(all_results['full']['avg_costs']) if len(all_results['full']['avg_costs']) > 0 else 0
    
    print(f"{'Variant':<20} {'AvgLatency (ms)':<18} {'AvgCost ($)':<15} {'Δ Latency':<15} {'Δ Cost':<12}")
    print("-" * 80)
    
    for mode in modes:
        if len(all_results[mode]['avg_latencies']) > 0:
            avg_lat = np.mean(all_results[mode]['avg_latencies'])
            avg_cost = np.mean(all_results[mode]['avg_costs'])
            
            if mode == 'full':
                delta_lat = "-"
                delta_cost = "-"
            else:
                delta_lat = f"+{((avg_lat - base_lat) / base_lat * 100):.1f}%"
                delta_cost = f"{((avg_cost - base_cost) / base_cost * 100):+.1f}%"
            
            print(f"{mode:<20} {avg_lat:<18.2f} {avg_cost:<15.4f} {delta_lat:<15} {delta_cost:<12}")
    
    print("=" * 80)
    print(f"Results saved to: {output_path}")
    print(f"\n推理时长估算：3个消融变体 × 3个种子 × 200 episodes = 1800 episodes")
    print(f"在GPU上预计需要 5-10 分钟")


if __name__ == '__main__':
    main()
