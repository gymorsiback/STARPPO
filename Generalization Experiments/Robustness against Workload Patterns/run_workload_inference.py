
import os
import sys
import numpy as np
import torch
import random
from typing import List, Dict, Tuple
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

from env import WorkflowDataset, WorkflowMoEEnv
from utils import haversine_km

DATA_ROOT = os.path.join(PROJECT_ROOT, 'data1')
REGION = 'Server1_Trap'
NUM_EPISODES = 200  
SEEDS = [42, 43, 44]

STAR_PPO_MODELS = {
    42: os.path.join(PROJECT_ROOT, 'results/STAR_PPO/models/LATEST_Server1_Trap_seed42_final.pt'),
    43: os.path.join(PROJECT_ROOT, 'results/STAR_PPO/models/LATEST_Server1_Trap_seed43_final.pt'),
    44: os.path.join(PROJECT_ROOT, 'results/STAR_PPO/models/LATEST_Server1_Trap_seed44_final.pt'),
}

class TrafficGenerator:
    BASE_INTERVAL = 2500.0  
    
    @staticmethod
    def generate_uniform(num_tasks: int, interval_ms: float = None) -> List[float]:
        interval = interval_ms or TrafficGenerator.BASE_INTERVAL
        return [i * interval for i in range(num_tasks)]
    
    @staticmethod
    def generate_poisson(num_tasks: int, lam: float = 0.4, seed: int = 42) -> List[float]:
        np.random.seed(seed)
        intervals = np.random.exponential(1000.0 / lam, num_tasks)
        arrival_times = np.cumsum(intervals)
        return arrival_times.tolist()
    
    @staticmethod
    def generate_bursty(num_tasks: int, burst_size: int = 10, burst_interval: float = 500.0, 
                        gap_interval: float = 20000.0, seed: int = 42) -> List[float]:
        np.random.seed(seed)
        arrival_times = []
        current_time = 0.0
        
        while len(arrival_times) < num_tasks:
            for _ in range(min(burst_size, num_tasks - len(arrival_times))):
                arrival_times.append(current_time)
                current_time += burst_interval + np.random.uniform(0, 100)
      
            current_time += gap_interval + np.random.uniform(0, 5000)
        
        return arrival_times[:num_tasks]
    
    @staticmethod
    def generate_on_off(num_tasks: int, on_duration: float = 10000.0, off_duration: float = 15000.0,
                        on_interval: float = 1000.0, seed: int = 42) -> List[float]:
        np.random.seed(seed)
        arrival_times = []
        current_time = 0.0
        is_on = True
        on_start = 0.0
        
        while len(arrival_times) < num_tasks:
            if is_on:
                while current_time - on_start < on_duration and len(arrival_times) < num_tasks:
                    arrival_times.append(current_time)
                    current_time += on_interval + np.random.uniform(0, 200)
 
                is_on = False
                current_time += off_duration
            else:
                is_on = True
                on_start = current_time
        
        return arrival_times[:num_tasks]


class TimedWorkflowEnv:
    def __init__(self, env: WorkflowMoEEnv):
        self.env = env
        self.global_time = 0.0  
        self.busy_until = {sid: 0.0 for sid in env.servers.keys()}  
    
    def reset(self, arrival_times: List[float]):
        self.global_time = 0.0
        self.busy_until = {sid: 0.0 for sid in self.env.servers.keys()}
        self.arrival_times = arrival_times
    
    def execute_step(self, task: Dict, action_idx: int, step_idx: int, 
                      prev_server_id: str, current_time: float) -> Tuple[float, str]:
        mi = self.env.actions[action_idx]
        server = self.env.servers[mi.server_id]

        if step_idx < len(self.env.cur_steps):
            _, req_id, _ = self.env.cur_steps[step_idx]
            if req_id is not None and req_id in self.env.ds.req_tokens:
                in_tok, out_tok = self.env.ds.req_tokens[req_id]
            else:
                size = float(task['TaskSize'])
                in_tok = int(0.6 * size)
                out_tok = int(0.4 * size)
        else:
            size = float(task['TaskSize'])
            in_tok = int(0.6 * size)
            out_tok = int(0.4 * size)
        tokens = in_tok + out_tok

        if prev_server_id is None:
            d_km = haversine_km(task['TaskLongitude'], task['TaskLatitude'], server.lon, server.lat)
        else:
            prev_server = self.env.servers[prev_server_id]
            d_km = haversine_km(prev_server.lon, prev_server.lat, server.lon, server.lat)
        
        network_ms = self.env._compute_channel_latency(d_km, tokens)
  
        if server.server_id in self.env.trap_server_ids and self.env.trap_latency > 0:
            if np.random.random() < self.env.trap_packet_loss_prob:
                network_ms += self.env.trap_bad_latency
            else:
                network_ms += self.env.trap_good_latency

        speed_tps = max(server.normalized_compute, 1e-6) * self.env.base_speed_tps
        compute_ms = (tokens / speed_tps) * 1000.0
  
        available_time = max(current_time, self.busy_until[server.server_id])
        queue_ms = available_time - current_time

        execution_time = network_ms + compute_ms
        self.busy_until[server.server_id] = available_time + execution_time

        step_latency = queue_ms + execution_time
        
        return step_latency, server.server_id


def run_star_ppo_inference(env: WorkflowMoEEnv, timed_env: TimedWorkflowEnv,
                           model_path: str, arrival_times: List[float],
                           tasks: List[Dict], seed: int) -> List[float]:
    from STAR_PPO.model import StarActor
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_servers = len(env.servers)
    server_ids = sorted(list(env.servers.keys()))
    
    actor = StarActor(state_dim=10, num_servers=num_servers).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    actor.load_state_dict(checkpoint)
    actor.eval()
    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
  
    caps = np.array([env.servers[sid].normalized_compute for sid in server_ids], dtype=np.float32)
    cost_mults = np.array([env.servers[sid].cost_multiplier for sid in server_ids], dtype=np.float32)
    cost_advantage = 1.0 - np.clip(cost_mults / 2.0, 0, 1.0)

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
    
    latencies = []
    timed_env.reset(arrival_times)
    timed_env.env.reset(tasks[0])  
    
    for i, (task, arrival_time) in enumerate(zip(tasks, arrival_times)):
        timed_env.env.reset(task)
   
        num_steps = len(timed_env.env.cur_steps)
        ep_latency = 0.0
        current_time = arrival_time
        prev_server_id = None
     
        for step_idx in range(num_steps):
            if step_idx < len(timed_env.env.cur_steps):
                _, _, req_type = timed_env.env.cur_steps[step_idx]
            else:
                req_type = task['RequiredModelTypes'][step_idx % len(task['RequiredModelTypes'])]
            
            available = timed_env.env.model_type_to_action_idxs.get(str(req_type), [])
            if not available:
                available = list(range(len(timed_env.env.actions)))
 
            w = [0.45, 0.40, 0.15]
            step_norm = step_idx / max(num_steps, 1)
            base_state = np.array([
                step_norm,
                task['TaskLongitude'],
                task['TaskLatitude'],
                0.0,  
                w[0], w[1], w[2]
            ], dtype=np.float32)
            aug_features = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            state_vec = np.concatenate([base_state, aug_features])
            
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)

            busy_times = np.array([max(0.0, timed_env.busy_until[sid] - current_time) 
                                   for sid in server_ids], dtype=np.float32)
            norm_queues = np.clip(busy_times / 5000.0, 0.0, 1.0)
 
            w2 = 0.30
            weights = caps / (1.0 + w2 * norm_queues)
            weights = weights * network_quality
            weights = weights * (0.5 + 0.5 * cost_advantage)
            weights = weights / (np.max(weights) + 1e-9)
            
            resource_weights = torch.FloatTensor(weights).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = actor(state_tensor, resource_weights).squeeze(0)

                valid_server_ids = set()
                server_to_action = {}
                for a_idx in available:
                    srv_id = timed_env.env.actions[a_idx].server_id
                    valid_server_ids.add(srv_id)
                    if srv_id not in server_to_action:
                        server_to_action[srv_id] = a_idx
                
                mask = torch.zeros(num_servers, device=device)
                for idx, sid in enumerate(server_ids):
                    if sid in valid_server_ids:
                        mask[idx] = 1.0
                
                masked_logits = logits + (1 - mask) * -1e9
          
                
                probs = torch.softmax(masked_logits, dim=0)
                top_k_probs, top_k_indices = torch.topk(probs, 10) 
                
                best_server_idx = -1
                best_score = -float('inf')
                
                for prob, idx in zip(top_k_probs, top_k_indices):
                    idx = idx.item()
                    sid = server_ids[idx]
    
                    queue_seconds = max(0.0, timed_env.busy_until[sid] - current_time) / 1000.0
      
                    congestion_penalty = 1.0 - np.exp(-1.5 * queue_seconds)
                    score = prob.item() * (1.0 - congestion_penalty)
                    
                    if score > best_score:
                        best_score = score
                        best_server_idx = idx
                
                selected_server = server_ids[best_server_idx]
            
            action_idx = server_to_action.get(selected_server)
            if action_idx is None:
                action_idx = random.choice(available)

            step_latency, prev_server_id = timed_env.execute_step(
                task, action_idx, step_idx, prev_server_id, current_time)
            
            ep_latency += step_latency
            current_time += step_latency
        
        latencies.append(ep_latency)
    
    return latencies


def main():
    print("=" * 70)
    print("🔬 Robustness against Workload Patterns 实验")
    print("    仅测试 STAR-PPO 在不同流量模式下的鲁棒性")
    print("=" * 70)

    print(f"\n[1] 加载数据集: {REGION}")
    ds = WorkflowDataset(DATA_ROOT, split='train', regions=[REGION])
    env = WorkflowMoEEnv(ds)
    timed_env = TimedWorkflowEnv(env)
    
    print(f"    服务器数: {len(env.servers)}")
    print(f"    任务数: {len(ds.tasks)}")

    traffic_patterns = {
        'Uniform': TrafficGenerator.generate_uniform,
        'Poisson': TrafficGenerator.generate_poisson,
        'Bursty': TrafficGenerator.generate_bursty,
        'On-Off': TrafficGenerator.generate_on_off,
    }
 
    results = {pattern: [] for pattern in traffic_patterns.keys()}
    
    tasks = ds.tasks[:NUM_EPISODES]
    
    for pattern_name, generator in traffic_patterns.items():
        print(f"\n[2] 测试流量模式: {pattern_name}")
        
        for seed in SEEDS:
            print(f"\n    Seed {seed}:")
   
            if pattern_name == 'Uniform':
                arrival_times = generator(NUM_EPISODES)
            else:
                arrival_times = generator(NUM_EPISODES, seed=seed)
   
            model_path = STAR_PPO_MODELS.get(seed)
            if model_path and os.path.exists(model_path):
                try:
                    latencies = run_star_ppo_inference(
                        env, timed_env, model_path, arrival_times, tasks, seed)
                    results[pattern_name].extend(latencies)
                    print(f"      STAR-PPO: avg={np.mean(latencies):.2f} ms, std={np.std(latencies):.2f} ms")
                except Exception as e:
                    import traceback
                    print(f"      STAR-PPO Error: {e}")
                    traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("📊 STAR-PPO 在不同流量模式下的表现")
    print("=" * 70)
    
    print(f"\n{'流量模式':<12} {'平均延迟(ms)':<15} {'标准差':<15} {'中位数':<15} {'P95':<15}")
    print("-" * 70)
    
    for pattern in traffic_patterns.keys():
        lats = results[pattern]
        if lats:
            p95 = np.percentile(lats, 95)
            print(f"{pattern:<12} {np.mean(lats):<15.2f} {np.std(lats):<15.2f} {np.median(lats):<15.2f} {p95:<15.2f}")

    baseline_mean = np.mean(results['Uniform']) if results['Uniform'] else 1.0
    print(f"\n相对于 Uniform（训练时流量模式）的性能变化：")
    for pattern in traffic_patterns.keys():
        if pattern != 'Uniform' and results[pattern]:
            pct_change = (np.mean(results[pattern]) - baseline_mean) / baseline_mean * 100
            print(f"  {pattern}: {pct_change:+.1f}%")
 
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'workload_pattern_results.npz')
    
    np.savez(output_path,
             patterns=list(traffic_patterns.keys()),
             uniform=results['Uniform'],
             poisson=results['Poisson'],
             bursty=results['Bursty'],
             on_off=results['On-Off'])
    
    print(f"\n✓ 结果已保存: {output_path}")


if __name__ == '__main__':
    main()

