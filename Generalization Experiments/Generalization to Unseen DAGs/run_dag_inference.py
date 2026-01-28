
import os
import sys
import numpy as np
import torch
import random
from typing import List, Dict, Tuple, Any
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

from env import WorkflowDataset, WorkflowMoEEnv
from utils import haversine_km

DATA_ROOT = os.path.join(PROJECT_ROOT, 'data1')
REGION = 'Server1_Trap'
NUM_EPISODES = 100  
SEEDS = [42, 43, 44]

STAR_PPO_MODELS = {
    42: os.path.join(PROJECT_ROOT, 'results/STAR_PPO/models/LATEST_Server1_Trap_seed42_final.pt'),
    43: os.path.join(PROJECT_ROOT, 'results/STAR_PPO/models/LATEST_Server1_Trap_seed43_final.pt'),
    44: os.path.join(PROJECT_ROOT, 'results/STAR_PPO/models/LATEST_Server1_Trap_seed44_final.pt'),
}

class DAGTopology:
    def __init__(self, name: str):
        self.name = name
    
    def get_structure(self) -> List[List[int]]:
        raise NotImplementedError
    
    def compute_latency(self, step_latencies: List[float]) -> float:
        raise NotImplementedError


class ChainTopology(DAGTopology):
    def __init__(self, num_steps: int = 5):
        super().__init__("Chain")
        self.num_steps = num_steps
    
    def get_structure(self) -> List[List[int]]:
        return [[i-1] if i > 0 else [] for i in range(self.num_steps)]
    
    def compute_latency(self, step_latencies: List[float]) -> float:
        return sum(step_latencies)


class DiamondTopology(DAGTopology):
    def __init__(self):
        super().__init__("Diamond")
        self.num_steps = 4
    
    def get_structure(self) -> List[List[int]]:
        return [
            [],      
            [0],     
            [0],     
            [1, 2]   
        ]
    
    def compute_latency(self, step_latencies: List[float]) -> float:
        lat_A = step_latencies[0]
        lat_B = step_latencies[1]
        lat_C = step_latencies[2]
        lat_D = step_latencies[3]
        return lat_A + max(lat_B, lat_C) + lat_D


class TreeTopology(DAGTopology):
    def __init__(self):
        super().__init__("Tree")
        self.num_steps = 4
    
    def get_structure(self) -> List[List[int]]:
        return [
            [],   
            [0],  
            [0],  
            [0]   
        ]
    
    def compute_latency(self, step_latencies: List[float]) -> float:
        lat_A = step_latencies[0]
        return lat_A + max(step_latencies[1:])


class HybridTopology(DAGTopology):
    def __init__(self):
        super().__init__("Hybrid")
        self.num_steps = 5
    
    def get_structure(self) -> List[List[int]]:
        return [
            [],      
            [0],     
            [0],     
            [1, 2],  
            [3]      
        ]
    
    def compute_latency(self, step_latencies: List[float]) -> float:
        lat_A = step_latencies[0]
        lat_B = step_latencies[1]
        lat_C = step_latencies[2]
        lat_D = step_latencies[3]
        lat_E = step_latencies[4]
        return lat_A + max(lat_B, lat_C) + lat_D + lat_E


class DAGEnvironment:
    def __init__(self, base_env: WorkflowMoEEnv, topology: DAGTopology):
        self.env = base_env
        self.topology = topology
        self.structure = topology.get_structure()
        self.step_latencies = []
        self.step_costs = []
        self.step_servers = []  
    
    def reset(self, task: Dict[str, Any]):
        self.env.reset(task)
        self.step_latencies = []
        self.step_costs = []
        self.step_servers = []
        return self.env._get_state()
    
    def step(self, action_idx: int, step_idx: int) -> Tuple[float, float]:
        mi = self.env.actions[action_idx]
        server = self.env.servers[mi.server_id]

        if step_idx < len(self.env.cur_steps):
            _, req_id, _ = self.env.cur_steps[step_idx]
            if req_id is not None and req_id in self.env.ds.req_tokens:
                in_tok, out_tok = self.env.ds.req_tokens[req_id]
            else:
                size = float(self.env.cur_task['TaskSize'])
                in_tok = int(0.6 * size)
                out_tok = int(0.4 * size)
        else:
            size = float(self.env.cur_task['TaskSize'])
            in_tok = int(0.6 * size)
            out_tok = int(0.4 * size)
        tokens = in_tok + out_tok

        dependencies = self.structure[step_idx]
        network_ms = 0.0
        
        if len(dependencies) == 0:
            d_km = haversine_km(
                self.env.cur_task['TaskLongitude'], 
                self.env.cur_task['TaskLatitude'],
                server.lon, server.lat
            )
            network_ms = self.env._compute_channel_latency(d_km, tokens)
        elif len(dependencies) == 1:
            prev_server_id = self.step_servers[dependencies[0]]
            prev_server = self.env.servers[prev_server_id]
            d_km = haversine_km(prev_server.lon, prev_server.lat, server.lon, server.lat)
            network_ms = self.env._compute_channel_latency(d_km, tokens)
            if prev_server_id in self.env.trap_server_ids or server.server_id in self.env.trap_server_ids:
                if self.env.trap_latency > 0:
                    if np.random.random() < self.env.trap_packet_loss_prob:
                        network_ms += self.env.trap_bad_latency
                    else:
                        network_ms += self.env.trap_good_latency
        else:
            max_network = 0.0
            for dep_idx in dependencies:
                prev_server_id = self.step_servers[dep_idx]
                prev_server = self.env.servers[prev_server_id]
                d_km = haversine_km(prev_server.lon, prev_server.lat, server.lon, server.lat)
                net_ms = self.env._compute_channel_latency(d_km, tokens)
                if prev_server_id in self.env.trap_server_ids or server.server_id in self.env.trap_server_ids:
                    if self.env.trap_latency > 0:
                        if np.random.random() < self.env.trap_packet_loss_prob:
                            net_ms += self.env.trap_bad_latency
                        else:
                            net_ms += self.env.trap_good_latency
                max_network = max(max_network, net_ms)
            network_ms = max_network

        speed_tps = max(server.normalized_compute, 1e-6) * self.env.base_speed_tps
        compute_ms = (tokens / speed_tps) * 1000.0
        step_latency = network_ms + compute_ms

        cost = (tokens / 1000.0) * mi.cost_per_token * server.cost_multiplier

        self.step_latencies.append(step_latency)
        self.step_costs.append(cost)
        self.step_servers.append(server.server_id)
        
        return step_latency, cost
    
    def get_total_latency(self) -> float:
        return self.topology.compute_latency(self.step_latencies)
    
    def get_total_cost(self) -> float:
        return sum(self.step_costs)


def run_star_ppo_inference(env: WorkflowMoEEnv, dag_env: DAGEnvironment, 
                           model_path: str, topology: DAGTopology,
                           num_episodes: int, seed: int) -> Tuple[List[float], List[float]]:
    """运行 STAR-PPO 推理"""
    from STAR_PPO.model import StarActor
    from STAR_PPO.env_augmented import AugmentedWorkflowEnv

    aug_env = AugmentedWorkflowEnv(env.ds)

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
    costs = []
    
    tasks = env.ds.tasks[:num_episodes]
    
    for task in tasks:
        aug_env.reset(task)
        dag_env.reset(task)
        
        num_steps = topology.num_steps
        
        for step_idx in range(num_steps):
            if step_idx < len(aug_env.cur_steps):
                _, _, req_type = aug_env.cur_steps[step_idx]
            else:
                req_type = task['RequiredModelTypes'][step_idx % len(task['RequiredModelTypes'])]
            
            available = aug_env.model_type_to_action_idxs.get(str(req_type), [])
            if not available:
                available = list(range(len(aug_env.actions)))

            w = [0.45, 0.40, 0.15]
            state_dict = aug_env._get_state()
            base_state = np.array([
                state_dict['step_norm'],
                state_dict['task_lon'],
                state_dict['task_lat'],
                float(state_dict['prev_region_id']),
                w[0], w[1], w[2]
            ], dtype=np.float32)
            aug_features = np.array([0.5, 0.5, 0.5], dtype=np.float32)  
            state_vec = np.concatenate([base_state, aug_features])
            
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
 
            current_time = aug_env.current_time_ms
            busy_times = np.array([max(0.0, aug_env.busy_until.get(sid, 0) - current_time) for sid in server_ids], dtype=np.float32)
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
                    srv_id = aug_env.actions[a_idx].server_id
                    valid_server_ids.add(srv_id)
                    if srv_id not in server_to_action:
                        server_to_action[srv_id] = a_idx
                
                mask = torch.zeros(num_servers, device=device)
                for idx, sid in enumerate(server_ids):
                    if sid in valid_server_ids:
                        mask[idx] = 1.0
                
                masked_logits = logits + (1 - mask) * -1e9
                server_idx = torch.argmax(masked_logits).item()
                selected_server = server_ids[server_idx]
 
            action_idx = server_to_action.get(selected_server)
            if action_idx is None:
                action_idx = random.choice(available)

            dag_env.step(action_idx, step_idx)
 
            try:
                aug_env.step(action_idx)
            except:
                aug_env.step_idx = step_idx + 1
                aug_env.prev_server_id = aug_env.actions[action_idx].server_id
        
        latencies.append(dag_env.get_total_latency())
        costs.append(dag_env.get_total_cost())
    
    return latencies, costs


def run_greedy_inference(env: WorkflowMoEEnv, dag_env: DAGEnvironment,
                         topology: DAGTopology, num_episodes: int, 
                         seed: int) -> Tuple[List[float], List[float]]:
    np.random.seed(seed)
    random.seed(seed)
    
    latencies = []
    costs = []
    
    tasks = env.ds.tasks[:num_episodes]
    
    for task in tasks:
        dag_env.reset(task)
        env.reset(task)  
        num_steps = topology.num_steps
        task_lon = task['TaskLongitude']
        task_lat = task['TaskLatitude']
        
        prev_server_id = None
        
        for step_idx in range(num_steps):
            if step_idx < len(env.cur_steps):
                _, _, req_type = env.cur_steps[step_idx]
            else:
                req_type = task['RequiredModelTypes'][step_idx % len(task['RequiredModelTypes'])]
            
            available = env.model_type_to_action_idxs.get(str(req_type), [])
            if not available:
                available = list(range(len(env.actions)))
  
            best_action = None
            best_score = float('inf')
            
            for a_idx in available:
                server = env.servers[env.actions[a_idx].server_id]
 
                if prev_server_id is None:
                    d_km = haversine_km(task_lon, task_lat, server.lon, server.lat)
                else:
                    prev_server = env.servers[prev_server_id]
                    d_km = haversine_km(prev_server.lon, prev_server.lat, server.lon, server.lat)
  
                compute_power = max(server.normalized_compute, 0.1)
                score = d_km / compute_power
                
                if score < best_score:
                    best_score = score
                    best_action = a_idx
            
            if best_action is None:
                best_action = random.choice(available)
            
            dag_env.step(best_action, step_idx)
            prev_server_id = env.actions[best_action].server_id
        
        latencies.append(dag_env.get_total_latency())
        costs.append(dag_env.get_total_cost())
    
    return latencies, costs


def run_random_inference(env: WorkflowMoEEnv, dag_env: DAGEnvironment,
                         topology: DAGTopology, num_episodes: int,
                         seed: int) -> Tuple[List[float], List[float]]:
    np.random.seed(seed)
    random.seed(seed)
    
    latencies = []
    costs = []
    
    tasks = env.ds.tasks[:num_episodes]
    
    for task in tasks:
        dag_env.reset(task)
        num_steps = topology.num_steps
        
        for step_idx in range(num_steps):
            if step_idx < len(env.cur_steps):
                _, _, req_type = env.cur_steps[step_idx]
            else:
                req_type = task['RequiredModelTypes'][step_idx % len(task['RequiredModelTypes'])]
            
            available = env.model_type_to_action_idxs.get(str(req_type), [])
            if not available:
                available = list(range(len(env.actions)))

            action_idx = random.choice(available)
            dag_env.step(action_idx, step_idx)

            env.prev_server_id = env.actions[action_idx].server_id
            if step_idx < len(env.cur_steps) - 1:
                env.step_idx = step_idx + 1
        
        latencies.append(dag_env.get_total_latency())
        costs.append(dag_env.get_total_cost())
    
    return latencies, costs


def main():
    print("=" * 70)
    print("🔬 Generalization to Unseen DAGs 实验")
    print("=" * 70)

    print(f"\n[1] 加载数据集: {REGION}")
    ds = WorkflowDataset(DATA_ROOT, split='train', regions=[REGION])
    env = WorkflowMoEEnv(ds)
    print(f"    服务器数: {len(env.servers)}")
    print(f"    任务数: {len(ds.tasks)}")

    topologies = [
        ChainTopology(num_steps=5),
        DiamondTopology(),
        TreeTopology(),
        HybridTopology()
    ]

    results = {
        'STAR_PPO': defaultdict(list),
        'Greedy': defaultdict(list),
        'Random': defaultdict(list)
    }
 
    for topo in topologies:
        print(f"\n[2] 测试拓扑: {topo.name}")
        print(f"    结构: {topo.get_structure()}")
        
        dag_env = DAGEnvironment(env, topo)
        
        for seed in SEEDS:
            print(f"\n    Seed {seed}:")
  
            if seed in STAR_PPO_MODELS and os.path.exists(STAR_PPO_MODELS[seed]):
                try:
                    lats, costs = run_star_ppo_inference(
                        env, dag_env, STAR_PPO_MODELS[seed], 
                        topo, NUM_EPISODES, seed
                    )
                    avg_lat = np.mean(lats)
                    results['STAR_PPO'][topo.name].append(avg_lat)
                    print(f"      STAR-PPO: {avg_lat:.2f} ms")
                except Exception as e:
                    print(f"      STAR-PPO Error: {e}")
                    results['STAR_PPO'][topo.name].append(np.nan)
            else:
                print(f"      STAR-PPO: 模型不存在")
                results['STAR_PPO'][topo.name].append(np.nan)
   
            env.reset(ds.tasks[0])  
            lats, costs = run_greedy_inference(env, dag_env, topo, NUM_EPISODES, seed)
            avg_lat = np.mean(lats)
            results['Greedy'][topo.name].append(avg_lat)
            print(f"      Greedy: {avg_lat:.2f} ms")
 
            env.reset(ds.tasks[0])
            lats, costs = run_random_inference(env, dag_env, topo, NUM_EPISODES, seed)
            avg_lat = np.mean(lats)
            results['Random'][topo.name].append(avg_lat)
            print(f"      Random: {avg_lat:.2f} ms")

    print("\n" + "=" * 70)
    print("📊 实验结果汇总")
    print("=" * 70)
    
    summary = {}
    for algo in ['STAR_PPO', 'Greedy', 'Random']:
        summary[algo] = {}
        for topo in topologies:
            vals = results[algo][topo.name]
            valid_vals = [v for v in vals if not np.isnan(v)]
            if valid_vals:
                summary[algo][topo.name] = np.mean(valid_vals)
            else:
                summary[algo][topo.name] = np.nan
    
    print(f"\n{'拓扑':<12} {'STAR-PPO':<15} {'Greedy':<15} {'Random':<15} {'改进 vs Greedy':<15}")
    print("-" * 70)
    for topo in topologies:
        star = summary['STAR_PPO'][topo.name]
        greedy = summary['Greedy'][topo.name]
        rand = summary['Random'][topo.name]
        
        if not np.isnan(star) and not np.isnan(greedy):
            improvement = (greedy - star) / greedy * 100
            print(f"{topo.name:<12} {star:<15.2f} {greedy:<15.2f} {rand:<15.2f} {improvement:>+.2f}%")
        else:
            print(f"{topo.name:<12} {'N/A':<15} {greedy:<15.2f} {rand:<15.2f} N/A")

    output_dir = os.path.dirname(os.path.abspath(__file__))
    result_path = os.path.join(output_dir, 'dag_generalization_results.npz')
    
    np.savez(result_path,
             topologies=[t.name for t in topologies],
             star_ppo=np.array([summary['STAR_PPO'][t.name] for t in topologies]),
             greedy=np.array([summary['Greedy'][t.name] for t in topologies]),
             random=np.array([summary['Random'][t.name] for t in topologies]))
    
    print(f"\n✓ 结果已保存: {result_path}")


if __name__ == '__main__':
    main()

