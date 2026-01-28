
import os
import csv
import math
import re
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np

try:
    from utils import haversine_km, parse_required_models
except ImportError:
    from .utils import haversine_km, parse_required_models



def load_csv(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def discover_regions(data_root: str) -> List[str]:
    regions = []
    for name in os.listdir(data_root):
        full = os.path.join(data_root, name)
        if os.path.isdir(full) and name.startswith('Server'):
            regions.append(name)
    regions.sort()
    return regions


@dataclass
class Server:
    server_id: str
    region: str
    lon: float
    lat: float
    normalized_compute: float
    cost_multiplier: float = 1.0  


@dataclass
class ModelInstance:
    idx: int
    model_instance_id: str
    model_type: str
    server_id: str
    cost_per_token: float


class WorkflowDataset:
    def __init__(self, data_root: str, split: str = 'train', regions: List[str] = None):
        self.data_root = data_root
        self.split = split
        if regions is not None:
            self.regions = regions
        else:
            self.regions = discover_regions(data_root)
        self.servers: Dict[str, Server] = {}
        self.trap_server_ids: set = set()  
        self.trap_latency: float = 0.0  
        for region in self.regions:
            trap_config_path = os.path.join(data_root, region, 'trap_config.json')
            trap_server_ids = set()
            trap_cost_mult = None  
            if os.path.exists(trap_config_path):
                import json
                with open(trap_config_path, 'r') as f:
                    trap_config = json.load(f)
                trap_server_ids = set(trap_config.get('trap_server_ids', []))
                trap_cost_mult = trap_config.get('trap_cost_multiplier', None)  
                self.trap_server_ids.update(trap_server_ids)
                self.trap_latency = trap_config.get('trap_latency', 500.0)
                cost_info = f"cost_mult={trap_cost_mult}" if trap_cost_mult else "成本不变"
                print(f"[ENV] 检测到陷阱配置: {len(trap_server_ids)} 个陷阱服务器, {cost_info}")
            
            srows = load_csv(os.path.join(data_root, region, 'servers.csv'))
            for r in srows:
                sid = r['ServerID']
                
                lon = float(r['Longitude'])
                lat = float(r['Latitude'])
                norm_compute = float(r.get('NormalizedCompute', '1.0'))
   
                compute_factor = 3.5 - (norm_compute - 0.25) * 4.0  
                hash_val = (lon * 1000 + lat * 100) % 100
                random_factor = 0.8 + (hash_val / 100) * 0.4  
                
                cost_mult = compute_factor * random_factor
                cost_mult = max(0.3, min(3.5, cost_mult))
         
                if sid in trap_server_ids and trap_cost_mult is not None:
                    cost_mult = trap_cost_mult
                
                self.servers[sid] = Server(
                    server_id=sid,
                    region=region,
                    lon=lon,
                    lat=lat,
                    normalized_compute=float(r.get('NormalizedCompute', '1.0')),
                    cost_multiplier=cost_mult,
                )
        self.model_instances: List[ModelInstance] = []
        idx = 0
        for region in self.regions:
            mrows = load_csv(os.path.join(data_root, region, 'model_instances.csv'))
            for r in mrows:
                mt = r.get('ModelTypeName') or r.get('ModelType')
                if mt is None:
                    mt = 'Unknown'
                mi = ModelInstance(
                    idx=idx,
                    model_instance_id=r['ModelInstanceID'],
                    model_type=str(mt),
                    server_id=r['ServerID'],
                    cost_per_token=float(r['CostPerToken']),
                )
                self.model_instances.append(mi)
                idx += 1
        self.num_actions = len(self.model_instances)
        self.tasks: List[Dict[str, Any]] = []
        for region in self.regions:
            tpath = os.path.join(data_root, region, 'tasks.csv')
            if not os.path.exists(tpath):
                continue
            trows = load_csv(tpath)
            for r in trows:
                if r['Split'] != split:
                    continue
                r['RequiredModelTypes'] = parse_required_models(r['RequiredModelTypes'])
                r['TaskLongitude'] = float(r['TaskLongitude'])
                r['TaskLatitude'] = float(r['TaskLatitude'])
                r['TaskSize'] = float(r['TaskSize'])
                if len(r['RequiredModelTypes']) >= 2:
                    self.tasks.append(r)
        self.req_tokens: Dict[str, Tuple[int, int]] = {}
        for region in self.regions:
            rlog_path = os.path.join(data_root, region, 'request_log.csv')
            if os.path.exists(rlog_path):
                with open(rlog_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for rr in reader:
                        rid = rr['RequestID']
                        try:
                            in_tok = int(rr['RequestLengthTokens'])
                            out_tok = int(rr['ResponseLengthTokens'])
                            self.req_tokens[rid] = (in_tok, out_tok)
                        except Exception:
                            continue
        self.task_to_requests: Dict[str, List[Tuple[int, str, str]]] = {}
        for region in self.regions:
            mpath = os.path.join(data_root, region, 'task_request_mapping.csv')
            if os.path.exists(mpath):
                mrows = load_csv(mpath)
                for r in mrows:
                    tid = r['TaskID']
                    step = int(r['WorkflowStepIndex'])
                    req_id = r['MappedRequestID']
                    req_type = r.get('RequiredModelType', '')
                    self.task_to_requests.setdefault(tid, []).append((step, req_id, req_type))
        for k in self.task_to_requests:
            self.task_to_requests[k].sort(key=lambda x: x[0])
        base_latencies = []
        self.link_latency: Dict[Tuple[str, str], float] = {}  
        self.link_bandwidth: Dict[Tuple[str, str], float] = {}  
        
        for region in self.regions:
            npath = os.path.join(data_root, region, 'network_links.csv')
            if os.path.exists(npath):
                nrows = load_csv(npath)
                for r in nrows:
                    try:
                        src = r.get('SrcServerID') or r.get('SourceServerID')
                        dst = r.get('DstServerID') or r.get('DestServerID')
                        latency = float(r.get('BaseLatencyMs', 0.5))
                        bandwidth = float(r.get('BandwidthMbps') or r.get('LinkBandwidth', 1000))
                        
                        if src and dst:
                            self.link_latency[(src, dst)] = latency
                            self.link_bandwidth[(src, dst)] = bandwidth
                            self.link_latency[(dst, src)] = latency
                            self.link_bandwidth[(dst, src)] = bandwidth
                        
                        base_latencies.append(latency)
                    except Exception:
                        pass
        
        self.mean_base_latency_ms = float(np.mean(base_latencies)) if base_latencies else 10.0
        print(f"[Dataset] 加载链路: {len(self.link_latency)//2} 条, 平均延迟: {self.mean_base_latency_ms:.2f}ms")



class WorkflowMoEEnv:
    def __init__(self,
                 dataset: WorkflowDataset,
                 base_speed_tps: float = 2000.0,
                 network_ms_per_km: float = 0.05,
                 lambda_switch_scale: float = 1.0,
                 intra_region_ms: float = 1.0,
                 device: str = 'cpu'):
        self.ds = dataset
        self.base_speed_tps = base_speed_tps
        self.network_ms_per_km = network_ms_per_km
        self.intra_region_ms = intra_region_ms
        self.device = device
        self.lambda_switch = lambda_switch_scale * self.ds.mean_base_latency_ms
   
        self.link_latency = dataset.link_latency
        self.link_bandwidth = dataset.link_bandwidth
        self.trap_server_ids = dataset.trap_server_ids  
        self.trap_latency = dataset.trap_latency 
        self.trap_packet_loss_prob = 0.50  
        self.trap_good_latency = 100.0     
        self.trap_bad_latency = 200.0     
        self.actions: List[ModelInstance] = dataset.model_instances
        self.model_type_to_action_idxs: Dict[str, List[int]] = {}
        for a in self.actions:
            self.model_type_to_action_idxs.setdefault(a.model_type, []).append(a.idx)
        self.servers = dataset.servers

        self.cur_task = None
        self.cur_steps: List[Tuple[int, str, str]] = []
        self.step_idx = 0
        self.prev_server_id = None
        self.current_time_ms = 0.0
        self.busy_until: Dict[str, float] = {sid: 0.0 for sid in self.servers.keys()}
        self.ep_latency: List[float] = []
        self.ep_cost: List[float] = []
        self.ep_switches: int = 0

        self.server_latency_history: Dict[str, float] = {
            sid: self.ds.mean_base_latency_ms * 5.0 for sid in self.servers.keys()
        }

    def _compute_channel_latency(self, dist_km: float, tokens: int) -> float:
        prop_delay_ms = dist_km * 0.005 
        data_bits = tokens * 32 
        bandwidth_hz = 20e6 * random.uniform(0.25, 1.0)

        tx_power_dbm = 23.0
        tx_power_watts = 10 ** (tx_power_dbm / 10.0) / 1000.0
  
        dist_km = max(dist_km, 0.1) 
        path_loss_linear = (0.1 / dist_km) ** 2.5

        fading_gain = random.expovariate(1.0)
 
        noise_floor_dbm = -174 + 10 * math.log10(bandwidth_hz) + 10
        noise_watts = 10 ** (noise_floor_dbm / 10.0) / 1000.0
  
        rx_power_watts = tx_power_watts * path_loss_linear * fading_gain
  
        interference_watts = noise_watts * random.uniform(0, 10)
        sinr = rx_power_watts / (noise_watts + interference_watts + 1e-20)

        capacity_bps = bandwidth_hz * math.log2(1 + sinr)
  
        capacity_bps = max(capacity_bps, 50000.0)
  
        tx_time_ms = (data_bits / capacity_bps) * 1000.0
 
        total_latency_ms = prop_delay_ms + tx_time_ms + 5.0
        
        return total_latency_ms

    def reset(self, task: Dict[str, Any]):
        self.cur_task = task
        tid = task['TaskID']
        self.cur_steps = self.ds.task_to_requests.get(tid, [])
        if not self.cur_steps:
            self.cur_steps = [(i, None, mt) for i, mt in enumerate(task['RequiredModelTypes'])]
        self.step_idx = 0
        self.prev_server_id = None
        self.current_time_ms = 0.0
        self.busy_until = {sid: 0.0 for sid in self.servers.keys()}
        self.ep_latency = []
        self.ep_cost = []
        self.ep_switches = 0
        self.server_latency_history = {
            sid: self.ds.mean_base_latency_ms * 5.0 for sid in self.servers.keys()
        }
        return self._get_state()

    def _get_state(self) -> Dict[str, Any]:
        total_steps = max(len(self.cur_steps), 1)
        step_norm = self.step_idx / total_steps
        task_lon = float(self.cur_task['TaskLongitude'])
        task_lat = float(self.cur_task['TaskLatitude'])
        prev_region = 0
        if self.prev_server_id is not None:
            region_str = self.servers[self.prev_server_id].region
            match = re.search(r'\d+', region_str)
            prev_region = int(match.group()) if match else 0

        state = {
            'step_norm': step_norm,
            'total_steps': total_steps,
            'task_lon': task_lon,
            'task_lat': task_lat,
            'prev_region_id': prev_region,
            'num_regions': len(self.ds.regions),
            'server_latency_history': self.server_latency_history.copy()
        }
        return state

    def available_actions(self) -> List[int]:
        _, _, req_type = self.cur_steps[self.step_idx]
        if req_type is None:
            req_type = self.cur_task['RequiredModelTypes'][self.step_idx]
        return self.model_type_to_action_idxs.get(str(req_type), [])

    def estimate_step(self, action_idx: int) -> Tuple[float, float, float]:
        mi = self.actions[action_idx]
        server = self.servers[mi.server_id]
        _, req_id, _ = self.cur_steps[self.step_idx]
        if req_id is not None and req_id in self.ds.req_tokens:
            in_tok, out_tok = self.ds.req_tokens[req_id]
        else:
            size = float(self.cur_task['TaskSize'])
            in_tok = int(0.6 * size)
            out_tok = int(0.4 * size)
        tokens = in_tok + out_tok

        d_km = 0.0
        link_latency_ms = 0.0
        
        if self.step_idx == 0:
            d_km = haversine_km(self.cur_task['TaskLongitude'], self.cur_task['TaskLatitude'], server.lon, server.lat)
            if self.trap_latency > 0 and server.server_id in self.trap_server_ids:
                link_latency_ms = (self.trap_packet_loss_prob * self.trap_bad_latency + 
                                  (1 - self.trap_packet_loss_prob) * self.trap_good_latency)
        else:
            if self.prev_server_id is not None:
                prev_server = self.servers[self.prev_server_id]
                d_km = haversine_km(prev_server.lon, prev_server.lat, server.lon, server.lat)
                if prev_server.region == server.region:
                     d_km = max(d_km, 0.5)
  
                involves_trap = (self.prev_server_id in self.trap_server_ids or 
                                server.server_id in self.trap_server_ids)
                
                if involves_trap and self.trap_latency > 0:
                    link_latency_ms = (self.trap_packet_loss_prob * self.trap_bad_latency + 
                                      (1 - self.trap_packet_loss_prob) * self.trap_good_latency)
                else:
                    link_key = (self.prev_server_id, server.server_id)
                    if link_key in self.link_latency:
                        link_latency_ms = self.link_latency[link_key]
            else:
                d_km = 1.0
                
        network_ms = self._compute_channel_latency(d_km, tokens) + link_latency_ms

        speed_tps = max(server.normalized_compute, 1e-6) * self.base_speed_tps
        compute_ms = (tokens / speed_tps) * 1000.0
        queue_ms = max(0.0, self.busy_until[server.server_id] - self.current_time_ms)
        step_latency_ms = network_ms + compute_ms + queue_ms
        cost = (tokens / 1000.0) * mi.cost_per_token * server.cost_multiplier
        switch_penalty_ms = 0.0
        if self.prev_server_id is not None:
            prev_region = self.servers[self.prev_server_id].region
            if prev_region != server.region:
                switch_penalty_ms = self.lambda_switch
        return step_latency_ms, cost, switch_penalty_ms

    def step(self, action_idx: int) -> Tuple[Dict[str, Any], Tuple[float, float, float], bool, Dict[str, Any]]:
        mi = self.actions[action_idx]
        server = self.servers[mi.server_id]
        _, req_id, _ = self.cur_steps[self.step_idx]
        if req_id is not None and req_id in self.ds.req_tokens:
            in_tok, out_tok = self.ds.req_tokens[req_id]
        else:
            size = float(self.cur_task['TaskSize'])
            in_tok = int(0.6 * size)
            out_tok = int(0.4 * size)
        tokens = in_tok + out_tok

        d_km = 0.0
        link_latency_ms = 0.0  
        
        if self.step_idx == 0:
            d_km = haversine_km(self.cur_task['TaskLongitude'], self.cur_task['TaskLatitude'], server.lon, server.lat)
            if self.trap_latency > 0 and server.server_id in self.trap_server_ids:
                if np.random.random() < self.trap_packet_loss_prob:
                    link_latency_ms = self.trap_bad_latency  
                else:
                    link_latency_ms = self.trap_good_latency  
        else:
            if self.prev_server_id is not None:
                prev_server = self.servers[self.prev_server_id]
                d_km = haversine_km(prev_server.lon, prev_server.lat, server.lon, server.lat)
                if prev_server.region == server.region:
                    d_km = max(d_km, 0.5)
   
                involves_trap = (self.prev_server_id in self.trap_server_ids or 
                                server.server_id in self.trap_server_ids)
                
                if involves_trap and self.trap_latency > 0:
                    if np.random.random() < self.trap_packet_loss_prob:
                        link_latency_ms = self.trap_bad_latency  
                    else:
                        link_latency_ms = self.trap_good_latency  
                else:
                    link_key = (self.prev_server_id, server.server_id)
                    if link_key in self.link_latency:
                        link_latency_ms = self.link_latency[link_key]
            else:
                d_km = 1.0
                
        network_ms = self._compute_channel_latency(d_km, tokens) + link_latency_ms

        speed_tps = max(server.normalized_compute, 1e-6) * self.base_speed_tps
        compute_ms = (tokens / speed_tps) * 1000.0
        queue_ms = max(0.0, self.busy_until[server.server_id] - self.current_time_ms)
        step_latency_ms = network_ms + compute_ms + queue_ms
        start_time = max(self.current_time_ms, self.busy_until[server.server_id])
        finish_time = start_time + compute_ms
        self.busy_until[server.server_id] = finish_time
        self.current_time_ms = start_time + step_latency_ms  
        cost = (tokens / 1000.0) * mi.cost_per_token * server.cost_multiplier
        switch_penalty_ms = 0.0
        if self.prev_server_id is not None:
            prev_region = self.servers[self.prev_server_id].region
            if prev_region != server.region:
                self.ep_switches += 1
                switch_penalty_ms = self.lambda_switch
        lat_best = 0.0
        lat_worst = 5000.0
        lat_normalized = (step_latency_ms - lat_best) / (lat_worst - lat_best)
        lat_normalized = np.clip(lat_normalized, 0.0, 1.0)
        r_L = -4.0 * lat_normalized
 
        cost_best = 0.00045  
        cost_worst = 0.15    
        cost_normalized = (cost - cost_best) / (cost_worst - cost_best)
        cost_normalized = np.clip(cost_normalized, 0.0, 1.0)
        r_C = -4.0 * cost_normalized
 
        r_S = 0.0
        if switch_penalty_ms > 0:
             r_S = -0.4

        self.prev_server_id = server.server_id
 
        alpha = 0.5
        self.server_latency_history[server.server_id] = (
            alpha * step_latency_ms + (1 - alpha) * self.server_latency_history[server.server_id]
        )

        self.ep_latency.append(step_latency_ms)
        self.ep_cost.append(cost)
        self.step_idx += 1
        done = (self.step_idx >= len(self.cur_steps))
        info = {
            'latency_ms': step_latency_ms,
            'cost': cost,
            'network_ms': network_ms,
            'compute_ms': compute_ms,
            'queue_ms': queue_ms,
            'switch_penalty_ms': switch_penalty_ms,
            'server_id': server.server_id,
            'region': server.region,
        }
        return self._get_state(), (r_L, r_C, r_S), done, info
