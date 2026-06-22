"""
Workflow MoE Environment
环境定义：包含数据加载、服务器路由环境逻辑
可被 PPO_algorithm 和 PPO_GNN 复用
"""
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


# ==========================
# Data Loading
# ==========================

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
    cost_multiplier: float = 1.0  # 服务器价格系数 (0.3 ~ 2.5)，对应 ε_n^comp
    p_n_rel: float = 1.0          # 节点可靠性估计 Pr{A_n(k+1)=1 | H_n(k)}，对应 system.tex §4.1


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
        # Load servers per region
        self.servers: Dict[str, Server] = {}
        self.trap_server_ids: set = set()  # 陷阱服务器ID集合
        self.trap_latency: float = 0.0  # 陷阱链路延迟
        for region in self.regions:
            # 检查是否有陷阱配置
            trap_config_path = os.path.join(data_root, region, 'trap_config.json')
            trap_server_ids = set()
            trap_cost_mult = None  # None 表示不修改陷阱服务器成本
            if os.path.exists(trap_config_path):
                import json
                with open(trap_config_path, 'r') as f:
                    trap_config = json.load(f)
                trap_server_ids = set(trap_config.get('trap_server_ids', []))
                trap_cost_mult = trap_config.get('trap_cost_multiplier', None)  # 可以为 None
                self.trap_server_ids.update(trap_server_ids)
                self.trap_latency = trap_config.get('trap_latency', 500.0)
                cost_info = f"cost_mult={trap_cost_mult}" if trap_cost_mult else "成本不变"
                print(f"[ENV] 检测到陷阱配置: {len(trap_server_ids)} 个陷阱服务器, {cost_info}")

            srows = load_csv(os.path.join(data_root, region, 'servers.csv'))
            for r in srows:
                sid = r['ServerID']
                # 生成服务器价格系数 - 基于算力（关键设计）
                #
                # Latency 组成：network_ms + compute_ms + queue_ms
                # - network_ms：取决于用户到服务器的距离（Agent学习选近的）
                # - compute_ms：取决于 normalized_compute（算力高=快）
                #
                # 关键洞察：让 cost_multiplier 与 normalized_compute 负相关
                # - 高算力服务器：便宜（规模效应）且快（compute_ms 低）
                # - 低算力服务器：贵（效率低）且慢（compute_ms 高）
                # 这样 Agent 选高算力服务器时，cost↓ 且 latency↓，目标对齐！

                lon = float(r['Longitude'])
                lat = float(r['Latitude'])
                norm_compute = float(r.get('NormalizedCompute', '1.0'))

                # 成本计算：基于算力 + 随机优惠
                # 高算力服务器：便宜 (0.3~0.6)
                # 低算力服务器：贵 (2.0~3.5)

                # 基础成本因子
                compute_factor = 3.5 - (norm_compute - 0.25) * 4.0  # 范围: 0.5 ~ 3.5

                # 随机因子
                hash_val = (lon * 1000 + lat * 100) % 100
                random_factor = 0.8 + (hash_val / 100) * 0.4  # 0.8 ~ 1.2

                cost_mult = compute_factor * random_factor
                cost_mult = max(0.3, min(3.5, cost_mult))

                # 简化设计：所有服务器保持自然成本分布
                # 陷阱服务器通过 trap_cost_mult 覆盖成本

                # 如果是陷阱服务器且指定了成本，覆盖成本
                # 如果 trap_cost_mult 为 None，则保持原成本（只用高算力诱惑）
                if sid in trap_server_ids and trap_cost_mult is not None:
                    cost_mult = trap_cost_mult

                # p_n_rel: 节点可靠性估计 (system.tex §4.1)
                # 陷阱服务器：50% 丢包率 → p_n_rel ≈ 0.50 (期望可用性)
                # 正常服务器：p_n_rel = 1.0
                p_rel = 0.50 if sid in trap_server_ids else 1.0

                self.servers[sid] = Server(
                    server_id=sid,
                    region=region,
                    lon=lon,
                    lat=lat,
                    normalized_compute=float(r.get('NormalizedCompute', '1.0')),
                    cost_multiplier=cost_mult,
                    p_n_rel=p_rel,
                )
        # Load model instances
        self.model_instances: List[ModelInstance] = []
        idx = 0
        for region in self.regions:
            mrows = load_csv(os.path.join(data_root, region, 'model_instances.csv'))
            for r in mrows:
                # Prefer textual model type name if present
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
        # Load tasks for split
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
                # Only keep multi-step workflows as per requirement
                if len(r['RequiredModelTypes']) >= 2:
                    self.tasks.append(r)
        # Load mapping and request log for token lengths
        self.req_tokens: Dict[str, Tuple[int, int]] = {}
        for region in self.regions:
            # request_log may be large; stream parse
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
        # sort steps
        for k in self.task_to_requests:
            self.task_to_requests[k].sort(key=lambda x: x[0])
        # Precompute mean BaseLatencyMs as lambda_switch base
        # 同时构建链路延迟查找表 (用于孤岛陷阱)
        base_latencies = []
        self.link_latency: Dict[Tuple[str, str], float] = {}  # (src, dst) -> latency_ms
        self.link_bandwidth: Dict[Tuple[str, str], float] = {}  # (src, dst) -> bandwidth_mbps

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
                            # 双向链路
                            self.link_latency[(dst, src)] = latency
                            self.link_bandwidth[(dst, src)] = bandwidth

                        base_latencies.append(latency)
                    except Exception:
                        pass

        self.mean_base_latency_ms = float(np.mean(base_latencies)) if base_latencies else 10.0
        print(f"[Dataset] 加载链路: {len(self.link_latency)//2} 条, 平均延迟: {self.mean_base_latency_ms:.2f}ms")


# ==========================
# Environment
# ==========================

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

        # 链路延迟/带宽查找表 (用于孤岛陷阱)
        self.link_latency = dataset.link_latency
        self.link_bandwidth = dataset.link_bandwidth
        self.trap_server_ids = dataset.trap_server_ids  # 陷阱服务器ID集合
        self.trap_latency = dataset.trap_latency  # 陷阱链路延迟

        # === 随机丢包陷阱参数 ===
        # 核心思想：link_latency 字典保持高值（供 STAR-PPO 感知）
        # 但实际 step() 时随机返回延迟（让缺少拓扑感知的算法受惩罚）
        # 参数设计：50% 丢包率 + 轻惩罚（主实验设置）
        self.trap_packet_loss_prob = 0.50  # 丢包概率（50%）
        self.trap_good_latency = 100.0     # 运气好时的延迟 (ms)
        self.trap_bad_latency = 200.0      # 运气差时的延迟 (ms)
        # 期望延迟 = 0.5×100 + 0.5×200 = 150ms
        # 轻惩罚设置，适合主实验

        # Build action lookup
        self.actions: List[ModelInstance] = dataset.model_instances
        self.model_type_to_action_idxs: Dict[str, List[int]] = {}
        for a in self.actions:
            self.model_type_to_action_idxs.setdefault(a.model_type, []).append(a.idx)
        # Server lookup
        self.servers = dataset.servers

        # State variables
        self.cur_task = None
        self.cur_steps: List[Tuple[int, str, str]] = []
        self.step_idx = 0
        self.prev_server_id = None
        self.current_time_ms = 0.0
        self.busy_until: Dict[str, float] = {sid: 0.0 for sid in self.servers.keys()}
        # 适应性实验：Region 宕机 / TrafficSpike（由 run_adaptability_inference 开启）
        self.offline_servers: set = set()
        self.traffic_spike_on_reset: bool = False
        self.traffic_spike_ratio: float = 0.40
        self.traffic_spike_queue_ms: Tuple[float, float] = (3200.0, 5200.0)
        self.traffic_spike_seed: int = 42
        # logging stats for episode
        self.ep_latency: List[float] = []
        self.ep_cost: List[float] = []
        self.ep_switches: int = 0
        # SLA threshold (ms)，对应 system.tex §4.6.3 中的 T_m^SLA
        self.sla_threshold_ms: float = 3000.0

        # New: Track observed latency for each server
        # Initialize with mean base latency to start neutral
        self.server_latency_history: Dict[str, float] = {
            sid: self.ds.mean_base_latency_ms * 5.0 for sid in self.servers.keys()
        }

    def _compute_channel_latency(self, dist_km: float, tokens: int) -> float:
        """
        Simulate a complex wireless/wired channel using Shannon Capacity and Fading.
        Returns Latency in ms.

        Formula: T = T_prop + DataSize / (Bandwidth * log2(1 + SINR))
        """
        # 1. Propagation Delay (Speed of light approx)
        # Keeping previous scaling factor for continuity, but adding realistic propagation
        prop_delay_ms = dist_km * 0.005 # ~1.5x speed of light in fiber + routing overhead

        # 2. Transmission Delay (Shannon Capacity)
        # Data Size (bits)
        data_bits = tokens * 32 # Assumption: 1 token ~ 4 bytes = 32 bits

        # Channel Physics Parameters
        # Bandwidth (Dynamic: Congestion Simulation)
        # Nominal 20MHz, fluctuates between 5MHz and 20MHz due to congestion
        bandwidth_hz = 20e6 * random.uniform(0.25, 1.0)

        # Transmission Power (dBm) -> Watts
        # 23 dBm (200mW)
        tx_power_dbm = 23.0
        tx_power_watts = 10 ** (tx_power_dbm / 10.0) / 1000.0

        # Path Loss (Urban/Long Distance Mix)
        # PL = (d/d0)^(-alpha)
        # Alpha 2.5 for "virtual link" approximation
        dist_km = max(dist_km, 0.1) # Avoid singularity
        path_loss_linear = (0.1 / dist_km) ** 2.5

        # Fading (Rayleigh - Exponential Distribution for Power Gain)
        # Mean 1.0
        fading_gain = random.expovariate(1.0)

        # Noise Power
        # Thermal Noise: -174 dBm/Hz + 10log(B) + NF(10dB)
        noise_floor_dbm = -174 + 10 * math.log10(bandwidth_hz) + 10
        noise_watts = 10 ** (noise_floor_dbm / 10.0) / 1000.0

        # Received Power
        rx_power_watts = tx_power_watts * path_loss_linear * fading_gain

        # SINR (Signal to Interference plus Noise Ratio)
        # Add random interference from other "users"
        interference_watts = noise_watts * random.uniform(0, 10)
        sinr = rx_power_watts / (noise_watts + interference_watts + 1e-20)

        # Channel Capacity (Shannon Limit)
        capacity_bps = bandwidth_hz * math.log2(1 + sinr)

        # Ensure minimum connectivity (Control Channel)
        # e.g., 50 kbps fallback
        capacity_bps = max(capacity_bps, 50000.0)

        # Transmission Time
        tx_time_ms = (data_bits / capacity_bps) * 1000.0

        # Combine
        # Scale factor to align roughly with previous 10-100ms range but with high variance
        # We add a base routing overhead of 5ms
        total_latency_ms = prop_delay_ms + tx_time_ms + 5.0

        return total_latency_ms

    def reset(self, task: Dict[str, Any]):
        self.cur_task = task
        tid = task['TaskID']
        self.cur_steps = self.ds.task_to_requests.get(tid, [])
        # fallback: infer steps from RequiredModelTypes
        if not self.cur_steps:
            self.cur_steps = [(i, None, mt) for i, mt in enumerate(task['RequiredModelTypes'])]
        self.step_idx = 0
        self.prev_server_id = None
        self.current_time_ms = 0.0
        self.busy_until = {sid: 0.0 for sid in self.servers.keys()}
        self.ep_latency = []
        self.ep_cost = []
        self.ep_switches = 0
        self.ep_total_latency: float = 0.0  # 累积端到端延迟，用于 SLA 检查
        # OR: Reset it to simulate new session?
        # For now, let's keep it persistent across steps but reset on episode start to avoid leakage between unrelated tasks
        # But wait, Trans wants to learn sequence. Within an episode, we have multiple steps.
        # So we reset history at start of episode.
        self.server_latency_history = {
            sid: self.ds.mean_base_latency_ms * 5.0 for sid in self.servers.keys()
        }
        if self.traffic_spike_on_reset:
            self._inject_traffic_spike_queues()
        return self._get_state()

    def _inject_traffic_spike_queues(self):
        """每个 episode 开始时向高算力热点注入排队 backlog，模拟并发突增。"""
        servers_sorted = sorted(
            self.servers.values(), key=lambda s: s.normalized_compute, reverse=True
        )
        n_hot = max(1, int(len(servers_sorted) * self.traffic_spike_ratio))
        rng = np.random.RandomState(self.traffic_spike_seed)
        for s in servers_sorted[:n_hot]:
            extra = float(rng.uniform(self.traffic_spike_queue_ms[0], self.traffic_spike_queue_ms[1]))
            self.busy_until[s.server_id] = max(self.busy_until[s.server_id], extra)

    def _get_state(self) -> Dict[str, Any]:
        # Build a compact state vector
        total_steps = max(len(self.cur_steps), 1)
        step_norm = self.step_idx / total_steps
        task_lon = float(self.cur_task['TaskLongitude'])
        task_lat = float(self.cur_task['TaskLatitude'])
        prev_region = 0
        if self.prev_server_id is not None:
            # 提取 region 中的数字部分 (兼容 Server1, Server1_Trap 等)
            region_str = self.servers[self.prev_server_id].region
            match = re.search(r'\d+', region_str)
            prev_region = int(match.group()) if match else 0

        # Get historical latency vector (aligned with server IDs if needed, but here we return dict or just raw)
        # We will let the environment wrapper/agent handle the mapping to feature vector

        # State dictionary for model input
        state = {
            'step_norm': step_norm,
            'total_steps': total_steps,
            'task_lon': task_lon,
            'task_lat': task_lat,
            'prev_region_id': prev_region,
            'num_regions': len(self.ds.regions),
            # Pass the history dict directly
            'server_latency_history': self.server_latency_history.copy()
        }
        return state

    def available_actions(self) -> List[int]:
        # Required model type for current step
        _, _, req_type = self.cur_steps[self.step_idx]
        if req_type is None:
            # fallback to RequiredModelTypes
            req_type = self.cur_task['RequiredModelTypes'][self.step_idx]
        actions = self.model_type_to_action_idxs.get(str(req_type), [])
        if not self.offline_servers:
            return actions
        return [
            idx for idx in actions
            if self.actions[idx].server_id not in self.offline_servers
        ]

    def estimate_step(self, action_idx: int) -> Tuple[float, float, float]:
        """Estimate (latency_ms, cost, switch_penalty_ms) for taking action at current state without mutating env."""
        mi = self.actions[action_idx]
        server = self.servers[mi.server_id]
        # tokens
        _, req_id, _ = self.cur_steps[self.step_idx]
        if req_id is not None and req_id in self.ds.req_tokens:
            in_tok, out_tok = self.ds.req_tokens[req_id]
        else:
            size = float(self.cur_task['TaskSize'])
            in_tok = int(0.6 * size)
            out_tok = int(0.4 * size)
        tokens = in_tok + out_tok

        # network (考虑链路数据 - 孤岛陷阱生效)
        d_km = 0.0
        link_latency_ms = 0.0

        if self.step_idx == 0:
            d_km = haversine_km(self.cur_task['TaskLongitude'], self.cur_task['TaskLatitude'], server.lon, server.lat)
            # 第一步也检查陷阱延迟（用户→陷阱服务器）
            # estimate_step 返回期望延迟（平均值），用于其他算法预估
            if self.trap_latency > 0 and server.server_id in self.trap_server_ids:
                # 期望延迟 = p * bad + (1-p) * good
                link_latency_ms = (self.trap_packet_loss_prob * self.trap_bad_latency +
                                  (1 - self.trap_packet_loss_prob) * self.trap_good_latency)
        else:
            if self.prev_server_id is not None:
                prev_server = self.servers[self.prev_server_id]
                d_km = haversine_km(prev_server.lon, prev_server.lat, server.lon, server.lat)
                if prev_server.region == server.region:
                     d_km = max(d_km, 0.5)

                # 查找链路延迟 (孤岛陷阱关键!)
                # === 随机丢包陷阱：estimate_step 返回期望延迟 ===
                involves_trap = (self.prev_server_id in self.trap_server_ids or
                                server.server_id in self.trap_server_ids)

                if involves_trap and self.trap_latency > 0:
                    # 期望延迟 = p * bad + (1-p) * good
                    link_latency_ms = (self.trap_packet_loss_prob * self.trap_bad_latency +
                                      (1 - self.trap_packet_loss_prob) * self.trap_good_latency)
                else:
                    link_key = (self.prev_server_id, server.server_id)
                    if link_key in self.link_latency:
                        link_latency_ms = self.link_latency[link_key]
            else:
                d_km = 1.0

        network_ms = self._compute_channel_latency(d_km, tokens) + link_latency_ms

        # compute and queue（estimate 也需与 step() 一致，含 p_n_rel 开销）
        speed_tps = max(server.normalized_compute, 1e-6) * self.base_speed_tps
        compute_ms = (tokens / speed_tps) * 1000.0
        effective_compute_ms = compute_ms / max(server.p_n_rel, 0.1)
        queue_ms = max(0.0, self.busy_until[server.server_id] - self.current_time_ms)
        step_latency_ms = network_ms + effective_compute_ms + queue_ms
        # cost (乘以服务器价格系数，不同服务器有不同运营成本)
        cost = (tokens / 1000.0) * mi.cost_per_token * server.cost_multiplier
        # switch
        switch_penalty_ms = 0.0
        if self.prev_server_id is not None:
            prev_region = self.servers[self.prev_server_id].region
            if prev_region != server.region:
                switch_penalty_ms = self.lambda_switch
        return step_latency_ms, cost, switch_penalty_ms

    def step(self, action_idx: int) -> Tuple[Dict[str, Any], Tuple[float, float, float], bool, Dict[str, Any]]:
        # Resolve action to model instance and server
        mi = self.actions[action_idx]
        server = self.servers[mi.server_id]
        # Determine tokens for this step
        _, req_id, _ = self.cur_steps[self.step_idx]
        if req_id is not None and req_id in self.ds.req_tokens:
            in_tok, out_tok = self.ds.req_tokens[req_id]
        else:
            # heuristic fallback based on task size
            size = float(self.cur_task['TaskSize'])
            in_tok = int(0.6 * size)
            out_tok = int(0.4 * size)
        tokens = in_tok + out_tok

        # Network latency (考虑链路数据 - 孤岛陷阱生效)
        d_km = 0.0
        link_latency_ms = 0.0  # 链路基础延迟

        if self.step_idx == 0:
            # user to first server (用户到服务器，使用信道模型)
            d_km = haversine_km(self.cur_task['TaskLongitude'], self.cur_task['TaskLatitude'], server.lon, server.lat)
            # 第一步也检查陷阱延迟（用户→陷阱服务器）
            # === 随机丢包陷阱：实际延迟随机化 ===
            if self.trap_latency > 0 and server.server_id in self.trap_server_ids:
                if np.random.random() < self.trap_packet_loss_prob:
                    link_latency_ms = self.trap_bad_latency  # 运气差，丢包
                else:
                    link_latency_ms = self.trap_good_latency  # 运气好，正常
        else:
            # server-to-server (服务器到服务器，使用链路数据)
            if self.prev_server_id is not None:
                prev_server = self.servers[self.prev_server_id]
                d_km = haversine_km(prev_server.lon, prev_server.lat, server.lon, server.lat)
                if prev_server.region == server.region:
                    d_km = max(d_km, 0.5)

                # === 随机丢包陷阱：实际延迟随机化 ===
                # 判断是否涉及陷阱服务器
                involves_trap = (self.prev_server_id in self.trap_server_ids or
                                server.server_id in self.trap_server_ids)

                if involves_trap and self.trap_latency > 0:
                    # 涉及陷阱服务器时，使用随机延迟
                    if np.random.random() < self.trap_packet_loss_prob:
                        link_latency_ms = self.trap_bad_latency  # 运气差，丢包
                    else:
                        link_latency_ms = self.trap_good_latency  # 运气好，正常
                else:
                    # 正常服务器，从 link_latency 字典读取
                    link_key = (self.prev_server_id, server.server_id)
                    if link_key in self.link_latency:
                        link_latency_ms = self.link_latency[link_key]
            else:
                d_km = 1.0

        network_ms = self._compute_channel_latency(d_km, tokens) + link_latency_ms

        # Compute latency components
        speed_tps = max(server.normalized_compute, 1e-6) * self.base_speed_tps
        compute_ms = (tokens / speed_tps) * 1000.0
        queue_ms = max(0.0, self.busy_until[server.server_id] - self.current_time_ms)
        # system.tex §4.1: p_n_rel 节点可靠性 → 期望重试开销
        # E[重试次数] = (1-p)/p，等效计算延迟系数 = 1/p_n_rel
        # p_n_rel=1.0 (正常) → 无额外开销；p_n_rel=0.5 (trap) → compute_ms 翻倍
        effective_compute_ms = compute_ms / max(server.p_n_rel, 0.1)
        step_latency_ms = network_ms + effective_compute_ms + queue_ms
        # Update server busy_until and current time
        start_time = max(self.current_time_ms, self.busy_until[server.server_id])
        finish_time = start_time + compute_ms
        self.busy_until[server.server_id] = finish_time
        self.current_time_ms = start_time + step_latency_ms  # includes network + queue + compute
        # Cost (乘以服务器价格系数，不同服务器有不同运营成本)
        cost = (tokens / 1000.0) * mi.cost_per_token * server.cost_multiplier
        # Switch penalty bookkeeping for this step
        switch_penalty_ms = 0.0
        if self.prev_server_id is not None:
            prev_region = self.servers[self.prev_server_id].region
            if prev_region != server.region:
                self.ep_switches += 1
                switch_penalty_ms = self.lambda_switch
        # REWARD SYSTEM - 严格单调上升设计
        # 目标：确保Reward从一开始就是负值，并且随着性能提升单调上升
        # 策略：设置Best值为"理论极限"（比实际最小值还小），确保所有episode都有负reward

        # === Latency Reward ===
        # 实际最小值 ~350ms，最大值 ~16000ms
        # 设置 lat_best=0.0 (理论极限)，确保 lat_normalized > 0
        # 设置 lat_worst=5000ms (覆盖绝大多数情况)
        lat_best = 0.0
        lat_worst = 5000.0
        # 线性映射到 [-4, 0]
        # 公式: -4 * ((lat - best) / (worst - best))
        lat_normalized = (step_latency_ms - lat_best) / (lat_worst - lat_best)
        lat_normalized = np.clip(lat_normalized, 0.0, 1.0)
        r_L = -4.0 * lat_normalized

        # === Cost Reward (Linear, with server cost multiplier) ===
        # 模型基础价格:
        # - ChatGPT: 0.0015
        # - GPT-4: 0.03
        # - GPT-4v: 0.06
        #
        # 服务器价格系数: 0.3 ~ 2.5 (大幅扩展，便于Agent优化)
        # 实际价格范围:
        # - 最便宜: ChatGPT × 0.3 = 0.00045
        # - 最贵: GPT-4v × 2.5 = 0.15
        #
        # Agent 可以通过选择便宜的服务器来优化 Cost！
        # 即使必须用 GPT-4，选 0.3x 的服务器比 2.5x 便宜 88%！
        cost_best = 0.00045  # ChatGPT × 0.3
        cost_worst = 0.15    # GPT-4v × 2.5
        cost_normalized = (cost - cost_best) / (cost_worst - cost_best)
        cost_normalized = np.clip(cost_normalized, 0.0, 1.0)
        r_C = -4.0 * cost_normalized

        # === Switch Penalty ===
        # 如果发生跨区域切换，给予 -0.4 的惩罚
        r_S = 0.0
        if switch_penalty_ms > 0:
             r_S = -0.4

        # Update prev server after computing r_S
        self.prev_server_id = server.server_id

        # Update Observed Latency History for this server
        # Simple exponential moving average to smooth out noise
        alpha = 0.5
        self.server_latency_history[server.server_id] = (
            alpha * step_latency_ms + (1 - alpha) * self.server_latency_history[server.server_id]
        )

        # track stats
        self.ep_latency.append(step_latency_ms)
        self.ep_cost.append(cost)
        self.ep_total_latency += step_latency_ms
        # Done?
        self.step_idx += 1
        done = (self.step_idx >= len(self.cur_steps))
        info = {
            'latency_ms': step_latency_ms,
            'cost': cost,
            'network_ms': network_ms,
            'compute_ms': effective_compute_ms,   # 含可靠性开销的有效计算延迟
            'compute_ms_base': compute_ms,        # 原始计算延迟（不含重试）
            'p_n_rel': server.p_n_rel,
            'queue_ms': queue_ms,
            'switch_penalty_ms': switch_penalty_ms,
            'server_id': server.server_id,
            'region': server.region,
        }
        # 在 episode 结束时追加聚合信息（对应 system.tex §4.6 中各目标函数）
        if done:
            info['episode_total_latency_ms'] = self.ep_total_latency
            info['episode_total_cost'] = float(sum(self.ep_cost))
            info['episode_switches'] = self.ep_switches
            # SLA 违约判定：端到端总延迟 > T_m^SLA，对应 system.tex §4.6.3 V_SLA
            info['sla_violated'] = self.ep_total_latency > self.sla_threshold_ms
        return self._get_state(), (r_L, r_C, r_S), done, info
