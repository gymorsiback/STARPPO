import os
import sys
import numpy as np
import torch
from typing import Dict, Any, List

# Add root path to access env.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env import WorkflowMoEEnv, WorkflowDataset

class AugmentedWorkflowEnv(WorkflowMoEEnv):
    """
    STAR-PPO 的核心组件：状态增强环境。
    继承自原始环境，但通过 Feature Engineering 注入拓扑感知能力。

    新增状态特征 (Topology Features):
    1. Avg Neighbor Load (邻居平均负载): 感知局部拥塞
    2. Avg Neighbor Bandwidth (邻居平均带宽): 感知局部连通性
    3. Hop Count to High-Compute Nodes (到高算力节点的跳数): 感知全局位置
    """
    def __init__(self, dataset: WorkflowDataset, **kwargs):
        super().__init__(dataset, **kwargs)

        # === 1. Pre-compute Static Topology Info ===
        # Build Adjacency List
        self.adj_list = {sid: [] for sid in self.servers.keys()}
        self.server_bandwidths = {sid: [] for sid in self.servers.keys()}

        for region in self.ds.regions:
            npath = os.path.join(self.ds.data_root, region, 'network_links.csv')
            if not os.path.exists(npath):
                # Fallback for single region root
                npath = os.path.join(self.ds.data_root, 'network_links.csv')

            if os.path.exists(npath):
                # Simple CSV parsing to avoid dependency on pandas if possible,
                # but dataset.load_csv is available
                from env import load_csv
                rows = load_csv(npath)
                for r in rows:
                    src = r['SrcServerID']
                    dst = r['DstServerID']
                    # Handle column name variations
                    bw_val = r.get('LinkBandwidth') or r.get('LinkBandwidthMbps') or '1000'
                    bw = float(bw_val)

                    if src in self.adj_list:
                        self.adj_list[src].append(dst)
                        self.server_bandwidths[src].append(bw)

        # Pre-compute Static Features per Server
        self.static_topo_feats = {}
        high_compute_nodes = [sid for sid, s in self.servers.items() if s.normalized_compute > 1.5]

        for sid in self.servers.keys():
            # Feature 2: Avg Outbound Bandwidth (Normalized)
            bws = self.server_bandwidths[sid]
            avg_bw = np.mean(bws) if bws else 100.0
            norm_bw = np.clip(avg_bw / 10000.0, 0.0, 1.0) # Assume max 10Gbps

            # Feature 3: Min Hops to High Compute (BFS)
            # Simple heuristic: 0 if self is high compute, else ...
            # For simplicity in this demo, we use a random proxy or simple distance if graph analysis is heavy.
            # Let's use simple geographic centrality proxy for now to keep it lightweight "without graph overhead".
            # Or just set to 0.5 as placeholder if we don't run full BFS.
            # Let's run a quick 1-hop check.
            is_near_high_compute = 0.0
            if sid in high_compute_nodes:
                is_near_high_compute = 1.0
            else:
                for nbr in self.adj_list[sid]:
                    if nbr in high_compute_nodes:
                        is_near_high_compute = 0.5
                        break

            self.static_topo_feats[sid] = (norm_bw, is_near_high_compute)

    def get_augmented_state(self, dwa_weights=None):
        """
        返回增强后的状态向量。
        Base State (7 dims) + Topology Features (3 dims) = 10 dims
        """
        # 1. Get Base State
        base_state = self._get_state() # Dict

        # 2. Extract Base Vector
        if dwa_weights is None:
            dwa_weights = [0.33, 0.33, 0.33]

        s_vec = np.array([
            base_state['step_norm'],
            base_state['task_lon'],
            base_state['task_lat'],
            float(base_state['prev_region_id']),
            dwa_weights[0],
            dwa_weights[1],
            dwa_weights[2]
        ], dtype=np.float32)

        # 3. Calculate Dynamic Topology Features
        # Context: Previous Server (where we are coming from)
        # If prev_server_id is None (start), we use a virtual "average" state
        prev_sid = self.prev_server_id

        if prev_sid is not None and prev_sid in self.servers:
            # Feature 1: Neighbor Congestion
            neighbors = self.adj_list.get(prev_sid, [])
            if neighbors:
                # Avg wait time of neighbors
                waits = []
                current_time = self.current_time_ms
                for nbr in neighbors:
                    busy = self.busy_until.get(nbr, 0.0)
                    waits.append(max(0.0, busy - current_time))
                avg_nbr_wait = np.mean(waits)
                norm_nbr_congestion = np.clip(avg_nbr_wait / 5000.0, 0.0, 1.0)
            else:
                norm_nbr_congestion = 0.0

            # Static features for previous server
            norm_bw, is_near_hc = self.static_topo_feats.get(prev_sid, (0.5, 0.0))

        else:
            # Start of episode: Neutral values
            norm_nbr_congestion = 0.0
            norm_bw = 1.0 # Assume good connectivity from user
            is_near_hc = 0.5

        topo_vec = np.array([
            norm_nbr_congestion,
            norm_bw,
            is_near_hc
        ], dtype=np.float32)

        return np.concatenate([s_vec, topo_vec])


