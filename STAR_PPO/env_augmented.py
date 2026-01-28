import os
import sys
import numpy as np
import torch
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env import WorkflowMoEEnv, WorkflowDataset

class AugmentedWorkflowEnv(WorkflowMoEEnv):
    def __init__(self, dataset: WorkflowDataset, **kwargs):
        super().__init__(dataset, **kwargs)
  
        self.adj_list = {sid: [] for sid in self.servers.keys()}
        self.server_bandwidths = {sid: [] for sid in self.servers.keys()}
        
        for region in self.ds.regions:
            npath = os.path.join(self.ds.data_root, region, 'network_links.csv')
            if not os.path.exists(npath):
                npath = os.path.join(self.ds.data_root, 'network_links.csv')
                
            if os.path.exists(npath):
                from env import load_csv
                rows = load_csv(npath)
                for r in rows:
                    src = r['SrcServerID']
                    dst = r['DstServerID']
                    bw_val = r.get('LinkBandwidth') or r.get('LinkBandwidthMbps') or '1000'
                    bw = float(bw_val)
                    
                    if src in self.adj_list:
                        self.adj_list[src].append(dst)
                        self.server_bandwidths[src].append(bw)
    
        self.static_topo_feats = {}
        high_compute_nodes = [sid for sid, s in self.servers.items() if s.normalized_compute > 1.5]
        
        for sid in self.servers.keys():
            bws = self.server_bandwidths[sid]
            avg_bw = np.mean(bws) if bws else 100.0
            norm_bw = np.clip(avg_bw / 10000.0, 0.0, 1.0) 
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
        base_state = self._get_state()
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
 
        prev_sid = self.prev_server_id
        
        if prev_sid is not None and prev_sid in self.servers:
            neighbors = self.adj_list.get(prev_sid, [])
            if neighbors:
                waits = []
                current_time = self.current_time_ms
                for nbr in neighbors:
                    busy = self.busy_until.get(nbr, 0.0)
                    waits.append(max(0.0, busy - current_time))
                avg_nbr_wait = np.mean(waits)
                norm_nbr_congestion = np.clip(avg_nbr_wait / 5000.0, 0.0, 1.0)
            else:
                norm_nbr_congestion = 0.0
  
            norm_bw, is_near_hc = self.static_topo_feats.get(prev_sid, (0.5, 0.0))
            
        else:
            norm_nbr_congestion = 0.0
            norm_bw = 1.0 
            is_near_hc = 0.5
            
        topo_vec = np.array([
            norm_nbr_congestion,
            norm_bw,
            is_near_hc
        ], dtype=np.float32)
        
        return np.concatenate([s_vec, topo_vec])


