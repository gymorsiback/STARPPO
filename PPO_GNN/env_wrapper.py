
import os
import sys
import torch
import numpy as np
from torch_geometric.data import Data

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env import WorkflowMoEEnv, WorkflowDataset
from utils import haversine_km

class GNNWorkflowEnv(WorkflowMoEEnv):
    def __init__(self, dataset: WorkflowDataset, device='cpu', **kwargs):
        super().__init__(dataset, device=device, **kwargs)
        self.edge_index = None
        self.edge_attr = None
        self.static_node_features = None
        self.server_to_idx = {}
        self.idx_to_server = {}
        self._precompute_graph_structure()
        
    def _precompute_graph_structure(self):
        server_ids = sorted(list(self.servers.keys()))
        num_servers = len(server_ids)
        self.server_to_idx = {sid: i for i, sid in enumerate(server_ids)}
        self.idx_to_server = {i: sid for i, sid in enumerate(server_ids)}

        server_feats = []
        for sid in server_ids:
            s = self.servers[sid]
            server_feats.append([
                s.lon / 180.0,
                s.lat / 180.0,
                s.normalized_compute
            ])
        self.static_node_features = torch.tensor(server_feats, dtype=torch.float32)

        K = 10
        edge_indices = []
        edge_attrs = []
        
        coords = self.static_node_features[:, :2] * 180.0 
        
        for i in range(num_servers):
            dists = []
            for j in range(num_servers):
                if i == j: continue
                d = haversine_km(coords[i,0].item(), coords[i,1].item(), 
                               coords[j,0].item(), coords[j,1].item())
                dists.append((j, d))
 
            dists.sort(key=lambda x: x[1])
            for j, d in dists[:K]:
                edge_indices.append([i, j])
                edge_attrs.append([np.exp(-d / 1000.0)]) 
                
        self.edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        self.edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

    def get_graph_data(self, state_dict, dwa_weights):
        num_nodes = self.static_node_features.shape[0]
 
        dynamic_feats = np.zeros((num_nodes, 3), dtype=np.float32)
        
        task_lon = state_dict['task_lon']
        task_lat = state_dict['task_lat']
        prev_sid = self.prev_server_id
        
        server_ids = sorted(list(self.servers.keys()))
        
        for i, sid in enumerate(server_ids):
            if sid == prev_sid:
                dynamic_feats[i, 0] = 1.0
  
            queue_ms = max(0.0, self.busy_until[sid] - self.current_time_ms)
            dynamic_feats[i, 1] = min(1.0, queue_ms / 1000.0)

            s = self.servers[sid]
            d = haversine_km(task_lon, task_lat, s.lon, s.lat)
            dynamic_feats[i, 2] = np.exp(-d / 1000.0)
            
        dynamic_feats_t = torch.tensor(dynamic_feats, dtype=torch.float32)

        x = torch.cat([self.static_node_features, dynamic_feats_t], dim=1)
   
        global_feat = torch.tensor([
            state_dict['step_norm'],
            dwa_weights[0], dwa_weights[1], dwa_weights[2]
        ], dtype=torch.float32).unsqueeze(0)
   
        valid_action_idxs = self.available_actions()
        valid_server_indices = []
        server_action_map = {} 
        server_to_actions = {}
        for aidx in valid_action_idxs:
            mi = self.actions[aidx]
            server_to_actions.setdefault(mi.server_id, []).append(mi)
            
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        for i, sid in enumerate(server_ids):
            if sid in server_to_actions:
                mask[i] = True
                best_mi = min(server_to_actions[sid], key=lambda m: m.cost_per_token)
                server_action_map[i] = best_mi.idx

        data = Data(
            x=x, 
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            global_feat=global_feat,
            candidate_mask=mask
        )
        return data, server_action_map


