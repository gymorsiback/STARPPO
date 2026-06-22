"""
Environment Wrapper for GNN-based PPO
将标准 WorkflowMoEEnv 的状态转换为 PyTorch Geometric 图数据
"""
import os
import sys
import torch
import numpy as np
from torch_geometric.data import Data

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env import WorkflowMoEEnv, WorkflowDataset
from utils import haversine_km

class GNNWorkflowEnv(WorkflowMoEEnv):
    def __init__(self, dataset: WorkflowDataset, device='cpu', **kwargs):
        super().__init__(dataset, device=device, **kwargs)
        # 预计算服务器间的地理距离，用于构建边
        self.edge_index = None
        self.edge_attr = None
        self.static_node_features = None
        self.server_to_idx = {}
        self.idx_to_server = {}
        self._precompute_graph_structure()

    def _precompute_graph_structure(self):
        """预计算静态的服务器图结构（假设服务器位置不变）"""
        # 服务器节点列表 (保持固定顺序)
        server_ids = sorted(list(self.servers.keys()))
        num_servers = len(server_ids)
        self.server_to_idx = {sid: i for i, sid in enumerate(server_ids)}
        self.idx_to_server = {i: sid for i, sid in enumerate(server_ids)}

        # 提取服务器静态特征: [lon, lat, normalized_compute]
        # 归一化经纬度到 [-1, 1]
        server_feats = []
        for sid in server_ids:
            s = self.servers[sid]
            server_feats.append([
                s.lon / 180.0,
                s.lat / 180.0,
                s.normalized_compute
            ])
        self.static_node_features = torch.tensor(server_feats, dtype=torch.float32)

        # 构建 K-NN 图 (K=10)
        K = 10
        edge_indices = []
        edge_attrs = []

        coords = self.static_node_features[:, :2] * 180.0 # 还原为经纬度用于计算距离

        for i in range(num_servers):
            dists = []
            for j in range(num_servers):
                if i == j: continue
                d = haversine_km(coords[i,0].item(), coords[i,1].item(),
                               coords[j,0].item(), coords[j,1].item())
                dists.append((j, d))

            # 排序取最近 K 个
            dists.sort(key=lambda x: x[1])
            for j, d in dists[:K]:
                edge_indices.append([i, j])
                # 边特征：距离
                # 同样使用高斯核/指数衰减转换距离为相似度
                edge_attrs.append([np.exp(-d / 1000.0)])

        self.edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        self.edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

    def get_graph_data(self, state_dict, dwa_weights):
        """
        将当前环境状态转换为 PyTorch Geometric Data 对象

        Data包含:
        - x: 节点特征 [num_nodes, 6] (static 3 + dynamic 3)
        - edge_index: 边索引 [2, num_edges]
        - edge_attr: 边属性 [num_edges, 1]
        - global_feat: 全局特征 [1, 4] (step_norm, dwa_weights)
        - candidate_mask: 有效节点掩码 [num_nodes] (bool)
        - action_mapping: [num_nodes, max_actions_per_node] (用于映射节点到具体 ActionID)
            注意：PPO 输出通常是 Action ID。但在 GNN 中，我们通常先选节点（服务器）。
            为了简化，我们假设模型会自动选择该服务器上最匹配的模型（或第一个）。
            或者，我们可以让 GNN 输出 Action Logits，但 Mask 掉无效的 Action。

            为了与 PPO_algorithm 对比，我们需要输出具体的 action_idx (ModelInstance)。
            这里我们构建一个 mask，长度为 num_servers。
            如果服务器 S 上有符合当前 req_type 的 ModelInstance，则 mask[S]=True。
            Agent 选定服务器后，我们在环境内部选择该服务器上符合要求的 ModelInstance。
            如果有多个，选 Cost 最低的（贪心）或随机。
        """
        num_nodes = self.static_node_features.shape[0]

        # 1. 构建动态节点特征 (3维: prev, queue, proximity)
        dynamic_feats = np.zeros((num_nodes, 3), dtype=np.float32)

        task_lon = state_dict['task_lon']
        task_lat = state_dict['task_lat']
        prev_sid = self.prev_server_id

        server_ids = sorted(list(self.servers.keys()))

        for i, sid in enumerate(server_ids):
            # Feature 1: Is previous server?
            if sid == prev_sid:
                dynamic_feats[i, 0] = 1.0

            # Feature 2: Queue time / Busy status (normalized)
            queue_ms = max(0.0, self.busy_until[sid] - self.current_time_ms)
            dynamic_feats[i, 1] = min(1.0, queue_ms / 1000.0)

            # Feature 3: Distance to task (proximity)
            s = self.servers[sid]
            d = haversine_km(task_lon, task_lat, s.lon, s.lat)
            # 使用 exp(-d/scale) 将距离转换为接近度 (0~1)
            dynamic_feats[i, 2] = np.exp(-d / 1000.0)

        dynamic_feats_t = torch.tensor(dynamic_feats, dtype=torch.float32)

        # 组合节点特征: Static (3) + Dynamic (3) = 6维
        x = torch.cat([self.static_node_features, dynamic_feats_t], dim=1)

        # 2. 构建 Global Feature
        # global_feat 保持 [1, 4] 形状，以便 PyG DataLoader 正确 cat 成 [batch_size, 4]
        global_feat = torch.tensor([
            state_dict['step_norm'],
            dwa_weights[0], dwa_weights[1], dwa_weights[2]
        ], dtype=torch.float32).unsqueeze(0)

        # 3. 构建 Candidate Mask & Action Mapping
        # 找出当前步骤需要的 Model Type
        valid_action_idxs = self.available_actions()
        valid_server_indices = []
        server_action_map = {} # server_idx -> best_action_idx

        # 预先分组 valid actions by server
        server_to_actions = {}
        for aidx in valid_action_idxs:
            mi = self.actions[aidx]
            server_to_actions.setdefault(mi.server_id, []).append(mi)

        mask = torch.zeros(num_nodes, dtype=torch.bool)

        for i, sid in enumerate(server_ids):
            if sid in server_to_actions:
                mask[i] = True
                # 策略：如果一个服务器有多个符合要求的模型，默认选 Cost 最低的
                # (因为 Latency 主要由服务器决定，同服务器模型间差异主要在 Cost)
                best_mi = min(server_to_actions[sid], key=lambda m: m.cost_per_token)
                server_action_map[i] = best_mi.idx

        # 创建 Data 对象（不包含 server_action_map，因为它无法被 PyG 批处理）
        # server_action_map 需要在外部单独处理
        data = Data(
            x=x,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            global_feat=global_feat,
            candidate_mask=mask
        )
        return data, server_action_map


