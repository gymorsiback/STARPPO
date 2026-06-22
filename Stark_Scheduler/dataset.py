import torch
import numpy as np
from torch.utils.data import Dataset

class ExpertPolicy:
    """
    Expert Policy: Weighted Best-Fit for WorkflowMoEEnv
    Selects model instance based on a heuristic score:
    Score = -w_lat * latency - w_cost * cost - w_switch * switch_penalty
    """
    def __init__(self, w_lat=1.0, w_cost=1.5, w_switch=0.2):
        # Increased w_cost to prioritize lower cost (aligned with PFAPPO's DWA)
        self.w_lat = w_lat
        self.w_cost = w_cost
        self.w_switch = w_switch

    def select_action(self, env):
        """
        Selects the best action (model instance index) for the current state.
        """
        available = env.available_actions()
        if not available:
            return 0

        best_score = -float('inf')
        best_action = available[0]

        for action_idx in available:
            # Estimate cost components
            latency_ms, cost, switch_penalty_ms = env.estimate_step(action_idx)

            # Calculate score (higher is better, so negate costs)
            # 注意：不再对 cost 乘以 1000，使权重与 RL reward 一致
            score = -self.w_lat * latency_ms - self.w_cost * cost - self.w_switch * switch_penalty_ms

            if score > best_score:
                best_score = score
                best_action = action_idx

        return best_action


def get_available_server_mask(env):
    """
    Returns a mask indicating which servers have the current required model type.
    This ensures the model only predicts from valid servers.
    """
    available = env.available_actions()
    server_ids = sorted(list(env.servers.keys()))

    # Create mask: 1 if server has available action, 0 otherwise
    mask = np.zeros(len(server_ids), dtype=np.float32)

    available_server_ids = set()
    for action_idx in available:
        mi = env.actions[action_idx]
        available_server_ids.add(mi.server_id)

    for i, sid in enumerate(server_ids):
        if sid in available_server_ids:
            mask[i] = 1.0

    return mask, available


class OnlineExpertDataset(Dataset):
    """
    Generates data on-the-fly using the Expert Policy in WorkflowMoEEnv.
    Returns (task_features, server_features, action_label) tuples.
    Now includes server availability mask to ensure valid predictions.
    """
    def __init__(self, env, steps_per_epoch=2000):
        self.env = env
        self.expert = ExpertPolicy()
        self.steps_per_epoch = steps_per_epoch
        self.data_buffer = []

    def __len__(self):
        return len(self.data_buffer)

    def __getitem__(self, idx):
        return self.data_buffer[idx]

    def generate_epoch_data(self):
        """Run the environment and collect (State, Action) pairs using expert"""
        self.data_buffer = []

        # Sample random tasks from dataset
        tasks = self.env.ds.tasks
        num_episodes = max(1, self.steps_per_epoch // 5)  # Assume ~5 steps per task on average

        for _ in range(num_episodes):
            # Random task
            task = np.random.choice(tasks)
            state_dict = self.env.reset(task)
            done = False

            while not done and len(self.data_buffer) < self.steps_per_epoch:
                # 1. Get Expert Action
                action = self.expert.select_action(self.env)

                # 2. Extract Features for Stark model (including availability mask)
                task_feat, server_feats, action_label, avail_mask = self.extract_structured_state(self.env, action)

                # 3. Store
                self.data_buffer.append({
                    'task_feat': torch.FloatTensor(task_feat),
                    'server_feats': torch.FloatTensor(server_feats),
                    'label': torch.tensor(action_label, dtype=torch.long),
                    'avail_mask': torch.FloatTensor(avail_mask)
                })

                # 4. Step Env with expert action
                next_state_dict, rewards, done, info = self.env.step(action)
                state_dict = next_state_dict

                if len(self.data_buffer) >= self.steps_per_epoch:
                    break

    def extract_structured_state(self, env, action_idx=None):
        """
        Extracts:
        1. Task Features: [step_norm, task_lon, task_lat, total_steps]
        2. Server Features: [num_servers, feature_dim] where features include
           [lon, lat, normalized_compute, cost_multiplier, latency_history, distance_to_task, is_available]
        3. Action Label: Map action_idx (model instance) to server index (only if action_idx provided)
        4. Availability Mask: Which servers have the required model type

        For inference (no action_idx), returns only (task_feat, server_feats).
        For training (with action_idx), returns (task_feat, server_feats, action_label, avail_mask).
        """
        state = env._get_state()

        # Get availability mask
        avail_mask, available_actions = get_available_server_mask(env)

        # Task features
        task_feat = [
            state['step_norm'],
            state['task_lon'] / 180.0,  # Normalize lon
            state['task_lat'] / 90.0,   # Normalize lat
            min(state['total_steps'] / 10.0, 1.0)  # Normalize steps (cap at 10)
        ]

        # Server features (now includes availability)
        server_feats = []
        server_ids = sorted(list(env.servers.keys()))

        # Get task location
        task_lon = state['task_lon']
        task_lat = state['task_lat']

        for i, sid in enumerate(server_ids):
            server = env.servers[sid]

            # Calculate distance to task
            from utils import haversine_km
            dist_km = haversine_km(task_lon, task_lat, server.lon, server.lat)

            # Get latency history for this server
            latency_hist = state['server_latency_history'].get(sid, env.ds.mean_base_latency_ms * 5.0)

            s_feat = [
                server.lon / 180.0,  # Normalized longitude
                server.lat / 90.0,   # Normalized latitude
                server.normalized_compute,  # Already normalized
                server.cost_multiplier / 4.0,  # Normalize by max (0.3~3.5 range)
                latency_hist / 1000.0,  # Normalize latency (in seconds)
                dist_km / 1000.0,  # Normalize distance (in 1000km)
                avail_mask[i]  # Is this server available for current model type?
            ]
            server_feats.append(s_feat)

        # For inference: return only state features
        if action_idx is None:
            return np.array(task_feat, dtype=np.float32), np.array(server_feats, dtype=np.float32)

        # For training: also return action label
        model_instance = env.actions[action_idx]
        server_id = model_instance.server_id
        action_label = server_ids.index(server_id)

        return np.array(task_feat, dtype=np.float32), np.array(server_feats, dtype=np.float32), action_label, avail_mask


def collate_fn(batch):
    """Collate batch of samples"""
    task_feats = torch.stack([item['task_feat'] for item in batch])
    server_feats = torch.stack([item['server_feats'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    avail_masks = torch.stack([item['avail_mask'] for item in batch])
    return task_feats, server_feats, labels, avail_masks

    """Collate batch of samples"""
    task_feats = torch.stack([item['task_feat'] for item in batch])
    server_feats = torch.stack([item['server_feats'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    avail_masks = torch.stack([item['avail_mask'] for item in batch])
    return task_feats, server_feats, labels, avail_masks

    """Collate batch of samples"""
    task_feats = torch.stack([item['task_feat'] for item in batch])
    server_feats = torch.stack([item['server_feats'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    avail_masks = torch.stack([item['avail_mask'] for item in batch])
    return task_feats, server_feats, labels, avail_masks
