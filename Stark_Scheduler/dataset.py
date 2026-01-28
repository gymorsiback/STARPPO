import torch
import numpy as np
from torch.utils.data import Dataset

class ExpertPolicy:
    def __init__(self, w_lat=1.0, w_cost=1.5, w_switch=0.2):
        self.w_lat = w_lat
        self.w_cost = w_cost
        self.w_switch = w_switch

    def select_action(self, env):
        available = env.available_actions()
        if not available:
            return 0
            
        best_score = -float('inf')
        best_action = available[0]
        
        for action_idx in available:
            latency_ms, cost, switch_penalty_ms = env.estimate_step(action_idx)
 
            score = -self.w_lat * latency_ms - self.w_cost * cost - self.w_switch * switch_penalty_ms
            
            if score > best_score:
                best_score = score
                best_action = action_idx
                
        return best_action


def get_available_server_mask(env):
    available = env.available_actions()
    server_ids = sorted(list(env.servers.keys()))

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
        self.data_buffer = []
    
        tasks = self.env.ds.tasks
        num_episodes = max(1, self.steps_per_epoch // 5)  
        
        for _ in range(num_episodes):
            task = np.random.choice(tasks)
            state_dict = self.env.reset(task)
            done = False
            
            while not done and len(self.data_buffer) < self.steps_per_epoch:
                action = self.expert.select_action(self.env)
    
                task_feat, server_feats, action_label, avail_mask = self.extract_structured_state(self.env, action)
   
                self.data_buffer.append({
                    'task_feat': torch.FloatTensor(task_feat),
                    'server_feats': torch.FloatTensor(server_feats),
                    'label': torch.tensor(action_label, dtype=torch.long),
                    'avail_mask': torch.FloatTensor(avail_mask)
                })
     
                next_state_dict, rewards, done, info = self.env.step(action)
                state_dict = next_state_dict
                
                if len(self.data_buffer) >= self.steps_per_epoch:
                    break
            
    def extract_structured_state(self, env, action_idx=None):
        state = env._get_state()
  
        avail_mask, available_actions = get_available_server_mask(env)
  
        task_feat = [
            state['step_norm'],
            state['task_lon'] / 180.0,  
            state['task_lat'] / 90.0,   
            min(state['total_steps'] / 10.0, 1.0)  
        ]

        server_feats = []
        server_ids = sorted(list(env.servers.keys()))

        task_lon = state['task_lon']
        task_lat = state['task_lat']
        
        for i, sid in enumerate(server_ids):
            server = env.servers[sid]
 
            from utils import haversine_km
            dist_km = haversine_km(task_lon, task_lat, server.lon, server.lat)
   
            latency_hist = state['server_latency_history'].get(sid, env.ds.mean_base_latency_ms * 5.0)
            
            s_feat = [
                server.lon / 180.0,  
                server.lat / 90.0,   
                server.normalized_compute,  
                server.cost_multiplier / 4.0,  
                latency_hist / 1000.0,  
                dist_km / 1000.0,  
                avail_mask[i]  
            ]
            server_feats.append(s_feat)

        if action_idx is None:
            return np.array(task_feat, dtype=np.float32), np.array(server_feats, dtype=np.float32)

        model_instance = env.actions[action_idx]
        server_id = model_instance.server_id
        action_label = server_ids.index(server_id)
        
        return np.array(task_feat, dtype=np.float32), np.array(server_feats, dtype=np.float32), action_label, avail_mask


def collate_fn(batch):
    task_feats = torch.stack([item['task_feat'] for item in batch])
    server_feats = torch.stack([item['server_feats'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    avail_masks = torch.stack([item['avail_mask'] for item in batch])
    return task_feats, server_feats, labels, avail_masks

    task_feats = torch.stack([item['task_feat'] for item in batch])
    server_feats = torch.stack([item['server_feats'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    avail_masks = torch.stack([item['avail_mask'] for item in batch])
    return task_feats, server_feats, labels, avail_masks

    task_feats = torch.stack([item['task_feat'] for item in batch])
    server_feats = torch.stack([item['server_feats'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    avail_masks = torch.stack([item['avail_mask'] for item in batch])
    return task_feats, server_feats, labels, avail_masks
