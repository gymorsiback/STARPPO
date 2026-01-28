import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch_geometric.loader import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from model import GNNActorCritic

class PPOAgent:
    def __init__(self, 
                 device='cpu', 
                 lr=3e-4, 
                 node_feat_dim=1,
                 global_feat_dim=7,
                 hidden_dim=64):
        self.device = device
        self.net = GNNActorCritic(node_feat_dim=node_feat_dim, 
                                  global_feat_dim=global_feat_dim,
                                  hidden_dim=hidden_dim).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
    
    def act(self, graph_data, deterministic=False):
        self.net.eval()
        with torch.no_grad():
            if not hasattr(graph_data, 'batch') or graph_data.batch is None:
                graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long, device=self.device)
            
            graph_data = graph_data.to(self.device)
                
            actions, log_probs, values, _ = self.net.act(graph_data, deterministic)
            
        return actions, log_probs, values

    def update(self, memory, batch_size=64, gamma=0.99, gae_lambda=0.95, clip_param=0.2, ent_coef=0.01, update_iters=10):
        self.net.train()
        
        dataset = memory
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        total_loss = 0
        total_pol_loss = 0
        total_val_loss = 0
        total_ent = 0
        updates = 0
        
        for _ in range(update_iters):
            for batch in loader:
                batch = batch.to(self.device)

                new_log_probs, entropy, new_values = self.net.evaluate_actions(batch, batch.action_node_idx)
 
                advs = batch.advantage
                if advs.dim() > 1:
                    advs = advs.squeeze(-1)
                
                old_log_prob = batch.old_log_prob
                if old_log_prob.dim() > 1:
                    old_log_prob = old_log_prob.squeeze(-1)
                
                ret = batch.ret
                if ret.dim() > 1:
                    ret = ret.squeeze(-1)

                advs = (advs - advs.mean()) / (advs.std() + 1e-8)
  
                ratio = torch.exp(new_log_probs - old_log_prob)

                surr1 = ratio * advs
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advs
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (ret - new_values).pow(2).mean()

                ent_mean = entropy.mean()

                loss = policy_loss + 0.5 * value_loss - ent_coef * ent_mean
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_pol_loss += policy_loss.item()
                total_val_loss += value_loss.item()
                total_ent += ent_mean.item()
                updates += 1
                
        return {
            'loss': total_loss / max(updates, 1),
            'pol_loss': total_pol_loss / max(updates, 1),
            'val_loss': total_val_loss / max(updates, 1),
            'entropy': total_ent / max(updates, 1)
        }
