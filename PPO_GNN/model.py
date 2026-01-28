import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from torch_geometric.data import Batch

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class GNNEncoder(nn.Module):
    def __init__(self, node_dim, hidden_dim, edge_dim=1, num_layers=2):
        super().__init__()
        
        self.convs = nn.ModuleList()

        self.convs.append(GINEConv(
            nn.Sequential(
                nn.Linear(node_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ), 
            train_eps=True,
            edge_dim=edge_dim  
        ))

        for _ in range(num_layers - 1):
            self.convs.append(GINEConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                ), 
                train_eps=True,
                edge_dim=edge_dim  
            ))

    def forward(self, x, edge_index, edge_attr):
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            
        return x

class GNNActorCritic(nn.Module):
    def __init__(self, 
                 node_feat_dim=3,  
                 global_feat_dim=7, 
                 hidden_dim=128,
                 gnn_layers=2):
        super().__init__()
        
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        total_node_dim = node_feat_dim + global_feat_dim
        
        self.encoder = GNNEncoder(total_node_dim, hidden_dim, edge_dim=1, num_layers=gnn_layers)
   
        self.actor_head = MLP(hidden_dim + global_feat_dim + node_feat_dim, hidden_dim, 1, num_layers=2)

        self.critic_head = MLP(hidden_dim * 2 + global_feat_dim, hidden_dim, 1, num_layers=2)

    def _manual_pool(self, node_embeds, num_graphs, num_nodes_per_graph):
        x_reshaped = node_embeds.view(num_graphs, num_nodes_per_graph, -1)
        mean_pool = x_reshaped.mean(dim=1)
        max_pool = x_reshaped.max(dim=1)[0]
        return mean_pool, max_pool

    def forward(self, batch_data):
        x, edge_index = batch_data.x, batch_data.edge_index
        edge_attr = batch_data.edge_attr
        
        if not hasattr(batch_data, 'batch') or batch_data.batch is None:
            batch_data.batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        batch = batch_data.batch
        global_feat = batch_data.global_feat 
        
        if hasattr(batch_data, 'num_graphs'):
            num_graphs = batch_data.num_graphs
        else:
            num_graphs = 1
        
        total_nodes = x.size(0)
        num_nodes_per_graph = total_nodes // num_graphs

        original_node_feat = x[:, :self.node_feat_dim]
  
        x_global = global_feat[batch]
        x_cat = torch.cat([x, x_global], dim=1)

        node_embeds = self.encoder(x_cat, edge_index, edge_attr)
 
        actor_input = torch.cat([node_embeds, x_global, original_node_feat], dim=1)
        logits = self.actor_head(actor_input).squeeze(-1)

        graph_embed_mean, graph_embed_max = self._manual_pool(node_embeds, num_graphs, num_nodes_per_graph)
        
        if global_feat.shape[0] != num_graphs:
            if global_feat.shape[0] == 1:
                critic_global = global_feat.expand(num_graphs, -1)
            else:
                critic_global = global_feat
        else:
            critic_global = global_feat
            
        critic_input = torch.cat([graph_embed_mean, graph_embed_max, critic_global], dim=1)
        value = self.critic_head(critic_input).squeeze(-1)
        
        return logits, value

    def act(self, batch_data, deterministic=False):
        logits, value = self.forward(batch_data)
        
        if hasattr(batch_data, 'num_graphs'):
            num_graphs = batch_data.num_graphs
        else:
            num_graphs = 1
            
        total_nodes = logits.size(0)
        num_nodes = total_nodes // num_graphs
        
        logits = logits.view(num_graphs, num_nodes)
        mask = batch_data.candidate_mask.view(num_graphs, num_nodes)
        
        logits[~mask] = -float('inf')
        
        probs = F.softmax(logits, dim=1)
        
        if torch.isnan(probs).any():
            probs[torch.isnan(probs)] = 0.0
            
        dist = torch.distributions.Categorical(probs=probs)
        
        if deterministic:
            actions_local = torch.argmax(probs, dim=1)
        else:
            actions_local = dist.sample()
            
        return actions_local, dist.log_prob(actions_local), value, probs

    def evaluate_actions(self, batch_data, actions):
        logits, values = self.forward(batch_data)
  
        if actions.dim() > 1:
            actions = actions.squeeze(-1)
        if actions.dim() == 0:
            actions = actions.unsqueeze(0)
        
        if hasattr(batch_data, 'num_graphs'):
            num_graphs = batch_data.num_graphs
        else:
            num_graphs = 1
            
        total_nodes = logits.size(0)
        num_nodes = total_nodes // num_graphs
        
        logits = logits.view(num_graphs, num_nodes)
        mask = batch_data.candidate_mask.view(num_graphs, num_nodes)
        
        logits[~mask] = -float('inf')
        
        probs = F.softmax(logits, dim=1)
        
        if torch.isnan(probs).any():
             probs[torch.isnan(probs)] = 0.0
             
        dist = torch.distributions.Categorical(probs=probs)
        
        log_probs = dist.log_prob(actions)
        entropies = dist.entropy()
        
        return log_probs, entropies, values
