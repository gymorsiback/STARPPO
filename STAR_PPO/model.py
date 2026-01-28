import torch
import torch.nn as nn
import torch.nn.functional as F

class StarActor(nn.Module):
    def __init__(self, state_dim, num_servers, hidden_dim=256):
        super(StarActor, self).__init__()
        input_dim = state_dim + num_servers
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.se_fc1 = nn.Linear(hidden_dim, hidden_dim // 4)
        self.se_fc2 = nn.Linear(hidden_dim // 4, hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.actor_head = nn.Linear(hidden_dim // 2, num_servers)

    def forward(self, state, resource_weights):
        
        x = torch.cat([state, resource_weights], dim=1)
        
        x = F.relu(self.fc1(x))
    
        se = F.avg_pool1d(x.unsqueeze(1), kernel_size=1).squeeze(1) 
        se = F.relu(self.se_fc1(x))
        se = torch.sigmoid(self.se_fc2(se))
        x = x * se
        
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        logits = self.actor_head(x)
        return logits

class StarCritic(nn.Module):
    def __init__(self, state_dim, num_servers, hidden_dim=256):
        super(StarCritic, self).__init__()
        input_dim = state_dim + num_servers
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.value_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, state, resource_weights):
        x = torch.cat([state, resource_weights], dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        value = self.value_head(x)
        return value

