import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, state_dim, num_servers, hidden_dim=256):
        super(Actor, self).__init__()
        input_dim = state_dim + num_servers
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.actor_head = nn.Linear(hidden_dim // 2, num_servers)

    def forward(self, state, resource_weights):
        x = torch.cat([state, resource_weights], dim=1)
        
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        
        logits = self.actor_head(x)
        return logits

class Critic(nn.Module):
    def __init__(self, state_dim, num_servers, hidden_dim=256):
        super(Critic, self).__init__()
        input_dim = state_dim + num_servers
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.value_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, state, resource_weights):
        x = torch.cat([state, resource_weights], dim=1)
        
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        
        value = self.value_head(x)
        return value

