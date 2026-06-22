import torch
import torch.nn as nn
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, state_dim, num_servers, hidden_dim=256):
        super(Actor, self).__init__()
        # PPO: 只接收 state_dim (7维)，不接收 resource_weights
        # 这是与 PFAPPO 的核心区别
        input_dim = state_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.actor_head = nn.Linear(hidden_dim // 2, num_servers)

    def forward(self, state):
        # state: [batch, state_dim]
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        logits = self.actor_head(x)
        return logits

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        # PPO: 只接收 state_dim (7维)
        input_dim = state_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.value_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        value = self.value_head(x)
        return value
