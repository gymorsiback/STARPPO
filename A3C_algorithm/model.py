import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, state_dim, num_servers, hidden_dim=256):
        super(ActorCritic, self).__init__()
   
        self.actor_fc1 = nn.Linear(state_dim, hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.actor_fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.actor_head = nn.Linear(hidden_dim // 2, num_servers)

        self.critic_fc1 = nn.Linear(state_dim, hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.critic_fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.value_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, state):

        x = torch.tanh(self.actor_fc1(state))
        x = torch.tanh(self.actor_fc2(x))
        x = torch.tanh(self.actor_fc3(x))
        logits = self.actor_head(x)

        v = torch.tanh(self.critic_fc1(state))
        v = torch.tanh(self.critic_fc2(v))
        v = torch.tanh(self.critic_fc3(v))
        value = self.value_head(v)
        
        return logits, value



