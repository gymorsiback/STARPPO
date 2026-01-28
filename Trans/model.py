import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  
        pos_embed = self.pos_embedding(positions)  
        return x + pos_embed

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=1, max_seq_len=20):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=128, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)  
        x = self.transformer(x)
        x = x[:, -1, :] 
        x = self.norm(x)
        return x

class TransformerActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, d_model=64):
        super(TransformerActorCritic, self).__init__()
        self.encoder = TransformerEncoder(state_dim, d_model)

        self.actor_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, action_dim)
        )

        self.critic_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )
        
    def forward(self, state_seq):
        feat = self.encoder(state_seq)
        return self.actor_head(feat), self.critic_head(feat)
    
    def get_action_logits(self, state_seq):
        feat = self.encoder(state_seq)
        return self.actor_head(feat)
    
    def get_value(self, state_seq):
        feat = self.encoder(state_seq)
        return self.critic_head(feat)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, d_model=64):
        super(Actor, self).__init__()
        self.encoder = TransformerEncoder(state_dim, d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, action_dim)
        )
        
    def forward(self, state_seq):
        feat = self.encoder(state_seq)
        return self.head(feat)

class Critic(nn.Module):
    def __init__(self, state_dim, d_model=64):
        super(Critic, self).__init__()
        self.encoder = TransformerEncoder(state_dim, d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )
        
    def forward(self, state_seq):
        feat = self.encoder(state_seq)
        return self.head(feat)
