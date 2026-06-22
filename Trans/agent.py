import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Categorical
from model import TransformerActorCritic

class TransPPOAgent:
    def __init__(self, state_dim, num_servers, lr=3e-4, gamma=0.99, clip_param=0.2, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.clip_param = clip_param

        # 使用共享Encoder的Actor-Critic
        self.model = TransformerActorCritic(state_dim, num_servers).to(device)

        # 单一优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.mse_loss = nn.MSELoss()

    def act(self, state_seq):
        """
        state_seq: [1, Seq, Feat]
        """
        if not isinstance(state_seq, torch.Tensor):
            state_seq = torch.tensor(state_seq, dtype=torch.float32)

        state_seq = state_seq.to(self.device)

        with torch.no_grad():
            logits, value = self.model(state_seq)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def update_from_batch(self, state_seqs, actions, old_log_probs, returns, advantages, entropy_coef=0.01):
        """
        state_seqs: [Batch, Seq, Feat]
        """
        state_seqs = state_seqs.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Forward pass (共享encoder)
        logits, values = self.model(state_seqs)
        values = values.squeeze()

        # Policy distribution
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy().mean()

        # PPO Ratio
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Clipped Surrogate Loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Critic Loss
        critic_loss = self.mse_loss(values, returns)

        # Total Loss
        loss = actor_loss + 0.5 * critic_loss - entropy_coef * dist_entropy

        # Single backward pass
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        return loss.item(), actor_loss.item(), critic_loss.item(), dist_entropy.item()
