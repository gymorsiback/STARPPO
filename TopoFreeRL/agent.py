import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Categorical
from TopoFreeRL.model import StarActor, StarCritic

class StarPPOAgent:
    def __init__(self, state_dim, num_servers, lr=3e-4, gamma=0.99, clip_param=0.2, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.clip_param = clip_param

        self.actor = StarActor(state_dim, num_servers).to(device)
        self.critic = StarCritic(state_dim, num_servers).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.mse_loss = nn.MSELoss()

    def act(self, state, resource_weights, guidance_alpha=0.0):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if not isinstance(resource_weights, torch.Tensor):
            resource_weights = torch.tensor(resource_weights, dtype=torch.float32)

        state = state.to(self.device)
        resource_weights = resource_weights.to(self.device)

        with torch.no_grad():
            logits = self.actor(state, resource_weights)

            # Curriculum Guidance (Optional, inherited from PFAPPO idea)
            if guidance_alpha > 0:
                # Normalize RW
                rw_centered = resource_weights - resource_weights.mean(dim=-1, keepdim=True)
                rw_std = resource_weights.std(dim=-1, keepdim=True) + 1e-6
                rw_norm = rw_centered / rw_std
                # Guide
                logits = logits + guidance_alpha * rw_norm

            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.critic(state, resource_weights)

        return action.item(), log_prob.item(), value.item()

    def update_from_batch(self, states, weights, actions, old_log_probs, returns, advantages, entropy_coef=0.01):
        states = states.to(self.device)
        weights = weights.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        logits = self.actor(states, weights)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy().mean()

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        values = self.critic(states, weights).squeeze()
        critic_loss = self.mse_loss(values, returns)

        loss = actor_loss + 0.5 * critic_loss - entropy_coef * dist_entropy

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return loss.item()

