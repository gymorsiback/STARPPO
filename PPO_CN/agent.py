import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from model import Actor, Critic

class PPO_CN_Agent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_param=0.2, target_kl=0.01, device='cpu', noise_explorer=None):
        self.device = device
        self.gamma = gamma
        self.clip_param = clip_param
        self.target_kl = target_kl
        self.noise_explorer = noise_explorer
        
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()

    def act(self, state, use_noise=True):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.device)
        
        with torch.no_grad():
            logits_clean = self.actor(state)

            dist_clean = Categorical(logits=logits_clean)

            if use_noise and self.noise_explorer is not None:
                current_scale = getattr(self.noise_explorer.noise_process, 'scale', 0)
                if current_scale > 1e-6:
                    logits_noisy = self.noise_explorer.add_noise_to_logits(logits_clean)
                    dist_noisy = Categorical(logits=logits_noisy)
                    action = dist_noisy.sample()
                else:
                    action = dist_clean.sample()
            else:
                action = dist_clean.sample()

            log_prob = dist_clean.log_prob(action)

            value = self.critic(state)
            
        return action.item(), log_prob.item(), value.item()

    def update_from_batch(self, states, actions, old_log_probs, returns, advantages, entropy_coef=0.01):
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        logits = self.actor(states)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy().mean()

        ratio = torch.exp(new_log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        values = self.critic(states).squeeze()
        critic_loss = self.mse_loss(values, returns)

        loss = actor_loss + 0.5 * critic_loss - entropy_coef * dist_entropy

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        return loss.item(), actor_loss.item(), critic_loss.item(), dist_entropy.item()

