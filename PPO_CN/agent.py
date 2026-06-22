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
        """
        state: [batch, state_dim]
        use_noise: Boolean, whether to apply colored noise

        标准的 Exploration Noise 实现（PPO 兼容）：
        1. 使用【干净的 logits】计算 log_prob（用于训练）
        2. 使用【加噪声的 logits】仅用于采样（增强探索）

        这样 old_log_prob 和 new_log_prob 都基于干净策略，ratio 计算正确。
        噪声只影响"探索什么动作"，不影响"策略梯度方向"。
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.device)

        with torch.no_grad():
            # 1. Get raw logits (clean policy)
            logits_clean = self.actor(state)

            # 2. Create clean distribution for log_prob calculation
            dist_clean = Categorical(logits=logits_clean)

            # 3. Sampling: use noisy logits if noise is enabled
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

            # 4. log_prob from CLEAN distribution (critical for PPO correctness)
            log_prob = dist_clean.log_prob(action)

            # 5. Value
            value = self.critic(state)

        return action.item(), log_prob.item(), value.item()

    def update_from_batch(self, states, actions, old_log_probs, returns, advantages, entropy_coef=0.01):
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Current policy (No noise during update! We want to learn the deterministic backbone)
        logits = self.actor(states)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy().mean()

        # Ratio
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Surrogate Loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Critic Loss
        values = self.critic(states).squeeze()
        critic_loss = self.mse_loss(values, returns)

        # Total Loss
        loss = actor_loss + 0.5 * critic_loss - entropy_coef * dist_entropy

        # Update
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return loss.item(), actor_loss.item(), critic_loss.item(), dist_entropy.item()

