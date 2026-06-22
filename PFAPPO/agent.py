import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Categorical
from model import Actor, Critic

class PFAPPOAgent:
    def __init__(self, state_dim, num_servers, lr=3e-4, gamma=0.99, clip_param=0.2, target_kl=0.01, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.clip_param = clip_param
        self.target_kl = target_kl

        self.actor = Actor(state_dim, num_servers).to(device)
        self.critic = Critic(state_dim, num_servers).to(device)

        # PPO usually uses same LR for both, or slightly different
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.mse_loss = nn.MSELoss()

    def act(self, state, resource_weights, guidance_alpha=1.0, guidance_temperature=0.5):
        """
        state: [1, state_dim]
        resource_weights: [1, num_servers]
        guidance_alpha: 引导强度 (0~1)
            - 0: 完全靠 Actor 网络决策
            - 1: resource_weights 有很强的引导作用
        guidance_temperature: 引导温度，控制 resource_weights 的影响力
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if not isinstance(resource_weights, torch.Tensor):
            resource_weights = torch.tensor(resource_weights, dtype=torch.float32)

        state = state.to(self.device)
        resource_weights = resource_weights.to(self.device)

        with torch.no_grad():
            logits = self.actor(state, resource_weights)

            # ========================================
            # 关键：直接用 resource_weights 调整 logits！
            # ========================================
            # 不是让网络"学习"如何使用 resource_weights，而是直接引导
            # resource_weights 高的服务器 → logits 更高 → 更容易被选中
            # guidance_alpha 控制引导强度
            if guidance_alpha > 0:
                # 将 resource_weights 归一化到 [-1, 1] 范围
                rw_centered = resource_weights - resource_weights.mean(dim=-1, keepdim=True)
                rw_std = resource_weights.std(dim=-1, keepdim=True) + 1e-6
                rw_normalized = rw_centered / rw_std

                # 直接加到 logits 上
                logits = logits + guidance_alpha * rw_normalized * guidance_temperature

            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.critic(state, resource_weights)

        return action.item(), log_prob.item(), value.item()

    def update(self, memory, batch_size=64, epochs=10, entropy_coef=0.01):
        # Unpack memory
        # memory is a list of tuples or dicts. Let's assume list of dicts for clarity.
        # But for efficiency, train.py should probably stack them into tensors before calling update
        # To keep it simple and consistent with other algos, let's assume tensors are passed directly or list of transitions.

        # We will restructure update to take tensors directly for efficiency
        pass # Will be implemented in train.py loop or helper function

    def update_from_batch(self, states, resource_weights, actions, old_log_probs, returns, advantages, entropy_coef=0.01):
        """
        Perform PPO update on a batch of collected data
        """
        states = states.to(self.device)
        resource_weights = resource_weights.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Current policy
        logits = self.actor(states, resource_weights)
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
        values = self.critic(states, resource_weights).squeeze()
        critic_loss = self.mse_loss(values, returns)

        # Total Loss
        loss = actor_loss + 0.5 * critic_loss - entropy_coef * dist_entropy

        # Update
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        # Clip grad norm? Optional but good for stability
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return loss.item(), actor_loss.item(), critic_loss.item(), dist_entropy.item()

