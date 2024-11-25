import torch
from torch.distributions import Categorical

class PPO:
    def __init__(self, policy, optimizer, clip_param=0.2, gamma=0.99, lam=0.95):
        self.policy = policy
        self.optimizer = optimizer
        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam

    def compute_advantages(self, rewards, values):
        advantages = []
        returns = 0
        for r, v in zip(reversed(rewards), reversed(values)):
            returns = r + self.gamma * returns
            advantages.insert(0, returns - v)
        return torch.tensor(advantages, dtype=torch.float32)

    def update(self, trajectories):
        for states, actions, log_probs, rewards, values in trajectories:
            advantages = self.compute_advantages(rewards, values)
            for _ in range(10):  # Update steps
                new_log_probs, entropy = self.policy.evaluate(states, actions)
                ratio = torch.exp(new_log_probs - log_probs)
                surrogate = ratio * advantages
                clipped_surrogate = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
                loss = -torch.min(surrogate, clipped_surrogate).mean()
                loss += 0.01 * entropy.mean()  # Entropy bonus
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
