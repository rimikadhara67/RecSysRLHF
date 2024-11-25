import torch.optim as optim

class PPOOptimizer:
    def __init__(self, policy, learning_rate=3e-4):
        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    def step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
