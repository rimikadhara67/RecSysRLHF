import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

    def act(self, state):
        logits = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).squeeze()
        return action, torch.log(probs[action])

    def evaluate(self, states, actions):
        logits = self.forward(states)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs)
        entropy = -(probs * log_probs).sum(dim=-1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        return selected_log_probs, entropy
