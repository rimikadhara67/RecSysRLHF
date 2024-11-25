def compute_discounted_rewards(rewards, gamma=0.99):
    discounted_rewards = []
    cumulative = 0
    for r in reversed(rewards):
        cumulative = r + gamma * cumulative
        discounted_rewards.insert(0, cumulative)
    return discounted_rewards

def compute_entropy(probs):
    return -(probs * probs.log()).sum(dim=-1).mean()
