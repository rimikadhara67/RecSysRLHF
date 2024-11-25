import yaml
from rlhf.ppo import PPO
from rlhf.policy import PolicyNetwork
from rlhf.reward_model import RewardModel
from rlhf.environment import CustomEnv
from rlhf.optimizer import PPOOptimizer
from utils.logger import log_metrics
from utils.data_processing import preprocess_data
import torch
import gym

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train(config):
    """Main training loop for PPO."""
    # Load environment
    env = CustomEnv(config)
    observation_space = env.env.observation_space.shape[0]
    action_space = env.env.action_space.n

    # Initialize policy and reward model
    policy = PolicyNetwork(input_dim=observation_space, action_dim=action_space, hidden_dim=config['policy_hidden_dim'])
    reward_model = RewardModel(input_dim=observation_space, hidden_dim=config['reward_model_hidden_dim'])
    optimizer = PPOOptimizer(policy, learning_rate=config['learning_rate'])

    # Initialize PPO
    ppo = PPO(policy, optimizer, clip_param=config['clip_param'], gamma=config['gamma'], lam=config['lam'])

    # Training loop
    for epoch in range(config['num_epochs']):
        states, actions, log_probs, rewards, values = [], [], [], [], []
        state = env.reset()
        total_reward = 0

        for step in range(config['max_steps']):
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action, log_prob = policy.act(state_tensor)
            value = reward_model(state_tensor)

            next_state, reward, done, _ = env.step(action.item())

            # Store trajectory
            states.append(state_tensor)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)

            state = next_state
            total_reward += reward

            if done:
                break

        # Compute advantages and update policy
        trajectories = [(states, actions, log_probs, rewards, values)]
        ppo.update(trajectories)

        # Log metrics
        if epoch % config['log_interval'] == 0:
            log_metrics(epoch, total_reward, ppo.policy_loss)

    print("Training complete!")

if __name__ == "__main__":
    # Load configuration
    config_path = "config/config.yaml"
    config = load_config(config_path)

    # Train the PPO agent
    train(config)
