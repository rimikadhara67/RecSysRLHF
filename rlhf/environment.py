import gym

class CustomEnv:
    def __init__(self, config):
        self.env = gym.make(config['env_name'])

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()
