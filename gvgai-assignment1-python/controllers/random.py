import random

class RandomAgent:
    def __init__(self, env, tick_max):
        self.env = env
        self.tick_max = tick_max
        self.tick = 0

    def act(self, env):
        actions = self.env.action_space
        return random.choice(actions)