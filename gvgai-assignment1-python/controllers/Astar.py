class AstarAgent:
    def __init__(self, env, tick_max):
        self.env = env
        self.tick_max = tick_max
        self.tick = 0
        # Your can add new attributes if needed

    def astar(self):

        if self.tick > self.tick_max:
            assert 0

        raise NotImplementedError

    def solve(self):
        state = self.env.reset()  # Reset environment to start a new episode
        actions = self.env.action_space
        
        raise NotImplementedError
    
        action_sequence = self.astar()
        return action_sequence

    def act(self, env):
        
        raise NotImplementedError