class LimitedDFSAgent:
    def __init__(self, env, tick_max):
        self.env = env
        self.tick_max = tick_max
        self.tick = 0
        # Your can add new attributes if needed

    def limiteddfs(self):

        if self.tick > self.tick_max:
            assert 0

        raise NotImplementedError

        # next_state, reward, isOver, info = env_copy.step(action_id)
        # self.tick += 1
        # print(f"Used steps: {self.n} / {self.tick_max}")

    def solve(self):
        state = self.env.reset()  # Reset environment to start a new episode
        actions = self.env.action_space
        
        raise NotImplementedError
    
        action_sequence = self.limiteddfs()
        return action_sequence

    def act(self, env):
        
        raise NotImplementedError