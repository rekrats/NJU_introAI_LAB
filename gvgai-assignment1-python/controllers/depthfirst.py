import copy
from copy import deepcopy
from collections import deque

D_ACTIONS = [1, 2, 3, 4]

class DFSNode:
    def __init__(self, env, parent=None, action=None):
        self.env = env
        self.parent = parent
        self.action = action
        if parent:
            self.actionlist = parent.actionlist + [action]
            self.depth = parent.depth + 1
        else:
            self.actionlist = []
            self.depth = 0

class DFSAgent:
    def __init__(self, env, tick_max):
        self.env = env
        self.tick_max = tick_max
        self.tick = 0
        # Your can add new attributes if needed
        self.visited = []
    # tips: 用 stack 实现 dfs，子节点入栈时父节点会丢失，所以需要在入栈时同时存储父节点的信息
    # 否则到了 goal 之后无法回溯
    
    def state_has_visited(self, state):
        for visited_state in self.visited:
            flag = True
            for row in range(len(state)):
                for col in range(len(state[0])):
                    if state[row][col] != visited_state[row][col]:
                        flag = False
                        break
            if flag:
                return True
        return False

    def dfs(self):
        root_env = deepcopy(self.env)
        root_env.reset()
        init_state = root_env._get_observation()
        init_node = DFSNode(root_env)
        ssstack = deque()
        ssstack.append(init_node)
        self.visited.append(init_state)

        while ssstack: # true means not empty
            if self.tick > self.tick_max:
                assert 0
            now_node = ssstack.pop()
            if now_node.depth > 50:
                continue
            now_env = now_node.env
            if (now_env.done) and (not now_env.goal_exists):
                return now_node.actionlist
            
            for action_id in D_ACTIONS:
                env_copy = deepcopy(now_env)
                obs_, reward, done_, _ = env_copy.step(action_id)
                self.tick += 1
                if done_:
                    if not env_copy.goal_exists:
                        return now_node.actionlist + [action_id]
                    else:
                        continue
                if self.state_has_visited(obs_):
                    continue
                self.visited.append(obs_)
                new_node = DFSNode(env_copy, now_node, action_id)
                ssstack.append(new_node)
            if self.tick % 500 == 0:
                print(f"Used steps: {self.tick} / {self.tick_max}")
    

        

    def solve(self):
        state = self.env.reset()  # Reset environment to start a new episode
        actions = self.env.action_space
        action_sequence = self.dfs()
        return action_sequence