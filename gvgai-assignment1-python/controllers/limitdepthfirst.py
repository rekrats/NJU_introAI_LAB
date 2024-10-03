import copy
from copy import deepcopy
from collections import deque
D_ACTIONS = [1, 2, 3, 4]

class LDFSNode:
    def __init__(self, env, parent=None, action=None):
        self.env = env
        self.parent = parent
        self.action = action
        self.score = float(1e9)
        self.x, self.y = env.avatar_pos
        if 'avatar_withkey' in env.grid[self.y][self.x]:
            self.haskey = True
        else:
            self.haskey = False
        if parent:
            self.actionlist = parent.actionlist + [action]
            self.depth = parent.depth + 1
        else:
            self.actionlist = []
            self.depth = 0

class LimitedDFSAgent:
    def __init__(self, env, tick_max):
        self.env = env
        self.tick_max = tick_max
        self.tick = 0
        # Your can add new attributes if needed
        self.maxdepth = 4 # Maximum depth of the search tree
        self.visited = []
        self.map = self.env.map
        self.goalpos = None
        self.keypos = None
        for y, low in enumerate(self.map):
            for x, char in enumerate(low):
                if char == 'g':
                    self.goalpos = (x, y)
                if char == 'k':
                    self.keypos = (x, y)

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
    
    def get_node_score(self, node):
        nodepos = (node.x, node.y)
        if node.haskey:
            return abs(nodepos[0] - self.goalpos[0]) + abs(nodepos[1] - self.goalpos[1]) 
        else:
            return abs(nodepos[0] - self.keypos[0]) + abs(nodepos[1] - self.keypos[1]) + abs(self.goalpos[0] - self.keypos[0]) + abs(self.goalpos[1] - self.keypos[1])

    def limiteddfs(self):
        self.visited = []
        # 单纯在每次搜索前清空visited，或者根本不做这件事情都有缺陷
        # 这个问题应该在 Astar 的时候修一下。
        self.tick = 0
        root_state = self.env._get_observation()
        root_node = LDFSNode(self.env)
        root_node.score = 2e9 # root node cannot be the best node
        sstack = deque()
        sstack.append(root_node)
        self.visited.append(root_state)
        best_score = float(1e9)
        best_node = None
        while len(sstack) > 0:
            node = sstack.pop()
            # print("now node depth", node.depth)
            if self.tick > self.tick_max:
                break
            if node.depth > self.maxdepth:
                continue
            if node.depth == self.maxdepth:
                score = node.score
                # print("now node score", score)
                if score < best_score:
                    best_score = score
                    best_node = node
                continue
            for action in D_ACTIONS:
                newenv = deepcopy(node.env)
                obs_, reward, done, _ = newenv.step(action)
                self.tick += 1
                if done:
                    if not newenv.goal_exists:
                        return node.actionlist + [action]
                    else:
                        continue
                if self.state_has_visited(obs_):
                    continue
                self.visited.append(obs_)
                newnode = LDFSNode(newenv, node, action)
                newnode.score = self.get_node_score(newnode)
                sstack.append(newnode)
        assert best_node is not None
        return best_node.actionlist

    def solve(self):
        state = self.env.reset()  # Reset environment to start a new episode
        actions = self.env.action_space
        
        raise NotImplementedError
    
        action_sequence = self.limiteddfs()
        return action_sequence

    def act(self, env):
        self.env = env
        action_list = self.limiteddfs()
        return action_list[0]