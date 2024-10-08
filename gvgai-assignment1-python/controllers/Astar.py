import copy
from copy import deepcopy
import heapq
D_ACTIONS = [1, 2, 3, 4]

class AstarNode:
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
            # Atar 算法中涉及到父节点的修改，记得在合适的时候调整父节点的属性（really?）
            self.actionlist = parent.actionlist + [action]
            self.depth = parent.depth + 1
        else:
            self.actionlist = []
            self.depth = 0

    def __lt__(self, other):
        return self.score < other.score

class AstarAgent:
    def __init__(self, env, tick_max):
        self.env = env
        self.tick_max = tick_max
        self.tick = 0
        # Your can add new attributes if needed
        self.maxdepth = 15 # level1: 15 level2: 25 level3: 28
        self.have_reached = [] # Agent 真正走到过的状态
        self.openlist = [] # 优先队列
        self.closedlist = [] # 在每次搜索开始，应当被重置为 have_reached
        self.map = self.env.map
        self.goalpos = None
        self.keypos = None
        for y, low in enumerate(self.map):
            for x, char in enumerate(low):
                if char == 'g':
                    self.goalpos = (x, y)
                if char == 'k':
                    self.keypos = (x, y)

    def state_equals(self, state1, state2):
        for row in range(len(state1)):
            for col in range(len(state1[0])):
                if state1[row][col] != state2[row][col]:
                    return False
        return True

    def state_is_closed(self, state):
        for visited_state in self.closedlist:
            if self.state_equals(state, visited_state):
                return True
        return False

    def _open_node_search(self, node):
        for i in range(len(self.openlist)):
            if self.state_equals(node.env._get_observation(), self.openlist[i].env._get_observation()):
                return i
        return -1
    
    def get_node_score(self, node):
        nodepos = (node.x, node.y)
        if node.haskey:
            return abs(nodepos[0] - self.goalpos[0]) + abs(nodepos[1] - self.goalpos[1]) 
        else:
            return abs(nodepos[0] - self.keypos[0]) + abs(nodepos[1] - self.keypos[1]) \
                + abs(self.goalpos[0] - self.keypos[0]) + abs(self.goalpos[1] - self.keypos[1])

    def astar(self):
        self.openlist = []
        # use deepcopy, not use =, to avoid reference(to see in note)
        self.closedlist = deepcopy(self.have_reached)
        rootnode = AstarNode(self.env)
        rootnode.score = self.get_node_score(rootnode)
        heapq.heappush(self.openlist, rootnode)
        bestnode = rootnode
        bestscore = rootnode.score
        while len(self.openlist) > 0:
            current = heapq.heappop(self.openlist)
            # print("Current depth and score:", current.depth, current.score)           
            if current.depth > self.maxdepth:
                continue
            self.closedlist.append(current.env._get_observation())
            for action in D_ACTIONS:
                newenv = deepcopy(current.env)
                _obs, reward, done, _ = newenv.step(action)
                if done:
                    if not newenv.goal_exists:
                        return current.actionlist + [action]
                    else:
                        continue
                if self.state_is_closed(_obs):
                    continue
                newnode = AstarNode(newenv, current, action)
                newnode.score = self.get_node_score(newnode)
                # print("New node score:", newnode.score)
                if newnode.score < bestscore:
                    bestnode = newnode
                    bestscore = newnode.score
                openindex = self._open_node_search(newnode)
                if openindex == -1:
                    heapq.heappush(self.openlist, newnode)
                else:
                    if newnode.score < self.openlist[openindex].score:
                        self.openlist[openindex] = newnode
        assert bestnode.actionlist is not None
        return bestnode.actionlist
                

    def solve(self):
        state = self.env.reset()  # Reset environment to start a new episode
        actions = self.env.action_space
        
        raise NotImplementedError
    
        action_sequence = self.astar()
        return action_sequence

    def act(self, env):
        self.env = env;
        self.have_reached.append(self.env._get_observation())
        actionlist = self.astar();
        print("Action list:", actionlist)
        return actionlist[0]