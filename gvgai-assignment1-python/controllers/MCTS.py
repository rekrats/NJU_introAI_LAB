import random
import math
import copy

ACTIONS = [1, 2, 3, 4]

class Node:
    def __init__(self, move='', env='', parent=None):
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.score = 0
        self.env = env
        if parent is None:
            self.state = []
        else:
            self.state = self.parent.state + [move]

    def is_fully_expanded(self):
        return len(self.children) == len(ACTIONS)

    def expand(self):
        moves_tried = [child.move for child in self.children]
        untried_moves = [move for move in ACTIONS if move not in moves_tried]
        move = random.choice(untried_moves)
        env_child = copy.deepcopy(self.env)
        env_child.step(move)
        child_node = Node(move=move, env=env_child, parent=self)
        self.children.append(child_node)
        return child_node

    def simulate(self):
        if self.env.done:
            return (0, 1) if self.env.goal_exists else (5, 1)
        env = copy.deepcopy(self.env)
        score = 0
        done = False
        used_ticks = 0
        while not done:
            action = random.choice(ACTIONS)
            _, reward, done, _ = env.step(action)
            used_ticks += 1
            score += reward
            if done or len(self.state) + used_ticks > 20:
                break
        return score, used_ticks

    def backpropagate(self, result):
        self.visits += 1
        self.score += result
        if self.parent:
            self.parent.backpropagate(result)

    def print_tree(self, indent=0):
        prefix = '    ' * indent
        print(f"{prefix}Move: {self.move}, Visits: {self.visits}, Score: {self.score}")
        for child in self.children:
            child.print_tree(indent + 1)

class MCTS:
    def __init__(self, tick_max, env):
        self.root = Node(move=None, env=env)
        self.tick_max = tick_max

    def select(self):
        node = self.root
        while node.is_fully_expanded():
            node = max(node.children, key=self.ucb1)
        return node

    def ucb1(self, node):
        Q = node.score / (node.visits + 1e-5)
        N = node.parent.visits
        n = node.visits
        c = math.sqrt(2)
        value = Q + c * math.sqrt(math.log(N + 1) / (n + 1e-5))
        return value + random.random()

    def run(self):
        used_ticks_total = 0
        while used_ticks_total < self.tick_max:
            leaf = self.select()
            if not leaf.is_fully_expanded():
                leaf = leaf.expand()
            score, used_ticks = leaf.simulate()
            leaf.backpropagate(score)
            used_ticks_total += used_ticks

class MCTSAgent:
    def __init__(self, env, tick_max):
        self.env = env
        self.tick_max = tick_max

    def solve(self):
        self.env.reset()
        self.mcts = MCTS(self.tick_max, self.env)
        self.mcts.run()

        # Print the tree after the search
        print("MCTS Tree Structure:")
        self.mcts.root.print_tree()

        node = self.mcts.root
        action_sequence = []

        while node.children:
            node = max(node.children, key=lambda node: node.score)
            action_sequence.append(node.move)

        return action_sequence
    
    def act(self, env):

        self.mcts = MCTS(self.tick_max, copy.deepcopy(env))
        self.mcts.run()

        node = self.mcts.root

        node = max(node.children, key=lambda node: node.visits)

        return node.move

