import copy

from env import BaitEnv

from controllers.random import RandomAgent
from controllers.depthfirst import DFSAgent
from controllers.limitdepthfirst import LimitedDFSAgent
from controllers.Astar import AstarAgent
from controllers.MCTS import MCTSAgent

if __name__ == "__main__":
    
    print("Game start!")
    level = 1
    env = BaitEnv(level=level, render=False)
    
    # actions: 0 noop, 1 left, 2 right, 3 down, 4 up
    
    mode = "Astar" # "play", "random", "depthfirst", "limitdepthfirst", "Astar", "MCTS"
    action_lst = None
    if mode == "play":
        # input your own actions here
        tick_max = 30
        # action_lst = [3, 2, 3, 1, 3, 4, 4, 4, 1, 0]
        action_lst =  [3, 2, 2, 2, 4, 2, 2, 3, 1, 1, 1, 1, 4, 4, 1, 1, 3, 3, 2, 2, 4, 4, 1, 1, 3, 3, 1, 1, 4, 1, 1, 3, 2, 2, 2, 2, 4, 4, 2, 2, 3, 3, 2, 2, 1, 1, 4, 1, 3, 2, 2, 2, 4, 2, 2, 3, 1, 1, 1, 1, 4, 1, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4]
    elif mode == "random":
        tick_max = 30
        agent = RandomAgent(env, tick_max)
    elif mode == "depthfirst":
        tick_max = 10000
        agent = DFSAgent(env, tick_max)
        action_lst = agent.solve()
    elif mode == "limitdepthfirst":
        tick_max = 100
        agent = LimitedDFSAgent(env, tick_max)
    elif mode == "Astar":
        tick_max = 100
        agent = AstarAgent(env, tick_max)
    elif mode == "MCTS":
        tick_max = 1000
        agent = MCTSAgent(env, tick_max)

    print("Action list:", action_lst)
    action_lst_len = len(action_lst) if action_lst else 1e8
    print("Action list length:", action_lst_len)

    env = BaitEnv(level=level, render=True)
    env.reset()
    for step in range(min(100, action_lst_len)):
        if action_lst:
            action_id = action_lst[step]
        else:
            env_copy = copy.deepcopy(env)
            env_copy.render = False
            action_id = agent.act(env_copy)
        state, reward, isOver, info = env.step(action_id)
        print(f"Step: {step}, Action taken: {action_id}, Reward: {reward}, Done: {isOver}, Info: {info}")
        if isOver:
            break

    env.make_gif()