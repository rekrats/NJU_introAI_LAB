import gym
from collections import deque
import random
import argparse
import torch

from agent import DQNAgent, DDQNAgent

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_name", type=str, default="dqn")
    parser.add_argument("--num_episodes", type=int, default=600)
    parser.add_argument("--max_steps_per_episode", type=int, default=500)
    parser.add_argument("--epsilon_start", type=float, default=0.9)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--epsilon_decay_rate", type=float, default=0.99)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--update_frequency", type=int, default=10)

    args = parser.parse_args()
    return args


def eval_policy(agent):
    state = env.reset()
    done = False
    return_ = 0
    while not done:
        action = agent.act(state, eps=0.)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        return_ += reward
    # print(f"Return {return_}") 
    return return_


def train(args, agent, buffer):
    # Training loop
    for episode in range(args.num_episodes):
        # Reset the environment
        state = env.reset()
        epsilon = max(args.epsilon_end, args.epsilon_start * (args.epsilon_decay_rate ** episode))

        # Run one episode
        losses = []
        return_ = 0
        for step in range(args.max_steps_per_episode):
            # Choose and perform an action
            action = agent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            
            buffer.append((state, action, reward, next_state, done))
            
            if len(buffer) >= args.batch_size:
                batch = random.sample(buffer, args.batch_size)
                # Update the agent's knowledge
                loss = agent.learn(batch, args.gamma)
                losses.append(loss)
            return_ += reward
            
            state = next_state
            
            # Check if the episode has ended
            if done:
                break
        loss = torch.mean(torch.tensor(losses))
        eval_return = eval_policy(agent)

        print(f"Episode {episode + 1} Step {step + 1}: Training Loss {loss}, Return {eval_return}")

if __name__ == "__main__":
    args = parser()
    # Set up the environment
    env = gym.make("CartPole-v1")

    buffer = deque(maxlen=args.buffer_size)

    # Initialize the DQNAgent
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    if args.agent_name == "dqn":
        agent = DQNAgent(input_dim, output_dim, buffer_size=args.buffer_size, seed=1234, lr = args.lr)
    elif args.agent_name == "ddqn":
        agent = DDQNAgent(input_dim, output_dim, buffer_size=args.buffer_size, seed=1234, lr = args.lr)
    else:
        assert False, "Not Implement agent!"
    train(args, agent, buffer)