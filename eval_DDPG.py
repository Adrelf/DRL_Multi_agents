from brain.DDPG_agent import Agent
from brain.DDPG_agent import ReplayBuffer
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

# Simulation parameters
num_episodes = 1

# Load the environment
env = UnityEnvironment(file_name="./Tennis_Linux/Tennis.x86_64")

# Get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# Number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# Size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# Examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

# Create agents
random_seed = 0
agent =  []
for i in range(num_agents):
    agent.append(Agent(state_size=state_size, action_size=action_size, random_seed=random_seed))

for i_agent in range(num_agents):
  filename = './saved_models/ddpg_actor_' + str(i_agent) + '.pth'
  agent[i_agent].actor_local.load_state_dict(torch.load(filename))
  filename = './saved_models/ddpg_critic_' + str(i_agent) + '.pth'
  agent[i_agent].critic_local.load_state_dict(torch.load(filename))

for i in range(num_episodes):
  env_info  = env.reset(train_mode=False)[brain_name]
  states    = env_info.vector_observations  # get the initial state
  score     = 0
  while True:
    actions = []

    for i_agent in range(num_agents):
      actions.append(agent[i_agent].act(states[i_agent],add_noise=False))

    # Retrieve useful information
    env_info    = env.step(actions)[brain_name]     # send all actions to the environment
    states      = env_info.vector_observations      # get next state (for each agent)
    rewards     = env_info.rewards                  # get reward (for each agent)
    dones       = env_info.local_done               # see if episode finished

    score += np.max(rewards)
    if np.any(dones):
      break

  print('Episode = {}'.format(i))
  print('Score = {}'.format(score))

env.close()