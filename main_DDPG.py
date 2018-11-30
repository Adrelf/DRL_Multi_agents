from brain.DDPG_agent import Agent
from brain.DDPG_agent import ReplayBuffer
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment



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


random_seed = 0
BUFFER_SIZE  = int(1e6)   # replay buffer size
BATCH_SIZE    = 512       # minibatch size
memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

# Create agents
agent =  []
for i in range(num_agents):
    agent.append(Agent(state_size=state_size, action_size=action_size, random_seed=random_seed)) #, memory=memory)


""""
# Play game
for i in range(1, 6):                                      # play game for 5 episodes
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and +1
        env_info = env.step(actions)[brain_name]           # send all actions to the environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Score (max over agents) from episode {}: {}'.format(i, np.max(scores)))
"""


def ddpg(n_episodes=10000, max_t = 1000, num_agents=num_agents, solve=True):
  scores_deque = deque(maxlen=100)
  scores_list = []
  for i_episode in range(1, n_episodes + 1):
    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations
    scores = np.zeros(num_agents)
    for t in range(max_t):
      actions = []
      for i_agent in range(num_agents):
        agent[i_agent].reset()
        actions.append(agent[i_agent].act(states[i_agent]))

      # Retrieve useful information
      env_info    = env.step(actions)[brain_name]   # send all actions to tne environment
      next_states = env_info.vector_observations    # get next state (for each agent)
      rewards     = env_info.rewards                # get reward (for each agent)
      dones       = env_info.local_done             # see if episode finished
      for i_agent in range(num_agents):
        agent[i_agent].step(states[i_agent], actions[i_agent],
                   rewards[i_agent], next_states[i_agent],
                   dones[i_agent], t)
      #agent.update_t_step()
      #for i in range(num_agents):
      #  agent.step_learn()
      states = next_states
      scores += rewards

      if np.any(dones):
        break

    scores_deque.append(np.amax(scores))
    scores_list.append(np.amax(scores))
    print('\rEpisode {}\tAverage Score Last 100 Episodes: {:.5f}\tMax Score (All Agents) Last Episode: {:.2f}'.format(
      i_episode, np.mean(scores_deque), np.amax(scores)), end="")

    if i_episode % 100 == 0:
      print('\rEpisode {}\tAverage Score Last 100 Episodes: {:.5f}'.format(i_episode, np.mean(scores_deque)))
      for i_agent in range(num_agents):
        filename = './saved_models/ddpg_actor_' + str(i_agent) + '.pth'
        print(filename)
        torch.save(agent[i_agent].actor_local.state_dict(), filename)
        filename = './saved_models/ddpg_critic_' + str(i_agent) + '.pth'
        torch.save(agent[i_agent].critic_local.state_dict(), filename)
    if solve:
      if np.mean(scores_deque) >= 0.5:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.5f}'.format(i_episode - 100,
                                                                                     np.mean(scores_deque)))
        for i_agent in range(num_agents):
          filename = './saved_models/ddpg_actor_' + str(i_agent) + '.pth'
          torch.save(agent[i_agent].actor_local.state_dict(), filename)
          filename = './saved_models/ddpg_critic_' + str(i_agent) + '.pth'
          torch.save(agent[i_agent].critic_local.state_dict(), filename)
        break
  return scores_list


scores = ddpg(num_agents=num_agents)

env.close()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

plt.savefig('./figures/training_performance.png')