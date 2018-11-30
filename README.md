# DRL_Multi_agents
Train two agents to play tennis.

In this environment, two agents control rackets to bounce a ball over a net. The goal of each agent is to keep the ball in play.
The deep reinforcement learning algorithm is based on actor-critic method (DDPG).

![alt text](https://github.com/Adrelf/DRL_Multi_agents/blob/master/images/tennis.gif)

# The Environment 
The environment is determinist.

 + State:<br/> 
   The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.
   
 + Actions:<br/>  
   Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.
   
 + Reward strategy:<br/>
   If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.
   
 + Solved Requirements:<br/>
   Need an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). 

# Algorithm
DDPG (Deep Deterministic Policy Gradient) is an hybrid method: Actor Critic. We use two neural networks:

 * a Critic that measures how good the action taken is (value-based). The value function maps each state action pair to a value which quantifies how is good to be / go to another state. The value function calculates what is the maximum expected future reward given a state and an action.
 
 * an Actor that controls how our agent behaves (policy-based). We directly optimize the policy without using a value function. This is useful when the action space is continuous or stochastic.<br/>
 
Instead of waiting until the end of the episode, we make an update at each step (TD Learning). The Critic observes our action and provides feedback in order to update our policy and be better at playing that game.<br/>

Each agent has its own actor and critic network. They use also a separate memory.

# Getting started

## Dependencies
 * Python 3.6 or higher
 * PyTorch
 * Create (and activate) a new environment with Python 3.6:
 ```
     conda create --name drlnd python=3.6
     source activate drlnd
 ```
 * Install requirements:
 ```
     clone git https://github.com/Adrelf/DRL_Multi_agents.git
     cd DRL_Multi_agents
     pip install -e .
  ```
 * Download the [Unity Environment!] (https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)   
Then, place the file in the DRL_Multi_agents/ folder in this repository, and unzip (or decompress) the file.

# Instructions
 * To train the agents, please use the following command:
  ```
      python main_DDPG.py
  ```
  
   with the fowolling hyper-parameters:
 
 Parameters | Value | Description
----------- | ----- | -----------
BUFFER_SIZE | int(1e6) | replay buffer size
BATCH_SIZE | 512 | minibatch size
GAMMA | 0.99 | discount factor
TAU | 1e-3 | for soft update of target parameters
LR_ACTOR | 1e-4 | learning rate of the actor
LR_CRITIC | 3e-4 | learning rate of the critic
WEIGHT_DECAY | 0.0001 | L2 weight decay
UPDATE_EVERY | 1 | how often to update the network
fc1_units | 128 | Number of nodes in first hidden layer for actor
fc2_units | 56 | Number of nodes in second hidden layer for actor
fc1_units |256 | Number of nodes in first hidden layer for critic
fc2_units | 128 | Number of nodes in second hidden layer for critic

 * To assess the performance of agents, please use the following command:
  ```
      python eval_DDPG.py
  ```
