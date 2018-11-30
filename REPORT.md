# Algorithm
We use a DDPG algorithm (Deep Deterministic Policy Gradient) ==>  hybrid method: Actor Critic.
Each agent has its own actor and critic networks.

The model is very simple and composed of 2 fully connected layer with leaky relu activation for each network.

* Actor model architecture
> state => leaky_relu(FC1(state)) ==> tanh(leaky_relu(FC2(FC1))) ==> action \
By applying a tanh function in output, we ensure that the action values are in the range [-1,1]

* Critc model architecture
> state + action => leaky_relu(FC1(batchnorm(state)) ==> leaky_relu(FC2(FC1+action))) ==> value function

# Hyperparameters tuning
The most important parameter to tune is the learnin rate.
A basic grid search method can be applied to find the optimal value for the learning rate.

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
fc1_units | 128 | number of nodes in first hidden layer for actor
fc2_units | 56 | number of nodes in second hidden layer for actor
fc1_units |256 | number of nodes in first hidden layer for critic
fc2_units | 128 | number of nodes in second hidden layer for critic

All hyperparemeters has been tuned to reach this performance.

# Performance assessment

Solved Requirements: Considered solved when the average reward is greater than or equal to +30 over 100 consecutive trials.
  + Performance for DDPG Agent.
Environment is solved in 1421 episodes.
![alt text](https://github.com/Adrelf/DRL_Multi_agents/blob/master/images/training_performance.png)

# Future improvements
  - MADDPG algorithm: Multi-agent Actor-Critic for Mixed Cooperatie-Competitive Environments ==> https://arxiv.org/pdf/1706.02275.pdf <br/>
  They use a centralized critic for each agent. There is communication between agents. In our context, we can also provide a global state containing the state of each agent in input of critic and action networks. We can also implement a shared memory with PER to speed-up the training phase.
  - D4PG algorithm: Distributed Distributional Deterministic Policy Gradients ==> https://arxiv.org/pdf/1804.08617.pdf <br/>
  They use a distributional critic update and a distributed parallel actors to speed-up the training phase. Like Rainbow algorithm, they use N-step returns and PER.
  
