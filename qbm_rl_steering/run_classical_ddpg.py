from qbm_rl_steering.classical_ddpg.agent import Agent
import gym
from gym import wrappers
import os
import numpy as np
import matplotlib.pyplot as plt

N_EPISODES = 600



params = {
  'quantum_ddpg': False,
  'n_episodes': 200,
  'env/n_dims': 2,
  'env/max_steps_per_episode': 30,
  'env/required_steps_above_reward_threshold': 1,
  'trainer/batch_size': 32,
  'trainer/n_exploration_steps': 100,
  'trainer/n_episodes_early_stopping': 15,
  'agent/gamma': 0.99,
  'agent/tau_critic': 0.001,
  'agent/tau_actor': 0.001,
  'lr_critic/init': 1e-3,
  'lr_critic/decay_factor': 1.,
  'lr_actor/init': 1e-4,
  'lr_actor/decay_factor': 1.,
  'action_noise/init': 0.2,
  'action_noise/final': 0.2,
  'epsilon_greedy/init': 0.2,
  'epsilon_greedy/final': 0.,
  'anneals/n_pieces': 2,
  'anneals/init': 1,
  'anneals/final': 2,
}

from qbm_rl_steering.environment.rms_env_nd import RmsSteeringEnv

env = RmsSteeringEnv(
  n_dims=params['env/n_dims'],
  max_steps_per_episode=params['env/max_steps_per_episode'],
  required_steps_above_reward_threshold=params['env/required_steps_above_reward_threshold'])

#get simulation environment
# env = gym.make("Pendulum-v0")
state_dims = [len(env.observation_space.low)]
action_dims = [len(env.action_space.low)]
action_boundaries = [env.action_space.low, env.action_space.high]
print(action_boundaries)
# create agent with environment parameters
agent = Agent(state_dims = state_dims, action_dims = action_dims,
            action_boundaries = action_boundaries, actor_lr = 1e-4,
            critic_lr = 1e-4, batch_size = 128, gamma = 0.99, rand_steps = 2,
            buf_size = int(1e6), tau = 0.001, fcl1_size = 400, fcl2_size = 600)
np.random.seed(0)
scores = []
# training loop: call remember on predicted states and train the models
episode = 0
for i in range(N_EPISODES):
    #get initial state
    state = env.reset()
    terminal = False
    score = 0
    #proceed until reaching an exit state
    while not terminal:
        #predict new action
        action = agent.get_action(state, episode)
        #perform the transition according to the predicted action
        state_new, reward, terminal, info = env.step(action)
        #store the transaction in the memory
        agent.remember(state, state_new, action, reward, terminal)
        #adjust the weights according to the new transaction
        agent.learn()
        #iterate to the next state
        state = state_new
        score += reward
        # env.render()
    scores.append(score)
    print("Iteration {:d} --> score {:.2f}. Running average {:.2f}".format( i, score, np.mean(scores)))
    episode += 1
plt.plot(scores)
plt.xlabel("Episode")
plt.ylabel("Cumulate reward")
plt.show()
