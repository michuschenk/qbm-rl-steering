from qbm_rl_steering.core.utils import generate_classical_actor

from cern_awake_env.simulation import SimulationEnv
from gym.wrappers import TimeLimit
from cernml.rltools.wrappers import LogRewards, RenderOnStep

import pickle
import numpy as np
import matplotlib.pyplot as plt


# RELOAD QBM-RL TRAINED AGENT
pathname = 'runs/indiv/2022-02-25_15:36:19/'  # Current BEST

actor_hidden_layers = [400, 300]
with open(pathname + 'actor_weights.pkl', 'rb') as fid:
    w = pickle.load(fid)

n_dims = 10
agent = generate_classical_actor(n_dims, n_dims, hidden_layers=actor_hidden_layers)
agent.set_weights(w['main_actor'])


# SET UP ENV
env = SimulationEnv(plane='H', remove_singular_devices=True)
env = TimeLimit(env, max_episode_steps=50)


# EVALUATION LOOP
n_episodes_eval = 50
init_rew = []
final_rew = []
episode_length = []

n_epi = 0
while n_epi < n_episodes_eval:
    rewards = []
    state = env.reset()
    rewards.append(env.compute_reward(state, None, {}))
    while True:
        a = agent.predict(state.reshape(1, -1))[0]
        state, reward, done, _ = env.step(a)
        # env.render()
        rewards.append(reward)
        if done:
            init_rew.append(rewards[0])
            final_rew.append(rewards[-1])
            episode_length.append(len(rewards) - 1)
            n_epi += 1
            break


# PLOT RESULTS
init_rew = np.array(init_rew)
final_rew = np.array(final_rew)
episode_length = np.array(episode_length)

fig = plt.figure(1, figsize=(7, 5))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212, sharex=ax1)

ax1.plot(episode_length, c='tab:blue')

# Undo all scalings that have been applied to the reward and multiply by
# 1'000 to get from [m] -> [mm]
scaling = 1. / (env.state_scale) * 1000

ax2.plot(init_rew * scaling, c='tab:red')
ax2.plot(final_rew * scaling, c='tab:green')
ax2.axhline(env.reward_objective * scaling, c='k', ls='--')

ax2.set_xlabel('Episode')
ax2.set_ylabel('Reward (mm)')
ax1.set_ylabel('# steps')
plt.show()

