from qbm_rl_steering.environment.orig_awake_env import e_trajectory_simENV
from qbm_rl_steering.core.utils import generate_classical_actor

import pickle
import matplotlib.pyplot as plt

params = {
    'env/n_dims': 10
}

env = e_trajectory_simENV()

# Reload actor
pathname = 'runs/indiv/2022-01-24_12:12:43/'
actor_hidden_layers = [400, 300]
with open(pathname + 'actor_weights.pkl', 'rb') as fid:
    w = pickle.load(fid)
actor = generate_classical_actor(params['env/n_dims'], params['env/n_dims'], hidden_layers=actor_hidden_layers)
actor.set_weights(w['main_actor'])

init_rew = []
final_rew = []
episode_length = []

n_episodes_eval = 100
n_epi = 0
while n_epi < n_episodes_eval:
    rewards = []
    state = env.reset()
    rewards.append(env._get_reward(state))

    while True:
        a = actor.predict(state.reshape(1, -1))[0]
        state, reward, done, _ = env.step(a)
        rewards.append(reward)
        if done:
            init_rew.append(rewards[0])
            final_rew.append(rewards[-1])
            episode_length.append(len(rewards) - 1)
            n_epi += 1
            break

fig = plt.figure(1, figsize=(7, 5))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212, sharex=ax1)

ax1.plot(episode_length, c='tab:blue')

ax2.plot(init_rew, c='tab:red')
ax2.plot(final_rew, c='tab:green')
ax2.axhline(env.threshold, c='k', ls='--')

ax2.set_xlabel('Episode')
ax2.set_ylabel('Reward (mm)')
ax1.set_ylabel('# steps')
plt.show()
