from stable_baselines3.common.noise import NormalActionNoise

import numpy as np
import matplotlib.pyplot as plt

from qbm_rl_steering.environment.target_steering_1d import TargetSteeringEnv
from cern_awake_env.simulation import SimulationEnv
from stable_baselines3 import SAC, DDPG

# env = TargetSteeringEnv()
env = SimulationEnv(plane='H', remove_singular_devices=True)
noise = NormalActionNoise(np.array([0.1]), np.array([0.01]))
model = SAC("MlpPolicy", env, verbose=1, action_noise=noise, gamma=0.85,
            learning_rate=1e-3)
model.learn(10000, log_interval=4)


n_evals = 100
init_rewards = np.zeros(n_evals)
final_rewards = np.zeros(n_evals)

ctr = 0
obs = env.reset()
obs = np.array([obs])
init_rewards[ctr] = env.compute_reward(obs, goal=None, info={})
# _, intensity = env.get_pos_at_bpm_target(env.mssb_angle)
# init_rewards[ctr] = env.get_reward(intensity)
# print('initial reward', env.get_reward(intensity))

while ctr < n_evals:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    obs = np.array([obs])
    # print('reward', reward)
    if done:
        final_rewards[ctr] = reward
        # print('================ done!')
        obs = env.reset()
        obs = np.array([obs])
        init_rewards[ctr] = env.compute_reward(obs, goal=None, info={})
        ctr += 1


plt.plot(init_rewards, label='init')
plt.plot(final_rewards, label='final')
plt.legend()
plt.show()

