import os

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.logger import configure

import datetime
import shutil
import pickle

import numpy as np


from qbm_rl_steering.environment.rms_env_nd import RmsSteeringEnv
from qbm_rl_steering.core.run_utils import (evaluator,
                                            plot_training_log,
                                            plot_evaluation_log)

# nst = 0
# for k in df.keys():
#   if 'step' in k:
#     nst += 1s


params = {
    'quantum_ddpg': False,
    'n_episodes': 200,
    'n_timesteps': 1000,
    'env/n_dims': 6,
    'env/max_steps_per_episode': 20,
    'env/required_steps_above_reward_threshold': 1,
    'trainer/batch_size': 100,
    'trainer/n_exploration_steps': 100,
    'trainer/n_episodes_early_stopping': 15,
    'agent/gamma': 0.99,
    'agent/tau': 0.001,
    'lr/init': 1e-3,
    'lr/decay_factor': 1.,
    'action_noise/init': 0.2,
    'action_noise/final': 0.,
    'epsilon_greedy/init': 0.,
    'epsilon_greedy/final': 0.,
    'anneals/n_pieces': 2,
    'anneals/init': 1,
    'anneals/final': 2,
}


import pandas as pd

new_logger = configure('/tmp/sb3_log', ["stdout"])


env = RmsSteeringEnv(
    n_dims=params['env/n_dims'],
    max_steps_per_episode=params['env/max_steps_per_episode'],
    required_steps_above_reward_threshold=
    params['env/required_steps_above_reward_threshold'])

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions),
    sigma=params['action_noise/init'] * np.ones(n_actions))
agent = DDPG('MlpPolicy', env, action_noise=action_noise,
             batch_size=params['trainer/batch_size'],
             learning_rate=params['lr/init'],
             learning_starts=params['trainer/n_exploration_steps'],
             gamma=params['agent/gamma'],
             tau=params['agent/tau'], verbose=1)

# PREPARE OUTPUT FOLDER
date_time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
out_path = './runs/indiv/' + date_time_now
os.makedirs(out_path)
shutil.copy('./run_ddpg_stablebaselines.py', out_path + '/run_ddpg_stablebaselines.py')
shutil.copy('./core/ddpg_agents.py', out_path + '/ddpg_agents.py')
shutil.copy('./environment/rms_env_nd.py', out_path + '/rms_env_nd.py')
with open(out_path + '/params_dict.pkl', 'wb') as fid:
    pickle.dump(params, fid)

# AGENT TRAINING
# episode_log = trainer(
#     env=env, agent=agent, action_noise_schedule=action_noise_schedule,
#     epsilon_greedy_schedule=epsilon_greedy_schedule,
#     n_anneals_schedule=n_anneals_schedule, n_episodes=params['n_episodes'],
#     max_steps_per_episode=params['env/max_steps_per_episode'],
#     batch_size=params['trainer/batch_size'],
#     n_exploration_steps=params['trainer/n_exploration_steps'],
#     n_episodes_early_stopping=params['trainer/n_episodes_early_stopping']
# )
# plot_training_log(env, agent, episode_log, save_path=out_path)
# df_train_log = pd.DataFrame(episode_log)
agent.learn(total_timesteps=params['n_timesteps'])

import matplotlib.pyplot as plt
plt.figure()
plt.plot(agent.mylogger['actor_loss'], c='r')
plt.figure()
plt.plot(agent.mylogger['critic_loss'], c='b')
plt.show()
#
# plt.plot(agent.replay_buffer.observations[:, 0])
# plt.show()

# df_train_log = pd.DataFrame({'n_episodes': agent._episode_num,
#                              'n_steps': agent.num_timesteps}, index=[0])
# df_train_log.to_csv(out_path + '/train_log')
#
# # AGENT EVALUATION
# # a) Random state inits
# env = RmsSteeringEnv(
#     n_dims=params['env/n_dims'],
#     max_steps_per_episode=params['env/max_steps_per_episode'],
#     required_steps_above_reward_threshold=
#     params['env/required_steps_above_reward_threshold'])
# eval_log_random = evaluator(env, agent, n_episodes=100, reward_scan=False)
# try:
#     df_eval_log = pd.DataFrame({'rewards': eval_log_random})
# except ValueError:
#     print('Issue creating eval df ... probably all evaluations '
#           'used same number of steps')
#
#     n_stp = eval_log_random.shape[1]
#     res_dict = {}
#     for st in range(n_stp):
#         res_dict[f'step_{st}'] = eval_log_random[:, st]
#     df_eval_log = pd.DataFrame(res_dict)
#
# df_eval_log.to_csv(out_path + '/eval_log_random')
# plot_evaluation_log(env, params['env/max_steps_per_episode'],
#                     eval_log_random, type='random')
#
# # b) Systematic state inits
# env = RmsSteeringEnv(
#     n_dims=params['env/n_dims'],
#     max_steps_per_episode=params['env/max_steps_per_episode'],
#     required_steps_above_reward_threshold=
#     params['env/required_steps_above_reward_threshold'])
# eval_log_scan = evaluator(env, agent, n_episodes=100, reward_scan=True)
# try:
#     df_eval_log = pd.DataFrame({'rewards': eval_log_scan})
# except ValueError:
#     print('Issue creating eval df ... probably all evaluations used '
#           'same number of steps')
#
#     n_stp = eval_log_scan.shape[1]
#     res_dict = {}
#     for st in range(n_stp):
#         res_dict[f'step_{st}'] = eval_log_scan[:, st]
#
#     df_eval_log = pd.DataFrame(res_dict)
#
# df_eval_log.to_csv(out_path + '/eval_log_scan')
# print("SAVING B TO FILE")
# plot_evaluation_log(env, params['env/max_steps_per_episode'],
#                     eval_log_scan, type='scan')
