import os

import datetime
import shutil
import pickle

import numpy as np
import matplotlib
matplotlib.use('qt5agg')

params = {
    'quantum_ddpg': True,  # False
    'n_steps': 200,  # 800
    'env/n_dims': 6,
    'env/max_steps_per_episode': 20,
    'env/required_steps_above_reward_threshold': 1,
    'trainer/batch_size': 16,  # 128,
    'trainer/n_exploration_steps': 100,  # 200,
    'trainer/n_episodes_early_stopping': 20,
    'agent/gamma': 0.99,
    'agent/tau_critic': 0.1,  # 0.001,
    'agent/tau_actor': 0.1,  # 0.001,
    'lr_critic/init': 1e-3,
    'lr_critic/decay_factor': 1.,
    'lr_actor/init': 1e-3,
    'lr_actor/decay_factor': 1.,
    'lr/final': 1e-5,
    'action_noise/init': 0.1,
    'action_noise/final': 0.,
    'epsilon_greedy/init': 0.1,
    'epsilon_greedy/final': 0.,
    'anneals/n_pieces': 2,
    'anneals/init': 1,
    'anneals/final': 2,
}

# params = {
#   'quantum_ddpg': False,  # False
#   'n_steps': 1000,
#   'env/n_dims': 6,
#   'env/max_steps_per_episode': 20,
#   'env/required_steps_above_reward_threshold': 1,
#   'trainer/batch_size': 100,  # 128,
#   'trainer/n_exploration_steps': 100,  # 200,
#   'trainer/n_episodes_early_stopping': 20,
#   'agent/gamma': 0.99,
#   'agent/tau_critic': 0.001,  # 0.0008,
#   'agent/tau_actor': 0.001,  # 0.0008,
#   'lr_critic/init': 1e-3,
#   'lr_critic/decay_factor': 1.,
#   'lr_actor/init': 1e-3,
#   'lr_actor/decay_factor': 1.,
#   'lr/final': 1e-5,
#   'action_noise/init': 0.2,
#   'action_noise/final': 0.,
#   'epsilon_greedy/init': 0.,
#   'epsilon_greedy/final': 0.,
#   'anneals/n_pieces': 2,
#   'anneals/init': 1,
#   'anneals/final': 2,
# }


process_id = None
from tensorflow.keras.optimizers.schedules import PolynomialDecay, PiecewiseConstantDecay
from qbm_rl_steering.core.ddpg_agents import ClassicalDDPG, QuantumDDPG
from qbm_rl_steering.environment.rms_env_nd import RmsSteeringEnv
from qbm_rl_steering.core.run_utils import trainer, evaluator, plot_training_log, plot_evaluation_log
import pandas as pd

env = RmsSteeringEnv(
    n_dims=params['env/n_dims'],
    max_steps_per_episode=params['env/max_steps_per_episode'],
    required_steps_above_reward_threshold=params['env/required_steps_above_reward_threshold'])

# Learning rate schedules: lr_critic = 5e-4, lr_actor = 1e-4
lr_schedule_critic = PolynomialDecay(params['lr_critic/init'],
                                     params['n_steps'],
                                     end_learning_rate=params['lr/final'])
lr_schedule_actor = PolynomialDecay(params['lr_actor/init'],
                                    params['n_steps'],
                                    end_learning_rate=params['lr/final'])

if params['quantum_ddpg']:
    agentMy = QuantumDDPG(env, state_space=env.observation_space,
                          action_space=env.action_space,
                          learning_rate_schedule_critic=lr_schedule_critic,
                          learning_rate_schedule_actor=lr_schedule_actor,
                          grad_clip_actor=1e4, grad_clip_critic=1.,
                          gamma=params['agent/gamma'],
                          tau_critic=params['agent/tau_critic'],
                          tau_actor=params['agent/tau_actor'],
                          )
else:
    agentMy = ClassicalDDPG(env, state_space=env.observation_space,
                            action_space=env.action_space,
                            learning_rate_critic=params['lr_critic/init'],
                            learning_rate_actor=params['lr_actor/init'],
                            grad_clip_actor=np.inf, grad_clip_critic=np.inf,
                            gamma=params['agent/gamma'],
                            tau_critic=params['agent/tau_critic'],
                            tau_actor=params['agent/tau_actor'],
                            )

# Action noise schedule
action_noise_schedule = PolynomialDecay(
    params['action_noise/init'], params['n_steps'],
    params['action_noise/final'])

# Epsilon greedy schedule
epsilon_greedy_schedule = PolynomialDecay(
    params['epsilon_greedy/init'], params['n_steps'],
    params['epsilon_greedy/final'])

# Schedule n_anneals
t_transition = [int(x * params['n_steps']) for x in
                np.linspace(0, 1., params['anneals/n_pieces'] + 1)][1:-1]
y_transition = [int(n) for n in np.linspace(params['anneals/init'],
                                            params['anneals/final'],
                                            params['anneals/n_pieces'])]
n_anneals_schedule = PiecewiseConstantDecay(t_transition, y_transition)

# PREPARE OUTPUT FOLDER
date_time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
out_path = './runs/indiv/' + date_time_now
if process_id is not None:
    out_path = out_path + f'_pid_{process_id}'
os.makedirs(out_path)
shutil.copy('./run_ddpg.py', out_path + '/run_ddpg.py')
shutil.copy('./core/ddpg_agents.py', out_path + '/ddpg_agents.py')
shutil.copy('./environment/rms_env_nd.py', out_path + '/rms_env_nd.py')

with open(out_path + '/params_dict.pkl', 'wb') as fid:
    pickle.dump(params, fid)

# AGENT TRAINING
episode_log = trainer(
    env=env, agent=agentMy, action_noise_schedule=action_noise_schedule,
    epsilon_greedy_schedule=epsilon_greedy_schedule,
    n_anneals_schedule=n_anneals_schedule, n_steps=params['n_steps'],
    max_steps_per_episode=params['env/max_steps_per_episode'],
    batch_size=params['trainer/batch_size'],
    n_exploration_steps=params['trainer/n_exploration_steps'],
    n_episodes_early_stopping=params['trainer/n_episodes_early_stopping']
)

# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(agentMy.losses_log['Q'], c='b')
# plt.show()
#
# plt.figure()
# plt.plot(agentMy.losses_log['Mu'], c='r')
# plt.show()

plot_training_log(env, agentMy, episode_log)  # , save_path=out_path)
df_train_log = pd.DataFrame(episode_log)
df_train_log.to_csv(out_path + '/train_log')

# AGENT EVALUATION
# a) Random state inits
env = RmsSteeringEnv(
    n_dims=params['env/n_dims'],
    max_steps_per_episode=params['env/max_steps_per_episode'],
    required_steps_above_reward_threshold=
    params['env/required_steps_above_reward_threshold'])

eval_log_random = evaluator(env, agentMy, n_episodes=100, reward_scan=False)

try:
    df_eval_log = pd.DataFrame({'rewards': eval_log_random})
except ValueError:
    print('Issue creating eval df ... probably all evaluations '
          'used same number of steps')

# n_stp = eval_log_random.shape[1]
# res_dict = {}
# for st in range(n_stp):
#     res_dict[f'step_{st}'] = eval_log_random[:, st]
# df_eval_log = pd.DataFrame(res_dict)

# df_eval_log.to_csv(out_path + '/eval_log_random')
plot_evaluation_log(env, params['env/max_steps_per_episode'],
                    eval_log_random, type='random')  # save_path=out_path

# b) Systematic state inits
env = RmsSteeringEnv(
    n_dims=params['env/n_dims'],
    max_steps_per_episode=params['env/max_steps_per_episode'],
    required_steps_above_reward_threshold=
    params['env/required_steps_above_reward_threshold'])

eval_log_scan = evaluator(env, agentMy, n_episodes=100, reward_scan=True)

try:
    df_eval_log = pd.DataFrame({'rewards': eval_log_scan})
except ValueError:
    print('Issue creating eval df ... probably all evaluations used '
          'same number of steps')

n_stp = eval_log_scan.shape[1]
res_dict = {}
for st in range(n_stp):
    res_dict[f'step_{st}'] = eval_log_scan[:, st]
df_eval_log = pd.DataFrame(res_dict)

df_eval_log.to_csv(out_path + '/eval_log_scan')
plot_evaluation_log(env, params['env/max_steps_per_episode'],
                    eval_log_scan, type='scan')  # save_path=out_path)

"""
# EVALUATE AGENT FROM MIN. REWARD STATE. TAKES MULTIPLE STEPS
done = False
state = env.reset(init_specific_reward_state=target_init_rewards[0])
print('Init min. reward state', state)
print('Init reward', env.calculate_reward(state / env.state_scale))
ctr = 0
while not done:
    act = agent.get_proposed_action(state, 0.)
    print(f'\nSTEP: {ctr}')
    print(f'Env. kick_angles: {env.kick_angles}')
    print(f'Proposed action: {act/env.action_scale}')
    state, reward, done, _ = env.step(act)
    print(f'New state: {state}')
    print(f'Reward: {reward}')
    print(f'Done: {done}')
    ctr += 1
"""
