import os

import datetime
import shutil
import pickle

import numpy as np
import matplotlib
import pandas as pd

from tensorflow.keras.optimizers.schedules import PolynomialDecay, PiecewiseConstantDecay

from qbm_rl_steering.core.ddpg_agents import ClassicalDDPG, QuantumDDPG
from qbm_rl_steering.environment.orig_awake_env import e_trajectory_simENV
from qbm_rl_steering.core.run_utils import trainer, evaluator
from qbm_rl_steering.core.run_utils import plot_training_log, plot_evaluation_log


# BASIC PARAMETERS FOR TRAINING
params = {
    'quantum_ddpg': True,  # True: use quantum critic (QBM), False: use classical critic
    'n_steps': 500,  # Total number of interactions between agent and env.
    'trainer/n_exploration_steps': 250,  # Number of pure exploration steps at beginning of training
                                         # No weight updates during that phase.
    'env/max_steps_per_episode': 20,  # Max. number of steps in an episode (if reward objective not
                                      # reached within that time frame, abort episode).
    'trainer/batch_size': 32,  # Number of interactions sampled from replay buffer
    'agent/gamma': 0.99,  # RL discount factor
    'agent/tau_critic': 0.01,  # Soft update factor for main and target critic nets
    'agent/tau_actor': 0.01,  # Soft update factor for main and target actor nets
    'lr_critic/init': 1e-3,  # Initial learning rate for critic net
    'lr_actor/init': 1e-3,  # Initial learning rate for actor net
    'lr/final': 5e-5,  # Final learning rate for critic net (quantum DDPG only)
    'action_noise/init': 0.2,  # Initial action noise (normalized)
    'action_noise/final': 0.,  # Final action noise (normalized)
    'epsilon_greedy/init': 0.,  # Initial probability for random actions
    'epsilon_greedy/final': 0.,  # Final probability for random actions
    'env/required_steps_above_reward_threshold': 1,  # obsolete ...
    'trainer/n_episodes_early_stopping': 10000000, # obsolete ...
}

# TRAINING
# Init. env, agent with parameters from params dict and run trainer.
env = e_trajectory_simENV()
lr_schedule_critic = PolynomialDecay(params['lr_critic/init'],
                                     params['n_steps'],
                                     end_learning_rate=params['lr/final'])
lr_schedule_actor = PolynomialDecay(params['lr_actor/init'],
                                    params['n_steps'],
                                    end_learning_rate=params['lr/final'])

if params['quantum_ddpg']:
    agentMy = QuantumDDPG(state_space=env.observation_space,
                          action_space=env.action_space,
                          learning_rate_schedule_critic=lr_schedule_critic,
                          learning_rate_schedule_actor=lr_schedule_actor,
                          grad_clip_actor=1e4, grad_clip_critic=1.,
                          gamma=params['agent/gamma'],
                          tau_critic=params['agent/tau_critic'],
                          tau_actor=params['agent/tau_actor'],
                          )
else:
    agentMy = ClassicalDDPG(state_space=env.observation_space,
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
                np.linspace(0, 1., 2)][1:-1]
y_transition = [int(n) for n in np.linspace(1, 1, 2)]
n_anneals_schedule = PiecewiseConstantDecay(t_transition, y_transition)

# Prepare output folder
date_time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
out_path = './runs/indiv/' + date_time_now
os.makedirs(out_path)
with open(out_path + '/params_dict.pkl', 'wb') as fid:
    pickle.dump(params, fid)

# Run trainer
episode_log = trainer(
    env=env, agent=agentMy, action_noise_schedule=action_noise_schedule,
    epsilon_greedy_schedule=epsilon_greedy_schedule,
    n_anneals_schedule=n_anneals_schedule, n_steps=params['n_steps'],
    max_steps_per_episode=params['env/max_steps_per_episode'],
    batch_size=params['trainer/batch_size'],
    n_exploration_steps=params['trainer/n_exploration_steps'],
    n_episodes_early_stopping=params['trainer/n_episodes_early_stopping'])

# Plot and save training log
plot_training_log(env, agentMy, episode_log, apply_scaling=True, save_path=out_path)

episode_log_2 = {}
for k, v in episode_log.items():
    if len(v) != 0:
        episode_log_2[k] = v[:]
episode_log = episode_log_2
df_train_log = pd.DataFrame(episode_log)
df_train_log.to_csv(out_path + '/train_log')

# Save agent weights
if params['quantum_ddpg']:
    weights = {'main_critic': {'w_vh': agentMy.main_critic_net.w_vh,
                               'w_hh': agentMy.main_critic_net.w_hh},
               'target_critic': {'w_vh': agentMy.target_critic_net.w_vh,
                                 'w_hh': agentMy.target_critic_net.w_hh}}
else:
    weights = {'main_critic': agentMy.main_critic_net_1.get_weights(),
               'target_critic': agentMy.target_critic_net_1.get_weights()}

with open(out_path + '/critic_weights.pkl', 'wb') as fid:
    pickle.dump(weights, fid)

weights = {'main_actor': agentMy.main_actor_net.get_weights(),
           'target_actor': agentMy.target_actor_net.get_weights()}
with open(out_path + '/actor_weights.pkl', 'wb') as fid:
    pickle.dump(weights, fid)


# EVALUATION
env = e_trajectory_simENV()
eval_log_scan = evaluator(env, agentMy, n_episodes=100, reward_scan=False)

try:
    df_eval_log = pd.DataFrame({'rewards': eval_log_scan})
except ValueError:
    print('Issue creating eval df ... probably all evaluations used '
          'same number of steps')

plot_evaluation_log(env, params['env/max_steps_per_episode'],
                    eval_log_scan, type='random', apply_scaling=True,
                    save_path=out_path)
