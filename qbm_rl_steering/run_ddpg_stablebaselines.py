from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

from qbm_rl_steering.environment.rms_env_nd import RmsSteeringEnv
from qbm_rl_steering.core.run_utils import (evaluator,
                                        plot_training_log,
                                        plot_evaluation_log)

import numpy as np


params = {
  'quantum_ddpg': False,
  'n_episodes': 200,
  'env/n_dims': 6,
  'env/max_steps_per_episode': 20,
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
  'action_noise/final': 0.,
  'epsilon_greedy/init': 0.,
  'epsilon_greedy/final': 0.,
  'anneals/n_pieces': 2,
  'anneals/init': 1,
  'anneals/final': 2,
}

env = RmsSteeringEnv(
  n_dims=params['env/n_dims'],
  max_steps_per_episode=params['env/max_steps_per_episode'],
  required_steps_above_reward_threshold=params['env/required_steps_above_reward_threshold'])

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                 sigma=0.1*np.ones(n_actions))
agent = DDPG('MlpPolicy', env, action_noise=action_noise, verbose=1)
agent.learn(total_timesteps=600, log_interval=10)

env = RmsSteeringEnv(
  n_dims=params['env/n_dims'],
  max_steps_per_episode=params['env/max_steps_per_episode'],
  required_steps_above_reward_threshold=params['env/required_steps_above_reward_threshold'])
eval_log_scan = evaluator(env, agent, n_episodes=100, reward_scan=True)
plot_evaluation_log(env, params['env/max_steps_per_episode'],
                eval_log_scan, save_path='./', type='scan')
