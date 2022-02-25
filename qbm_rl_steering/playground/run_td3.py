from stable_baselines3 import DDPG
from qbm_rl_steering.environment.rms_env_nd import RmsSteeringEnv



params = {
  'quantum_ddpg': False,  # False
  'n_episodes': 2000,  # 800
  'env/n_dims': 6,
  'env/max_steps_per_episode': 30,
  'env/required_steps_above_reward_threshold': 1,
  'trainer/batch_size': 64,  # 128,
  'trainer/n_exploration_steps': 100,  # 200,
  'trainer/n_episodes_early_stopping': 15,
  'agent/gamma': 0.99,
  'agent/tau_critic': 0.001,  # 0.0008,
  'agent/tau_actor': 0.001,  # 0.0008,
  'lr_critic/init': 5e-4,
  'lr_critic/decay_factor': 1.,
  'lr_actor/init': 5e-4,
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

env = RmsSteeringEnv(
  n_dims=params['env/n_dims'],
  max_steps_per_episode=params['env/max_steps_per_episode'],
  required_steps_above_reward_threshold=params['env/required_steps_above_reward_threshold'])

agent = DDPG('MlpPolicy', env)
agent.train(30*100, 128)


