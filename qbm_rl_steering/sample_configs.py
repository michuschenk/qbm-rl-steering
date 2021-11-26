import pickle

import numpy as np

from qbm_rl_steering.core.qbm import QFunction, get_visible_nodes_array, create_general_qubo_dict, \
    get_average_effective_hamiltonian
from qbm_rl_steering.core.utils import generate_classical_actor
from qbm_rl_steering.environment.rms_env_nd import RmsSteeringEnv


params = {
    'quantum_ddpg': True,  # False
    'n_steps': 1000,  # 800
    'env/n_dims': 6,
    'env/max_steps_per_episode': 20,
    'env/required_steps_above_reward_threshold': 1,
    'trainer/batch_size': 32,  # 128,
    'trainer/n_exploration_steps': 100,  # 200,
    'trainer/n_episodes_early_stopping': 30,
    'agent/gamma': 0.99,
    'agent/tau_critic': 0.1,  # 0.001,
    'agent/tau_actor': 0.1,  # 0.001,
    'lr_critic/init': 2e-3,
    'lr_critic/decay_factor': 1.,
    'lr_actor/init': 1e-3,
    'lr_actor/decay_factor': 1.,
    'lr/final': 5e-5,
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

state_space = env.observation_space
action_space = env.action_space

kwargs_q_func = dict(
    sampler_type='SQA',
    state_space=state_space,
    action_space=action_space,
    small_gamma=params['agent/gamma'],
    n_replicas=1,
    big_gamma=(20., 0.), beta=2,
    n_annealing_steps=200,
    n_meas_for_average=1,
    kwargs_qpu={})


# TODO: do we need main or target nets here?
# Reload critic
with open('runs/indiv/2021-11-26_15:47:13/critic_weights.pkl', 'rb') as fid:
    w = pickle.load(fid)

critic = QFunction(**kwargs_q_func)
critic.w_hh = w['main_critic']['w_hh']
critic.w_vh = w['main_critic']['w_vh']

# Reload actor
actor_hidden_layers = [400, 300]
with open('runs/indiv/2021-11-26_15:47:13/actor_weights.pkl', 'rb') as fid:
    w = pickle.load(fid)
actor = generate_classical_actor(params['env/n_dims'], params['env/n_dims'], hidden_layers=actor_hidden_layers)
actor.set_weights(w['main_actor'])

# Try it out for 1 episode
state = env.reset(init_outside_threshold=True)
rewards = [env.calculate_reward(env.calculate_state(env.kick_angles))]

while True:
    a = actor.predict(state.reshape(1, -1))[0]
    state, reward, done, _ = env.step(a)
    rewards.append(reward)
    if done:
        break

print('rewards', rewards)
print('n_steps_eps', len(rewards) - 1)

# SEE SPIN CONFIGURATIONS FOR ONE STATE-ACTION PAIR
state = env.reset(init_outside_threshold=True)
action = actor.predict(state.reshape(1, -1))[0]

# Define QUBO
visible_nodes = get_visible_nodes_array(state=state, action=action, state_space=state_space, action_space=action_space)
qubo_dict = create_general_qubo_dict(critic.w_hh, critic.w_vh, visible_nodes)

# Run the annealing process
spin_configurations = critic.sampler.sample(qubo_dict=qubo_dict, n_meas_for_average=10000,
                                            n_steps=critic.n_annealing_steps)

# Return the number of occurrences of unique spin configurations along
# axis 0, i.e. along index of independent measurements
spin_config_unique, n_occurrences = np.unique(spin_configurations, axis=0, return_counts=True)
mean_n_occurrences = n_occurrences / float(np.sum(n_occurrences))
hamiltonians = []

for sc in spin_config_unique:
    sc_3d = np.atleast_3d(sc)
    sc_3d = np.rollaxis(sc_3d, 2, 1)
    eff_hamil = get_average_effective_hamiltonian(sc_3d, critic.w_hh, critic.w_vh, visible_nodes, big_gamma_final=0.,
                                                  beta_final=2.)
    hamiltonians.append(eff_hamil)

print('unique_configs', spin_config_unique)
print('n_occurrences', n_occurrences)
print('mean_n_occurrences', mean_n_occurrences)
print('hamiltonians', hamiltonians)

sc_all1 = [1.] * 32
sc_all1 = np.atleast_3d(sc_all1)
sc_all1 = np.rollaxis(sc_all1, 2, 1)
print('For comparison config with all 1s, Hamiltonian:', get_average_effective_hamiltonian(
    sc_all1, critic.w_hh, critic.w_vh, visible_nodes, big_gamma_final=0., beta_final=2.))
