import itertools
import pickle

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from qbm_rl_steering.core.qbm import QFunction, get_visible_nodes_array, create_general_qubo_dict, \
    get_average_effective_hamiltonian
from qbm_rl_steering.core.utils import generate_classical_actor
from qbm_rl_steering.environment.rms_env_nd import RmsSteeringEnv


def calc_f(qubo_dict, x):
    f = 0.
    for i in range(len(x)):
        for j in range(len(x)):
            try:
                qij = qubo_dict[(i, j)]
            except KeyError:
                qij = 0

            if i == j:
                f += qij * x[i]
            else:
                f += qij * x[i] * x[j]
    return f


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

beta = 2.  # 2.
big_gamma_final = 0.
kwargs_q_func = dict(
    sampler_type='SQA',
    state_space=state_space,
    action_space=action_space,
    small_gamma=params['agent/gamma'],
    n_replicas=1,
    big_gamma=(200., big_gamma_final), beta=beta,
    n_annealing_steps=800,  # 200
    n_meas_for_average=1,
    kwargs_qpu={})

# TODO: do we need main or target nets here?
# pathname = 'runs/indiv/2021-11-26_15:47:13/'
# pathname = 'runs/indiv/2021-12-08_16:42:23/'
# pathname = 'runs/indiv/2021-12-13_16:52:47/'
pathname = 'runs/indiv/2021-12-13_19:45:48/'

# 1x3
# pathname = 'runs/indiv/2021-12-16_14:42:19/'


# Reload critic
with open(pathname + 'critic_weights.pkl', 'rb') as fid:
    w = pickle.load(fid)

critic = QFunction(**kwargs_q_func)
critic.w_hh = w['main_critic']['w_hh']
critic.w_vh = w['main_critic']['w_vh']

kwargs_q_func.update({'sampler_type': 'QAOA'})
critic_qaoa = QFunction(**kwargs_q_func)
critic_qaoa.w_hh = w['main_critic']['w_hh']
critic_qaoa.w_vh = w['main_critic']['w_vh']

# Reload actor
actor_hidden_layers = [400, 300]
with open(pathname + 'actor_weights.pkl', 'rb') as fid:
    w = pickle.load(fid)
actor = generate_classical_actor(params['env/n_dims'], params['env/n_dims'], hidden_layers=actor_hidden_layers)
actor.set_weights(w['main_actor'])

# Try it out for 1 episode
# state = env.reset(init_outside_threshold=True)
# rewards = [env.calculate_reward(env.calculate_state(env.kick_angles))]
#
# while True:
#     a = actor.predict(state.reshape(1, -1))[0]
#     state, reward, done, _ = env.step(a)
#     rewards.append(reward)
#     if done:
#         break
#
# print('rewards', rewards)
# print('n_steps_eps', len(rewards) - 1)


state = env.reset(init_outside_threshold=True)
action = actor.predict(state.reshape(1, -1))[0]

# state = np.array([-0.01042566, 0.0089064, 0.07676773, 0.16209732, 0.3443751, -0.39691028])
# action = np.array([0.00752933, 0.01214568, 0.03700168, -0.01813428, -0.00270007, -0.00986744])

print('state', state)
print('action', action)


# Define QUBO
visible_nodes = get_visible_nodes_array(state=state, action=action, state_space=state_space,
                                        action_space=action_space)
qubo_dict, ising = create_general_qubo_dict(critic.w_hh, critic.w_vh, visible_nodes)

# Run the annealing process
spin_configs = critic.sampler.sample(qubo_dict=qubo_dict, n_meas_for_average=2000,
                                     n_steps=critic.n_annealing_steps)

print('=================')
print("Annealing")
print('spin_configs', spin_configs)

# Return the number of occurrences of unique spin configurations along
# axis 0, i.e. along index of independent measurements
spin_config_unique, n_occurrences = np.unique(spin_configs, axis=0, return_counts=True)
mean_n_occurrences = n_occurrences / float(np.sum(n_occurrences))

f_vals = []
sc_3d = None
for sc in spin_config_unique:
    # sc_3d = np.atleast_3d(sc)
    # sc_3d = np.rollaxis(sc_3d, 2, 1)
    # eff_hamil = get_average_effective_hamiltonian(sc_3d, critic.w_hh, critic.w_vh, visible_nodes,
    #                                               big_gamma_final=big_gamma_final, beta_final=beta)
    # hamiltonians.append(eff_hamil)

    sc[sc == -1] = 0
    f_vals.append(calc_f(qubo_dict, sc.flatten().T))

print('unique_configs', spin_config_unique)
print('n_occurrences', n_occurrences)
print('mean_n_occurrences', mean_n_occurrences)
print('f_vals', f_vals)
print('total # unique configs', len(spin_config_unique))

# QAOA
print('=================')
print('QAOA')

# spin_configs_qaoa = critic_qaoa.sampler.sample(qubo_dict=qubo_dict, n_meas_for_average=1)
qubo_problem = critic_qaoa.sampler._reformulate_qubo(qubo_dict)
num_reads = kwargs_q_func['n_meas_for_average'] * critic_qaoa.sampler.n_replicas

# spin_configurations = []
# for i in range(num_reads):
#     spin_configurations.append(
#         list(critic_qaoa.sampler.solver.solve(qubo_problem).x))
# num_reads = kwargs_q_func['n_meas_for_average']
# num_reads = 10
res = critic_qaoa.sampler.solver.solve(qubo_problem)
# sc_dict = res.min_eigen_solver_result['eigenstate'].sample(num_reads, reverse_endianness=True)

# for k, v in sc_dict.items():
#     sc_ = [np.int(i) for i in k]
#     for i in range(int(num_reads * v)):
#         spin_configurations.append(sc_)

# spin_configurations = []
# for i in range(num_reads):
#     spin_configurations.append(
#         list(critic_qaoa.sampler.solver.solve(qubo_problem).x))

spin_configurations = [res.x]

# Convert to np array and flip all the 0s to -1s
spin_configurations = np.array(spin_configurations)
# spin_configurations[spin_configurations == 0] = -1

spin_configs_qaoa = spin_configurations.reshape(
    (num_reads, critic_qaoa.sampler.n_replicas, critic_qaoa.sampler.n_nodes))

# Return the number of occurrences of unique spin configurations along
# axis 0, i.e. along index of independent measurements
print('spin_configs_qaoa', spin_configs_qaoa)
spin_config_unique_qaoa, n_occurrences_qaoa = np.unique(spin_configs_qaoa, axis=0, return_counts=True)
mean_n_occurrences_qaoa = n_occurrences_qaoa / float(np.sum(n_occurrences_qaoa))

f_vals_qaoa = []
sc_3d = None
for sc in spin_config_unique_qaoa:
    # sc_3d = np.atleast_3d(sc)
    # sc_3d = np.rollaxis(sc_3d, 2, 1)
    # eff_hamil = get_average_effective_hamiltonian(sc_3d, critic_qaoa.w_hh, critic_qaoa.w_vh, visible_nodes,
    #                                               big_gamma_final=big_gamma_final, beta_final=beta)
    # hamiltonians_qaoa.append(eff_hamil)
    f_vals_qaoa.append(calc_f(qubo_dict, sc.flatten()))

print('unique_configs', spin_config_unique_qaoa)
print('n_occurrences', n_occurrences_qaoa)
print('mean_n_occurrences', mean_n_occurrences_qaoa)
print('f_vals', f_vals_qaoa)
print('total # unique configs', len(spin_config_unique_qaoa))


print('=================')
print('Brute force')
n_qubits = critic_qaoa.sampler.n_nodes
lst = [list(i) for i in itertools.product([0, 1], repeat=n_qubits)]

f_min = np.inf
sc_min = None
f_vals_bf = []
for i in trange(2**n_qubits):
    sc = lst[i]
    # sc_ = np.atleast_3d(sc)
    # sc_ = np.rollaxis(sc_, 2, 1)
    # ham = get_average_effective_hamiltonian(
    #     sc_, critic.w_hh, critic.w_vh, visible_nodes, big_gamma_final=big_gamma_final, beta_final=beta)
    # all_energies.append(ham)
    fv = calc_f(qubo_dict, sc)
    f_vals_bf.append(fv)

    if fv < f_min:
        f_min = fv
        sc_min = sc

print('f_val', f_min)
print('Corresp. config', sc_min)

sc_min = np.array(sc_min)
# sc_min[sc_min==0] = -1
f_min = calc_f(qubo_dict, sc_min)

_, bins, _ = plt.hist(np.array(f_vals_bf), bins=1000, color='grey')
plt.hist(np.array(f_vals), bins=bins, color='red')
# plt.hist(np.array(f_vals_qaoa), bins=bins, color='green')

idx_max = np.argmax(n_occurrences)
plt.axvline(f_vals[idx_max], color='tab:blue', label='Annealing', lw=3)

idx_max = np.argmax(n_occurrences_qaoa)
plt.axvline(f_vals_qaoa[idx_max], color='tab:green', label='NumPyEigensolver')

plt.axvline(f_min, color='tab:red', ls='--', label='Brute force')
plt.legend()
plt.xlabel('f value')
plt.show()