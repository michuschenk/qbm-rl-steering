from qbm_rl_steering.agents.qbmq_utils import QFunction,\
    create_general_qubo_matrix, get_average_effective_hamiltonian,\
    get_free_energy

import math
from neal import SimulatedAnnealingSampler
import random
import numpy as np


def create_general_Q_from_mircea(Q_hh, Q_vh, visible_iterable):
    Q = dict()

    for k_pair, w in Q_hh.items():
        Q[k_pair] = Q[(k_pair[1],k_pair[0],)] = w

    for k_pair, w in Q_vh.items():

        if (k_pair[1],k_pair[1],) not in Q:
            Q[(k_pair[1],k_pair[1],)] = w * visible_iterable[k_pair[0]]
        else:
            Q[(k_pair[1],k_pair[1],)] += w * visible_iterable[k_pair[0]]
    return Q


def update_weights_mircea(
        Q_hh, Q_vh, samples, reward, future_F, current_F, visible_iterable,
        learning_rate, small_gamma):

        prob_dict = dict()

        for s in samples:
            for k_pair in Q_hh.keys():
                if k_pair in prob_dict:
                    prob_dict[k_pair] += \
                        (-1 if s[k_pair[0]] == 0 else 1) \
                        * (-1 if s[k_pair[1]] == 0 else 1)
                else:
                    prob_dict[k_pair] = \
                        (-1 if s[k_pair[0]] == 0 else 1) \
                        * (-1 if s[k_pair[1]] == 0 else 1)

            for k in s.keys():
                if k in prob_dict:
                    prob_dict[k] += \
                        (-1 if s[k] == 0 else 1)
                else:
                    prob_dict[k] = (-1 if s[k] == 0 else 1)

        for k_pair in Q_hh.keys():
            Q_hh[k_pair] = Q_hh[k_pair] - learning_rate \
                           * (reward + small_gamma * future_F - current_F) \
                           * prob_dict[k_pair] / len(samples)

        for k_pair in Q_vh.keys():
            Q_vh[k_pair] = Q_vh[k_pair] - learning_rate \
                           * (reward + small_gamma * future_F - current_F) \
                           * visible_iterable[k_pair[0]] \
                           * prob_dict[k_pair[1]] / len(samples)

        return Q_hh, Q_vh


def get_3d_hamiltonian_average_value_mircea(
        samples, Q, replica_count, average_size, big_gamma, beta):

    i_sample = 0

    h_sum = 0

    w_plus =\
        math.log10(math.cosh( big_gamma * beta / replica_count ) / math.sinh(
        big_gamma * beta / replica_count )) / ( 2 * beta )

    for _ in range(average_size):

        new_h_0 = new_h_1 = 0

        j_sample = i_sample

        a = i_sample + replica_count - 1

        while j_sample < a:

            added_set = set()

            for k_pair, v_weight in Q.items():

                if k_pair[0] == k_pair[1]:
                    new_h_0 = new_h_0 + v_weight * ( -1 if samples[j_sample][k_pair[0]] == 0 else 1 )
                else:
                    if k_pair not in added_set and ( k_pair[1] , k_pair[0] , ) not in added_set:
                    # if True:
                        new_h_0 = new_h_0 + v_weight\
                            * ( -1 if samples[j_sample][k_pair[0]] == 0 else 1 )\
                            * ( -1 if samples[j_sample][k_pair[1]] == 0 else 1 )

                        added_set.add( k_pair )

            for node_index in samples[j_sample].keys():

                new_h_1 = new_h_1\
                    + ( -1 if samples[j_sample][node_index] == 0 else 1 )\
                    * ( -1 if samples[j_sample + 1][node_index] == 0 else 1 )

            j_sample += 1

        added_set = set()

        for k_pair, v_weight in Q.items():

            if k_pair[0] == k_pair[1]:
                new_h_0 = new_h_0 + v_weight * ( -1 if samples[j_sample][k_pair[0]] == 0 else 1 )
            else:
                if k_pair not in added_set and ( k_pair[1] , k_pair[0] , ) not in added_set:
                # if True:
                    new_h_0 = new_h_0 + v_weight\
                        * ( -1 if samples[j_sample][k_pair[0]] == 0 else 1 )\
                        * ( -1 if samples[j_sample][k_pair[1]] == 0 else 1 )

                    added_set.add( k_pair )


        for node_index in samples[j_sample].keys():

            new_h_1 = new_h_1\
                + ( -1 if samples[j_sample][node_index] == 0 else 1 )\
                * ( -1 if samples[i_sample][node_index] == 0 else 1 )

        h_sum = h_sum + new_h_0 / replica_count + w_plus * new_h_1

        i_sample += replica_count

    return -1 * h_sum / average_size


def get_free_energy_mircea(average_hamiltonina, samples, replica_count, beta):
    key_list = sorted(samples[0].keys())

    prob_dict = dict()

    for i_sample in range(0,len(samples),replica_count):
        c_iterable = list()

        for s in samples[i_sample : i_sample + replica_count]:
            for k in key_list:
                c_iterable.append( s[k] )

        c_iterable = tuple(c_iterable)

        if c_iterable in prob_dict:
            prob_dict[c_iterable] += 1
        else:
            prob_dict[c_iterable] = 1

    a_sum = 0

    div_factor = len(samples) // replica_count

    for c in prob_dict.values():
        a_sum = a_sum + c * math.log10( c / div_factor ) / div_factor

    return average_hamiltonina + a_sum / beta


# Dummy parameters for the test
n_meas_avg = 100
n_replicas = 10
num_reads = n_meas_avg * n_replicas

big_gamma = 0.5
beta = 2.
learning_rate = 1e-1  # set high to see some differences should they occur
reward = 0.8
small_gamma = 0.99


# A random input state-action vector
visible_nodes = np.array([1, -1, 1, 1, 1, -1, 1, 1, 1, -1])

q_function = QFunction(8, 2, [0, 1, 2], learning_rate, small_gamma, n_replicas,
                       n_meas_avg, big_gamma, beta)

qubo_matrix = create_general_qubo_matrix(q_function.w_hh, q_function.w_vh,
                                         visible_nodes)
# gen_Q = create_general_Q_from_mircea(Q_hh, Q_vh, visible_nodes)

# Generate samples
samples = list(SimulatedAnnealingSampler().sample_qubo(
    Q=qubo_matrix, num_reads=num_reads).samples())

random.shuffle(samples)
samples_np = np.array([list(s.values()) for s in samples])
samples_np[samples_np == 0] = -1
samples_np = samples_np.reshape((n_meas_avg, n_replicas, -1))

# Average Hamiltonian
avg_eff_hamiltonian = get_average_effective_hamiltonian(
    samples_np, q_function.w_hh, q_function.w_vh, visible_nodes, big_gamma,
    beta)
avg_eff_hamiltonian_mircea = get_3d_hamiltonian_average_value_mircea(
    samples, qubo_matrix, n_replicas, n_meas_avg, big_gamma, beta)
print('\nAvg Hamiltonian', avg_eff_hamiltonian)
print('Avg Hamiltonian, Mircea', avg_eff_hamiltonian_mircea)

# Free energy
F_value = get_free_energy(samples_np, avg_eff_hamiltonian, beta)
F_value_mircea = get_free_energy_mircea(
    avg_eff_hamiltonian_mircea, samples, n_replicas, beta)
print('\nFree energy', F_value)
print('Free energy, Mircea', F_value_mircea)

# Weights update
Q_hh, Q_vh = q_function.w_hh.copy(), q_function.w_vh.copy()
q_value = -F_value
q_value_mircea = -F_value_mircea

# Dummy values
next_q_value = 23.
next_F_value = -next_q_value

print('\nWeights before', q_function.w_vh, q_function.w_hh)
q_function.update_weights(samples_np, visible_nodes, q_value,
                          next_q_value, reward)
print('Weights after', q_function.w_vh, q_function.w_hh)

print('Weights before, Mircea', Q_vh, Q_hh)
Q_hh, Q_vh = update_weights_mircea(
    Q_hh, Q_vh, samples, reward, next_F_value, F_value_mircea,
    visible_nodes, learning_rate, small_gamma)
print('Weights after, Mircea', Q_vh, Q_hh)

# Difference in weights comes from this update factor (sign in front of reward)
# If I flip this one sign, the updates are exactly the same (again up to ~1e-13)
print('\nWeights update factor',
      learning_rate * (
                reward + small_gamma * next_q_value - q_value))
print('Weights update factor, Mircea',
      -learning_rate * (
                reward + small_gamma * next_F_value - F_value_mircea))
