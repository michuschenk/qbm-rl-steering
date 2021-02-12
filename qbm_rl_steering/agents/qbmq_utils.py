import math
import numpy as np
from typing import Dict, Tuple, Optional, Union

from neal import SimulatedAnnealingSampler


def remap_binary_vector(
        vec: Union[Tuple, np.ndarray], mapping: Dict = {0: -1}) -> Tuple:
    """
    Remap values in vec (binary vector) according to the mapping dict. Default
    mapping means that 0s are transformed into -1s.
    :param vec: input np.ndarray or tuple that needs to be converted.
    :param mapping dictionary that contains the remapping of the values
    :return remapped vector as tuple. """
    vec = np.array(vec)
    for inp, out in mapping.items():
        vec[vec == inp] = out
    return tuple(vec)


def make_action_binary(
        action_index: int, n_bits_action_space: int = 2) -> Tuple:
    """
    Similar to make_state_discrete_binary. Convert action_index to a
    binary vector using 0s and 1s. Conversion of 0s to -1s will be done in a
    separate function.
    :param action_index: index of action (integer). See which index
    corresponds to which action in env.action_map.
    :param n_bits_action_space: number of bits required to describe action
    space in binary.
    :return binary vector that encodes the action_index """
    binary_fmt = f'0{n_bits_action_space}b'
    action_binary = tuple([int(i) for i in format(action_index, binary_fmt)])
    return action_binary


def get_visible_nodes_tuple(
        state: np.ndarray, action: int, n_bits_action_space: int = 2) -> Tuple:
    """
    Take state (e.g. directly from environment), and action_index (
    following env.action_map), and concatenate them to the visible_iterable
    tuple required for the create_general_Q_from(..) function. This also
    converts all 0s appearing in state and action_index (once made binary) to
    -1s.
    :param state: state in binary-encoded vector, as obtained directly from
    the environment (either through .reset(), or .step())
    :param action: index of action as used in environment
    :return tuple, concatenation of state and action, following the
    convention of Mircea's code on QBM. """
    s = remap_binary_vector(state)
    a = remap_binary_vector(make_action_binary(action, n_bits_action_space))
    return s + a


def create_general_qubo_matrix(
        w_hh: Dict, w_vh: Dict, visible_nodes: Tuple) -> Dict:
    """
    Creates a dictionary of the coupling weights that can be used with the
    DWAVE API. It corresponds to an upper triangular matrix, where the
    self-couplings (linear coefficients) are on the diagonal, i.e. (i,
    i) keys, and the quadratic coefficients are on the off-diagonal, i.e. (i, j)
    keys with i < j. As the visible nodes are clamped, they are incorporated
    into biases, i.e. self-couplings of the hidden nodes they are connected to.
    :param w_hh: Contains key pairs (i, j) where i < j and the values
    are the coupling weights between the hidden nodes i and j.
    :param w_vh: Contains key pairs (visible, hidden) and the values are the
    coupling weights between visible and hidden nodes.
    :param visible_nodes: tuple of inputs, i.e. visible nodes, given by
    binary vectors (with -1 and +1) of the states and action vectors
    concatenated (length: n_bits_observation_space + n_bits_action_space).
    :return Dictionary of the QUBO upper triangular Q-matrix that describes the
    quadratic equation to be minimized.
    """
    qubo_matrix = dict()

    # Add hidden-hidden couplings to QUBO matrix
    # (note that QUBO matrix is no longer made symmetrical to reduce the
    # number of couplings -> speeds up the sampling a bit)
    for k, w in w_hh.items():
        qubo_matrix[k] = w

    # Self-couplings; clamp visible nodes to hidden nodes they are connected to
    for k, w in w_vh.items():
        if (k[1], k[1]) not in qubo_matrix:
            qubo_matrix[(k[1], k[1])] = w * visible_nodes[k[0]]
        else:
            qubo_matrix[(k[1], k[1])] += w * visible_nodes[k[0]]
    return qubo_matrix


def get_qubo_samples(qubo_matrix: Dict, n_meas_for_average: int,
                     n_replicas: int) -> np.ndarray:
    """
    Samples are provided in a list of dictionaries. Each dictionary
    corresponds to 1 sample. The dictionary keys are the indices of the hidden
    nodes [0, 1,..., 15], and the values are the corresponding spins {0, 1}.
    We will work with {-1, 1} rather than {0, 1}, so we remap all the sampled
    spin configurations. We also turn the list of dictionaries into a 3D
    numpy array, where axis 0 is the index of the 'independent measurement'
    for averaging the effective Hamiltonian, axis 1 is index k (replicas,
    i.e. 3D extension of the Ising model), and axis 2 is the index of the
    hidden nodes, after reshaping.
    :param qubo_matrix: 
    :param num_reads: 
    :return: 
    """
    # TODO: what is the num_sweeps argument in the DWAVE sample method?
    # TODO: Is it OK to treat replicas in the same way as the 'independent
    #  measurements' (for avg.) that we do?
    num_reads = n_meas_for_average * n_replicas
    samples = list(SimulatedAnnealingSampler().sample_qubo(
        Q=qubo_matrix, num_reads=num_reads).samples())

    samples = np.array([list(s.values()) for s in samples])
    samples[samples == 0] = -1
    samples = samples.reshape((n_meas_for_average, n_replicas, -1))
    return samples


def get_average_effective_hamiltonian(
        samples: np.ndarray, w_hh: Dict, w_vh: Dict, visible_nodes: Tuple,
        big_gamma: float, beta: float) -> float:
    """
    This method calculates the average effective Hamiltonian as given in
    Eq. (9) in paper: https://arxiv.org/pdf/1706.00074.pdf , using samples
    obtained from DWAVE QUBO sample method.
    :param samples: samples returned by the DWAVE sample() method, but
    converted to numpy array and reshaped to
    (n_meas_for_average, n_replicas, n_nodes). The samples contain the spin
    states (1 or -1) of all the hidden nodes.
    :param w_hh: coupling dictionary (quadratic coefficients) between the
    hidden nodes. The key is (h, h') where h and h' are in range [0, ..,
    n_nodes] and h != h' (no self-coupling here).
    :param w_vh: couplings between the visible nodes (states and actions) and
    the corresponding hidden nodes. The key is (v, h), where v is the index
    of the visible node in range [0, .., n_bits_observation_space, .. ,
    n_bits_observation_space + n_bits_action_space]. h is in range [0, ..,
    n_nodes].
    :param visible_nodes: tuple of inputs, i.e. visible nodes, given by
    binary vectors (with -1 and +1) of the states and action vectors
    concatenated (length: n_bits_observation_space + n_bits_action_space).
    :param big_gamma: hyperparameter; strength of the transverse field (
    virtual, average
    value), see paper for more details.
    :param beta: hyperparameter; inverse temperature used for simulated
    annealing
    :return: single float; effective Hamiltonian averaged over the individual
    measurements.
    """
    _, n_replicas, _ = samples.shape

    # FIRST TERM, sum over h, h' in Eq. (9)
    h_sum_1 = 0.
    for (h, h_prime), w in w_hh.items():
        h_sum_1 += w * samples[:, :, h] * samples[:, :, h_prime]

    # SECOND term, sum over v, h in Eq. (9)
    h_sum_2 = 0.
    for (v, h), w in w_vh.items():
        h_sum_2 += w * visible_nodes[v] * samples[:, :, h]

    # Sum over replicas (sum over k), divide by n_replicas, and negate
    # This corresponds to the first line in Eq. (9)
    # h_sum_12 has shape (n_meas_for_average,)
    h_sum_12 = -np.sum(h_sum_1 + h_sum_2, axis=-1) / n_replicas

    # THIRD term, [-w_plus * (sum_hk_hkplus1 + sum_h1_hr)], in Eq. (9)
    # h_sum_3 has shape (n_meas_for_average,)
    x = big_gamma * beta / n_replicas
    coth_term = math.cosh(x) / math.sinh(x)
    w_plus = math.log10(coth_term) / (2. * beta)

    # I think there is a typo in Eq. (9). The summation index in w+(.. + ..)
    # of the first term should only go from k=1 to r-1.
    hk_hkplus1_sum = np.sum(np.sum(
        samples[:, :-1, :] * samples[:, 1:, :], axis=1), axis=-1)
    h1_hr_sum = np.sum(samples[:, -1, :] * samples[:, 0, :], axis=-1)
    h_sum_3 = -w_plus * (hk_hkplus1_sum + h1_hr_sum)

    # Take average over n_meas_for_average
    return float(np.mean(h_sum_12 + h_sum_3))


def _nodes_of_3d_bm_as_tuple(configuration):
    """
    Converts samples corresponding to all replica from 1 configuration = 3D
    Boltzmann machine into 1 tuple with all measurements
    :param configuration: samples corresponding to all replica from 1
    configuration = 3D Boltzmann machine
    :return: tuple with all measurements
    """
    config = ()
    for replica in configuration:
        config = config + tuple(replica.values())
    return config


def get_free_energy(average_effective_hamiltonian: float, samples: dict,
                    replica_count: int, average_size: int, beta: float):
    configurations = [samples[x:x + replica_count]
                      for x in range(0, len(samples), replica_count)]
    prob_dict = dict()

    for conf in configurations:
        conf_tuple = _nodes_of_3d_bm_as_tuple(conf)
        if conf_tuple in prob_dict:
            prob_dict[conf_tuple] += 1
        else:
            prob_dict[conf_tuple] = 1

    a_sum = 0
    for n_occurrence_of_conf in prob_dict.values():
        mean_occurence_of_conf = n_occurrence_of_conf / average_size
        a_sum = a_sum + (mean_occurence_of_conf * math.log10(
            mean_occurence_of_conf))

    return average_effective_hamiltonian + a_sum / beta


def update_weights(samples: np.ndarray, w_hh: Dict, w_vh: Dict,
                   visible_nodes: Tuple, current_Q: float, future_Q: float,
                   reward: float, learning_rate: float, small_gamma: float,
                   in_place: bool = True) -> Optional[Tuple[Dict, Dict]]:
    """
    Calculates the TD(0) learning step, i.e. the updates of the coupling
    dictionaries w_hh, w_vh according to Eqs. (11) and (12) in the paper:
    https://arxiv.org/pdf/1706.00074.pdf
    :param samples: samples returned by the DWAVE sample() method,
    but converted to numpy array and reshaped to
    (n_meas_for_average, n_replicas, n_nodes). The samples contain the spin
    states (1 or -1) of all the hidden nodes.
    :param w_hh: dictionary of couplings between the hidden nodes of the
    Chimera graph, Fig. 2 in paper.
    :param w_vh: dictionary of couplings between visible nodes (state and
    action nodes) and corresponding hidden nodes of the Chimera graph,
    Fig. 2 in paper.
    :param visible_nodes: tuple of inputs, i.e. visible nodes, given by
    binary vectors (with -1 and +1) of the states and action vectors
    concatenated (length: n_bits_observation_space + n_bits_action_space).
    :param current_Q: Q function value at time step n, Q(s_n, a_n)
    :param future_Q: Q function value at time step n+1, Q(s_n+1, a_n+1)
    :param reward: RL reward of current step, r_n(s_n, a_n)
    :param learning_rate: note that this corresponds to the parameter epsilon in
    the paper
    :param small_gamma: discount factor
    :param in_place: flag to decide whether update should be done in place
    (i.e. nothing is returned), or not (new coupling dictionaries will be
    returned, original dictionaries are left untouched).
    :return: Either None or tuple of the new coupling dictionaries depending
    on the flag in_place.
    """
    # If operation not done in place, need to make a copy first
    w_hh_, w_vh_ = w_hh, w_vh
    if not in_place:
        w_hh_, w_vh_ = w_hh.copy(), w_vh.copy()

    # This term is the same for both weight updates w_hh and w_vh
    update_factor = learning_rate * (
            reward + small_gamma * future_Q - current_Q)

    # Update of w_vh, Eq. (11)
    h_avg = np.mean(np.mean(samples, axis=0), axis=0)
    for v, h in w_vh_.keys():
        w_vh_[(v, h)] += update_factor * visible_nodes[v] * h_avg[h]

    # Update of w_hh, Eq. (12)
    for h, h_prime in w_hh_.keys():
        w_hh_[(h, h_prime)] += update_factor * np.mean(
            samples[:, :, h] * samples[:, :, h_prime])

    if not in_place:
        return w_hh_, w_vh_
