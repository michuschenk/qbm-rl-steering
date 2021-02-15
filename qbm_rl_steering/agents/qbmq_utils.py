import math
import random
import numpy as np
from typing import Dict, Tuple

from neal import SimulatedAnnealingSampler


def get_visible_nodes_array(state: np.ndarray, action: int,
                            n_bits_action_space: int = 2) -> np.ndarray:
    """
    Take state (e.g. directly from environment, in binary encoding), and action
    index (following env.action_map), and concatenate them to the
    visible_nodes_tuple. Then convert all 0s to -1s to be compatible with the
    spin states {-1, +1}.
    :param state: state as binary-encoded vector {0, 1}, obtained directly from
    the environment (either through .reset(), or .step()).
    :param action: index of action as used in environment, see env.action_map
    keys.
    :param n_bits_action_space: number of bits used to encode the discrete
    action space [given by ceil(log2(n_actions))].
    :return binary state-action numpy array with entries {-1, +1}.
    """
    # Turn action index into binary vector
    binary_fmt = f'0{n_bits_action_space}b'
    action_binary = [int(i) for i in format(action, binary_fmt)]
    visible_nodes = np.array(list(state) + action_binary)

    # Turn all 0s in binary state-action vector into -1s
    visible_nodes[visible_nodes == 0] = -1
    return visible_nodes


def create_general_qubo_matrix(
        w_hh: Dict, w_vh: Dict, visible_nodes: np.ndarray) -> Dict:
    """
    Creates a dictionary of the coupling weights that can be used with the
    DWAVE API. It corresponds to an upper triangular matrix, where the
    self-coupling weights (linear coefficients) are on the diagonal, i.e.
    (i, i) keys, and the quadratic coefficients are on the off-diagonal,
    i.e. (i, j) keys with i < j. As the visible nodes are clamped, they are
    incorporated into biases, i.e. self-coupling weights of the hidden nodes
    they are connected to.
    :param w_hh: Contains key pairs (i, j) where i < j and the values
    are the coupling weights between the hidden nodes i and j.
    :param w_vh: Contains key pairs (visible, hidden) and the values are the
    coupling weights between visible and hidden nodes.
    :param visible_nodes: numpy array of inputs, i.e. visible nodes, given by
    binary vectors (with -1 and +1) of the states and action vectors
    concatenated (length: n_bits_observation_space + n_bits_action_space).
    :return Dictionary of the QUBO upper triangular Q-matrix that describes the
    quadratic equation to be minimized.
    """
    qubo_matrix = dict()

    # Add hidden-hidden coupling weights to QUBO matrix
    # (note that QUBO matrix is no longer made symmetrical to reduce the
    # number of coupling weights -> speeds up the sampling a bit)
    for k, w in w_hh.items():
        qubo_matrix[k] = w

    # Self-coupling weights; clamp visible nodes to hidden nodes they are
    # connected to
    for k, w in w_vh.items():
        if (k[1], k[1]) not in qubo_matrix:
            qubo_matrix[(k[1], k[1])] = w * visible_nodes[k[0]]
        else:
            qubo_matrix[(k[1], k[1])] += w * visible_nodes[k[0]]
    return qubo_matrix


def get_qubo_samples(qubo_matrix: Dict, n_meas_for_average: int,
                     n_replicas: int) -> np.ndarray:
    """
    Run the AnnealingSampler with the DWAVE QUBO method and generate all the
    samples (= spin configurations at hidden nodes of Chimera graph,
    with values {-1, +1}). The DWAVE sample() method provides samples in a
    list of dictionaries. Each dictionary corresponds to 1 sample. The
    dictionary keys are the indices of the hidden nodes [0, 1,..., 15],
    and the values are the corresponding spins {0, 1}. We will work with {-1, 1}
    rather than {0, 1}, so we remap all the sampled spin configurations. We
    also turn the list of dictionaries into a 3D numpy array.
    :param qubo_matrix: Dictionary of coupling weights that corresponds to an
    upper triangular matrix, where the self-couplings (linear coefficients)
    are on the diagonal and the quadratic coefficients are on the off-diagonal.
    :param n_meas_for_average: number of 'independent measurements' that will
    then be used to calculate the average effective Hamiltonian.
    :param n_replicas: number of replicas in the 3D extension of the Ising
    model, see Fig. 1 in paper: https://arxiv.org/pdf/1706.00074.pdf .
    :return: 3D numpy array of all the samples. axis 0 is the index of the
    'independent measurement' for averaging the effective Hamiltonian,
    axis 1 is index k (replicas), and axis 2 is the index of the hidden nodes.
    """
    # TODO: what is the num_sweeps argument in the DWAVE sample method?
    # TODO: Is it OK to treat replicas in the same way as the 'independent
    #  measurements' (for avg.) that we do?
    num_reads = n_meas_for_average * n_replicas
    samples = list(SimulatedAnnealingSampler().sample_qubo(
        Q=qubo_matrix, num_reads=num_reads).samples())

    # TODO: why do we shuffle them?
    random.shuffle(samples)
    samples = np.array([list(s.values()) for s in samples])
    samples[samples == 0] = -1
    samples = samples.reshape((n_meas_for_average, n_replicas, -1))
    return samples


def get_average_effective_hamiltonian(
        samples: np.ndarray, w_hh: Dict, w_vh: Dict, visible_nodes: np.ndarray,
        big_gamma: float, beta: float) -> float:
    """
    This method calculates the average effective Hamiltonian as given in
    Eq. (9) in paper: https://arxiv.org/pdf/1706.00074.pdf , using samples
    obtained from DWAVE QUBO sample method.
    :param samples: samples returned by the DWAVE sample() method, but
    converted to numpy array and reshaped to
    (n_meas_for_average, n_replicas, n_hidden_nodes). The samples contain the
    spin states (1 or -1) of all the hidden nodes.
    :param w_hh: dictionary of coupling weights (quadratic coefficients)
    between the hidden nodes of the Chimera graph. The key is (h, h') where h
    and h' are in range [0, .., n_hidden_nodes] and h != h' (no self-coupling
    here).
    :param w_vh: dictionary of coupling weights between the visible nodes
    (states-action vector) and the corresponding hidden nodes they are
    connected to. The key is (v, h), where v is the index of the visible node
    in range [0, .., n_bits_observation_space, .. , n_bits_observation_space
    + n_bits_action_space]. h is in range [0, .., n_hidden_nodes].
    :param visible_nodes: numpy array of inputs, i.e. visible nodes, given by
    binary vectors ({-1, +1}) of the concatenated states and action vectors
    (length: n_bits_observation_space + n_bits_action_space).
    :param big_gamma: hyperparameter; strength of the transverse field
    (virtual, average value), see paper for details.
    :param beta: hyperparameter; inverse temperature used for simulated
    annealing, see paper for details.
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


def get_free_energy(samples: np.ndarray, avg_eff_hamiltonian: float,
                    beta: float) -> float:
    """
    We count the number of unique spin configurations on the 3D extended
    Ising model (torus), i.e. on the n_replicas * n_hidden_nodes nodes and
    calculate the probability of occurrence for each spin configuration
    through mean values.
    :param samples: samples returned by the DWAVE sample() method, but
    converted to numpy array and reshaped to
    (n_meas_for_average, n_replicas, n_hidden_nodes). The samples contain the
    spin states (1 or -1) of all the hidden nodes.
    :param avg_eff_hamiltonian: average effective Hamiltonian according to
    Eq. (9) in paper: https://arxiv.org/pdf/1706.00074.pdf .
    :param beta: hyperparameter; inverse temperature used for simulated
    annealing, see paper for more details.
    :return: free energy of the QBM defined according to the paper.
    """
    # Return the number of occurrences of unique spin configurations along
    # axis 0, i.e. along index of independent measurements
    _, n_occurrences = np.unique(samples, axis=0, return_counts=True)
    mean_n_occurrences = n_occurrences / float(np.sum(n_occurrences))
    a_sum = np.sum(mean_n_occurrences * np.log10(mean_n_occurrences))

    return avg_eff_hamiltonian + a_sum / beta


class QFunction(object):
    def __init__(self, n_bits_observation_space: int,
                 n_bits_action_space: int, possible_actions: list,
                 learning_rate: float, small_gamma: float, n_replicas: int,
                 n_meas_for_average: int, big_gamma: float,
                 beta: float) -> None:
        """
        Implementation of the Q function (state-action value function) using
        DWAVE neal sampler.
        :param n_bits_observation_space: number of bits used to encode
        observation space of environment
        :param n_bits_action_space: number of bits required to encode the
        actions that are possible in the given environment
        :param possible_actions: list of possible action indices
        :param learning_rate: RL. parameter, learning rate for update of
        coupling weights of the Chimera graph.
        :param small_gamma: RL parameter, discount factor cumulative rewards.
        :param n_replicas: number of replicas in the 3D extension of the Ising
        model, see Fig. 1 in paper: https://arxiv.org/pdf/1706.00074.pdf
        :param n_meas_for_average: number of 'independent measurements' that
        will then be used to calculate the average effective Hamiltonian.
        :param big_gamma:  hyperparameter; strength of the transverse field
        (virtual, average value), see paper for more details.
        :param beta: hyperparameter; inverse temperature used for simulated
        annealing, see paper for details.
        """
        self.n_replicas = n_replicas
        self.n_meas_for_average = n_meas_for_average
        self.big_gamma = big_gamma
        self.beta = beta

        self.learning_rate = learning_rate
        self.small_gamma = small_gamma

        self.n_bits_observation_space = n_bits_observation_space
        self.n_bits_action_space = n_bits_action_space
        self.possible_actions = possible_actions
        self.w_hh, self.w_vh = self._initialise_weights()

    def _initialise_weights(self) -> Tuple[Dict, Dict]:
        """
        Initialise the coupling weights of the Chimera graph, i.e. both
        hidden-hidden couplings and visible-hidden couplings.
        """
        # ==============================
        # COUPLINGS BETWEEN HIDDEN NODES
        # This loop initializes weights to fully connect the nodes in the two
        # unit cells of the Chimera graph (see Fig. 2 in the paper:
        # https://arxiv.org/pdf/1706.00074.pdf). The indexing of the nodes is
        # starting at the top left (node 0) and goes down vertically (blue
        # nodes), and then to the right (first red node is index 4). These
        # are 32 couplings = 2 * 4**2.
        w_hh = dict()
        for i, ii in zip(tuple(range(4)), tuple(range(8, 12))):
            for j, jj in zip(tuple(range(4, 8)), tuple(range(12, 16))):
                w_hh[(i, j)] = 2 * random.random() - 1
                w_hh[(ii, jj)] = 2 * random.random() - 1

        # This loop connects the 4 red nodes of the first unit cell of the
        # Chimera graph on the left (Fig. 2) to the blue nodes of the second
        # unit on the right, i.e. node 4 to node 12; node 5 to node 13,
        # etc. These are 4 additional couplings.
        for i, j in zip(tuple(range(4, 8)), tuple(range(12, 16))):
            w_hh[(i, j)] = 2 * random.random() - 1

        # We get a total of 32 + 4 = 36 hidden couplings defined by w_hh.

        # ==============================
        # COUPLINGS BETWEEN VISIBLE [the 'input' (= state layer) and the
        # 'output' (= action layer) of a 'classical' Q-net] AND THE HIDDEN NODES
        w_vh = dict()

        # Dense connection between the state nodes (visible) and the BLUE
        # hidden nodes (all 8 of them) of the Chimera graph. Blue nodes have
        # indices [0, 1, 2, 3, 12, 13, 14, 15]. We hence have connections
        # between the state nodes [0, 1, ..., n_bits_observation_space] to
        # all of the blue nodes. This is n_bits_observation_space * 8 = 64
        # couplings (here).
        for j in (tuple(range(4)) + tuple(range(12, 16))):
            for i in range(self.n_bits_observation_space):
                w_vh[(i, j)] = 2 * random.random() - 1

        # Dense connection between the action nodes (visible) and the RED hidden
        # nodes (all 8 of them) of the Chimera graph. Red nodes have indices
        # [4, 5, 6, 7, 8, 9, 10, 11]. We hence have connections between the
        # action nodes [n_bits_observation_space, ..,
        # n_bits_observation_space + n_bits_action_space] (here: [8, 9]) to all
        # of the red nodes. This is n_bits_action_space * 8 = 16 couplings
        # (here).
        for j in (tuple(range(4, 8)) + tuple(range(8, 12))):
            for i in range(
                    self.n_bits_observation_space,
                    self.n_bits_observation_space + self.n_bits_action_space):
                w_vh[(i, j)] = 2 * random.random() - 1

        # We get a total of 64 + 16 = 80 couplings (here) defined by w_vh.
        return w_hh, w_vh

    def calculate_q_value(self, state: np.ndarray, action: int) -> \
            Tuple[float, np.ndarray, np.ndarray]:
        """
        Based on state and chosen action, calculate the free energy,
        samples and vis_iterable.
        :param state: state the environment is in (binary vector, directly
        obtained from either env.reset(), or env.step())
        :param action: chosen action (index)
        :return free energy, samples, and visible_nodes.
        """
        visible_nodes = get_visible_nodes_array(state=state, action=action)
        qubo_matrix = create_general_qubo_matrix(
            self.w_hh, self.w_vh, visible_nodes)

        samples = get_qubo_samples(
            qubo_matrix, self.n_meas_for_average, self.n_replicas)

        avg_eff_hamiltonian = get_average_effective_hamiltonian(
            samples, self.w_hh, self.w_vh, visible_nodes, self.big_gamma,
            self.beta)

        free_energy = get_free_energy(samples, avg_eff_hamiltonian, self.beta)
        q_value = -free_energy

        return q_value, samples, visible_nodes

    def follow_policy(
            self, state: np.ndarray, epsilon: float) -> \
            Tuple[int, float, np.ndarray, np.ndarray]:
        """
        Follow the epsilon-greedy policy to get the next action. With
        probability epsilon we pick a random action, and with probability
        (1 - epsilon) we pick the action greedily, i.e. such that
        a = argmax_a Q(s, a).
        :param state: state that the environment is in (binary vector, directly
        obtained from either env.reset(), or env.step()).
        :param epsilon: probability for choosing random action.
        :return: chosen action index, q_value, QUBO samples, state-action
        values of visible_nodes
        """
        if np.random.random() < epsilon:
            # Pick action randomly
            action = random.choice(self.possible_actions)
            q_value, samples, visible_nodes = self.calculate_q_value(
                state, action)
            return action, q_value, samples, visible_nodes
        else:
            # Pick action greedily
            # Since the Chimera graph (QBM) does not offer an input ->
            # output layer structure in the classical Q-net sense, we have to
            # loop through all the actions to calculate the Q value for every
            # action to then pick the argmax_a Q
            max_dict = {'q_value': float('-inf'), 'action': None,
                        'samples': None, 'visible_nodes': None}

            for action in self.possible_actions:
                q_value, samples, visible_nodes = self.calculate_q_value(
                    state, action)

                # TODO: what needs to be done is to pick action randomly in
                #  case there are several actions with the same Q values.
                if max_dict['q_value'] < q_value:
                    max_dict['q_value'] = q_value
                    max_dict['action'] = action
                    max_dict['samples'] = samples
                    max_dict['visible_nodes'] = visible_nodes

            return (max_dict['action'], max_dict['q_value'],
                    max_dict['samples'], max_dict['visible_nodes'])

    def update_weights(
            self, samples: np.ndarray, visible_nodes: np.ndarray,
            current_Q: float, future_Q: float, reward: float) -> None:
        """
        Calculates the TD(0) learning step, i.e. the updates of the coupling
        dictionaries w_hh, w_vh according to Eqs. (11) and (12) in the paper:
        https://arxiv.org/pdf/1706.00074.pdf
        :param samples: samples returned by the DWAVE sample() method,
        but converted to numpy array and reshaped to
        (n_meas_for_average, n_replicas, n_hidden_nodes). The samples contain the
        spin states (1 or -1) of all the hidden nodes.
        :param visible_nodes: numpy array of visible nodes, given by binary
        vectors (with -1 and +1) of the states and action vectors concatenated
        (length: n_bits_observation_space + n_bits_action_space).
        :param current_Q: Q function value at time step n, Q(s_n, a_n)
        :param future_Q: Q function value at time step n+1, Q(s_n+1, a_n+1)
        :param reward: RL reward of current step, r_n(s_n, a_n)
        :return None
        """
        # This term is the same for both weight updates w_hh and w_vh
        update_factor = self.learning_rate * (
                reward + self.small_gamma * future_Q - current_Q)

        # Update of w_vh, Eq. (11)
        h_avg = np.mean(np.mean(samples, axis=0), axis=0)
        for v, h in self.w_vh.keys():
            self.w_vh[(v, h)] += update_factor * visible_nodes[v] * h_avg[h]

        # Update of w_hh, Eq. (12)
        for h, h_prime in self.w_hh.keys():
            self.w_hh[(h, h_prime)] += update_factor * np.mean(
                samples[:, :, h] * samples[:, :, h_prime])
