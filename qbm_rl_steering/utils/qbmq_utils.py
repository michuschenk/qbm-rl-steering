import math
import random
import numpy as np
from typing import Dict, Tuple

# The DWAVE-neal is NOT a quantum annealing simulator (SQA, transverse field
# decay) in my understanding, but it does simulated annealing (SA, temperature
# decay)
from neal import SimulatedAnnealingSampler

# Found instead this library that does SQA
# import sqaod as sq
# Can also run on an nvidia GPU (CUDA) => see utils/sqa.py
from qbm_rl_steering.utils.sqa import SQA


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


def create_general_qubo_dict(
        w_hh: Dict, w_vh: Dict, visible_nodes: np.ndarray) -> Dict:
    """
    Creates a dictionary of the coupling weights of the graph. It corresponds
    to an upper triangular matrix, where the self-coupling weights (linear
    coefficients) are on the diagonal, i.e. (i, i) keys, and the quadratic
    coefficients are on the off-diagonal, i.e. (i, j) keys with i < j. As the
    visible nodes are clamped, they are incorporated into biases,
    i.e. self-coupling weights of the hidden nodes they are connected to.
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
    qubo_dict = dict()

    # Add hidden-hidden coupling weights to QUBO matrix
    # (note that QUBO matrix is no longer made symmetrical to reduce the
    # number of coupling weights -> speeds up the sampling a bit)
    for k, w in w_hh.items():
        qubo_dict[k] = w

    # Self-coupling weights; clamp visible nodes to hidden nodes they are
    # connected to
    for k, w in w_vh.items():
        if (k[1], k[1]) not in qubo_dict:
            qubo_dict[(k[1], k[1])] = w * visible_nodes[k[0]]
        else:
            qubo_dict[(k[1], k[1])] += w * visible_nodes[k[0]]

    return qubo_dict


def get_average_effective_hamiltonian(
        samples: np.ndarray, w_hh: Dict, w_vh: Dict, visible_nodes: np.ndarray,
        big_gamma_final: float, beta: float) -> float:
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
    :param big_gamma_final: hyperparameter; final, i.e. at the end of the SQA,
    strength of the transverse field (virtual, average value), see paper for
    details.
    :param beta: Inverse temperature (note that this parameter is kept
    constant in SQA other than in SA).
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
    x = big_gamma_final * beta / n_replicas
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


def get_free_energy(spin_configurations: np.ndarray, avg_eff_hamiltonian: float,
                    beta: float) -> float:
    """
    We count the number of unique spin configurations on the 3D extended
    Ising model (torus), i.e. on the n_replicas * n_hidden_nodes nodes and
    calculate the probability of occurrence for each spin configuration
    through mean values.
    :param spin_configurations: samples returned by the DWAVE sample() method,
    but converted to numpy array and reshaped to
    (n_meas_for_average, n_replicas, n_hidden_nodes). The samples contain the
    spin configurations (1 or -1) of all the hidden nodes (including the
    replicas, i.e. Trotter slices).
    :param avg_eff_hamiltonian: average effective Hamiltonian according to
    Eq. (9) in paper: https://arxiv.org/pdf/1706.00074.pdf .
    :param beta: Inverse temperature (note that this parameter is kept
    constant in SQA other than in SA).
    :return: free energy of the QBM defined according to the paper.
    """
    # Return the number of occurrences of unique spin configurations along
    # axis 0, i.e. along index of independent measurements
    _, n_occurrences = np.unique(spin_configurations, axis=0,
                                 return_counts=True)
    mean_n_occurrences = n_occurrences / float(np.sum(n_occurrences))
    a_sum = np.sum(mean_n_occurrences * np.log10(mean_n_occurrences))

    return avg_eff_hamiltonian + a_sum / beta


class QFunction(object):
    def __init__(self, n_bits_observation_space: int,
                 n_bits_action_space: int, small_gamma: float, n_replicas: int,
                 big_gamma: Tuple[float, float], beta: float,
                 n_annealing_steps: int, n_meas_for_average: int) -> None:
        """
        Implementation of the Q function (state-action value function) using
        an SQA method to update / train.
        :param n_bits_observation_space: number of bits used to encode
        observation space of environment
        :param n_bits_action_space: number of bits required to encode the
        actions that are possible in the given environment
        :param small_gamma: RL parameter, discount factor for
        cumulative future rewards.
        :param n_replicas: number of replicas (aka. Trotter slices) in the 3D
        extension of the Ising model, see Fig. 1 in paper:
        https://arxiv.org/pdf/1706.00074.pdf
        :param big_gamma: Transverse field; first entry is initial gamma and
        second entry is final gamma at end of annealing process.
        :param beta: Inverse temperature (note that this parameter is kept
        constant in SQA other than in SA).
        :param n_meas_for_average: number of times we run an independent
        annealing process from start to end
        :param n_annealing_steps: number of steps that one annealing
        process should take (~annealing time).
        """
        self.sqa = SQA(big_gamma, beta, n_replicas)
        self.n_annealing_steps = n_annealing_steps
        self.n_meas_for_average = n_meas_for_average

        self.small_gamma = small_gamma

        self.n_bits_observation_space = n_bits_observation_space
        self.n_bits_action_space = n_bits_action_space
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
        spin_configurations and vis_iterable.
        :param state: state the environment is in (binary vector, directly
        obtained from either env.reset(), or env.step())
        :param action: chosen action (index)
        :return free energy, spin_configurations, and visible_nodes.
        """
        # Define QUBO
        visible_nodes = get_visible_nodes_array(
            state=state, action=action,
            n_bits_action_space=self.n_bits_action_space)
        qubo_dict = create_general_qubo_dict(
            self.w_hh, self.w_vh, visible_nodes)

        # Run SQA to get spin configurations
        spin_configurations = self.sqa.anneal(
            qubo_dict=qubo_dict,
            n_meas_for_average=self.n_meas_for_average,
            n_steps=self.n_annealing_steps)

        # Based on sampled spin configurations calculate free energy
        avg_eff_hamiltonian = get_average_effective_hamiltonian(
            spin_configurations, self.w_hh, self.w_vh, visible_nodes,
            self.sqa.big_gamma_final, self.sqa.beta)

        free_energy = get_free_energy(
            spin_configurations, avg_eff_hamiltonian, self.sqa.beta)
        q_value = -free_energy

        return q_value, spin_configurations, visible_nodes

    def update_weights(
            self, spin_configurations: np.ndarray, visible_nodes: np.ndarray,
            current_q: float, future_q: float, reward: float,
            learning_rate: float) -> None:
        """
        Calculates the TD(0) learning step, i.e. the updates of the coupling
        dictionaries w_hh, w_vh according to Eqs. (11) and (12) in the paper:
        https://arxiv.org/pdf/1706.00074.pdf
        :param spin_configurations: spin configurations returned by the SQA.
        np array of shape (n_meas_for_average, n_replicas, n_hidden_nodes).
        The spin configurations contain the spin states (1 or -1) of all
        the hidden nodes (that includes replicas, i.e. Trotter slices).
        :param visible_nodes: numpy array of visible nodes, given by binary
        vectors (with -1 and +1) of the states and action vectors concatenated
        (length: n_bits_observation_space + n_bits_action_space).
        :param current_q: Q function value at time step n, Q(s_n, a_n)
        :param future_q: Q function value at time step n+1, Q(s_n+1, a_n+1)
        :param reward: RL reward of current step, r_n(s_n, a_n)
        :param learning_rate: RL. parameter, learning rate for update of
        coupling weights of the Chimera graph.
        :return None
        """
        # This term is the same for both weight updates w_hh and w_vh
        update_factor = learning_rate * (
                reward + self.small_gamma * future_q - current_q)

        # Update of w_vh, Eq. (11)
        h_avg = np.mean(np.mean(spin_configurations, axis=0), axis=0)
        for v, h in self.w_vh.keys():
            self.w_vh[(v, h)] += update_factor * visible_nodes[v] * h_avg[h]

        # Update of w_hh, Eq. (12)
        for h, h_prime in self.w_hh.keys():
            self.w_hh[(h, h_prime)] += update_factor * np.mean(
                spin_configurations[:, :, h] *
                spin_configurations[:, :, h_prime])


# OLD / NO LONGER IN USE
def dwave_anneal(
        qubo_dict: Dict, n_meas_for_average: int, n_replicas: int,
        beta: Tuple[float, float]) -> np.ndarray:
    """
    I THINK THIS IS JUST SIMULATED ANNEALING (SA), NOT SIMULATED QUANTUM
    ANNEALING (SQA). WE NEED THE LATTER.
    Run the AnnealingSampler with the DWAVE QUBO method and generate all the
    samples (= spin configurations at hidden nodes of Chimera graph,
    with values {-1, +1}). The DWAVE sample() method provides samples in a
    list of dictionaries. Each dictionary corresponds to 1 sample. The
    dictionary keys are the indices of the hidden nodes [0, 1,..., 15],
    and the values are the corresponding spins {0, 1}. We will work with {-1, 1}
    rather than {0, 1}, so we remap all the sampled spin configurations. We
    also turn the list of dictionaries into a 3D numpy array.
    :param qubo_dict: Dictionary of coupling weights that corresponds to an
    upper triangular matrix, where the self-couplings (linear coefficients)
    are on the diagonal and the quadratic coefficients are on the off-diagonal.
    :param n_meas_for_average: number of 'independent measurements' that will
    then be used to calculate the average effective Hamiltonian.
    :param n_replicas: number of replicas in the 3D extension of the Ising
    model, see Fig. 1 in paper: https://arxiv.org/pdf/1706.00074.pdf .
    :param beta: hyperparameter; inverse temperature used for simulated
    annealing, from start to end, according to linear (or geometric)
    schedule, see paper for details.
    :return: 3D numpy array of all the samples. axis 0 is the index of the
    'independent measurement' for averaging the effective Hamiltonian,
    axis 1 is index k (replicas), and axis 2 is the index of the hidden nodes.
    """
    # TODO: Is it OK to treat replicas in the same way as the 'independent
    #  measurements' (for avg.) that we do?
    num_reads = n_meas_for_average * n_replicas
    spin_configurations = list(SimulatedAnnealingSampler().sample_qubo(
        Q=qubo_dict, num_reads=num_reads,
        beta_schedule_type='linear', beta_range=beta).samples())

    # TODO: why do we shuffle them?
    random.shuffle(spin_configurations)
    spin_configurations = np.array([
        list(s.values()) for s in spin_configurations])
    spin_configurations[spin_configurations == 0] = -1
    spin_configurations = spin_configurations.reshape((n_meas_for_average,
                                                       n_replicas, -1))
    return spin_configurations
