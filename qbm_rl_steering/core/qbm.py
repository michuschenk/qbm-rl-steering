import math
import random
import numpy as np
from typing import Dict, Tuple, Union
import gym

import dimod

# SQA (simulated quantum annealing)
try:
    from qbm_rl_steering.samplers.sqa_annealer import SQA
except ImportError:
    print('! Cannot import libraries (sqaod) required for SQA...')

# Quantum annealing on proper QPU using Amazon Braket
try:
    from qbm_rl_steering.samplers.qpu_annealer import QPU
except ImportError:
    print('! Cannot import libraries required for QPU ...')

# Qiskit samplers
from qbm_rl_steering.samplers.qaoa_solver import QAOA


def get_visible_nodes_array(state: np.ndarray, action: int,
                            state_space: gym.spaces.Space,
                            action_space: gym.spaces.Discrete) -> np.ndarray:
    """Takes state (e.g. directly from environment) and action as inputs and
    concatenates them to visible_nodes_tuple that will be input to clamped QBM.
    :param state: state as np.ndarray from the environment obtained either through
    env.reset() or env.step().
    :param action: action as used in environment.
    :param state_space: openAI gym state space, either Box or MultiBinary type.
    :param action_space: openAI gym action space of Discrete type.
    :returns normalized state-action vector corresponding to visible nodes of
    clamped QBM."""
    if isinstance(state_space, gym.spaces.Box):
        # Continuous state space
        state_normalized = 2. * np.array(state) / (state_space.high - state_space.low)
        if (np.abs(state_normalized) > 2.).any():
            print("STATE WELL OUT OF NORMALIZATION RANGE")
        all_actions = np.arange(action_space.n)
        action_normalized = (2. * action - np.max(all_actions)) / np.max(all_actions)
        visible_nodes = np.array(list(state_normalized) + [action_normalized])

    elif isinstance(state_space, gym.spaces.MultiBinary):
        # Discrete binary state space
        n_bits_action_space = math.ceil(math.log2(action_space.n))

        # This is in case where discrete env.action_space.n == 1 (does not make much
        # sense from RL point-of-view)
        if action_space.n < 2:
            n_bits_action_space = 1

        # Turn action index into binary vector
        binary_fmt = f'0{n_bits_action_space}b'
        action_binary = [int(i) for i in format(action, binary_fmt)]
        visible_nodes = np.array(list(state) + action_binary)
        # Turn all 0s in binary state-action vector into -1s
        # (this is the convention for the Ising model).
        visible_nodes[visible_nodes == 0] = -1
    else:
        raise TypeError("State space has to be either gym.spaces.Box or "
                        "gym.spaces.MultiBinary")
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
    the states and action vectors concatenated.
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
    ising = qubo_dict.copy()

    J = {}
    h = {}

    # TODO: not so robust (nor elegant) maybe
    keylist = []
    for i, j in ising.keys():
        keylist.append(i)
        keylist.append(j)
    N = max(keylist) + 1
    for i in range(N):
        try:
            h[i] = ising[(i, i)]
        except KeyError:
            h[i] = 0

        for j in range(i + 1, N):
            try:
                J[i, j] = ising[(i, j)]
            except KeyError:
                J[i, j] = 0

    model = dimod.BinaryQuadraticModel(h, J, 0.0, vartype='SPIN')
    qubo, offset = model.to_qubo()

    return qubo


def get_gradient_average_effective_hamiltonian(
        spin_configurations: np.ndarray, w_vh: Dict,
        visible_nodes: np.ndarray):
    """
    Calculate gradient with respect to visible nodes
    (note that we only take gradient of Heff wrt. v)
    """
    n_meas_for_average, n_replicas, _ = spin_configurations.shape

    h_derivative_wrt_v = np.zeros((len(visible_nodes), n_meas_for_average))
    for (v, h), w in w_vh.items():
        h_derivative_wrt_v[v, :] -= np.sum(
            w * spin_configurations[:, :, h], axis=-1) / n_replicas

    return np.float_(np.mean(h_derivative_wrt_v, axis=-1))


def get_average_effective_hamiltonian(
        spin_configurations: np.ndarray, w_hh: Dict, w_vh: Dict,
        visible_nodes: np.ndarray, big_gamma_final: float, beta_final: float) \
        -> float:
    """
    This method calculates the average effective Hamiltonian as given in
    Eq. (9) in paper: https://arxiv.org/pdf/1706.00074.pdf , using samples
    obtained from DWAVE QUBO sample method.
    :param spin_configurations: samples returned by the DWAVE sample() method,
    but converted to numpy array and reshaped to
    (n_meas_for_average, n_replicas, n_hidden_nodes). The samples contain the
    spin states (1 or -1) of all the hidden nodes.
    :param w_hh: dictionary of coupling weights (quadratic coefficients)
    between the hidden nodes of the Chimera graph. The key is (h, h') where h
    and h' are in range [0, .., n_hidden_nodes] and h != h' (no self-coupling
    here).
    :param w_vh: dictionary of coupling weights between the visible nodes
    (states-action vector) and the corresponding hidden nodes they are
    connected to. The key is (v, h), where v is the index of the visible node.
    h is in range [0, .., n_hidden_nodes].
    :param visible_nodes: numpy array of inputs, i.e. visible nodes, given by
    the concatenated states and action vectors.
    :param big_gamma_final: final, i.e. at the end of the SQA, strength of
    the transverse field (virtual, average value), see paper for details.
    :param beta_final: Inverse temperature (note that this parameter is kept
    constant in SQA other than in SA).
    :return: single float; effective Hamiltonian averaged over the individual
    measurements.
    """
    _, n_replicas, _ = spin_configurations.shape

    # FIRST TERM, sum over h, h' in Eq. (9)
    h_sum_1 = 0.
    for (h, h_prime), w in w_hh.items():
        h_sum_1 += w * (spin_configurations[:, :, h] *
                        spin_configurations[:, :, h_prime])

    # SECOND term, sum over v, h in Eq. (9)
    h_sum_2 = 0.
    for (v, h), w in w_vh.items():
        h_sum_2 += w * visible_nodes[v] * spin_configurations[:, :, h]

    # Sum over replicas (sum over k), divide by n_replicas, and negate
    # This corresponds to the first line in Eq. (9)
    # h_sum_12 has shape (n_meas_for_average,)
    h_sum_12 = -np.sum(h_sum_1 + h_sum_2, axis=-1) / n_replicas

    # THIRD term, [-w_plus * (sum_hk_hkplus1 + sum_h1_hr)], in Eq. (9)
    # h_sum_3 has shape (n_meas_for_average,)
    if big_gamma_final == 0:
        # This is to remove the w_plus term in case we use classical SA.
        # TODO: is this correct?
        coth_term = 1.
    else:
        x = big_gamma_final * beta_final / n_replicas
        coth_term = math.cosh(x) / math.sinh(x)
    w_plus = math.log10(coth_term) / (2. * beta_final)

    # I think there is a typo in Eq. (9). The summation index in w+(.. + ..)
    # of the first term should only go from k=1 to r-1.
    hk_hkplus1_sum = np.sum(np.sum(
        spin_configurations[:, :-1, :] * spin_configurations[:, 1:, :],
        axis=1), axis=-1)
    h1_hr_sum = np.sum(
        spin_configurations[:, -1, :] * spin_configurations[:, 0, :],
        axis=-1)
    h_sum_3 = -w_plus * (hk_hkplus1_sum + h1_hr_sum)

    # Take average over n_meas_for_average
    return float(np.mean(h_sum_12 + h_sum_3))


def get_free_energy(spin_configurations: np.ndarray, avg_eff_hamiltonian: float,
                    beta_final: float) -> float:
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
    :param beta_final: Inverse temperature (note that this parameter is kept
    constant in SQA other than in SA).
    :return: free energy of the QBM defined according to the paper.
    """
    # Return the number of occurrences of unique spin configurations along
    # axis 0, i.e. along index of independent measurements
    _, n_occurrences = np.unique(spin_configurations, axis=0,
                                 return_counts=True)
    mean_n_occurrences = n_occurrences / float(np.sum(n_occurrences))
    a_sum = np.sum(mean_n_occurrences * np.log10(mean_n_occurrences))

    return avg_eff_hamiltonian + a_sum / beta_final


class QFunction(object):
    def __init__(self, sampler_type: str, state_space: gym.spaces.Space,
                 action_space: gym.spaces.Discrete, small_gamma: float,
                 n_replicas: int, big_gamma: Union[Tuple[float, float], float],
                 beta: Union[float, Tuple[float, float]], n_annealing_steps: int,
                 n_meas_for_average: int, n_rows_qbm: int, n_columns_qbm: int,
                 kwargs_qpu: Dict, qfunc_it=0) -> None:
        """Implements the Q function (state-action value function)
        represented by a quantum Boltzmann machine (QBM) that is trained
        using (simulated) quantum annealing or QAOA. The Boltzmann machine
        couplings are following the Chimera graph topology of the D-Wave 2000Q
        QPU with 8 qubits per unit cell.
        :param sampler_type: choose between simulated quantum annealing (SQA),
        quantum annealing on hardware (QA), or QAOA.
        :param state_space: gym state space as initialized in the openAI gym
        environment (Box or MultiBinary type).
        :param action_space: gym action space as initialized in the openAI gym
        environment (Discrete type).
        :param small_gamma: discount factor for cumulative future rewards.
        :param n_replicas: number of Trotter slices in 3D extension of the
        Ising spin model to account for non-zero transverse field Gamma at the
        end of the annealing process, see Fig. 1 in paper:
        https://arxiv.org/pdf/1706.00074.pdf
        :param big_gamma: Transverse field; first entry is initial gamma and
        second entry is final gamma at end of annealing process.
        :param beta: inverse annealing temperature (constant during quantum
        annealing).
        :param n_meas_for_average: number of times we run an independent
        sampling process from start to end
        :param n_annealing_steps: number of steps that one annealing
        process should take (~annealing time).
        :param n_rows_qbm: number of unit cells along x-axis of the Chimera graph.
        :param n_columns_qbm: number of unit cells along y-axis of the Chimera
        graph.
        :param kwargs_qpu: additional keyword arguments required for the
        initialization of the D-Wave QPU through Amazon Braket."""
        self.n_nodes_per_unit_cell = 8
        self.n_rows = n_rows_qbm
        self.n_columns = n_columns_qbm
        self.n_unit_cells = self.n_rows * self.n_columns
        n_graph_nodes = self.n_unit_cells * self.n_nodes_per_unit_cell

        if sampler_type == 'SQA':
            self.sampler = SQA(
                big_gamma=big_gamma, beta=beta, n_replicas=n_replicas,
                n_nodes=n_graph_nodes)
        elif sampler_type == 'QPU':
            print('SETTING PROPER QPU AS SAMPLER')
            self.sampler = QPU(
                big_gamma=big_gamma, beta=beta, n_replicas=n_replicas, n_nodes=n_graph_nodes,
                qfunc_it=qfunc_it,
                dwave_token=kwargs_qpu["token"], dwave_solver=kwargs_qpu["solver"])
        elif sampler_type == 'QAOA':
            self.sampler = QAOA(n_nodes=n_graph_nodes, simulator='qasm',
                                n_shots=10, beta_final=beta)
        else:
            raise ValueError("Sampler type must be either 'SQA', 'QA', or 'QAOA'.")

        self.sampler_type = sampler_type

        self.n_annealing_steps = n_annealing_steps
        self.n_meas_for_average = n_meas_for_average
        self.n_replicas = n_replicas

        self.small_gamma = small_gamma

        # For normalization purposes
        self.state_space = state_space
        self.action_space = action_space

        self.w_hh, self.w_vh = self._initialise_weights()
        self.m, self.v = self._initialise_adam_params()

        # To keep track of how weights evolve
        self.w_hh_history, self.w_vh_history = {}, {}
        for k in self.w_hh.keys():
            self.w_hh_history[k] = []
            self.w_hh_history[k].append(self.w_hh[k])
        for k in self.w_vh.keys():
            self.w_vh_history[k] = []
            self.w_vh_history[k].append(self.w_vh[k])

    def _initialise_weights(self, scale: float = 5.) -> Tuple[Dict, Dict]:
        """Initialises coupling weights of the Chimera graph, i.e. both
        hidden-hidden couplings and visible-hidden couplings.
        :param scale: scaling of initial weights.
        :returns tuple of dictionaries with hidden-hidden and visible-hidden
        couplings of the QBM in Ising convention."""
        # ==============================
        # COUPLINGS BETWEEN HIDDEN NODES
        # This loop initializes hidden-hidden coupling weights to fully connect
        # nodes in a number of unit cells, e.g. of the Chimera graph (see Fig. 2
        # in paper: https://arxiv.org/pdf/1706.00074.pdf). The number of
        # unit cells can be arbitrary. The indexing of the nodes is starting
        # at the top left (node 0) and goes down vertically (blue nodes), and
        # then to the right (first red node is index 4). Note that inter-cell
        # couplings will be added in a separate loop below.
        w_hh = dict()
        for unit_idx in range(self.n_unit_cells):
            first_node = self.n_nodes_per_unit_cell * unit_idx
            for i in range(first_node, first_node + 4):
                for j in range(first_node + 4, first_node + 8):
                    w_hh[(i, j)] = (2 * random.random() - 1) * scale

        # This loop creates inter-cell couplings based on the chosen
        # architecture (i.e. n_rows vs. n_columns). Unit cells are horizontally
        # connected as 4 -> 12 -> 20 -> etc., 5 -> 13 -> 21 -> etc.,
        # while they are vertically connected as (if n_columns = 2) 0 -> 16
        # -> 32 -> etc., 1 -> 17 -> 33 -> etc.
        # 1) Horizontal inter-cell couplings
        for col in range(self.n_columns - 1):
            for row in range(self.n_rows):
                first_node = (col * self.n_nodes_per_unit_cell +
                              row * self.n_columns * self.n_nodes_per_unit_cell)
                for i, j in zip(tuple(range(first_node + 4, first_node + 8)),
                                tuple(range(first_node + 12, first_node + 16))):
                    w_hh[(i, j)] = (2 * random.random() - 1) * scale

        # 2) Vertical inter-cell couplings
        for row in range(self.n_rows - 1):
            for col in range(self.n_columns):
                first_node = (col * self.n_nodes_per_unit_cell +
                              row * self.n_columns * self.n_nodes_per_unit_cell)
                for i, j in zip(tuple(range(first_node, first_node + 4)),
                                tuple(range(
                                    first_node + 8 * self.n_columns,
                                    first_node + 8 * self.n_columns + 4))):
                    w_hh[(i, j)] = (2 * random.random() - 1) * scale

        # ==============================
        # COUPLINGS BETWEEN VISIBLE [the 'input' (= state layer) and the
        # 'output' (= action layer) of a 'classical' Q-net] AND THE HIDDEN NODES
        w_vh = dict()

        # TODO: needs to be cleaned up: we assume that if the state space is
        #  discrete, binary that also the action space is binary (note that for
        #   Q-learning it is always Discrete anyway).

        # TODO: THIS LOOKS WRONG! IS A DISCRETE ACTION_SPACE NOT ALWAYS REPRESENTED BY JUST 1 NODE?
        #  OR IF IT'S A TUPLE OF DISCRETE SPACES (e.g. with several discrete controls elements), IT WOULD BE
        #  ONE NODE PER DIMENSION.
        if isinstance(self.state_space, gym.spaces.Box):
            n_state_nodes = len(self.state_space.high)
            # n_action_nodes = self.action_space.n
            n_action_nodes = len(self.action_space.shape) + 1
        elif isinstance(self.state_space, gym.spaces.MultiBinary):
            n_state_nodes = self.state_space.n
            n_action_nodes = math.ceil(math.log2(self.action_space.n))
            if self.action_space.n < 2:
                n_action_nodes = 1
        else:
            raise TypeError("State space must be of Box or of MultiBinary type.")

        # Dense connection between the state nodes (visible) and the BLUE
        # hidden nodes (all 8 of them) of the Chimera graph. Blue nodes have
        # indices [0, 1, 2, 3, 12, 13, 14, 15, 16, 17, 18, 19, ...]. We hence
        # have connections between the state nodes [0, 1, ...,
        # len(state_space.high)] to all of the blue nodes.
        blue_nodes = tuple()
        for i in range(self.n_unit_cells):
            pick_col = i % 2
            first_node = i * self.n_nodes_per_unit_cell + pick_col * 4
            blue_nodes += tuple(range(first_node, first_node + 4))

        for j in blue_nodes:
            for i in range(n_state_nodes):
                w_vh[(i, j)] = (2 * random.random() - 1) * scale

        # Dense connection between the action nodes (visible) and the RED hidden
        # nodes of the Chimera graph. Red nodes have indices [4, 5, 6, 7, 8,
        # 9, 10, 11, 20, 21, 22, 23, ...]. We hence have connections between the
        # action nodes [len(state_space.high), ..,
        # len(state_space.high) + len(action_space.high)] to all of the red
        # nodes.
        red_nodes = tuple()
        for i in range(self.n_unit_cells):
            pick_col = (i + 1) % 2
            first_node = i * self.n_nodes_per_unit_cell + pick_col * 4
            red_nodes += tuple(range(first_node, first_node + 4))
        for j in red_nodes:
            for i in range(n_state_nodes, n_state_nodes + n_action_nodes):
                w_vh[(i, j)] = (2 * random.random() - 1) * scale

        return w_hh, w_vh

    def calculate_q_value(self, state: np.ndarray, action: int) -> \
            Tuple[float, np.ndarray, np.ndarray]:
        """Based on state and chosen action, calculates free energy,
        spin_configurations and visible nodes array.
        :param state: state the environment is in.
        :param action: chosen action.
        :returns free energy, spin configurations, and visible nodes."""
        # Define visible nodes array (state-action input vector) and
        # and convert into QUBO problem in preparation for sampler.
        visible_nodes = get_visible_nodes_array(
            state=state, action=action,
            state_space=self.state_space,
            action_space=self.action_space)
        qubo_dict = create_general_qubo_dict(
            self.w_hh, self.w_vh, visible_nodes)

        # Run sampling process
        spin_configurations = self.sampler.sample(
            qubo_dict=qubo_dict,
            n_meas_for_average=self.n_meas_for_average,
            n_steps=self.n_annealing_steps)

        # Based on sampled spin configurations calculate free energy
        avg_eff_hamiltonian = get_average_effective_hamiltonian(
            spin_configurations, self.w_hh, self.w_vh, visible_nodes,
            self.sampler.big_gamma_final, self.sampler.beta_final)

        free_energy = get_free_energy(
            spin_configurations, avg_eff_hamiltonian,
            self.sampler.beta_final)
        q_value = -free_energy

        return q_value, spin_configurations, visible_nodes

    def update_weights(
            self, spin_configurations: np.ndarray, visible_nodes: np.ndarray,
            current_q: float, future_q: float, reward: float,
            learning_rate: float, grad_clip: float = 100000.,
            adam_params: Dict = None, step: int = None) -> None:
        """Calculates TD(0) learning step, i.e. the updates of the coupling
        dictionaries w_hh, w_vh according to Eqs. (11) and (12) in paper:
        https://arxiv.org/pdf/1706.00074.pdf
        :param step: current iteration.
        :param adam_params: dictionary with parameters for Adam optimizer.
        :param spin_configurations: spin configurations returned by the sampler.
        array of shape (n_meas_for_average, n_replicas, n_hidden_nodes).
        The spin configurations contain the spin states (1 or -1) of all
        the hidden nodes (that includes replicas, i.e. Trotter slices).
        :param visible_nodes: array of visible nodes.
        :param current_q: Q function value at time step n, Q(s_n, a_n)
        :param future_q: Q function value at time step n+1, Q(s_n+1, a_n+1)
        :param reward: RL reward of current step, r_n(s_n, a_n)
        :param learning_rate: RL. parameter, learning rate for update of
        coupling weights of the Chimera graph.
        :param grad_clip: gradient clipping value (absolute value).
        :returns None
        """
        if not adam_params:
            adam_params = {
                "epsilon": 1E-8,
                "gamma_1": 0.9,
                "gamma_2": 0.999
            }

        # This term is the same for both weight updates w_hh and w_vh
        update_factor = reward + self.small_gamma * future_q - current_q

        # Update of w_vh, Eq. (11)
        h_avg = np.mean(np.mean(spin_configurations, axis=0), axis=0)
        for v, h in self.w_vh.keys():
            grads = update_factor * visible_nodes[v] * h_avg[h]

            m_t = self.m["vh"][(v, h)]
            v_t = self.v["vh"][(v, h)]
            delta, self.m["vh"][(v, h)], self.v["vh"][(v, h)] = self._calc_adam_delta(
                adam_params, m_t, v_t, grads, step)

            delta = np.clip(delta, -delta, grad_clip)
            self.w_vh[(v, h)] += learning_rate * delta

        # Update of w_hh, Eq. (12)
        for h, h_prime in self.w_hh.keys():
            grads = update_factor * np.mean(
                spin_configurations[:, :, h] *
                spin_configurations[:, :, h_prime])

            m_t = self.m["hh"][(h, h_prime)]
            v_t = self.v["hh"][(h, h_prime)]
            delta, self.m["hh"][(h, h_prime)], self.v["hh"][(h, h_prime)] = self._calc_adam_delta(
                adam_params, m_t, v_t, grads, step)

            delta = np.clip(delta, -grad_clip, grad_clip)
            self.w_hh[(h, h_prime)] += learning_rate * delta

        # Keep track of the weights
        for k in self.w_hh.keys():
            self.w_hh_history[k].append(self.w_hh[k])
        for k in self.w_vh.keys():
            self.w_vh_history[k].append(self.w_vh[k])

    def _initialise_adam_params(self):
        """Initialise first and second moment parameters for adam optimizer."""
        v = {"vh": {}, "hh": {}}
        m = {"vh": {}, "hh": {}}

        for v_, h_ in self.w_vh.keys():
            m["vh"][(v_, h_)] = 0.
            v["vh"][(v_, h_)] = 0.

        for h, h_prime in self.w_hh.keys():
            m["hh"][(h, h_prime)] = 0.
            v["hh"][(h, h_prime)] = 0.

        return m, v

    def _calc_adam_delta(self, adam_params, m_t, v_t, grads, step):
        """Calculate the update using Adam optimization."""
        gamma_1 = adam_params["gamma_1"]
        gamma_2 = adam_params["gamma_2"]
        epsilon = adam_params["epsilon"]

        m_t = gamma_1 * m_t + (1. - gamma_1) * grads
        v_t = gamma_2 * v_t + (1. - gamma_2) * grads ** 2

        m_hat = m_t / (1. - gamma_2 ** (step+1))
        v_hat = v_t / (1. - gamma_2 ** (step+1))

        delta = m_hat / (np.sqrt(v_hat) + epsilon)

        return delta, m_t, v_t
