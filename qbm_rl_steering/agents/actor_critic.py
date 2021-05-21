import math
import random
from typing import Tuple, Union, Dict

import numpy as np
import matplotlib.pyplot as plt
import sys

import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.optimizers as KO
import tensorflow.keras as K
from tensorflow.python.framework.ops import disable_eager_execution

from qbm_rl_steering.utils.helpers import plot_log
from qbm_rl_steering.environment.env_desc import TargetSteeringEnv

# What we need for QBM
from qbm_rl_steering.utils.sqa_annealer import SQA
import gym

try:
    import matplotlib
    matplotlib.use('qt5agg')
except ImportError as err:
    print(err)

disable_eager_execution()


def get_visible_nodes_array(state: np.ndarray, action: np.ndarray,
                            state_space: gym.spaces.Box,
                            action_space: gym.spaces.Box) -> np.ndarray:
    """
    Take state (e.g. directly from environment), and action index (following
    env.action_map), and concatenate them to the visible_nodes_tuple.
    :param state: state as np.array from the environment (either through
    .reset(), or .step()).
    :param action: index of action as used in environment, see env.action_map
    keys.
    :param state_space: openAI gym state space
    :param action_space: openAI gym action space
    :return normalized state-action vector == visible nodes of QBM
    """
    # TODO: fix documentation
    # print('action in visible nodes', action)
    state_normalized = 2. * state / (state_space.high - state_space.low)
    action_normalized = 2. * action / (action_space.high - action_space.low)
    # print('action_normalized in visible nodes', action_normalized)
    visible_nodes = np.array(list(state_normalized) + list(action_normalized))
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

    return qubo_dict


def get_average_effective_hamiltonian(
        spin_configurations: np.ndarray, w_hh: Dict, w_vh: Dict,
        visible_nodes: np.ndarray, big_gamma_final: float, beta_final: float)\
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
        # This is to remove the w_plus term
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

    # TODO: drop the a_sum entropy term for QAOA?
    return avg_eff_hamiltonian + a_sum / beta_final


class QFunction(object):
    def __init__(self, sampler_type: str, state_space: gym.spaces.Box,
                 action_space: gym.spaces.Box, small_gamma: float,
                 n_graph_nodes: int, n_replicas: int,
                 big_gamma: Union[Tuple[float, float], float],
                 beta: Union[float, Tuple[float, float]],
                 n_annealing_steps: int, n_meas_for_average: int,
                 kwargs_qpu) -> None:
        """
        Implementation of the Q function (state-action value function) using
        an SQA method to update / train.
        :param sampler_type: choose between simulated quantum annealing (SQA),
        classical annealing (SA), or Quantum annealing on hardware (QPU) (use
        big_gamma = 0 with SA)
        :param state_space: gym state space as initialized in the openAI gym
        environment (Box type).
        :param action_space: gym action space as initialized in the openAI gym
        environment (Discrete type).
        :param small_gamma: RL parameter, discount factor for
        cumulative future rewards.
        :param n_graph_nodes: number of nodes of the graph structure. E.g. for
        2 unit cells of the DWAVE-2000 chip, it's 16 nodes (8 per unit).
        :param n_replicas: number of replicas (aka. Trotter slices) in the 3D
        extension of the Ising model, see Fig. 1 in paper:
        https://arxiv.org/pdf/1706.00074.pdf
        :param big_gamma: Transverse field; first entry is initial gamma and
        second entry is final gamma at end of annealing process (when using
        SQA). When sampler_type is SA, set big_gamma = 0.
        :param beta: Inverse temperature, either a float (for SQA, or QAOA),
        or a tuple of floats (for SA). For SQA, the temperature is kept
        constant.
        :param n_meas_for_average: number of times we run an independent
        sampling process from start to end
        :param n_annealing_steps: number of steps that one annealing
        process should take (~annealing time).
        :param kwargs_qpu: additional keyword arguments required for the
        initialization of the DWAVE QPU on Amazon Braket.
        """
        # TODO: adapt documentation

        if sampler_type == 'SQA':
            self.sampler = SQA(
                big_gamma=big_gamma, beta=beta, n_replicas=n_replicas,
                n_nodes=n_graph_nodes)
        else:
            raise ValueError("sampler_type must be 'SQA'.")

        self.sampler_type = sampler_type

        self.n_annealing_steps = n_annealing_steps
        self.n_meas_for_average = n_meas_for_average
        self.n_replicas = n_replicas

        self.small_gamma = small_gamma

        # For normalization purposes
        self.state_space = state_space
        self.action_space = action_space

        self.w_hh, self.w_vh = self._initialise_weights()

        # Keep track of the weights
        self.w_hh_history, self.w_vh_history = {}, {}
        for k in self.w_hh.keys():
            self.w_hh_history[k] = []
            self.w_hh_history[k].append(self.w_hh[k])
        for k in self.w_vh.keys():
            self.w_vh_history[k] = []
            self.w_vh_history[k].append(self.w_vh[k])

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
        # 'output' (= action layer) of a 'classical' Q-net] AND HIDDEN NODES
        w_vh = dict()

        # Dense connection between the state node(s) (visible) and the BLUE
        # hidden nodes (all 8 of them) of the Chimera graph. Blue nodes have
        # indices [0, 1, 2, 3, 12, 13, 14, 15]. We hence have connections
        # between the state nodes [0, ..., len(state_space.high)] to all of the
        # blue nodes (if state is just a float, we only have 1 node). This is
        # 8 couplings here since we only work with 1 float describing the state.
        for j in (tuple(range(4)) + tuple(range(12, 16))):
            for i in range(len(self.state_space.high)):
                w_vh[(i, j)] = 2 * random.random() - 1

        # Dense connection between the action nodes (visible) and the RED hidden
        # nodes (all 8 of them) of the Chimera graph. Red nodes have indices
        # [4, 5, 6, 7, 8, 9, 10, 11]. We hence have connections between the
        # action node [len(state_space), len(state_space) + 1] to all of the
        # red nodes. This is 1 * 8 = 8 couplings here.
        for j in (tuple(range(4, 8)) + tuple(range(8, 12))):
            for i in range(len(self.state_space.high),
                           len(self.state_space.high) + 1):  # + 1 for action
                w_vh[(i, j)] = 2 * random.random() - 1

        # We get a total of 64 + 16 = 80 couplings (here) defined by w_vh.
        return w_hh, w_vh

    def calculate_q_value_on_batch(self, states, actions):
        q_values = []
        spin_configurations = []
        visible_nodes = []
        for i in range(len(states)):
            q, sc, vn = self.calculate_q_value(states[i], actions[i])
            q_values.append(q)
            spin_configurations.append(sc)
            visible_nodes.append(vn)
        q_values = np.array(q_values)
        spin_configurations = np.array(spin_configurations)
        visible_nodes = np.array(visible_nodes)

        return q_values, spin_configurations, visible_nodes

    def calculate_q_value(self, state: np.ndarray, action: np.ndarray) -> \
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
            state_space=self.state_space, action_space=self.action_space)
        # print('visible_nodes', visible_nodes)
        qubo_dict = create_general_qubo_dict(
            self.w_hh, self.w_vh, visible_nodes)

        # Run the sampling process (will be either annealing: SA, SQA,
        # or QPU, or QAOA)
        spin_configurations = self.sampler.sample(
            qubo_dict=qubo_dict,
            n_meas_for_average=self.n_meas_for_average,
            n_steps=self.n_annealing_steps)

        # Based on sampled spin configurations calculate free energy
        avg_eff_hamiltonian = get_average_effective_hamiltonian(
            spin_configurations, self.w_hh, self.w_vh, visible_nodes,
            self.sampler.big_gamma_final, self.sampler.beta_final)

        free_energy = get_free_energy(
            spin_configurations, avg_eff_hamiltonian, self.sampler.beta_final)
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
        :param visible_nodes: numpy array of visible nodes, given by the states
        and action vectors concatenated.
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
        # print('reward', reward)
        # print('future_q', future_q)
        # print('current_q', current_q)

        # print('update_factor', update_factor)
        # print('visible_nodes', visible_nodes)

        # Update of w_vh, Eq. (11)
        h_avg = np.mean(np.mean(spin_configurations, axis=0), axis=0)
        # print('h_avg', h_avg)
        for v, h in self.w_vh.keys():
            self.w_vh[(v, h)] += update_factor * visible_nodes[v] * h_avg[h]

        # Update of w_hh, Eq. (12)
        for h, h_prime in self.w_hh.keys():
            self.w_hh[(h, h_prime)] += update_factor * np.mean(
                spin_configurations[:, :, h] *
                spin_configurations[:, :, h_prime])

        # Keep track of the weights
        for k in self.w_hh.keys():
            self.w_hh_history[k].append(self.w_hh[k])
        for k in self.w_vh.keys():
            self.w_vh_history[k].append(self.w_vh[k])


class Memory:
    """A FIFO experiene replay buffer.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.states = np.zeros([size, obs_dim], dtype=np.float32)
        self.actions = np.zeros([size, act_dim], dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.next_states = np.zeros([size, obs_dim], dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def get_sample(self, batch_size=32):
        if self.size < batch_size:
            idxs = np.random.randint(0, self.size, size=self.size)
        else:
            idxs = np.random.randint(0, self.size, size=batch_size)
        return self.states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs], self.dones[idxs]

class ClassicACAgent(object):
    def __init__(self, GAMMA, env):
        # env intel
        self.env = env
        self.action_n = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape
        self.state_n = self.state_dim[0]
        # constants
        self.ACT_LIMIT = max(env.action_space.high)  # required for clipping prediciton aciton
        self.GAMMA = GAMMA  # discounted reward factor
        self.TAU = 0.1  # soft update factor
        self.BUFFER_SIZE = int(1e6)
        self.BATCH_SIZE = 10  # training batch size.
        self.ACT_NOISE_SCALE = 0.2

        # QBM related stuff
        self.n_annealing_steps = 100
        self.n_meas_for_average = 50
        self.learning_rate = 1e-3

        # create networks
        self.dummy_Q_target_prediction_input = np.zeros((self.BATCH_SIZE, 1))
        self.dummy_dones_input = np.zeros((self.BATCH_SIZE, 1))

        self.critic = self._gen_critic_network()
        self.critic_target = self._gen_critic_network()
        self.actor = self._gen_actor_network()  # the local actor wich is trained on.
        self.actor_target = self._gen_actor_network()  # the target actor which is slowly updated toward optimum

        self.memory = Memory(self.state_n, self.action_n, self.BUFFER_SIZE)

    def _gen_actor_network(self):
        state_input = KL.Input(shape=self.state_dim)
        dense = KL.Dense(128, activation='relu')(state_input)
        dense = KL.Dense(128, activation='relu')(dense)
        out = KL.Dense(self.action_n, activation='tanh')(dense)
        model = K.Model(inputs=state_input, outputs=out)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.001,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True)
        model.compile(optimizer=K.optimizers.Adam(learning_rate=0.001),
                      loss=self._ddpg_actor_loss)
        model.summary()
        return model

    def get_action(self, states, noise=None, episode=1):
        if noise is None: noise = self.ACT_NOISE_SCALE
        if len(states.shape) == 1: states = states.reshape(1, -1)
        action = self.actor.predict_on_batch(states)
        if noise != 0:
            action += noise/episode * np.random.randn(self.action_n)
            action = np.clip(action, -self.ACT_LIMIT, self.ACT_LIMIT)
        return action

    def get_target_action(self, states):
        return self.actor_target.predict_on_batch(states)

    def train_actor(self, states, actions):
        self.actor.train_on_batch(states, states) # Q_predictions)

    def _gen_critic_network(self):
        # Define Q functions and their updates
        kwargs_q_func = dict(
            sampler_type='SQA',
            state_space=self.env.observation_space,
            action_space=self.env.action_space,
            small_gamma=self.GAMMA,
            n_graph_nodes=16,
            n_replicas=1,
            big_gamma=(20., 0.), beta=2.,
            n_annealing_steps=self.n_annealing_steps,
            n_meas_for_average=self.n_meas_for_average,
            kwargs_qpu={})

        q_function = QFunction(**kwargs_q_func)
        return q_function

    def _ddpg_actor_loss(self, y_true, y_pred):
        # y_pred is the action from the actor net. y_true is the state, we maximise the q
        q = self.q_custom_gradient(y_true, y_pred)
        return -K.backend.mean(q)

    @tf.custom_gradient
    def q_custom_gradient(self, y_true, y_pred):
        def get_q_value(y_true, y_pred):
            q_value, _, _ = (
                self.critic.calculate_q_value_on_batch(y_true, y_pred))
            dq_over_dstate = self.get_state_derivative(y_true, y_pred)
            dq_over_daction = self.get_action_derivative(y_true, y_pred)

            return np.float32(q_value), np.float32(dq_over_dstate),\
                   np.float32(dq_over_daction)
            # first is function, second is gradient

        z, dz_over_dstate, dz_over_daction = tf.numpy_function(
            get_q_value, [y_true, y_pred], [tf.float32, tf.float32, tf.float32])

        def grad(dy):
            return (tf.dtypes.cast(dy * dz_over_dstate, dtype=tf.float32),
                    tf.dtypes.cast(dy * dz_over_daction, dtype=tf.float32))
        return z, grad

    def get_state_derivative(self, y_true, y_pred, epsilon=0.04):
        # q0, _, _ = self.critic.calculate_q_value(y_true, y_pred)
        qeps_plus, _, _ = self.critic.calculate_q_value_on_batch(
            y_true + epsilon, y_pred)
        qeps_minus, _, _ = self.critic.calculate_q_value_on_batch(
            y_true - epsilon, y_pred)
        return np.float_((qeps_plus - qeps_minus) / (2*epsilon))

    def get_action_derivative(self, y_true, y_pred, epsilon=0.04):
        # q0, _, _ = self.critic.calculate_q_value(y_true, y_pred)
        qeps_plus, _, _ = self.critic.calculate_q_value_on_batch(
            y_true, y_pred + epsilon)
        qeps_minus, _, _ = self.critic.calculate_q_value_on_batch(
            y_true, y_pred - epsilon)
        return np.float_((qeps_plus - qeps_minus) / (2*epsilon))

    def train_critic(self, states, next_states, actions, rewards, dones):
        # Training the QBM
        # Use experiences in replay_buffer to update weights
        # n_replay_batch = self.replay_batch_size
        # if len(self.replay_buffer) < self.replay_batch_size:
        #     n_replay_batch = len(self.replay_buffer)
        # replay_samples = random.sample(self.replay_buffer, n_replay_batch)

        for jj in np.arange(len(states)):
            # Act only greedily here: should be OK to do that always
            # because we collect our experiences according to an
            # epsilon-greedy policy

            # Recalculate the q_value of the (sample.state, sample.action)
            # pair
            q_value, spin_configs, visible_nodes = (
                self.critic_target.calculate_q_value(states[jj], actions[jj]))

            # Now calculate the next_q_value of the greedy action, without
            # actually taking the action (to take actions in env.,
            # we don't follow purely greedy action).
            next_action = self.get_target_action(next_states[jj])
            # print('next action', next_action)
            next_q_value, spin_configurations, visible_nodes = (
                self.critic_target.calculate_q_value(
                    state=next_states[jj], action=next_action))

            # Update weights and update target Q-function if needed
            # TODO: change learning rate to fixed value...
            self.critic.update_weights(
                spin_configs, visible_nodes, q_value, next_q_value,
                rewards[jj], learning_rate=5e-3)

    def _soft_update_actor_and_critic(self):
        # Critic soft update:
        # TODO: check here if something doesn't work ...
        for k in self.critic.w_hh.keys():
            self.critic_target.w_hh[k] = (
                    self.TAU * self.critic.w_hh[k] +
                    (1.0 - self.TAU) * self.critic_target.w_hh[k])
        for k in self.critic.w_vh.keys():
            self.critic_target.w_vh[k] = (
                    self.TAU * self.critic.w_vh[k] +
                    (1.0 - self.TAU) * self.critic_target.w_vh[k])

        # Actor soft update
        weights_actor_local = np.array(self.actor.get_weights())
        weights_actor_target = np.array(self.actor_target.get_weights())
        self.actor_target.set_weights(
            self.TAU * weights_actor_local +
            (1.0 - self.TAU) * weights_actor_target)

    def store(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

    def train(self):
        """Trains the networks of the agent (local actor and critic) and soft-updates  target.
        """
        states, actions, rewards, next_states, dones = self.memory.get_sample(
            batch_size=self.BATCH_SIZE)
        self.train_critic(states, next_states, actions, rewards, dones)
        # print('states', states)
        # print('actions', actions)
        # print('memory buffer', states.shape)
        # print('states in memory', states)
        self.train_actor(states, actions)
        self._soft_update_actor_and_critic()


if __name__ == "__main__":

    GAMMA = 0.85
    EPOCHS = 16
    MAX_EPISODE_LENGTH = 10
    START_STEPS = 10
    INITIAL_REW = 0

    env = TargetSteeringEnv(max_steps_per_episode=MAX_EPISODE_LENGTH)
    agent = ClassicACAgent(GAMMA, env)

    # s = np.linspace(-1, 1, 15)
    # a = np.linspace(-1, 1, 13)
    # q = np.zeros((len(s), len(a)))
    # dqda = np.zeros((len(s), len(a)))
    # dqds = np.zeros((len(s), len(a)))
    # for i, s_ in enumerate(s):
    #     for j, a_ in enumerate(a):
    #         q[i, j], _, _ = agent.critic.calculate_q_value(s_, a_)
    #         dqda[i, j] = agent.get_action_derivative(s_, a_, epsilon=0.4)
    #         dqds[i, j] = agent.get_state_derivative(s_, a_, epsilon=0.4)
    #
    # fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(8, 10))
    # imq = axs[0].pcolormesh(s, a, q.T, shading='auto')
    # fig.colorbar(imq, ax=axs[0])
    # axs[0].set_title('Q')
    # axs[0].set_ylabel('action')
    #
    # imdqda = axs[1].pcolormesh(s, a, dqda.T, shading='auto')
    # fig.colorbar(imdqda, ax=axs[1])
    # axs[1].set_title('dq / da')
    # axs[1].set_ylabel('action')
    #
    # imdqds = axs[2].pcolormesh(s, a, dqds.T, shading='auto')
    # fig.colorbar(imdqds, ax=axs[2])
    # axs[2].set_title('dq / ds')
    # axs[2].set_xlabel('state')
    # axs[2].set_ylabel('action')
    # plt.show()
    #
    # plt.figure()
    # plt.suptitle('q vs dqda')
    # plt.plot(a, q[6, :], label='Q')
    # plt.plot(a, dqda[6, :], label='dQ/da')
    # plt.legend()
    # plt.xlabel('action')
    # plt.ylabel('Q and dq/da resp.')
    # plt.show()
    #
    # plt.figure()
    # plt.suptitle('q vs dqds')
    # plt.plot(s, q[:, 5], label='Q')
    # plt.plot(s, dqds[:, 5], label='dQ/ds')
    # plt.legend()
    # plt.xlabel('state')
    # plt.ylabel('Q and dq/ds resp.')
    # plt.show()

    state, reward, done, ep_rew, ep_len, ep_cnt = env.reset(), INITIAL_REW, \
                                                  False, [[]], 0, 0

    # Calculate reward in current state
    _, intensity = env.get_pos_at_bpm_target(env.mssb_angle)
    ep_rew[-1].append(env.get_reward(intensity))
    total_steps = MAX_EPISODE_LENGTH * EPOCHS

    # Main loop: collect experience in env and update/log each epoch
    to_exploitation = False
    for t in range(total_steps):

        # print('actor weights:', agent.actor.get_weights())
        # print('n nans actor weights:', np.sum(np.isnan(np.array(
        #       agent.actor.get_weights()).flatten())))

        if t > START_STEPS:
            # print('\n\n\n!!!!!!!! END OF RANDOM SAMPLING !!!!!!!!\n\n\n')
            if not to_exploitation:
                print('Now exploiting ...')
                to_exploitation = True
            action = agent.get_action(state, episode=1)
            action = np.squeeze(action)
        else:
            # print('\n!!!!!!!! USING RANDOM SAMPLING !!!!!!!!\n')
            action = env.action_space.sample()

        # Step the env
        # print('action before env.step', action)
        next_state, reward, done, _ = env.step(action)
        #print("reward ",reward,done)
        ep_rew[-1].append(reward) #keep adding to the last element till done
        ep_len += 1

        #print(done)
        done = False if ep_len==MAX_EPISODE_LENGTH else done

        # Store experience to replay buffer
        agent.store(state, action, reward, next_state, done)

        state = next_state

        if done or (ep_len == MAX_EPISODE_LENGTH):
            ep_cnt += 1
            if True: #ep_cnt % RENDER_EVERY == 0:
                print(f"Episode: {len(ep_rew)-1}, Reward: {ep_rew[-1][-1]}, "
                      f"Length: {len(ep_rew[-1])}")
            ep_rew.append([])

            for _ in range(ep_len):
                agent.train()

            state, reward, done, ep_ret, ep_len = (
                env.reset(), INITIAL_REW, False, 0, 0)

            _, intensity = env.get_pos_at_bpm_target(env.mssb_angle)
            ep_rew[-1].append(env.get_reward(intensity))

    init_rewards = []
    rewards = []
    reward_lengths = []
    for episode in ep_rew[:-1]:
        if(len(episode) > 0):
            rewards.append(episode[-1])
            init_rewards.append(episode[0])
            reward_lengths.append(len(episode)-1)
    print('Total number of interactions:', np.sum(reward_lengths))

    plot_log(env, fig_title='Training')
    # fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(8, 6))
    # axs[0].plot(reward_lengths)
    # axs[0].axhline(env.max_steps_per_episode, c='k', ls='-',
    #                label='Max. # steps')
    # axs[0].set_ylabel('# steps per episode')
    # axs[0].set_ylim(0, env.max_steps_per_episode + 0.5)
    # axs[0].legend(loc='upper right')
    #
    # axs[1].plot(init_rewards, c='r', marker='.', label='initial')
    # axs[1].plot(rewards, c='forestgreen', marker='x', label='final')
    # axs[1].legend(loc='lower right')
    # axs[1].set_xlabel('Episode')
    # axs[1].set_ylabel('Reward')
    # plt.show()


    # Agent evaluation
    n_episodes_eval = 50
    episode_counter = 0

    env = TargetSteeringEnv(max_steps_per_episode=MAX_EPISODE_LENGTH)
    while episode_counter < n_episodes_eval:
        state = env.reset(init_outside_threshold=True)
        while True:
            a = agent.get_action(state, noise=0)
            state, reward, done, _ = env.step(a)
            if done:
                episode_counter += 1
                break

    plot_log(env, fig_title='Evaluation')
