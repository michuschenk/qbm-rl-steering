import random
from qbm_rl_steering.environment.env_desc import TargetSteeringEnv
import qbm_rl_steering.agents.qbmq_utils as utl
import qbm_rl_steering.environment.helpers as hlp

from neal import SimulatedAnnealingSampler
import numpy as np
import math
import tqdm

from typing import Union


def make_action_binary(action_index: int) -> tuple:
    """ Similar to make_state_discrete_binary. Convert action_index to a
    binary vector using 0s and 1s. Conversion of 0s to -1s will be done in a
    separate function.
    :param action_index: index of action (integer). See which index
    corresponds to which action in env.action_map.
    :return binary vector that encodes the action_index """
    n_bits_action_space = 2  # can describe 2**n_bits_action_space actions
    binary_fmt = f'0{n_bits_action_space}b'
    action_binary = tuple([int(i) for i in format(action_index, binary_fmt)])
    return action_binary


def convert_bits(vec: Union[tuple, np.ndarray], mapping: dict = {0: -1}) \
        -> tuple:
    """ Remap values in vec (bits) according to the mapping dict. Default
    mapping means that 0s are transformed into -1s.
    :param vec input np.ndarray or tuple that needs to be converted.
    :param mapping dictionary that contains the remapping of the values
    :return remapped vector as tuple. """
    vec = np.array(vec)
    for inp, out in mapping.items():
        vec[vec == inp] = out
    return tuple(vec)


def create_visible_iterable(state: np.ndarray, action: int) -> tuple:
    """ Take state (e.g. directly from environment), and action_index (
    following env.action_map), and concatenate them to the visible_iterable
    tuple required for the create_general_Q_from(..) function. This also
    converts all 0s appearing in state and action_index (once made binary) to
    -1s.
    :param state: state in binary-encoded vector, as obtained directly from
    the environment (either through .reset(), or .step())
    :param action: index of action as used in environment
    :return tuple, concatenation of state and action, following the
    convention of Mircea's code on QBM. """
    s = convert_bits(state)
    a = convert_bits(make_action_binary(action))
    return s + a


class QFunction(object):
    def __init__(self, n_bits_observation_space: int,
                 n_bits_action_space: int, possible_actions: list,
                 replica_count: int, average_size: int, big_gamma: float,
                 beta: float) -> None:
        """ Implementation of the Q function using DWAVE neal sampler.
        :param n_bits_observation_space: number of bits used to encode
        observation space of environment
        :param n_bits_action_space: number of bits required to encode the
        actions that are possible in the given environment
        :param possible_actions: list of possible action indices
        :param replica_count: ?
        :param average_size: ?
        :param big_gamma: QBM param., see paper
        :param beta: QBM param., see paper
        """

        # TODO: rename Q_hh, Q_vh into w_hh, w_vh
        self.Q_hh = dict()
        self.Q_vh = dict()
        self.replica_count = replica_count
        self.average_size = average_size
        self.sample_count = replica_count * average_size
        self.big_gamma = big_gamma
        self.beta = beta

        self.n_bits_observation_space = n_bits_observation_space
        self.n_bits_action_space = n_bits_action_space
        self.possible_actions = possible_actions
        self._initialise_weights()

    def _initialise_weights(self) -> None:
        """ Initialise the weights of the Q function network. """
        for i, ii in zip(tuple(range(4)), tuple(range(8, 12))):
            for j, jj in zip(tuple(range(4, 8)), tuple(range(12, 16))):
                self.Q_hh[(i, j)] = 2 * random.random() - 1
                self.Q_hh[(ii, jj)] = 2 * random.random() - 1
        for i, j in zip(tuple(range(4, 8)), tuple(range(12, 16))):
            self.Q_hh[(i, j)] = 2 * random.random() - 1

        # Fully connection between state and blue nodes
        for j in (tuple(range(4)) + tuple(range(12, 16))):
            for i in range(self.n_bits_observation_space):
                self.Q_vh[(i, j,)] = 2 * random.random() - 1
            # Fully connection between action and red nodes
        for j in (tuple(range(4, 8)) + tuple(range(8, 12))):
            for i in range(
                    self.n_bits_observation_space,
                    self.n_bits_observation_space + self.n_bits_action_space):
                self.Q_vh[(i, j,)] = 2 * random.random() - 1

    def calculate_F(self, state: np.ndarray, action: int) -> tuple:
        """ Based on state and chosen action, calculate the free energy,
        samples and vis_iterable.
        :param state: state the environment is in (binary vector, directly
        obtained from either env.reset(), or env.step())
        :param action: chosen action (index)
        :return free energy, samples, and vis_iterable """
        vis_iterable = create_visible_iterable(state=state, action=action)

        general_Q = utl.create_general_Q_from(
            Q_hh=self.Q_hh, Q_vh=self.Q_vh, visible_iterable=vis_iterable)

        samples = list(SimulatedAnnealingSampler().sample_qubo(
            general_Q, num_reads=self.sample_count).samples())

        # TODO: why do we need to shuffle them?
        random.shuffle(samples)
        avg_hamiltonian = utl.get_3d_hamiltonian_average_value(
            samples, general_Q, self.replica_count, self.average_size,
            big_gamma=self.big_gamma, beta=self.beta)

        free_energy = utl.get_free_energy(
            avg_hamiltonian, samples, self.replica_count, beta=self.beta)

        return free_energy, samples, vis_iterable

    def calculate_and_predict(self, state: np.ndarray, epsilon: float) -> tuple:
        """ Get a2 = argmax_a Q(s2, a), but following epsilon-greedy policy
        :param state: state the environment is in (binary vector, directly
        obtained from either env.reset(), or env.step())
        :param possible_actions: list of possible actions (here [0, 1, 2])
        :param epsilon: probability for choosing random action (epsilon-greedy)
        :return: chosen action, free energy, samples, vis_iterable """
        # Epsilon greedy implementation
        if np.random.random() < epsilon:
            # Pick action randomly
            action = random.choice(self.possible_actions)
            free_energy, samples, vis_iterable = self.calculate_F(
                state, action)
            return action, free_energy, samples, vis_iterable
        else:
            # Pick action greedily
            # Do I really have to loop through all the actions to calculate
            # the Q values for every action to then pick the argmax_a Q?
            max_dict = {'free_energy': float('inf'), 'action': None,
                        'samples': None, 'vis_iterable': None}

            for action in self.possible_actions:
                free_energy, samples, vis_iterable = self.calculate_F(
                    state, action)

                # If the now calculated F is smaller than the previous minimum
                # (i.e. max. Q), update values in dictionary
                if max_dict['free_energy'] > free_energy:
                    max_dict['free_energy'] = free_energy
                    max_dict['action'] = action
                    max_dict['samples'] = samples
                    max_dict['vis_iterable'] = vis_iterable

            return (max_dict['action'], max_dict['free_energy'],
                    max_dict['samples'], max_dict['vis_iterable'])


class QBMQN(object):
    def __init__(self, env: TargetSteeringEnv, replica_count: int,
                 average_size: int, big_gamma: float = 0.5, beta: float = 2.0,
                 learning_rate: float = 1e-4,
                 exploration_fraction: float = 0.8,
                 exploration_initial_eps: float = 1.0,
                 exploration_final_eps: float = 0.05,
                 small_gamma: float = 0.99) -> None:
        """ Implementation of the QBM Q learning agent, following the paper:
        https://arxiv.org/pdf/1706.00074.pdf
        :param env: OpenAI gym environment
        :param replica_count: ?
        :param average_size: ?
        :param big_gamma: QBM param., see paper
        :param beta: QBM param., 'temperature', see paper
        :param learning_rate: RL. param., learning rate for update of weights
        :param exploration_fraction: RL param., fraction of total number of
        time steps over which the epsilon decays
        :param exploration_initial_eps: RL param., initial epsilon for
        epsilon-greedy param.
        :param exploration_final_eps: RL param., final epsilon for
        epsilon-greedy param.
        :param small_gamma: RL param., discount factor """
        self.env = env

        # Learning parameters
        self.learning_rate = learning_rate
        self.small_gamma = small_gamma
        self.exploration_fraction = exploration_fraction
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps

        # Q function
        n_bits_observation_space = env.n_bits_observation_space
        n_bits_action_space = math.ceil(math.log2(env.action_space.n))
        possible_actions = [i for i in range(env.action_space.n)]
        self.q_function = QFunction(
            n_bits_observation_space, n_bits_action_space, possible_actions,
            replica_count, average_size, big_gamma, beta)

    def learn(self, total_timesteps: int) -> None:
        """ Train the agent for the specified number of iterations.
        :param total_timesteps: number of training steps """
        epsilon = self._get_epsilon_schedule(total_timesteps)

        state_1 = env.reset()
        for it in tqdm.trange(total_timesteps):
            # Step 1: given the current_state, pick an action randomly (this is
            # to compute current_F). This corresponds to (s1, a1).
            # TODO: is this really what we have to do? Random action here?
            action_1 = env.action_space.sample()

            current_F, current_samples, current_vis_iterable = (
                self.q_function.calculate_F(state_1, action_1)
            )

            # Step 2: take the step in the environment
            state_2, reward_1, done, _ = env.step(action=action_1)

            # Step 3: get action_2 = argmax_act Q(state_2, act)
            # Need to take max Q, resp. min. F, or random action if
            # epsilon-greedy is fulfilled. Do I really need to loop through
            # all the actions to calculate the Q values for every action to
            # then pick the argmax  Q?
            action_2, future_F, future_samples, future_vis_iterable = (
                self.q_function.calculate_and_predict(state_2, epsilon[it])
            )

            # Step 4: update weights
            self.q_function.Q_hh, self.q_function.Q_vh = (
                utl.update_weights(
                    self.q_function.Q_hh,
                    self.q_function.Q_vh,
                    current_samples,  # why current_samples?
                    reward_1,
                    future_F,
                    current_F,
                    current_vis_iterable,
                    self.learning_rate,
                    self.small_gamma)
            )

            state_1 = state_2
            if done:
                state_1 = env.reset()
        # Reset environment after training to save episode logging to all logs.
        env.reset()

    def predict(self, state, deterministic):
        """ Based on the given state, we pick the best action (here we always
        pick the action greedily, i.e. epsilon = 0., as we are assuming that
        the agent has been trained). This method is required to evaluate the
        trained agent.
        :param state: state encoded as binary-encoded vector as obtained from
        environment .reset() or .step()
        :param deterministic: another argument used by stable-baselines3
        :return next action, None: need to fulfill the stable-baselines3
        interface """
        action, free_energy, samples, vis_iterable = (
            self.q_function.calculate_and_predict(state=state, epsilon=0.))

        return action, None

    def _get_epsilon_schedule(self, total_timesteps: int) -> np.ndarray:
        """ Define epsilon schedule as linear decay between time step 0 and
        time step exploration_fraction * total_timesteps, starting from
        exploration_initial_eps and ending at exploration_final_eps.
        :param total_timesteps: total number of training steps
        :return epsilon array including decay """
        n_steps_decay = int(self.exploration_fraction * total_timesteps)
        eps_step = (
            (self.exploration_final_eps - self.exploration_initial_eps) /
            n_steps_decay)
        eps_decay = np.arange(
            self.exploration_initial_eps, self.exploration_final_eps, eps_step)
        eps = np.ones(total_timesteps) * self.exploration_final_eps
        eps[:n_steps_decay] = eps_decay
        return eps


if __name__ == "__main__":
    # TODO: Logging needs to be changed potentially?
    # TODO: where should these variables go? Into QBMQN?

    N_BITS_OBSERVATION_SPACE = 8

    # Agent training
    replica_count = 10
    average_size = 50
    total_timesteps = 100

    env = TargetSteeringEnv(n_bits_observation_space=N_BITS_OBSERVATION_SPACE)
    agent = QBMQN(env, replica_count=replica_count, average_size=average_size,
                  big_gamma=0.5, beta=2., exploration_fraction=0.8,
                  exploration_initial_eps=1.0, exploration_final_eps=0.,
                  small_gamma=0.98, learning_rate=1e-3)
    agent.learn(total_timesteps=total_timesteps)
    hlp.plot_log(env, fig_title='Agent training')

    # Agent evaluation
    env = TargetSteeringEnv(n_bits_observation_space=N_BITS_OBSERVATION_SPACE)
    hlp.evaluate_agent(env, agent, n_episodes=6,
                       make_plot=True, fig_title='Agent evaluation')
