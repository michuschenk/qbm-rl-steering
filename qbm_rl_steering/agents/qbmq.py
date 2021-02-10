import gym
import random
from qbm_rl_steering.environment.env_desc import TargetSteeringEnv
import qbm_rl_steering.agents.qbmq_utils as utl

from neal import SimulatedAnnealingSampler
import numpy as np
import math
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


def create_visible_iterable(state: np.ndarray, action_index: int) -> tuple:
    """ Take state (e.g. directly from environment), and action_index (
    following env.action_map), and concatenate them to the visible_iterable
    tuple required for the create_general_Q_from(..) function. This also
    converts all 0s appearing in state and action_index (once made binary) to
    -1s. """
    s = convert_bits(state)
    a = convert_bits(make_action_binary(action_index))
    return s + a


class QFunction(object):
    def __init__(self, n_bits_observation_space: int,
                 n_bits_action_space: int, possible_actions: list,
                 replica_count: int, average_size: int, big_gamma: float,
                 beta: float) -> None:

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

    # TODO: define epsilon, that can decay over time. predict returns random or
    #  Q value
    def _initialise_weights(self):
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

    def calculate_F(self, state, action):
        """ Based on state and chosen action, calculate the free energy,
        samples and vis_iterable.
        :param state: state the environment is in (binary vector, directly
        obtained from either env.reset(), or env.step())
        :param action: chosen action (index)
        :return free energy, samples, and vis_iterable """
        vis_iterable = create_visible_iterable(state=state, action_index=action)

        general_Q = utl.create_general_Q_from(Q_hh=self.Q_hh, Q_vh=self.Q_vh,
                                              visible_iterable=vis_iterable)

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

    def calculate_and_predict(self, state, epsilon):
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

    def update_weights(self):
        pass


class QBMQN(object):
    def __init__(self, env: TargetSteeringEnv, replica_count, average_size,
                 big_gamma, beta):
        self.env = env
        n_bits_observation_space = env.n_bits_observation_space
        # how many bits do we need to describe the action space?
        n_bits_action_space = math.ceil(math.log2(env.action_space.n))
        possible_actions = [i for i in range(env.action_space.n)]

        self.q_function = QFunction(
            n_bits_observation_space, n_bits_action_space, possible_actions,
            replica_count, average_size, big_gamma, beta)

    def learn(self, n_iterations):
        pass
        # state_1 = self.env.reset()
        # all_possible_actions = env.something
        # for i in range(n_iterations):
        #     action_1 = self.q_function.predict(state_1,all_possible_actions) # --> current_F,
        #     state_2,reward, done,_ = env.step(action_1)
        #     action_2 = self.q_function.predict(state_2,all_possible_actions) # --> future_F

        # check what is required for weight update
        # check how to get handle on current and future F.
        # at the end state_1 = state_2


if __name__ == "__main__":
    debug = True

    # TODO: where should these variables go? Into QBMQN?
    replica_count = 10
    average_size = 50
    sample_count = replica_count * average_size

    beta = 2.
    big_gamma = 0.5

    learning_rate = 1e-3
    small_gamma = 0.98
    epsilon = 0.1

    env = TargetSteeringEnv(n_bits_observation_space=8)
    qbmqn = QBMQN(env, replica_count, average_size, big_gamma, beta)

    # Initialize environment
    current_state = env.reset()

    # Step 1: given the current_state, pick an action randomly (this is to
    # compute current_F). This corresponds to (s1, a1).
    # TODO: is this really what we have to do? Random action here?
    state_1 = current_state
    action_1 = env.action_space.sample()

    current_F, current_samples, current_vis_iterable = (
        qbmqn.q_function.calculate_F(state_1, action_1)
    )

    # Step 2: take the step in the environment
    state_2, reward_1, done, _ = env.step(action=action_1)

    # Step 3: get action_2 = argmax_act Q(state_2, act)
    # Need to take max Q, resp. min. F, or random action if epsilon-greedy is
    # fulfilled. Do I really need to loop through all the actions to
    # calculate the Q values for every action to then pick the argmax  Q?
    action_2, future_F, future_samples, future_vis_iterable = (
        qbmqn.q_function.calculate_and_predict(state_2, epsilon)
    )

    # Step 4: update weights
    # TODO: Rename Q_hh, Q_vh to w_hh, w_vh
    qbmqn.q_function.Q_hh, qbmqn.q_function.Q_vh = (
        utl.update_weights(
            qbmqn.q_function.Q_hh,
            qbmqn.q_function.Q_vh,
            current_samples,  # why current_samples?
            reward_1,
            future_F,
            current_F,
            current_vis_iterable,
            learning_rate,
            small_gamma)
    )
