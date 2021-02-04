import gym

import math
from scipy.integrate import quad
from math import pi, exp
import numpy as np


class TwissElement:
    def __init__(self, beta, alpha, d, mu):
        """
        :param beta: beta Twiss function value at element
        :param alpha: alpha Twiss function value at element
        :param d: ?
        :param mu: phase advance
        """
        self.beta = beta
        self.alpha = alpha
        self.mu = mu


def transport(element1: TwissElement, element2: TwissElement, x: float,
              px: float) -> (float, float):
    """ Transport (x, xp) coordinate-momentum pair from element 1 to element 2
    using linear transport. """
    mu = element2.mu - element1.mu
    alpha1 = element1.alpha
    alpha2 = element2.alpha
    beta1 = element1.beta
    beta2 = element2.beta

    m11 = math.sqrt(beta2 / beta1) * (math.cos(mu) + alpha1 * math.sin(mu))
    m12 = math.sqrt(beta1 * beta2) * math.sin(mu)
    m21 = ((alpha1 - alpha2) * math.cos(mu) - (1 + alpha1 * alpha2) * math.sin(
        mu)) / math.sqrt(beta1 * beta2)
    m22 = math.sqrt(beta1 / beta2) * (math.cos(mu) - alpha2 * math.sin(mu))

    return m11 * x + m12 * px, m21 * x + m22 * px


class TargetSteeringEnv(gym.Env):
    def __init__(self, n_bits_observation_space=8, debug=False):
        """
        :param n_bits_observation_space: number of bits used to represent the
        observation space (now discrete)

        x0: beam position at origin, i.e. before entering MSSB
        state: beam position at BPM (observation / state)
        mssb_angle: dipole kick angle (rad)
        mssb_min, mssb_max: min. and max. dipole strengths for init. (rad)
        mssb_delta: amount of discrete change in mssb_angle upon action (rad)
        """
        super(TargetSteeringEnv, self).__init__()

        # MSSB (dipole) kicker
        self.mssb_angle = None  # not set, will be init. with self.reset()
        self.mssb_angle_min = -160e-6
        self.mssb_angle_max = 160e-6
        self.mssb_delta = 3e-5  # discrete action step (rad)

        # Beam position
        self.x0 = 0.
        self.state = None  # not set, will be init. with self.reset()

        # Define transfer line
        self.mssb = TwissElement(16.1, -0.397093117, 0.045314011, 1.46158005)
        self.bpm1 = TwissElement(339.174497, -6.521184683, 2.078511443,
                                 2.081365696)
        self.target = TwissElement(7.976311944, -0.411639485, 0.30867161,
                                   2.398031982)

        # Get possible range of observations to define observation_space
        self.x_max, _ = self.get_pos_at_bpm_target(self.mssb_angle_max)
        self.x_min, _ = self.get_pos_at_bpm_target(self.mssb_angle_min)

        # find change in x for change mssb_delta in dipole to define the
        # threshold for canceling an episode and to define the discretization
        # range required (the latter has to be larger than the former by at
        # least x_delta)
        x_min_plus_delta, _ = self.get_pos_at_bpm_target(
            self.mssb_angle_min + self.mssb_delta)
        x_delta = x_min_plus_delta - self.x_min
        self.x_delta = x_delta
        self.x_margin_discretisation = 8 * x_delta
        self.x_margin_abort_episode = (
                self.x_margin_discretisation - 1.5 * x_delta)

        # Gym requirements
        # Define action space with 3 discrete actions.
        self.action_space = gym.spaces.Discrete(3)

        # This will create a discrete state space with
        # length n_bits_observation_space, i.e. the Q-network will have
        # n_bits_observation_space nodes at its input layer. You can find
        # that by checking agent.q_net once initialized.
        self.observation_space = gym.spaces.MultiBinary(
            n_bits_observation_space)
        self.observation_bin_width = (
            (2. * self.x_margin_discretisation + self.x_max - self.x_min) /
            2**n_bits_observation_space)
        self.n_bits_observation_space = n_bits_observation_space

        # For cancellation when beyond certain number of steps in an epoch
        self.step_count = None
        self.max_steps_per_epoch = 30
        self.reward_threshold = 0.9 * self.get_max_reward()

        # Logging
        self.log_all = []
        self.log_episode = []
        self.done_reason_map = {
            -1: '',
            0: 'Max. # steps',
            1: 'Reward thresh.',
            2: 'State OOB'}
        self.debug = debug

    def step(self, action):
        """ Action is discrete here and is an integer number in set {0, 1, 2}.
        Action_map shows how action is mapped to change of self.mssb_angle. """

        err_msg = f"{action} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        x_binary = self.state

        # Apply action and update environment
        action_map = {0: 0, 1: self.mssb_delta, 2: -self.mssb_delta}
        total_angle = self.mssb_angle + action_map[action]
        x_new, reward = self.get_pos_at_bpm_target(total_angle)
        x_new_binary = self._make_state_discrete_binary(x_new)
        self.state = x_new_binary
        self.mssb_angle = total_angle

        if self.debug:
            x_new_discrete = self._make_state_discrete(x_new)
            print('x_new', x_new)
            print('x_new_discrete', x_new_discrete)
            print('x_new_binary', x_new_binary)
            print('x_new_binary_float',
                  self._make_binary_state_float(x_new_binary))

        self.step_count += 1

        # Episode done?
        done = bool(
            self.step_count > self.max_steps_per_epoch
            or reward > self.reward_threshold
            or x_new > (self.x_max + self.x_margin_abort_episode)
            or x_new < (self.x_min - self.x_margin_abort_episode)
        )

        # Keep track of reason for episode abort
        done_reason = -1
        if self.step_count > self.max_steps_per_epoch:
            done_reason = 0
        elif reward > self.reward_threshold:
            done_reason = 1
        elif (x_new > (self.x_max + self.x_margin_abort_episode) or
              x_new < (self.x_min - self.x_margin_abort_episode)):
            done_reason = 2
        else:
            pass

        # Log history
        self.log_episode.append([x_binary, action, reward, x_new_binary,
                                 done, done_reason])
        if done:
            self.log_all.append(self.log_episode)

        return self.state, reward, done, {}

    def reset(self):
        """ Reset the environment. Initialize self.mssb_angle as a multiple of
        self.mssb_delta. """
        idx_max = (self.mssb_angle_max - self.mssb_angle_min) / self.mssb_delta
        idx = np.random.randint(idx_max)
        self.mssb_angle = self.mssb_angle_min + idx * self.mssb_delta
        x_init, _ = self.get_pos_at_bpm_target(self.mssb_angle)

        x_init_binary = self._make_state_discrete_binary(x_init)
        self.state = x_init_binary

        self.step_count = 0
        self.log_episode = []

        if self.debug:
            print('x_init', x_init)
            print('x_init_discrete', self._make_state_discrete(x_init))
            print('x_init_binary', x_init_binary)
            print('x_init_binary_float',
                  self._make_binary_state_float(x_init_binary))

        return self.state

    def clear_log(self):
        """ Delete log / history of the environment. """
        self.log_episode = []
        self.log_all = []

    def _get_reward(self, beam_pos: float):
        """
        Calculate reward of environment state: reward is defined by integrated
        intensity on target (assuming Gaussian beam, integration range +/- 3
        sigma.
        :param beam_pos: beam position on target (not known to RL agent)
        :return: reward, float in [0, 1].
        """
        emittance = 1.1725E-08
        sigma = math.sqrt(self.target.beta * emittance)
        self.intensity_on_target = quad(
            lambda x: 1 / (sigma * (2 * pi) ** 0.5) * exp(
                (x - beam_pos) ** 2 / (-2 * sigma ** 2)),
            -3*sigma, 3*sigma)

        reward = self.intensity_on_target[0]
        return reward

    def get_pos_at_bpm_target(self, total_angle: float) -> (float, float):
        """
        Transports beam through the transfer line and calculates the position at
        the BPM and at the target. These are required for the reward
        calculation.
        :param total_angle: total kick angle from the MSSB dipole (rad)
        :return: position at BPM and reward as a tuple
        """
        x_bpm, px_bpm = transport(self.mssb, self.bpm1, self.x0, total_angle)
        x_target, px_target = transport(
            self.mssb, self.target, self.x0, total_angle)

        reward = self._get_reward(x_target)
        return x_bpm, reward

    def _make_state_discrete_binary(self, x):
        """ Discretize state into 2**self.n_bits_observation_space bins and
        convert to binary format.
        :param x: input BPM position (float), to be converted to binary
        :return x_binary: list of length self.n_bits_observation
        encoding the state in discrete, binary format.
        """
        bin_idx = self._make_state_discrete(x)

        # Make sure that we never go above or below range available in binary
        assert bin_idx < (2**self.n_bits_observation_space - 1)
        assert bin_idx > -1

        # Convert integer to binary with a fixed length
        binary_fmt = f'0{self.n_bits_observation_space}b'
        binary_string = format(bin_idx, binary_fmt)

        # Convert binary_string to list
        x_binary = [int(i) for i in binary_string]
        return x_binary

    def _make_state_discrete(self, x):
        """ Take input x (BPM position) and discretize / bin.
        :param x: BPM position (float)
        :return: bin_idx (bin index)
        """
        bin_idx = int((x - self.x_min + self.x_margin_discretisation) /
                      self.observation_bin_width)
        return bin_idx

    def _make_binary_state_float(self, x_binary):
        """ This is the inverse operation of _make_state_discrete_binary(..).
        I.e. we take a binary input list of length
        self.n_bits_observation_space and convert it into a float
        corresponding to the BPM position. """
        binary_string = ''.join([str(i) for i in x_binary])
        bin_idx = int(binary_string, 2)
        x = ((bin_idx + 0.5) * self.observation_bin_width +
             self.x_min - self.x_margin_discretisation)
        return x

    def get_max_reward(self):
        """ Calculate maximum reward. This is used to define the threshold
        for cancellation of an episode. Note that in reality this is usually
        not known. """
        angles = np.linspace(self.mssb_angle_min, self.mssb_angle_max, 200)
        max_r = -1.
        for i, ang in enumerate(angles):
            _, r = self.get_pos_at_bpm_target(total_angle=ang)
            max_r = max(r, max_r)
        return max_r

    def get_max_n_steps_optimal_behaviour(self):
        """
        :return: maximum number of steps required to solve the problem from any
        initial condition assuming optimal behaviour of agent.
        """
        return np.ceil(self.x_max / self.x_delta - 1)
