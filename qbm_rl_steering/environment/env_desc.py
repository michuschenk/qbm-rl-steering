from typing import Tuple, List, Dict

import numpy as np
import math
from scipy.integrate import quad
import gym

from .logger import Logger


class TwissElement:
    def __init__(self, beta: float, alpha: float, d: float, mu: float) -> None:
        """ Initialize a Twiss element as part of a beam transfer line.
        :param beta: beta Twiss function value at element
        :param alpha: alpha Twiss function value at element
        :param d: ?
        :param mu: phase advance """
        self.beta = beta
        self.alpha = alpha
        self.mu = mu


def transport(element1: TwissElement, element2: TwissElement, x: float,
              px: float) -> Tuple[float, float]:
    """ Transport (x, xp) coordinate-momentum pair from element 1 to element 2
    using linear transport.
    :param element1: the first Twiss element, starting point
    :param element2: the second Twiss element, end point
    :param x: initial coordinate (m), at element1
    :param px: initial momentum (?), at element1
    :return (x, px): tuple of coordinate and momentum at element2 """
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
    def __init__(self, n_bits_observation_space: int = 8,
                 max_steps_per_episode: int = 20, n_actions: int = 2,
                 simple_reward: bool = True, debug: bool = False) -> None:
        """
        :param n_bits_observation_space: number of bits used to represent the
        observation space (will be discretized into
        2**n_bits_observation_space bins)
        :param max_steps_per_episode: max number of steps we allow agent to
        'explore' per episode. After this number of steps, episode is aborted.
        :param n_actions: number of actions. Here only values 2 (up or down),
        and 3 (up, down, stay) are possible.
        :param debug: Flag for debugging, adds some prints here and there. """
        super(TargetSteeringEnv, self).__init__()

        # DEFINE TRANSFER LINE
        self.mssb = TwissElement(16.1, -0.397093117, 0.045314011, 1.46158005)
        self.bpm1 = TwissElement(339.174497, -6.521184683, 2.078511443,
                                 2.081365696)
        self.target = TwissElement(7.976311944, -0.411639485, 0.30867161,
                                   2.398031982)

        # MSSB DIPOLE / KICKER
        # mssb_angle: dipole kick angle (rad)
        # mssb_min, mssb_max: min. and max. dipole strengths for init. (rad)
        # mssb_delta: amount of discrete change in mssb_angle upon action (rad)
        self.mssb_angle = None  # not set, will be init. with self.reset()
        self.mssb_angle_min = -160e-6  # (rad)
        self.mssb_angle_max = 160e-6  # (rad)
        self.mssb_delta = 10e-6  # discrete action step (rad)

        # BEAM POSITION
        # x0: position at origin, i.e. before entering MSSB
        # state: position at BPM(observation / state)
        # x_min, x_max: possible range of observations to define
        # observation_space given mssb_angle_min, mssb_angle_max
        self.x0 = 0.
        self.state = None  # not set, will be init. with self.reset()
        self.x_max, _ = self.get_pos_at_bpm_target(self.mssb_angle_max)
        self.x_min, _ = self.get_pos_at_bpm_target(self.mssb_angle_min)

        # Find change in x for change mssb_delta in dipole to define the
        # threshold for canceling an episode and to define the discretization
        # range required (the latter has to be larger than the former by at
        # least x_delta)
        x_min_plus_delta, _ = self.get_pos_at_bpm_target(
            self.mssb_angle_min + self.mssb_delta)
        self.x_delta = x_min_plus_delta - self.x_min
        self.x_margin_discretization = 3 * self.x_delta
        self.x_margin_abort_episode = (
                self.x_margin_discretization - 2. * self.x_delta)

        # GYM REQUIREMENTS
        # Define action space with 2 or 3 discrete actions. Action_map defines
        # how action is mapped to change of self.mssb_angle.
        if n_actions == 3:
            self.action_space = gym.spaces.Discrete(3)
            self.action_map = {0: 0., 1: self.mssb_delta, 2: -self.mssb_delta}
        elif n_actions == 2:
            self.action_space = gym.spaces.Discrete(2)
            self.action_map = {0: self.mssb_delta, 1: -self.mssb_delta}
        else:
            raise ValueError("Set n_actions either to 2 or 3.")

        # This will create a discrete observation / state space with
        # length n_bits_observation_space, i.e. the Q-network will have
        # n_bits_observation_space nodes at its input layer. You can verify
        # that by checking agent.q_net once initialized.
        self.observation_space = gym.spaces.MultiBinary(
            n_bits_observation_space)
        self.observation_bin_width = (
                (2. * self.x_margin_discretization + self.x_max - self.x_min) /
                2 ** n_bits_observation_space)
        self.n_bits_observation_space = n_bits_observation_space

        # For cancellation when beyond certain number of steps in an episode
        self.simple_reward = simple_reward
        self.step_count = None
        self.max_steps_per_episode = max_steps_per_episode
        self.reward_threshold = 0.98 * self.get_max_reward()

        # Logging and debugging
        self.logger = Logger()
        self.debug = debug

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """ Perform one step in the environment (take an action, update
        parameters in environment, receive reward, check if episode ends,
        append all info to logger, return new state, reward, etc.
        :param action: is discrete here and is an integer number in {0, 1, 2}.
        :return tuple of the new state, reward, whether episode is done,
        and dictionary with additional info (not used at the moment). """
        err_msg = f"{action} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        x_binary = self.state

        # Apply action and update environment (get new position at BPM and
        # convert into a binary vector)
        total_angle = self.mssb_angle + self.action_map[action]
        x_new, reward = self.get_pos_at_bpm_target(total_angle)
        x_new_binary = self._make_state_discrete_binary(x_new)
        self.state = x_new_binary
        self.mssb_angle = total_angle

        self.step_count += 1

        # Is episode done?
        done = bool(
            self.step_count > self.max_steps_per_episode
            or reward > self.reward_threshold
            or x_new > (self.x_max + self.x_margin_abort_episode)
            or x_new < (self.x_min - self.x_margin_abort_episode)
        )

        # Keep track of reason for episode abort
        done_reason = -1
        if self.step_count > self.max_steps_per_episode:
            done_reason = 0
        elif reward > self.reward_threshold:
            done_reason = 1
        elif (x_new > (self.x_max + self.x_margin_abort_episode) or
              x_new < (self.x_min - self.x_margin_abort_episode)):
            done_reason = 2
        else:
            pass

        if self.simple_reward:
            reward = self.simplify_reward(reward)

        # Log history
        self.logger.log_episode.append(
            [x_binary, action, reward, x_new_binary, done, done_reason])
        if done:
            self.logger.log_all.append(self.logger.log_episode)

        return self.state, reward, done, {}

    def reset(self, init_state: float = None) -> np.ndarray:
        """ Reset the environment. Initialize self.mssb_angle as a multiple of
        self.mssb_delta, get the initial state, and reset logger. This method
        gets called e.g. at the end of an episode.
        :return an initial state """
        if init_state is None:
            # Initialize mssb_angle within self.mssb_angle_min and
            # self.mssb_angle_max
            self.mssb_angle = np.random.uniform(
                low=self.mssb_angle_min, high=self.mssb_angle_max)
        else:
            # If init_state is set, we have to calc. the mssb_angle that puts
            # env in that state (that's a bit messy ...)
            # TODO: this needs major cleanup
            n_points = 1000
            x_pos_array = np.zeros(n_points)
            mssb_array = np.linspace(
                    self.mssb_angle_min-self.mssb_delta,
                    self.mssb_angle_max+self.mssb_delta, n_points)
            for i, mssb in enumerate(mssb_array):
                x_pos_array[i], _ = self.get_pos_at_bpm_target(mssb)
            idx = np.argmin(np.abs(init_state - x_pos_array))
            self.mssb_angle = mssb_array[idx]

        x_init, _ = self.get_pos_at_bpm_target(self.mssb_angle)

        # Convert state into binary vector
        x_init_binary = self._make_state_discrete_binary(x_init)
        self.state = x_init_binary

        self.step_count = 0
        self.logger.episode_reset()

        if self.debug:
            print('x_init', x_init)
            print('x_init_discrete', self._make_state_discrete(x_init))
            print('x_init_binary', x_init_binary)
            print('x_init_binary_float',
                  self.make_binary_state_float(x_init_binary))

        return self.state

    def clear_log(self) -> None:
        """ Delete all log / history of the logger of this environment. """
        self.logger.clear_all()

    def _get_reward(self, beam_pos: float) -> float:
        """ Calculate reward of environment given beam position on target.
        Reward is defined by integrated intensity on target (assuming
        Gaussian beam, integration range +/- 3 sigma).
        :param beam_pos: beam position on target (not known to RL agent)
        :return: reward, float in range ~[0, 1]. """
        emittance = 1.1725E-08
        sigma = math.sqrt(self.target.beta * emittance)
        self.intensity_on_target = quad(
            lambda x: 1 / (sigma * (2 * math.pi) ** 0.5) * math.exp(
                (x - beam_pos) ** 2 / (-2 * sigma ** 2)),
            -3*sigma, 3*sigma)

        reward = self.intensity_on_target[0]

        return reward

    def simplify_reward(self, reward: float) -> float:
        """
        Simplify, i.e. discretize the reward. Give only positive reward when
        episode is solved
        :param reward: input reward
        :return discretized simplified reward
        """
        if reward > self.reward_threshold:
            reward = 100.
        else:
            reward = 0.
        return reward

    def get_pos_at_bpm_target(self, total_angle: float) -> Tuple[float, float]:
        """ Transports beam through the transfer line and calculates the
        position at the BPM and at the target. These are required for the reward
        calculation and to get the state based on the currently set dipole
        angle.
        :param total_angle: total kick angle of the MSSB dipole (rad)
        :return position at BPM and reward as a tuple """
        x_bpm, px_bpm = transport(self.mssb, self.bpm1, self.x0, total_angle)
        x_target, px_target = transport(
            self.mssb, self.target, self.x0, total_angle)

        reward = self._get_reward(x_target)
        return x_bpm, reward

    def _make_state_discrete_binary(self, x: float) -> np.ndarray:
        """
        Discretize state into 2**self.n_bits_observation_space bins and
        convert to binary format.
        :param x: input BPM position (float), to be converted to binary
        :return x_binary: np.array of length self.n_bits_observation_space
        encoding the state in discrete, binary format. """
        bin_idx = self._make_state_discrete(x)

        # Make sure that we never go above or below range available in binary
        assert bin_idx < (2**self.n_bits_observation_space - 1)
        assert bin_idx > -1

        return self.make_binary(bin_idx)

    def make_binary(self, val: int) -> np.ndarray:
        """
        Converts an integer to a binary vector that describes the state.
        :param val: integer number to be converted
        :return binary encoded vector (0s are replaced by -1s)
        """
        binary_fmt = f'0{self.n_bits_observation_space}b'
        binary_string = format(val, binary_fmt)

        # Convert binary_string to np array
        state_binary = np.array([int(i) for i in binary_string])
        state_binary[state_binary == 0] = -1
        return state_binary

    def _make_state_discrete(self, x: float) -> int:
        """ Take input x (BPM position) and discretize / bin.
        :param x: BPM position
        :return: bin index """
        bin_idx = int((x - self.x_min + self.x_margin_discretization) /
                      self.observation_bin_width)
        return bin_idx

    def make_binary_state_float(self, x_binary: np.ndarray) -> float:
        """ This is the inverse operation of _make_state_discrete_binary(..).
        I.e. we take a binary input np.array of length
        self.n_bits_observation_space and convert it back to a float
        corresponding to the BPM position (modulo the loss of precision due
        to the discretization performed earlier).
        :param x_binary: state as binary encoded vector, i.e. np.array of
        length self.n_bits_observation_space
        :return x position converted back to a float """
        x_binary[x_binary == -1] = 0
        binary_string = ''.join([str(i) for i in x_binary])
        bin_idx = int(binary_string, 2)
        x = ((bin_idx + 0.5) * self.observation_bin_width +
             self.x_min - self.x_margin_discretization)
        return x

    def get_max_reward(self) -> float:
        """ Calculate maximum reward. This is used to define the threshold
        for cancellation of an episode. Note that this is usually not known.
        :return maximum reward """
        _, _, rewards = self.get_response()
        return np.max(rewards)

    def get_response(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Calculate response of the environment, i.e. the x_position and
        reward dependence on the dipole kick angle (for the full available
        range of angles).
        :return Tuple of np.ndarrays containing dipole angles, x positions,
        and rewards. """
        angles = np.linspace(self.mssb_angle_min, self.mssb_angle_max, 1000)
        x_pos = np.zeros_like(angles)
        rewards = np.zeros_like(angles)
        for i, ang in enumerate(angles):
            x, r = self.get_pos_at_bpm_target(total_angle=ang)
            x_pos[i] = x
            rewards[i] = r
        return angles, x_pos, rewards

    def get_all_states(self) -> Tuple[np.ndarray, List]:
        """
        Return a list of all the possible states that the system can be in
        in binary encoded form. This is only considering the actual range
        that we can reach with dipole angles in [mssb_angle_min,
        mssb_angle_max].
        :return tuple of np array and List of all possible states in float
        and binary form {-1, 1} resp.
        """
        # Allowed range of states (float)
        states_float = np.arange(
            self.x_min,
            self.x_max + self.observation_bin_width / 2.,
            self.observation_bin_width)

        # Convert to binary vectors
        states_binary = []
        for s in states_float:
            states_binary.append(self._make_state_discrete_binary(s))

        return states_float, states_binary

    def get_max_n_steps_optimal_behaviour(self) -> int:
        """ Calculate maximum number of steps required to solve the problem
        from any initial condition assuming optimal behaviour of agent. We
        take into account the reward threshold (above which the problem is
        assumed to be solved).
        :return upper bound for number of required steps (int). """
        _, x, r = self.get_response()
        idx = np.where(r > self.reward_threshold)[0][-1]
        x_up_reward_thresh = x[idx]
        return np.ceil((self.x_max - x_up_reward_thresh) / self.x_delta - 1)
