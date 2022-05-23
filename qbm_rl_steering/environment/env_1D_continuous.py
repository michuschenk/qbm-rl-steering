from typing import Tuple, Dict

import numpy as np
import math
from scipy.integrate import quad
import gym

from qbm_rl_steering.environment.logger import Logger


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
    def __init__(self, max_steps_per_episode: int = 20, n_actions: int = 2) \
            -> None:
        """
        :param max_steps_per_episode: max number of steps we allow agent to
        'explore' per episode. After this number of steps, episode is aborted.
        :param n_actions: number of actions. Here only values 2 (up or down),
        and 3 (up, down, stay) are possible. """
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
        self.mssb_angle_max = 140e-6  # (rad)
        self.mssb_angle_min = -self.mssb_angle_max  # (rad)
        self.mssb_delta = 15e-6  # discrete action step (rad)

        # BEAM POSITION
        # x0: position at origin, i.e. before entering MSSB
        # state: position at BPM(observation / state)
        # x_min, x_max: possible range of observations to define
        # observation_space given mssb_angle_min, mssb_angle_max
        self.x0 = 0.
        self.state = None  # not set, will be init. with self.reset()
        x_max, _ = self.get_pos_at_bpm_target(self.mssb_angle_max)
        x_min, _ = self.get_pos_at_bpm_target(self.mssb_angle_min)

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

        # This will create a continuous observation space
        self.observation_space = gym.spaces.Box(
            low=np.array([1.05*x_min]), high=np.array([1.05*x_max]))

        # For cancellation when beyond certain number of steps in an episode
        self.step_count = None
        self.max_steps_per_episode = max_steps_per_episode
        self.reward_threshold = 0.85 * self.get_max_intensity()

        # Logging
        self.interaction_logger = Logger()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """ Perform one step in the environment (take an action, update
        parameters in environment, receive reward, check if episode ends,
        append all info to logger, return new state, reward, etc.
        :param action: is discrete here and is an integer number in {0, 1, 2}.
        :return tuple of the new state, reward, whether episode is done,
        and dictionary with additional info (not used at the moment). """
        err_msg = f"{action} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        x = self.state

        # Apply action and update environment (get new position at BPM and
        # convert into a binary vector)
        total_angle = self.mssb_angle + self.action_map[action]
        x_new, intensity = self.get_pos_at_bpm_target(total_angle)

        self.state = np.array([x_new])
        self.mssb_angle = total_angle

        self.step_count += 1

        # Is episode done?
        done = bool(
            self.step_count > self.max_steps_per_episode
            or intensity > self.reward_threshold
            or x_new > self.observation_space.high
            or x_new < self.observation_space.low
        )

        # Keep track of reason for episode abort
        done_reason = -1
        if self.step_count > self.max_steps_per_episode:
            done_reason = 0
        elif intensity > self.reward_threshold:
            done_reason = 1
        elif (x_new > self.observation_space.high or
              x_new < self.observation_space.low):
            done_reason = 2
        else:
            pass

        reward = self.get_reward(intensity)

        # Interaction log
        self.interaction_logger.log_episode.append(
            [x, action, reward, x_new, done, done_reason])
        if done:
            self.interaction_logger.log_all.append(
                self.interaction_logger.log_episode)

        return self.state, reward, done, {}

    def _init_random_state(self, init_outside_threshold: bool):
        """
        Initialize environment in random state (= BPM position).
        :param init_outside_threshold: bool flag. If True, will only accept
        random initial state outside of intensity threshold. If False accept
        any initial state.
        :return: corresponding dipole strength for initial state
        """
        mssb_angle = None
        init_intensity = 1.1 * self.reward_threshold
        while init_intensity > 0.8*self.reward_threshold:
            mssb_angle = np.random.uniform(low=self.mssb_angle_min,
                                           high=self.mssb_angle_max)
            x_init, init_intensity = self.get_pos_at_bpm_target(mssb_angle)

            if not init_outside_threshold:
                break

        return mssb_angle

    def _init_specific_state(self, init_state: float):
        """
        Alternative way to initialize environment, but to a specific state.
        We have to calc. the mssb_angle that puts env in that state (the way
        that's done is a bit messy ...)
        :param init_state:
        :return:
        """
        # TODO: this needs major cleanup
        x_pos_array = np.zeros(1000)
        mssb_array = np.linspace(
            self.mssb_angle_min - self.mssb_delta,
            self.mssb_angle_max + self.mssb_delta,
            len(x_pos_array))

        for i, mssb in enumerate(mssb_array):
            x_pos_array[i], _ = self.get_pos_at_bpm_target(mssb)
        idx = np.argmin(np.abs(init_state - x_pos_array))
        return mssb_array[idx]

    def reset(self, init_state: float = None,
              init_outside_threshold: bool = False) -> np.ndarray:
        """
        Reset the environment. Initialize self.mssb_angle, get the initial
        state, and reset logger. This method gets called e.g. at the end of
        an episode or at the very beginning of a training.
        :param init_state: if not None, sets environment into specific state.
        :param init_outside_thresh: bool flag. If True, will only accept
        random initial state outside of intensity threshold. If False accept
        any initial state.
        :return Initial state as np.ndarray
        """
        if init_state is None:
            self.mssb_angle = self._init_random_state(init_outside_threshold)
        else:
            self.mssb_angle = self._init_specific_state(init_state)

        x_init, _ = self.get_pos_at_bpm_target(self.mssb_angle)
        self.state = np.array([x_init])

        # Logging
        self.step_count = 0
        self.interaction_logger.episode_reset()

        return self.state

    def clear_log(self) -> None:
        """ Delete all log / history of the logger of this environment. """
        self.interaction_logger.clear_all()

    def _get_integrated_intensity(self, beam_position: float) -> float:
        """ Calculate integrated intensity given beam position on target
        assuming Gaussian beam (integration range +/- 3 sigma).
        :param beam_position: beam position on target (not known to RL agent)
        :return: integrated intensity, float in range ~[0, 1]. """
        emittance = 1.1725E-08
        sigma = math.sqrt(self.target.beta * emittance)
        self.intensity_on_target = quad(
            lambda x: 1 / (sigma * (2 * math.pi) ** 0.5) * math.exp(
                (x - beam_position) ** 2 / (-2 * sigma ** 2)),
            -3*sigma, 3*sigma)

        return self.intensity_on_target[0]

    def get_reward(self, intensity: float) -> float:
        """
        Calculate reward from integrated intensity. Give additional higher
        reward when episode is finished.
        :param intensity: integrated intensity on target
        :return corresponding reward
        """
        return -100. * (1. - intensity)

    def get_pos_at_bpm_target(self, total_angle: float) -> Tuple[float, float]:
        """ Transports beam through the transfer line and calculates the
        position at the BPM and at the target. These are required for the
        intensity calculation and to get the state based on the currently set
        dipole angle.
        :param total_angle: total kick angle of the MSSB dipole (rad)
        :return position at BPM and intensity as a tuple """
        x_bpm, px_bpm = transport(self.mssb, self.bpm1, self.x0, total_angle)
        x_target, px_target = transport(
            self.mssb, self.target, self.x0, total_angle)

        intensity = self._get_integrated_intensity(x_target)
        return x_bpm, intensity

    def get_max_intensity(self) -> float:
        """ Calculate maximum integrated intensity. This is used to define the
        threshold for cancellation of an episode. Note that this is potentially
        not known for all environments.
        :return maximum integrated intensity """
        _, _, intensities = self.get_response()
        return np.max(intensities)

    def get_response(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Calculate response of the environment, i.e. the x_position and
        reward dependence on the dipole kick angle (for the full available
        range of angles).
        :return Tuple of np.ndarrays containing dipole angles, x positions,
        and rewards. """
        angles = np.linspace(self.mssb_angle_min, self.mssb_angle_max, 1000)
        x_pos = np.zeros_like(angles)
        intensities = np.zeros_like(angles)
        for i, ang in enumerate(angles):
            x, intens = self.get_pos_at_bpm_target(total_angle=ang)
            x_pos[i] = x
            intensities[i] = intens
        return angles, x_pos, intensities

    def get_max_n_steps_optimal_behaviour(self) -> int:
        """ Calculate maximum number of steps required to solve the problem
        from any initial condition assuming optimal behaviour of agent. We
        take into account the intensity threshold (above which the problem is
        assumed to be solved).
        :return upper bound for number of required steps (int). """
        _, x, intensity = self.get_response()
        x1, _ = self.get_pos_at_bpm_target(self.mssb_angle_min)
        x2, _ = self.get_pos_at_bpm_target(self.mssb_angle_min +
                                           self.mssb_delta)
        x_delta = x2 - x1
        idx = np.where(intensity > self.reward_threshold)[0][-1]
        x_up_intensity_threshold = x[idx]
        return np.ceil(
            (self.observation_space.high - x_up_intensity_threshold)
            / x_delta - 1)
