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


class TargetSteeringEnv2D(gym.Env):
    def __init__(self, max_steps_per_episode: int = 20) -> None:
        """
        :param max_steps_per_episode: max number of steps we allow agent to
        'explore' per episode. After this number of steps, episode is aborted.
        :param action_scale: scaling factor for the action. """
        super(TargetSteeringEnv2D, self).__init__()

        # DEFINE TRANSFER LINE
        self.mssb = TwissElement(16.1, -0.397093117, 0.045314011,
                                       1.46158005)
        self.mbb = TwissElement(8.88303879, -0.374085208, 0.057623602,
                                      1.912555325)
        self.bpm1 = TwissElement(339.174497, -6.521184683, 2.078511443,
                                       2.081365696)
        self.bpm2 = TwissElement(30.82651983, 1.876067844, 0.929904474,
                                       2.163823492)
        self.target = TwissElement(7.976311944, -0.411639485, 0.30867161,
                                         2.398031982)

        # MSSB DIPOLE / KICKER
        # mssb_angle: dipole kick angle (rad)
        # mssb_min, mssb_max: min. and max. dipole strengths for init. (rad)
        self.mssb_angle = None  # not set, will be init. with self.reset()
        self.mssb_angle_max = 140e-6  # 140e-6  # (rad)
        self.mssb_angle_min = -self.mssb_angle_max  # (rad)
        self.mssb_angle_margin = 30e-6

        self.mbb_angle = None  # not set, will be init. with self.reset()
        self.mbb_angle_max = 140e-6  # 140e-6  # (rad)
        self.mbb_angle_min = -self.mbb_angle_max  # (rad)
        self.mbb_angle_margin = 30e-6

        # Use same action_scale for mbb and mssb
        self.action_scale = (2. / (self.mssb_angle_max - self.mssb_angle_min +
                                   2 * self.mssb_angle_margin))

        # BEAM POSITION
        # x0: position at origin, i.e. before entering MSSB
        # state: position at BPM(observation / state)
        # x_min, x_max: possible range of observations to define
        # observation_space given mssb_angle_min, mssb_angle_max
        self.x0 = 0.
        self.state = None  # not set, will be init. with self.reset()
        mssb_angle = self.mssb_angle_max + self.mssb_angle_margin
        mbb_angle = self.mbb_angle_max + self.mbb_angle_margin

        x_max_wMargin, _ = self.get_pos_at_bpm_target(mssb_angle, mbb_angle)

        mssb_angle = self.mssb_angle_min - self.mssb_angle_margin
        mbb_angle = self.mbb_angle_min - self.mbb_angle_margin
        x_min_wMargin, _ = self.get_pos_at_bpm_target(mssb_angle, mbb_angle)

        self.state_scale = 2. / (np.max(x_max_wMargin) - np.min(x_min_wMargin))
        print('self.state_scale', self.state_scale)


        # GYM REQUIREMENTS
        # Define continuous action space
        self.action_space = gym.spaces.Box(
            low=np.array([self.mssb_angle_min - self.mssb_angle_margin,
                          self.mbb_angle_min - self.mbb_angle_margin]) *
            self.action_scale,
            high=np.array([self.mssb_angle_max + self.mssb_angle_margin,
                           self.mbb_angle_max + self.mbb_angle_margin]) *
            self.action_scale)

        # Define continuous observation space
        self.observation_space = gym.spaces.Box(
            low=x_min_wMargin * self.state_scale,
            high=x_max_wMargin * self.state_scale)

        # For cancellation when beyond certain number of steps in an episode
        self.step_count = None
        self.max_steps_per_episode = max_steps_per_episode
        self.intensity_threshold = 0.85

        # Logging
        self.interaction_logger = Logger()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """ Perform one step in the environment (take an action, update
        parameters in environment, receive reward, check if episode ends,
        append all info to logger, return new state, reward, etc.
        :param action: continuous action
        :return tuple of the new state, reward, whether episode is done,
        and dictionary with additional info (not used at the moment). """
        state = self.state

        # Apply action and update environment (get new position at BPM and
        # convert into a binary vector)
        total_mssb_angle = self.mssb_angle + action[0] / self.action_scale
        total_mbb_angle = self.mbb_angle + action[1] / self.action_scale

        new_state, intensity = self.get_pos_at_bpm_target(total_mssb_angle,
                                                          total_mbb_angle)

        self.mssb_angle = total_mssb_angle
        self.mbb_angle = total_mbb_angle
        # x1_new, x2_new = new_state

        self.state = new_state * self.state_scale
        self.step_count += 1

        # Is episode done?
        done = bool(
            self.step_count > self.max_steps_per_episode
            or intensity > self.intensity_threshold
        )

        # Keep track of reason for episode abort
        done_reason = -1

        reward = self.get_reward(intensity)

        # Interaction log
        self.interaction_logger.log_episode.append(
            [state, action / self.action_scale, reward, new_state, done,
             done_reason])
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
        self.mssb_angle = None
        self.mbb_angle = None
        init_intensity = 1.1 * self.intensity_threshold
        while init_intensity > 0.8 * self.intensity_threshold:
            mssb_angle = np.random.uniform(low=self.mssb_angle_min,
                                           high=self.mssb_angle_max)
            mbb_angle = np.random.uniform(low=self.mbb_angle_min,
                                           high=self.mbb_angle_max)
            x_init, init_intensity = self.get_pos_at_bpm_target(
                mssb_angle, mbb_angle
            )
            if not init_outside_threshold:
                break

        return mssb_angle, mbb_angle

    def reset(self, init_state: float = None,
              init_outside_threshold: bool = False) -> np.ndarray:
        """
        Reset the environment. Initialize self.mssb_angle, get the initial
        state, and reset logger. This method gets called e.g. at the end of
        an episode or at the very beginning of a training.
        :param init_state: if not None, sets environment into specific state.
        :param init_outside_threshold: bool flag. If True, will only accept
        random initial state outside of intensity threshold. If False accept
        any initial state.
        :return Initial state as np.ndarray
        """
        if init_state is None:
            self.mssb_angle, self.mbb_angle = self._init_random_state(
                init_outside_threshold)

        x_init, init_intensity = self.get_pos_at_bpm_target(self.mssb_angle,
                                                            self.mbb_angle)
        self.state = x_init * self.state_scale

        # Logging
        self.step_count = 0
        self.interaction_logger.episode_reset()

        init_reward = self.get_reward(init_intensity)
        self.interaction_logger.log_episode.append(
            [x_init, None, init_reward, None, False, None])

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
            -3 * sigma, 3 * sigma)

        return self.intensity_on_target[0]

    def get_reward(self, intensity: float) -> float:
        """
        Calculate reward from integrated intensity. Give additional higher
        reward when episode is finished.
        :param intensity: integrated intensity on target
        :return corresponding reward
        """
        return -100. * (1. - intensity)

    def get_pos_at_bpm_target(self, mssb_angle, mbb_angle) -> Tuple:
        """ Transports beam through the transfer line and calculates the
        position at the BPM and at the target. These are required for the
        intensity calculation and to get the state based on the currently set
        dipole angle.
        :param total_angle: total kick angle of the MSSB dipole (rad)
        :return position at BPM and intensity as a tuple """
        # x_bpm, px_bpm = transport(self.mssb, self.bpm1, self.x0, total_angle)
        # x_target, px_target = transport(
        #     self.mssb, self.target, self.x0, total_angle)

        x1, px1 = transport(self.mssb, self.mbb, self.x0, mssb_angle)
        px1 += mbb_angle

        bpm1_x, bpm1_px = transport(self.mbb, self.bpm1, x1, px1)
        bpm2_x, bpm2_px = transport(self.bpm1, self.bpm2, bpm1_x, bpm1_px)
        target_x, target_px = transport(self.bpm2, self.target, bpm2_x, bpm2_px)
        intensity = self._get_integrated_intensity(target_x)
        state = np.array([bpm1_x, bpm2_x])
        return state, intensity