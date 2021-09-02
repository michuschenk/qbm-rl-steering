from typing import Tuple, Dict

import numpy as np
import math
from scipy.integrate import quad
import gym

from qbm_rl_steering.environment.logger import Logger

# AWAKE BEAM LINE DATA: BPMs
bpm_alfx = [-1.349372930, 2.312786519, -2.744905014,
            1.593613746, -5.291911960, -44.054983693, -102.304054653,
            87.910598503, 11.848629081, 7.185660896]
bpm_betx = [5.583291691, 6.584555767, 5.423499998, 9.661065623,
            19.449279543, 33.935518869, 156.835824893, 92.460419437,
            34.764047558, 12.941234099]
bpm_mux = [0.036705998, 0.063892908, 0.111230584, 0.131370180,
           0.150838314, 0.644947045, 0.646806477, 0.647676457, 0.650588345,
           0.659195366]

# AWAKE BEAM LINE DATA: KICKERs
kicker_alfx = [-0.057834000, -1.404947459, 2.199008181, -2.918002716,
               1.553312133, -5.455952820, -50.463792668, -109.778860845,
               78.547904519, 11.393110165]
kicker_betx = [5.016723858, 5.886266934, 6.052163992, 6.046419848, 9.314903776,
               20.631544669, 44.521621821, 180.589111428, 73.817067099,
               32.160972762]
kicker_mux = [0.009194325, 0.039760035, 0.066868057, 0.114287972, 0.133215709,
              0.151712285, 0.645405637, 0.646912395, 0.647892223, 0.651121445]

# Mu target: kind of calibrated (note with number of kickers, we have more
# freedom to go completely off target ... trying to compensate with distance
# of last kicker from target or by adjusting magnet strength range depending
# on how many there kickers we use.
# TODO: to be discussed...
# target_delta_mux = {1: 0.3, 2: 0.25, 3: 0.2, 4: 0.15, 5: 0.1}


class TwissElement:
    def __init__(self, beta: float, alpha: float, mu: float) -> None:
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


class TargetSteeringEnvND(gym.Env):
    def __init__(self, n_dims=2, max_steps_per_episode: int = 20) -> None:
        """
        :param n_dims: choose how many dimensions (#kickers and #bpms to use)
        :param max_steps_per_episode: max number of steps we allow agent to
        'explore' per episode. After this number of steps, episode is aborted.
        """
        super(TargetSteeringEnvND, self).__init__()

        self.n_dims = n_dims
        if self.n_dims > 10:
            raise ValueError("n_dim cannot be larger than 10.")

        # DEFINE TRANSFER LINE (flexible dimension)
        self.kickers = []
        self.bpms = []

        for i in range(self.n_dims):
            self.kickers.append(TwissElement(
                kicker_betx[i], kicker_alfx[i], 2 * np.pi * kicker_mux[i]))
            self.bpms.append(TwissElement(
                bpm_betx[i], bpm_alfx[i], 2 * np.pi * bpm_mux[i]))

        target_mux = self.bpms[-1].mu + 2. * np.pi * 0.07
        self.target = TwissElement(7.976311944, -0.411639485, target_mux)

        # KICKERS
        self.kick_angles = [None] * self.n_dims  # will be init. at self.reset()
        self.kick_angle_max = 200e-6 - self.n_dims * 15e-6  # 100e-6  # (rad)
        self.kick_angle_min = -self.kick_angle_max  # (rad)
        self.kick_angle_margin = 0.1*self.kick_angle_max  # (rad)

        # Use same action_scale for all dipoles
        self.action_scale = (2. / (self.kick_angle_max - self.kick_angle_min +
                                   2 * self.kick_angle_margin))

        # BEAM POSITION
        # x0: position at origin, i.e. before entering first kicker
        # state: position at BPMs (observation / state)
        # x_min, x_max: possible range of observations to define
        # observation_space given kick_angle_min, kick_angle_max
        self.x0 = 0.
        self.state = [None] * self.n_dims  # will be init. with self.reset()
        max_angle = self.kick_angle_max + self.kick_angle_margin
        x_max_margin, _ = self.get_pos_at_bpm_target([max_angle] * self.n_dims)

        min_angle = self.kick_angle_min - self.kick_angle_margin
        x_min_margin, _ = self.get_pos_at_bpm_target([min_angle] * self.n_dims)

        self.state_scale = 2. / (np.max(x_max_margin) - np.min(x_min_margin))

        # GYM REQUIREMENTS
        # Define continuous action space
        self.action_space = gym.spaces.Box(
            low=np.array([min_angle] * self.n_dims) * self.action_scale,
            high=np.array([max_angle] * self.n_dims) * self.action_scale)

        # Define continuous observation space
        self.observation_space = gym.spaces.Box(
            low=x_min_margin * self.state_scale,
            high=x_max_margin * self.state_scale)

        # For cancellation when beyond certain number of steps in an episode
        self.step_count = None
        self.max_steps_per_episode = max_steps_per_episode
        self.intensity_threshold = 0.9

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

        # Apply action and update environment (get new position at BPMs)
        total_kick_angles = self.kick_angles + action / self.action_scale
        new_state, intensity = self.get_pos_at_bpm_target(total_kick_angles)
        self.kick_angles = total_kick_angles

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
        self.kick_angles = [None] * self.n_dims
        temp_angles = None
        init_intensity = 1.1 * self.intensity_threshold
        while init_intensity > 0.8 * self.intensity_threshold:
            temp_angles = np.random.uniform(
                low=self.kick_angle_min, high=self.kick_angle_max,
                size=self.n_dims)
            x_init, init_intensity = self.get_pos_at_bpm_target(temp_angles)
            if not init_outside_threshold:
                break

        return temp_angles

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
            self.kick_angles = self._init_random_state(init_outside_threshold)

        x_init, init_intensity = self.get_pos_at_bpm_target(self.kick_angles)
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

    def get_pos_at_bpm_target(self, kick_angles) -> Tuple:
        """ Transports beam through the transfer line and calculates the
        position at BPMs and target. These are required for the intensity
        calculation and to get the state based on the currently set dipole
        angles.
        :param kick_angles: list of kicks (rad) for all the dipoles in the line
        :return position at BPMs (=state) and intensity (~reward) as a tuple """
        state = []
        xi, pxi = self.x0, 0
        for i in range(len(kick_angles) - 1):
            bpm_x, bpm_px = transport(self.kickers[i], self.bpms[i],
                                      xi, pxi + kick_angles[i])
            state.append(bpm_x)
            xi, pxi = transport(self.bpms[i], self.kickers[i+1],
                                bpm_x, bpm_px)

        # Transport from last kicker to last BPM
        bpm_x, bpm_px = transport(self.kickers[-1], self.bpms[-1],
                                  xi, pxi + kick_angles[-1])
        state.append(bpm_x)

        # Transport from last BPM to target
        target_x, target_px = transport(self.bpms[-1], self.target,
                                        bpm_x, bpm_px)

        intensity = self._get_integrated_intensity(target_x)
        state = np.array(state)
        return state, intensity
