from typing import Tuple, Dict

import numpy as np
import math
import gym

from qbm_rl_steering.environment.logger import Logger

# TODO: implement target trajectory, rather than assuming 0 traj. - does it
#  make a difference?


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

state_scale_correction = np.array([1.05, 1.04, 1.01, 1.00, 0.94, 1.15, 2.55,
                                   2.58, 2.53, 2.58])


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


class RmsSteeringEnv(gym.Env):
    def __init__(self, n_dims: int = 2, max_steps_per_episode: int = 20,
                 reward_scale: float = 1e6, kick_angle_max: float = 300e-6,
                 required_steps_above_reward_threshold: int = 4) -> None:
        """
        :param n_dims: choose how many dimensions (#kickers and #bpms to use)
        :param max_steps_per_episode: max number of steps we allow agent to
        'explore' per episode. After this number of steps, episode is aborted.
        :param reward_scale: scalar to scale up reward
        :param kick_angle_max: maximum kick angle per corrector (m rad)
        :param required_steps_above_reward_threshold: number of steps we want
        agent to stay above reward objective before ending episode. Can be > 1
        during training, but typically == 1 during evaluation.
        """
        super(RmsSteeringEnv, self).__init__()

        self.n_dims = n_dims
        self.reward_scale = reward_scale
        if self.n_dims > 10:
            raise ValueError("n_dims cannot be larger than 10.")

        # DEFINE TRANSFER LINE (flexible dimension)
        self.kickers = []
        self.bpms = []

        for i in range(self.n_dims):
            self.kickers.append(TwissElement(
                kicker_betx[i], kicker_alfx[i], 2 * np.pi * kicker_mux[i]))
            self.bpms.append(TwissElement(
                bpm_betx[i], bpm_alfx[i], 2 * np.pi * bpm_mux[i]))

        # KICKERS
        self.kick_angles = [None] * self.n_dims  # will be init. at self.reset()
        self.kick_angle_max = kick_angle_max  # (rad)
        self.kick_angle_min = -self.kick_angle_max  # (rad)

        # TODO: get rid of margin. Not really used anymore
        # self.kick_angle_margin = 0.1 * self.kick_angle_max  # (rad)

        # Use same action_scale for all dipoles
        self.action_scale = 2. / (self.kick_angle_max - self.kick_angle_min)

        # BEAM POSITION
        # x0: position at origin, i.e. before entering first kicker
        # state: position at BPMs (observation / state)
        # x_min, x_max: possible range of observations to define
        # observation_space given kick_angle_min, kick_angle_max
        self.x0 = 0.
        self.state = [None] * self.n_dims  # will be init. with self.reset()
        max_angle = self.kick_angle_max  # + self.kick_angle_margin
        x_max_margin = self.calculate_state([max_angle] * self.n_dims)

        min_angle = self.kick_angle_min  # - self.kick_angle_margin
        x_min_margin = self.calculate_state([min_angle] * self.n_dims)

        self.state_scale = 2. / (np.max(x_max_margin) - np.min(x_min_margin))
        self.state_scale /= state_scale_correction[n_dims - 1]

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

        # Reward threshold is adjusted such that we require 0.0016 m at 10-D,
        # as in the original AWAKE environment. In n-D the reward_threshold is
        # scaled linearly with n. Hence at 1-D, threshold will be  0.00016 m.
        self.reward_threshold = -self.reward_scale * self.n_dims * 1.6e-4
        self.required_steps_above_reward_threshold = \
            required_steps_above_reward_threshold
        self.steps_above_reward_threshold = None
        self.max_steps_above_reward_threshold = None
        # self.cancel_on_reward = self.reward_threshold * 20

        # Logging
        self.interaction_logger = Logger()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """ Perform one step in the environment (take an action, update
        parameters in environment, receive reward, check if episode ends,
        append all info to logger, return new state, reward, etc.
        :param action: continuous action, in normalized units [-1, 1].
        :return tuple of the new state, reward, whether episode is done,
        and dictionary with additional info (not used at the moment). """

        # Note: self.state is in normalized units, i.e. in [-1, 1],
        # while self.kick_angles is in (m rad). The action that comes in is
        # again in normalized units. Hence we scale the action parameter by
        # 1/self.action_scale before adding it to self.kick_angles.
        state = self.state

        # Apply action and update environment (get new position at BPMs) and
        # bring new_state to normalized state space.
        total_kick_angles = self.kick_angles + action / self.action_scale
        new_state = self.calculate_state(total_kick_angles)
        self.kick_angles = total_kick_angles
        self.state = new_state * self.state_scale

        self.step_count += 1

        # Is episode done?
        # Can be either because 1) reached max number of steps per episode,
        # 2) reached reward target and stayed inside for a while,
        # or 3) kick angles went out of bounds...
        # Note that reward is calculated using the non-normalized state,
        # i.e. units of (m).
        reward = self.calculate_reward(new_state)

        # Keep stats for how many consecutive steps agent has stayed within
        # reward objective at any point during episode
        if reward > self.reward_threshold:
            self.steps_above_reward_threshold += 1
        else:
            self.steps_above_reward_threshold = 0

        self.max_steps_above_reward_threshold = max(
            self.steps_above_reward_threshold,
            self.max_steps_above_reward_threshold)

        # Is episode over?
        done = bool(
            self.step_count >= self.max_steps_per_episode
            or (self.steps_above_reward_threshold >=
                self.required_steps_above_reward_threshold)
            # or any([angle < 10. * (self.kick_angle_min - self.kick_angle_margin)
            #         for angle in self.kick_angles])
            # or any([angle > 10. * (self.kick_angle_max + self.kick_angle_margin)
            #         for angle in self.kick_angles])
            # or reward < self.cancel_on_reward
        )

        # Why did episode end?
        done_reason = 'NA'
        if self.step_count >= self.max_steps_per_episode:
            done_reason = 'MaxSteps'
        elif (self.steps_above_reward_threshold >=
              self.required_steps_above_reward_threshold):
            done_reason = 'RewardObjective'
        # elif (any([angle < 10. * (self.kick_angle_min - self.kick_angle_margin)
        #            for angle in self.kick_angles]) or
        #       any([angle > 10. * (self.kick_angle_max + self.kick_angle_margin)
        #            for angle in self.kick_angles])):
        #     done_reason = 'OutOfBounds'

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
        Initialize environment in random state (= BPM positions) by
        initializing random corrector kicks.
        :param init_outside_threshold: bool flag. If True, will only accept
        random initial state outside of reward threshold. If False accept
        any initial state. In case set to True, will iteratively try out
        random kick angles until condition is satisfied. If False,
        break after first iteration in any case.
        :return: corresponding dipole strengths (kick angles) for initial state.
        """
        self.kick_angles = [None] * self.n_dims
        temp_angles = None
        temp_reward = 0
        while temp_reward > 1.2 * self.reward_threshold:
            temp_angles = np.random.uniform(
                low=self.kick_angle_min, high=self.kick_angle_max,
                size=self.n_dims)
            x_init = self.calculate_state(temp_angles)
            temp_reward = self.calculate_reward(x_init)

            if not init_outside_threshold:
                break
        return temp_angles

    def _init_specific_reward(self, init_reward: float, margin: float = 5.):
        self.kick_angles = [None] * self.n_dims
        temp_angles = None
        temp_reward = 1e39
        while abs(temp_reward - init_reward) > margin:
            temp_angles = np.random.uniform(
                low=self.kick_angle_min, high=self.kick_angle_max,
                size=self.n_dims)
            x_init = self.calculate_state(temp_angles)
            temp_reward = self.calculate_reward(x_init)
        return temp_angles

    def reset(self, init_outside_threshold: bool = False,
              init_specific_reward_state: float = None) -> np.ndarray:
        """
        Reset the environment: initialize self.kick_angles, get initial
        state, step_count to 0, and reset logger. This method gets called
        at beginning of new episode.
        :param init_outside_threshold: bool flag. If True, will only accept
        random initial state outside of reward threshold. If False accept
        any initial state. See _init_random_state(..) method.
        :param init_specific_reward_state: can specify a specific reward that
        you want the env to start in (note that the state is typically not well
        defined.
        :return Initial state as np.ndarray of dim self.n_dims.
        """
        if init_specific_reward_state:
            self.kick_angles = self._init_specific_reward(
                init_specific_reward_state)
        else:
            self.kick_angles = self._init_random_state(init_outside_threshold)

        x_init = self.calculate_state(self.kick_angles)
        self.state = x_init * self.state_scale

        # Logging and counters
        self.step_count = 0
        self.steps_above_reward_threshold = 0
        self.max_steps_above_reward_threshold = 0
        self.interaction_logger.episode_reset()

        # Note that reward is calculated using the non-normalized state,
        # i.e. units of (m).
        init_reward = self.calculate_reward(x_init)
        self.interaction_logger.log_episode.append(
            [x_init, None, init_reward, None, False, None])

        return self.state

    def clear_log(self) -> None:
        """ Delete all log / history of the logger of this environment. """
        self.interaction_logger.clear_all()

    def calculate_reward(self, state: np.ndarray) -> float:
        """ Calculate reward as rms of trajectory / state (i.e. positions at
        BPMs).
        :param state: non-normalized positions at BPMs (units (m)).
        :return corresponding scaled reward (scaled negative rms).
        """
        return -self.reward_scale * np.sqrt(np.mean(np.array(state) ** 2))

    def calculate_state(self, kick_angles) -> np.ndarray:
        """ Transports beam through the transfer line and calculates the
        position at BPMs, using currently set dipole corrector angles.
        :param kick_angles: list of kicks (rad) for all the dipoles in the line
        :return position at BPMs, non-normalized (= state). """
        state = []
        xi, pxi = self.x0, 0
        for i in range(len(kick_angles) - 1):
            bpm_x, bpm_px = transport(self.kickers[i], self.bpms[i],
                                      xi, pxi + kick_angles[i])
            state.append(bpm_x)
            xi, pxi = transport(self.bpms[i], self.kickers[i + 1],
                                bpm_x, bpm_px)

        # Transport from last kicker to last BPM
        bpm_x, bpm_px = transport(self.kickers[-1], self.bpms[-1],
                                  xi, pxi + kick_angles[-1])
        state.append(bpm_x)
        return np.array(state)
