import gym

import math
from scipy.integrate import quad
from math import pi, exp
import numpy as np


class TwissElement():

    def __init__(self, beta, alpha, d, mu):
        self.beta = beta
        self.alpha = alpha
        self.mu = mu


def transport(element1, element2, x, px):
    mu = element2.mu - element1.mu
    alpha1 = element1.alpha
    alpha2 = element2.alpha
    beta1 = element1.beta
    beta2 = element2.beta

    m11 = math.sqrt(beta2 / beta1) * (math.cos(mu) + alpha1 * math.sin(mu))
    m12 = math.sqrt(beta1 * beta2) * math.sin(mu)
    m21 = ((alpha1 - alpha2) * math.cos(mu) - (1 + alpha1 * alpha2) * math.sin(mu)) / math.sqrt(beta1 * beta2)
    m22 = math.sqrt(beta1 / beta2) * (math.cos(mu) - alpha2 * math.sin(mu))

    return m11 * x + m12 * px, m21 * x + m22 * px


class TargetSteeringEnv(gym.Env):
    def __init__(self):
        """
        x0: beam position at origin, i.e. before entering MSSB
        state: beam position at BPM (observation / state)
        mssb_angle: dipole kick angle (rad)
        mssb_min, mssb_max: min. and max. dipole strengths for initialization (rad)
        mssb_delta: amount of discrete change in mssb_angle upon action (rad) """
        super(TargetSteeringEnv, self).__init__()

        # MSSB (dipole) kicker
        self.mssb_angle = None  # not set, will be init. with self.reset()
        self.mssb_angle_min = -400e-6  # rad, BPM pos. of ~ -16 mm
        self.mssb_angle_max = 400e-6  # rad, BPM pos. of ~ +16 mm
        self.mssb_delta = 5e-5  # 50 urad as delta

        # Beam position
        self.x0 = 0.
        self.state = None  # not set, will be init. with self.reset()

        # Define transfer line
        self.mssb = TwissElement(16.1, -0.397093117, 0.045314011, 1.46158005)
        self.bpm1 = TwissElement(339.174497, -6.521184683, 2.078511443, 2.081365696)
        self.target = TwissElement(7.976311944, -0.411639485, 0.30867161, 2.398031982)

        # Get possible range of observations to define observation_space
        self.x_min, _ = self._get_pos_at_bpm_target(self.mssb_angle_min)
        self.x_max, _ = self._get_pos_at_bpm_target(self.mssb_angle_max)

        # Gym requirements
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            np.array([self.x_min]), np.array([self.x_max]), dtype=np.float32)
        self.steps_beyond_done = None

        self.log = []

        # First initialization
        self.reset()

    def step(self, action):
        """ Action is discrete here and is an integer number in set {0, 1, 2}.
        Action_map shows how action is mapped to change of self.mssb_angle. """

        err_msg = f"{action} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        x, = self.state

        # Apply action and update environment
        action_map = {0: 0, 1: self.mssb_delta, 2: -self.mssb_delta}
        total_angle = self.mssb_angle + action_map[action]
        x_new, reward = self._get_pos_at_bpm_target(total_angle)

        self.state = np.array([x_new])
        self.mssb_angle = total_angle

        # Done?
        # TODO: introduce max number of episodes? e.g. if not successful after 30 steps
        # TODO: target: if intensity high enough, cancel as well.
        done = bool(
            x_new > self.x_max
            or x_new < self.x_min)

        # Keep history
        self.log.append([x, action, reward, x_new, done])

        # from cartpole.py
        if not done:
            pass
        elif self.steps_beyond_done is None:
            # Beam outside allowed position
            self.steps_beyond_done = 0
        else:
            if self.steps_beyond_done == 0:
                gym.logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.

        return self.state, reward, done, {}

    def reset(self):
        pass


    def _get_reward(self, beam_pos):
        emittance = 1.1725E-08
        sigma = math.sqrt(self.target.beta * emittance)
        self.intensity_on_target = quad(
            lambda x: 1 / (sigma * (2 * pi) ** 0.5) * exp((x - beam_pos) ** 2 / (-2 * sigma ** 2)), -3 * sigma,
            3 * sigma)

        reward = self.intensity_on_target[0]
        return reward

    def _get_pos_at_bpm_target(self, total_angle):
        x_bpm, px_bpm = transport(self.mssb, self.bpm1, self.x0, total_angle)
        x_target, px_target = transport(self.mssb, self.target, self.x0, total_angle)

        reward = self._get_reward(x_target)
        return x_bpm, reward
