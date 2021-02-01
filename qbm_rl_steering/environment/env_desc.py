import gym

import math
from scipy.integrate import quad
from math import pi, exp

class twissElement():

    def __init__(self,beta,alpha,d,mu):

        self.beta = beta
        self.alpha = alpha
        self.mu = mu

def transport(element1,element2,x,px):
    mu = element2.mu - element1.mu
    alpha1 = element1.alpha
    alpha2 = element2.alpha
    beta1 = element1.beta
    beta2 = element2.beta


    m11 = math.sqrt(beta2/beta1)*(math.cos(mu)+alpha1*math.sin(mu))
    m12 = math.sqrt(beta1*beta2)*math.sin(mu)
    m21 = ((alpha1-alpha2)*math.cos(mu)-(1+alpha1*alpha2)*math.sin(mu))/math.sqrt(beta1*beta2)
    m22 = math.sqrt(beta1/beta2)*(math.cos(mu)-alpha2*math.sin(mu))

    return m11*x+m12*px, m21*x+m22*px



class TargetSteeringEnv(gym.Env):
    def __init__(self):
        self.x0 = 0.
        self.mssb = twissElement(16.1, -0.397093117, 0.045314011, 1.46158005)
        self.bpm1 = twissElement(339.174497, -6.521184683, 2.078511443, 2.081365696)
        self.target = twissElement(7.976311944, -0.411639485, 0.30867161, 2.398031982)

        self.mssb_angle = 0. #radian: will have to be randomly set; e.g. some multiples of a delta
        self.delta = 5e-5 # 50 urad as delta




    def step(self, action):
        pass

    def reset(self):
        pass


    def _get_reward(self,beam_pos):

        emittance = 1.1725E-08
        sigma = math.sqrt(self.target.beta*emittance)
        self.intensity_on_target = quad(lambda x: 1 / (sigma * (2 * pi) ** 0.5) * exp((x-beam_pos) ** 2 / (-2 * sigma ** 2)), -3*sigma,3*sigma)

        reward = self.intensity_on_target[0]
        return reward

    #
    def _get_pos_at_bpm_target(self,total_angle):
        x_bpm, px_bpm = transport(self.mssb, self.bpm1, self.x0,total_angle)
        x_target,px_target = transport(self.mssb, self.target, self.x0,total_angle)

        reward = self._get_reward(x_target)
        return x_bpm,reward


