import os

import datetime
import shutil
import pickle

import numpy as np
import matplotlib

from tensorflow.keras.optimizers.schedules import PolynomialDecay, PiecewiseConstantDecay
from qbm_rl_steering.core.ddpg_agents import ClassicalDDPG, QuantumDDPG
from qbm_rl_steering.environment.rms_env_nd import RmsSteeringEnv
from qbm_rl_steering.environment.orig_awake_env import e_trajectory_simENV

from qbm_rl_steering.core.run_utils import trainer, evaluator, plot_training_log, plot_evaluation_log
import pandas as pd


def get_val(params):

    env = e_trajectory_simENV()

    # Learning rate schedules: lr_critic = 5e-4, lr_actor = 1e-4
    lr_schedule_critic = PolynomialDecay(params['lr_critic/init'],
                                         params['n_steps'],
                                         end_learning_rate=params['lr/final'])
    lr_schedule_actor = PolynomialDecay(params['lr_actor/init'],
                                        params['n_steps'],
                                        end_learning_rate=params['lr/final'])

    if params['quantum_ddpg']:
        agentMy = QuantumDDPG(state_space=env.observation_space,
                              action_space=env.action_space,
                              learning_rate_schedule_critic=lr_schedule_critic,
                              learning_rate_schedule_actor=lr_schedule_actor,
                              grad_clip_actor=1e4, grad_clip_critic=1.,
                              # grad_clip_actor=np.inf, grad_clip_critic=np.inf,
                              gamma=params['agent/gamma'],
                              tau_critic=params['agent/tau_critic'],
                              tau_actor=params['agent/tau_actor'],
                              )
    else:
        agentMy = ClassicalDDPG(state_space=env.observation_space,
                                action_space=env.action_space,
                                learning_rate_critic=params['lr_critic/init'],
                                learning_rate_actor=params['lr_actor/init'],
                                grad_clip_actor=np.inf, grad_clip_critic=np.inf,
                                gamma=params['agent/gamma'],
                                tau_critic=params['agent/tau_critic'],
                                tau_actor=params['agent/tau_actor'],
                                )

    # Action noise schedule
    action_noise_schedule = PolynomialDecay(
        params['action_noise/init'], params['n_steps'],
        params['action_noise/final'])

    # Epsilon greedy schedule
    epsilon_greedy_schedule = PolynomialDecay(
        params['epsilon_greedy/init'], params['n_steps'],
        params['epsilon_greedy/final'])

    # Schedule n_anneals
    t_transition = [int(x * params['n_steps']) for x in
                    np.linspace(0, 1., params['anneals/n_pieces'] + 1)][1:-1]
    y_transition = [int(n) for n in np.linspace(params['anneals/init'],
                                                params['anneals/final'],
                                                params['anneals/n_pieces'])]
    n_anneals_schedule = PiecewiseConstantDecay(t_transition, y_transition)

    # AGENT TRAINING
    episode_log = trainer(
        env=env, agent=agentMy, action_noise_schedule=action_noise_schedule,
        epsilon_greedy_schedule=epsilon_greedy_schedule,
        n_anneals_schedule=n_anneals_schedule, n_steps=params['n_steps'],
        max_steps_per_episode=params['env/max_steps_per_episode'],
        batch_size=params['trainer/batch_size'],
        n_exploration_steps=params['trainer/n_exploration_steps'],
        n_episodes_early_stopping=params['trainer/n_episodes_early_stopping']
    )

    env = e_trajectory_simENV()
    eval_log_scan = evaluator(env, agentMy, n_episodes=200, reward_scan=False)  # reward_scan=True
    n_tot_eval_steps = np.sum(np.array([(len(r) - 1) for r in eval_log_scan]))
    
    return n_tot_eval_steps

