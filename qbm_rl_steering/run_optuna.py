from qbm_rl_steering.core.ddpg_agents import QuantumDDPG
from qbm_rl_steering.core.run_utils import trainer, evaluator

import numpy as np

from tensorflow.keras.optimizers.schedules import PolynomialDecay, PiecewiseConstantDecay
from qbm_rl_steering.environment.orig_awake_env import e_trajectory_simENV

import optuna


def get_val(params):
    # Learning rate schedules: lr_critic = 5e-4, lr_actor = 1e-4
    lr_schedule_critic = PolynomialDecay(params['lr_critic/init'],
                                         params['n_steps'],
                                         end_learning_rate=params['lr/final'])
    lr_schedule_actor = PolynomialDecay(params['lr_actor/init'],
                                        params['n_steps'],
                                        end_learning_rate=params['lr/final'])

    env = e_trajectory_simENV()

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
    trainer(
        env=env, agent=agentMy, action_noise_schedule=action_noise_schedule,
        epsilon_greedy_schedule=epsilon_greedy_schedule,
        n_anneals_schedule=n_anneals_schedule, n_steps=params['n_steps'],
        max_steps_per_episode=params['env/max_steps_per_episode'],
        batch_size=params['trainer/batch_size'],
        n_exploration_steps=params['trainer/n_exploration_steps'],
        n_episodes_early_stopping=params['trainer/n_episodes_early_stopping']
    )

    env = e_trajectory_simENV()
    eval_log_scan = evaluator(env, agentMy, n_episodes=500, reward_scan=False)  # reward_scan=True
    n_tot_eval_steps = np.sum(np.array([(len(r) - 1) for r in eval_log_scan]))

    return n_tot_eval_steps


def objective(trial):
    lr_i = trial.suggest_float('lr_i', 1e-4, 1e-1, log=True)
    lr_f = trial.suggest_float('lr_f', 1e-6, 1e-3, log=True)
    max_steps = trial.suggest_int("max_steps", 8, 60, step=5)
    batch_size = trial.suggest_int("batch_size", 8, 48, step=8)
    n_exp = trial.suggest_int("n_exploration", 10, 200, step=20)
    gamma = trial.suggest_float("gamma", 0.7, 0.99, log=False)
    tau = trial.suggest_float("tau", 0.001, 0.2, log=True)
    act_noise_i = trial.suggest_float("act_noise_i", 0., 0.3, log=False)
    # act_noise_f = trial.suggest_float("act_noise_f", 0., 0.3, log=False)
    eps_greedy_i = trial.suggest_float("eps_greedy_i", 0., 0.9, log=False)
    # exp_frac = trial.suggest_float("exp_frac", 0.7, 0.99, log=False)
    # anneal_steps = trial.suggest_int("anneal_steps", 60, 200, step=10)
    # big_gamma = trial.suggest_float("big_gamma", 10, 50, log=True)
    # beta = trial.suggest_float("beta", 0.02, 5., log=True)
    # n_meas_avg = trial.suggest_int("n_meas_avg", 1, 30, step=5)

    params = {
        'quantum_ddpg': True,
        'n_steps': 200,
        'env/n_dims': 10,
        'env/max_steps_per_episode': max_steps,  # 50,
        'env/required_steps_above_reward_threshold': 1,
        'trainer/batch_size': batch_size,  # 32
        'trainer/n_exploration_steps': n_exp,  # 150    400: works well, too...  , 500,  # 100,
        'trainer/n_episodes_early_stopping': 20,
        'agent/gamma': gamma,
        'agent/tau_critic': tau,  # 0.001,
        'agent/tau_actor': tau,  # 0.001,
        'lr_critic/init': lr_i,  # 1e-3
        'lr_critic/decay_factor': 1.,
        'lr_actor/init': lr_i,
        'lr_actor/decay_factor': 1.,
        'lr/final': lr_f,  # 5e-5,
        'action_noise/init': act_noise_i,  # 0.2,
        'action_noise/final': 0.,
        'epsilon_greedy/init': eps_greedy_i,  # 0.1
        'epsilon_greedy/final': 0.,
        'anneals/n_pieces': 2,
        'anneals/init': 1,
        'anneals/final': 1,
    }

    n_runs = 5
    val = 0
    for i in range(n_runs):
        val += get_val(params)
    return val


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
