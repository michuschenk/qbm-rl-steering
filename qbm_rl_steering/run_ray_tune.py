import time

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.optuna import OptunaSearch

from ray.tune import register_trainable

from qbm_rl_steering.core.ddpg_agents import QuantumDDPG
from qbm_rl_steering.core.run_utils import trainer, evaluator

import numpy as np

from tensorflow.keras.optimizers.schedules import PolynomialDecay, PiecewiseConstantDecay
from qbm_rl_steering.environment.orig_awake_env import e_trajectory_simENV


def get_val(params):

    # Learning rate schedules: lr_critic = 5e-4, lr_actor = 1e-4
    lr_schedule_critic = PolynomialDecay(params['lr_critic/init'],
                                         params['n_steps'],
                                         end_learning_rate=params['lr/final'])
    lr_schedule_actor = PolynomialDecay(params['lr_actor/init'],
                                        params['n_steps'],
                                        end_learning_rate=params['lr/final'])

    env = e_trajectory_simENV(MAX_TIME=params['env/max_steps_per_episode'])
    agentMy = QuantumDDPG(state_space=env.observation_space,
                          action_space=env.action_space,
                          learning_rate_schedule_critic=lr_schedule_critic,
                          learning_rate_schedule_actor=lr_schedule_actor,
                          # grad_clip_actor=1e4, grad_clip_critic=1.,
                          grad_clip_actor=np.inf, grad_clip_critic=np.inf,
                          gamma=params['agent/gamma'],
                          tau_critic=params['agent/tau_critic'],
                          tau_actor=params['agent/tau_actor'],
                          n_replicas=params['agent/n_replicas'],
                          big_gamma=params['agent/big_gamma'],
                          beta=params['agent/beta'],
                          n_annealing_steps=params['agent/n_annealing_steps'],
                          # n_meas_for_average=params['agent/n_meas_for_average'],
                          n_rows_qbm=params['agent/n_rows_qbm'],
                          n_columns_qbm=params['agent/n_columns_qbm']
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

    max_steps_eval = 30
    n_episodes_eval = 500
    env = e_trajectory_simENV(MAX_TIME=max_steps_eval)
    eval_log_scan = evaluator(env, agentMy, n_episodes=n_episodes_eval, reward_scan=False)
    n_tot_eval_steps = np.sum(np.array([(len(r) - 1) for r in eval_log_scan]))

    # Normalize -> we want metric to be avg. #steps needed to find objective
    n_tot_eval_steps / float(n_episodes)

    return n_tot_eval_steps


def objective_fun(config):
    # Hyperparameters
    params = {
        'quantum_ddpg': True,  # False
        'n_steps': 100,  # 500
        'env/n_dims': 10,
        'env/max_steps_per_episode': config["max_steps"],  # 50,
        'env/required_steps_above_reward_threshold': 1,
        'trainer/batch_size': config["batch_size"],  # 32
        'trainer/n_exploration_steps': config["n_exploration_steps"],  # 150
        'trainer/n_episodes_early_stopping': 20,
        'agent/gamma': config["gamma"],
        'agent/tau_critic': config["tau_critic"],  # 0.001,
        'agent/tau_actor': config["tau_actor"],  # 0.001,
        'lr_critic/init': config["lr_critic_i"],  # 1e-3
        'lr_critic/decay_factor': 1.,
        'lr_actor/init': config["lr_actor_i"],
        'lr_actor/decay_factor': 1.,
        'lr/final': config["lr_f"],  # 5e-5,
        'action_noise/init': config["action_noise_i"],  # 0.2,
        'action_noise/final': 0.,
        'epsilon_greedy/init': config["epsilon_greedy_i"],  # 0.1
        'epsilon_greedy/final': 0.,
        'anneals/n_pieces': 2,
        'anneals/init': config["n_meas_for_average"],
        'anneals/final': config["n_meas_for_average"],
        'agent/n_replicas': config["n_trotter"],
        'agent/big_gamma': (config["big_gamma_i"], 0),
        'agent/beta': config["beta"],
        'agent/n_annealing_steps': config["n_annealing_steps"],
        'agent/n_rows_qbm': config["n_rows_qbm"],
        'agent/n_columns_qbm': config["n_columns_qbm"]
    }

    val = 0
    n_independent_runs = 2
    for i in range(n_independent_runs):
        val += get_val(params)

    tune.report(n_total_steps_eval=val/n_independent_runs)
    time.sleep(0.2)


register_trainable("objective_fun", objective_fun)
ray.init(include_dashboard=True, dashboard_host='0.0.0.0')

algo = OptunaSearch()
algo = ConcurrencyLimiter(algo, max_concurrent=4)
scheduler = AsyncHyperBandScheduler()

analysis = tune.run(
    objective_fun,
    local_dir='./ray_tune_results',
    metric="n_total_steps_eval",
    mode="min",
    search_alg=algo,
    scheduler=scheduler,
    num_samples=5,
    config={
        "max_steps": tune.qrandint(8, 60, 4),
        "batch_size": tune.qrandint(8, 64, 8),
        "n_exploration_steps": tune.qrandint(0, 100, 5),
        "gamma": tune.quniform(0.7, 0.99, 0.01),
        "tau_critic": tune.loguniform(1e-4, 1.),
        "tau_actor": tune.loguniform(1e-4, 1.),
        "lr_critic_i": tune.loguniform(1e-4, 5e-1),
        "lr_actor_i": tune.loguniform(1e-4, 5e-1),
        "lr_f": tune.loguniform(1e-5, 1e-3),
        "action_noise_i": tune.quniform(0., 0.4, 0.04),
        "epsilon_greedy_i": tune.quniform(0, 1., 0.05),
        "n_meas_average": tune.qrandint(1, 31, 5),
        "n_trotter": tune.qrandint(1, 10, 1),
        "beta": tune.loguniform(0.01, 10.),
        "n_annealing_steps": tune.qrandint(30, 200, 10),
        "big_gamma_i": tune.loguniform(1, 50),
        "n_rows_qbm": tune.qrandint(2, 4, 1),
        "n_columns_qbm": tune.qrandint(2, 4, 1)
    },
)

print("Best hyperparameters found were: ", analysis.best_config)
