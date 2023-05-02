from typing import Dict, Tuple

import numpy as np
import gym

from qbm_rl_steering.agents.ferl import QBMQ


import time

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune import register_trainable


def hack_the_env(env: gym.Env) -> gym.Env:
    # TODO: something is not right with the normalization. How do they do it properly in SB3 / gym?
    #  The reason is that some of the bounds are +/- np.inf which my code does not like so much ...
    # For now we hack the lows and highs of the default observation_space, but keep in mind. Might cause
    # other problems as well ...
    low = env.observation_space.low
    high = env.observation_space.high

    low[1] = -2.5
    low[3] = -2.5
    high[1] = 2.5
    high[3] = 2.5

    env.observation_space = gym.spaces.Box(low, high)

    return env


def evaluate(agent: QBMQ, n_evals: int = 10) -> Dict:
    """Run n_evals episodes and take avg, std, min, and max reward. Note that the reward
    is simply given by the number of steps that the episode took."""
    n_steps_list = []
    for _ in range(n_evals):
        n_steps = run_episode(agent)
        n_steps_list.append(n_steps)

    n_steps_list = np.array(n_steps_list)
    avg_, std_ = np.mean(n_steps_list), np.std(n_steps_list)
    max_, min_ = np.max(n_steps_list), np.min(n_steps_list)
    return {"avg": avg_, "std": std_, "max": max_, "min": min_}


def run_episode(agent: QBMQ) -> int:
    """Run one episode."""
    env = agent.env
    obs = env.reset()
    total_steps = 0

    done = False
    while not done:
        act, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(act)
        total_steps += 1

    return total_steps


def fill_in_params(params: Dict) -> Tuple[Dict, Dict, Dict]:
    """Fill in dictionaries with parameter values of the optuna trial."""
    # RL settings
    kwargs_rl = {
        'learning_rate': (params['lr_init'], params['lr_final']),
        'small_gamma': params['agent/gamma'],
        'exploration_epsilon': (params['epsilon_greedy/init'], params['epsilon_greedy/final']),
        'exploration_fraction': 1.,
        'replay_batch_size': params['trainer/batch_size'],
        'target_update_frequency': 1,
        'soft_update_factor': params['agent/tau']
    }

    # Graph config and quantum annealing settings
    kwargs_anneal = {
        'sampler_type': 'SQA',
        'kwargs_qpu': {'aws_device':
                       'arn:aws:braket:::device/qpu/d-wave/DW_2000Q_6',
                       's3_location': None},
        'n_replicas': 1,
        'n_meas_for_average': params['anneal/n_meas_avg'],
        'n_annealing_steps': params['anneal/n_steps'],
        'big_gamma': (params['anneal/big_gamma'], 0.),
        'beta': params['anneal/beta']
    }

    # Q-function settings (QBM)
    kwargs_qbm = {
        "n_columns_qbm": params['qbm/n_cols'],
        "n_rows_qbm": params['qbm/n_rows']
    }

    return kwargs_anneal, kwargs_rl, kwargs_qbm


def get_val(params: Dict) -> float:
    """Core function to initialize, train, and evaluate the agent. Metric is defined as
    the improvement before vs after training (in terms of average episodic reward)."""
    kwargs_anneal, kwargs_rl, kwargs_qbm = fill_in_params(params)

    # Initialize, evaluate, train, and re-evaluate
    env = gym.make("CartPole-v1")
    env = hack_the_env(env)
    agent = QBMQ(env=env, **kwargs_anneal, **kwargs_rl, **kwargs_qbm)

    eval_before = evaluate(agent, n_evals=50)
    agent.learn(total_timesteps=params['trainer/n_steps'])
    eval_after = evaluate(agent, n_evals=10)

    metric = eval_after["avg"] - eval_before["avg"]
    return metric


def objective_fun(config: Dict) -> None:
    """Core optimizer function that reports to ray"""
    params = {
        'trainer/n_steps': 40,
        'trainer/batch_size': 16,  # config["batch_size"],
        'agent/gamma': 0.95,  # config["gamma"],
        'agent/tau': 1.,  # config["tau"],
        'lr_init': config["lr_i"],
        'lr_final': config["lr_i"],  # config["lr_f"],
        'epsilon_greedy/init': 1.,  # config["epsilon_greedy_i"],
        'epsilon_greedy/final': 0.,
        'anneal/n_steps': 150,
        'anneal/beta': 7.,  # config["beta"],
        'anneal/big_gamma': 1.5,  # config["big_gamma_i"],
        'anneal/n_meas_avg': 1,
        'qbm/n_rows': 2,  # config['n_rows_qbm'],
        'qbm/n_cols': 2,  # config['n_columns_qbm']
    }

    n_runs = 5
    val = 0
    for _ in range(n_runs):
        val += get_val(params)
        # # Another way to define a metric could be to check how many episodes were at maximum (500).
        # if eval_after["max"] == 500 and eval_after["min"] == 500:
        #     val += 1

    tune.report(average_episodic_reward=val/n_runs)
    time.sleep(0.2)


if __name__ == "__main__":
    register_trainable("objective_fun", objective_fun)
    ray.init(include_dashboard=True, dashboard_host='0.0.0.0')

    algo = OptunaSearch()
    algo = ConcurrencyLimiter(algo, max_concurrent=15)
    scheduler = AsyncHyperBandScheduler()

    analysis = tune.run(
        objective_fun,
        local_dir='./ray_tune_results_ferl',
        metric="average_episodic_reward",
        mode="max",
        search_alg=algo,
        scheduler=scheduler,
        num_samples=100,
        config={
            # "batch_size": tune.qrandint(16, 32, 16),
            # "gamma": tune.quniform(0.7, 0.99, 0.02),
            # "tau": tune.loguniform(1e-4, 1.),
            "lr_i": tune.loguniform(1e-4, 5e-1),
            # "lr_f": tune.loguniform(1e-5, 1e-3),
            # "epsilon_greedy_i": tune.quniform(0, 1., 0.05),
            # "beta": tune.loguniform(0.1, 10.),
            # "big_gamma_i": tune.loguniform(1, 20),
            # "n_annealing_steps": tune.qrandint(30, 200, 10),
            # "n_meas_average": tune.qrandint(1, 31, 5),
            # "n_trotter": tune.qrandint(1, 10, 1),
            # "n_rows_qbm": tune.qrandint(1, 3, 1),
            # "n_columns_qbm": tune.qrandint(1, 3, 1)
        },
    )

    print("Best hyper-params found were: ", analysis.best_config)
