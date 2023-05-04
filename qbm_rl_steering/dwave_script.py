import time
from typing import Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import gym
import pandas as pd
import pickle

from tqdm import trange

from qbm_rl_steering.agents.ferl import QBMQ


def hack_the_env(env: gym.Env) -> gym.Env:
    # TODO: something is not right with the normalization. How do they do it properly in SB3 gym?
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


def get_params() -> Tuple[Dict, Dict, Dict]:
    """Get parameters for RL, QBM, and annealing."""

    kwargs_rl = {
        'learning_rate': (0.01, 0.01),  # (0.016, 0.016),
        'small_gamma': 0.95,  # 0.95,  # 0.90,
        'exploration_epsilon': (0.95, 0.),
        'exploration_fraction': 0.9,
        'replay_batch_size': 4,  # 16,  # 32,
        'target_update_frequency': 1,
        'soft_update_factor': 0.8,  # 1.  # 0.6
    }

    # Graph config and quantum annealing settings
    kwargs_anneal = {
        'sampler_type': 'QPU',
        'kwargs_qpu': {
            'token': "DEV-0e07ab13594ade21154e9244ec6c6069198a5260",  #  "DEV-66ff199bc69a2ea5bb4223259859867c616de277",
            'solver': "Advantage_system6.1"},
        'n_replicas': 1,
        'n_meas_for_average': 1,
        'n_annealing_steps': 150,  # 100,
        'big_gamma': (10., 0.5),  # (8.5, 0.),
        'beta': 2.,  # 7.,  # 0.06,
    }

    # Q-function settings (QBM)
    kwargs_qbm = {
        "n_columns_qbm": 2,  # 1,
        "n_rows_qbm": 2,  # 1
    }

    return kwargs_rl, kwargs_anneal, kwargs_qbm


def evaluate(agent: QBMQ, n_evals: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate agent for n_evals episodes.
    :param agent: RL agent using QBM
    :param n_evals: Number of episodes to evaluate
    :return: stats on performance
    """
    n_steps_list = []
    for _ in range(n_evals):
        n_steps = run_episode(agent)
        n_steps_list.append(n_steps)

    n_steps_list = np.array(n_steps_list)
    avg_, std_ = np.mean(n_steps_list), np.std(n_steps_list)
    max_, min_ = np.max(n_steps_list), np.min(n_steps_list)
    return avg_, std_, max_, min_


def run_episode(agent: QBMQ, with_render: bool = False) -> int:
    """
    Run one episode of the agent.
    :param agent: RL agent using QBM.
    :param with_render: flag whether to render the environment.
    :return: total number of steps needed.
    """
    env = agent.env
    obs = env.reset()
    total_steps = 0

    done = False
    while not done:
        if total_steps % 5 == 0:
            print(f'Running episode, step {total_steps}')
        act, _ = agent.predict(obs, deterministic=True)
        # act = env.action_space.sample()
        obs, reward, done, info = env.step(act)
        if with_render:
            env.render()
            time.sleep(0.01)
        total_steps += 1

    if with_render:
        env.close()

    return total_steps


def full_run(n_training_steps: int = 20) -> Tuple[QBMQ, pd.DataFrame]:
    """Run entire training with evaluations before and after."""
    kwargs_rl, kwargs_anneal, kwargs_qbm = get_params()

    env = gym.make("CartPole-v1")
    env = hack_the_env(env)
    # kwargs_anneal.update({"sampler_type": "SQA"})
    agent = QBMQ(env=env, **kwargs_anneal, **kwargs_rl, **kwargs_qbm)

    # Evaluate agent first
    # print(f'BEFORE TRAINING')
    # avg_i, std_i, max_i, min_i = evaluate(agent, n_evals=1)
 
    # Train the agent
    print(f'TRAIN AGENT')
    agent.learn(total_timesteps=n_training_steps)
    
    # RELOAD WEIGHTS
    # with open("weights_100steps_030523/agent_whh.pkl", "rb") as f:
    #     w_hh = pickle.load(f)
    # with open("weights_100steps_030523/agent_wvh.pkl", "rb") as f:
    #     w_vh = pickle.load(f)
    # with open("weights_100steps_030523/agent_whh_t.pkl", "rb") as f:
    #     w_hh_t = pickle.load(f)
    # with open("weights_100steps_030523/agent_wvh_t.pkl", "rb") as f:
    #     w_vh_t = pickle.load(f)

    # agent.q_function.w_hh = w_hh
    # agent.q_function.w_vh = w_vh
    # agent.q_function_target.w_hh = w_hh_t
    # agent.q_function_target.w_vh = w_vh_t

    # Re-evaluate agent again
    # print(f'AFTER TRAINING')
    # avg_f, std_f, max_f, min_f = evaluate(agent, n_evals=1)
    # print(f'Avg +/- std: {avg_f:.1f} +/- {std_f:.1f}')
    # print(f'Min, max: {min_f:.0f}, {max_f:.0f}')

    res_df = pd.DataFrame()
    # res_df = pd.DataFrame([{
    #     # "avg_i": avg_i,
    #     "avg_f": avg_f,
    #     # "std_i": std_i,
    #     "std_f": std_f,
    #     # "max_i": max_i,
    #     "max_f": max_f,
    #     # "min_i": min_i,
    #     "min_f": min_f
    # }])

    return agent, res_df


if __name__ == "__main__":
    n_training_steps = 20

    n_runs = 1
    results_df = pd.DataFrame()
    for i in trange(n_runs):
        agent, res = full_run(n_training_steps)
        results_df = pd.concat([results_df, res])
    results_df.reset_index(inplace=True, drop=True)
    # results_df.to_pickle("res_eval_100steps_050323.pkl")

    # SAVE WEIGHTS
    with open("agent_whh.pkl", "wb") as fid:
        pickle.dump(agent.q_function.w_hh, fid)

    with open("agent_wvh.pkl", "wb") as fid:
        pickle.dump(agent.q_function.w_vh, fid)

    with open("agent_whh_t.pkl", "wb") as fid:
        pickle.dump(agent.q_function_target.w_hh, fid)

    with open("agent_wvh_t.pkl", "wb") as fid:
        pickle.dump(agent.q_function_target.w_vh, fid)

    with open("w_hh_history.pkl", "wb") as fid:
        pickle.dump(agent.q_function.w_hh_history, fid)

    with open("w_vh_history.pkl", "wb") as fid:
        pickle.dump(agent.q_function.w_vh_history, fid)
