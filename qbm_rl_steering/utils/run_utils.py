from typing import Tuple, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pickle
import gym

from qbm_rl_steering.agents.ferl import QBMQ


# TODO: move this function to helpers
def find_policy_from_q(agent: QBMQ, n_samples_states: int = 100) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get response of trained Q-net for all possible (state, action)
     pairs and calculate optimal policy according to learned Q-values."""
    states, q_values = agent.get_q_net_response(n_samples_states)

    best_action = np.ones(len(states), dtype=int) * -1
    for i in range(len(states)):
        best_action[i] = np.argmax(q_values[i, :])
    return states, q_values, best_action


# TODO: move function to helpers
def plot_agent_evaluation(
        states_q: np.ndarray, q_values: np.ndarray, best_action: np.ndarray,
        visited_states: List, x_reward_thresh: float) -> None:
    """Plot evaluation of agent after training."""
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 7))
    fig.suptitle('QBM agent evaluation')

    # Value functions
    cols = ['tab:red', 'tab:blue']
    for i in range(q_values.shape[1]):
        axs[0].plot(1e3 * states_q, q_values[:, i], c=cols[i],
                    label=f'Action {i}')
    axs[0].axvline(1e3*x_reward_thresh, c='k', ls='--', label='Success region')
    axs[0].axvline(-1e3*x_reward_thresh, c='k', ls='--')
    axs[0].legend(loc='upper right')

    # Plot policy
    for a in list(set(best_action)):
        msk_a = best_action == a
        axs[1].plot(1e3 * states_q[msk_a], best_action[msk_a],
                    marker='o', ms=3, ls='None', c=cols[a])
    axs[1].axvline(1e3*x_reward_thresh, c='k', ls='--')
    axs[1].axvline(-1e3*x_reward_thresh, c='k', ls='--')
    axs[1].set_ylabel('Best action')

    # What states have been visited and how often?
    axs[2].hist(1e3*np.array(visited_states), bins=100)
    axs[2].axvline(1e3*x_reward_thresh, c='k', ls='--')
    axs[2].axvline(-1e3*x_reward_thresh, c='k', ls='--')
    axs[2].set_xlabel('State, BPM pos. (mm)')
    axs[2].set_ylabel('# visits')

    plt.show()


# TODO: move function to helpers
def calculate_policy_optimality(env: gym.Env, states: np.ndarray,
                                best_action: np.ndarray) -> float:
    """Metric for optimality of policy: we can do this because we know the
    optimal policy already. Measure how many of the actions are correct
    according to the Q-functions that we learned. We only judge actions
    for states outside of reward threshold (inside episode is anyway over
    after 1 step and agent has no way to learn what's best there.
    :returns agent object and the performance metric"""
    _, x, r = env.get_response()
    idx = np.where(r > env.reward_threshold)[0][-1]
    x_reward_thresh = x[idx]
    n_states_total = np.sum(
        (states < -x_reward_thresh) | (states > x_reward_thresh))

    # How many of the actions that the agent would take are actually
    # according to optimal policy? (this is environment dependent and
    # something we can do because we know the optimal policy).
    n_correct_actions = np.sum(
        (best_action == 0) & (states < -x_reward_thresh))
    n_correct_actions += np.sum(
        (best_action == 1) & (states > x_reward_thresh))

    policy_eval = 100 * n_correct_actions / float(n_states_total)
    print(f'Optimality of policy: {policy_eval:.1f}%')
    return policy_eval


# TODO: could maybe be turned into more general version for both DQN and QBMQ.
def train_and_evaluate_agent(
        kwargs_env: Dict, kwargs_rl: Dict, kwargs_anneal: Dict,
        total_timesteps: int, make_plots: bool = True,
        calc_optimality: bool = False, n_samples_states: int = 100,
        save_eval: bool = True) -> Tuple[QBMQ, float]:
    """Initializes environment, trains an agent for given number of training
    steps and runs the evaluation of it to return  the trained agent and the
    corresponding optimality metric.
    :param kwargs_env: dictionary with environment-related parameters.
    :param kwargs_anneal: dictionary with annealing-related parameters.
    :param total_timesteps: number of training interactions.
    :param make_plots: boolean flag whether or not to create plots.
    :param calc_optimality: boolean flag whether or not to calculate the
    optimality metric of the agent (can be done independent of plotting).
    :param n_samples_states: number of states to evaluate when calculating
    optimality metric.
    :param save_eval: boolean flag whether or not to save the evaluation to
    a pickle file.
    :returns trained agent and corresponding optimality metric."""
    if make_plots:
        calc_optimality = True

    # Initialize environment
    if kwargs_env['type'] == 'discrete':
        from qbm_rl_steering.environment.env_1D_discrete import TargetSteeringEnv
        env = TargetSteeringEnv(**kwargs_env)
    elif kwargs_env['type'] == 'continuous':
        from qbm_rl_steering.environment.env_1D_continuous import TargetSteeringEnv
        env = TargetSteeringEnv(**kwargs_env)
    else:
        raise ValueError("kwargs_env['type'] has to be either 'discrete' or "
                         "'continuous'.")

    # Initialize agent and train
    agent = QBMQ(env=env, **kwargs_anneal, **kwargs_rl)
    # hlp.evaluate_agent(env, agent, n_episodes=50, make_plot=True,
    #                    fig_title='Agent evaluation before training')

    # _ = agent.env.reset()
    visited_states = agent.learn(total_timesteps=total_timesteps)
    # When using learn_systematic it's best to make sure you sweep through
    # all states at least once.
    # visited_states = agent.learn_systematic(total_timesteps=total_timesteps)

    # Plot learning evolution (note that this does not work when we either
    # set play_out_episode to True or when using learn_systematic.
    # hlp.plot_log(env, fig_title='Agent training')

    # Evaluate the agent
    # Evaluate agent on random initial states
    # env = TargetSteeringEnv(**kwargs_env)
    # hlp.evaluate_agent(env, agent, n_episodes=30, make_plot=True,
    #                    fig_title='Agent evaluation')

    if calc_optimality:
        states_q, q_values, best_action = find_policy_from_q(
            agent, n_samples_states=n_samples_states)

    # Get state (x pos.) where reward threshold is
    _, x, r = env.get_response()
    idx = np.where(r > env.reward_threshold)[0][-1]
    x_reward_thresh = x[idx]
    if make_plots:
        plot_agent_evaluation(
            states_q, q_values, best_action, visited_states, x_reward_thresh)

    if save_eval:
        dict_eval = {
            'states_q': states_q,
            'q_values': q_values,
            'best_action': best_action,
            'visited_states': visited_states,
            'x_reward_thresh': x_reward_thresh
        }

        with open('ferl_optimality.pkl', 'wb') as fid:
            pickle.dump(dict_eval, fid)

    policy_optimality = None
    if calc_optimality:
        policy_optimality = calculate_policy_optimality(
            env, states_q, best_action)
    return agent, policy_optimality
