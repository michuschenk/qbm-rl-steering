import numpy as np
import matplotlib.pyplot as plt
import torch as th
from typing import Tuple

import gym
from tqdm import tqdm

# TODO: currently using some methods that are not available for all envs.


def plot_log(env: gym.Env, fig_title: str = '',
             save_name: str = 'training_log.png') -> None:
    """ Plot the evolution of the state, action, and reward using the data
    stored in environment logger .
    :param env: OpenAI gym-based environment of transfer line
    :param fig_title: figure title """
    episodic_data = env.interaction_logger.extract_episodic_data()

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(7, 7))
    fig.suptitle(fig_title)

    # Episode abort reason
    axs[0].plot(episodic_data['episode_count'], episodic_data['done_reason'],
                c='tab:blue', ls='None', marker='.', ms=4)
    axs[0].set_yticks([
        i for i in env.interaction_logger.done_reason_map.keys()])
    axs[0].set_yticklabels(
        [s for s in env.interaction_logger.done_reason_map.values()],
        rotation=45)
    axs[0].set_ylim(
        -0.5, max(env.interaction_logger.done_reason_map.keys()) + 0.5)

    # Episode length (for statistics remove zero entries)
    msk = episodic_data['episode_length'] == 0
    n_steps_avg = np.mean(episodic_data['episode_length'][~msk])
    n_steps_std = np.std(episodic_data['episode_length'][~msk])
    axs[1].plot(episodic_data['episode_count'], episodic_data['episode_length'],
                c='tab:blue',
                label=f'#steps {n_steps_avg:.1f} +/- {n_steps_std:.1f}')
    axs[1].axhline(env.max_steps_per_episode, c='k', label='Max. # steps')
    axs[1].axhline(env.get_max_n_steps_optimal_behaviour(),
                   c='k', ls='--', label='UB optimal behaviour')
    axs[1].set_ylim(0, 1.1 * env.max_steps_per_episode)

    # Reward
    axs[2].plot(episodic_data['episode_count'], episodic_data['reward_initial'],
                'tab:red', ls='-', marker='o', ms=4, label='Initial')
    axs[2].plot(episodic_data['episode_count'], episodic_data['reward_final'],
                c='tab:green', ls='-', marker='x', ms=4, mew=1,
                label='Final')

    # Y-axis scaling
    rew_min = np.min(
        (episodic_data['reward_final'], episodic_data['reward_initial']))
    rew_max = np.max(
        (episodic_data['reward_final'], episodic_data['reward_initial']))
    rew_min = min(rew_min, -0.05 * (rew_max - rew_min))
    axs[2].set_ylim(1.05*rew_min, 1.05*rew_max)

    axs[0].set_ylabel('Abort reason')
    axs[1].set_ylabel('# steps per episode')
    axs[2].set_ylabel('Reward')

    axs[1].legend(loc='upper right', fontsize=10)
    axs[2].legend(loc='lower right', fontsize=10)
    axs[-1].set_xlabel('Episode')
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.show()


def evaluate_agent(env: gym.Env, agent, n_episodes: int = 100,
                   make_plot: bool = False, fig_title: str = 'Agent test') ->\
        None:
    """ Run agent for a number of episodes on environment and plot log.
    :param env: OpenAI gym-based environment
    :param agent: RL agent (trained or untrained)
    :param n_episodes: number of episodes used to evaluate agent
    :param make_plot: flag to decide whether to show plots or not
    :param fig_title: figure title """
    episode_count = 0
    env.clear_log()
    obs = env.reset(init_outside_thresh=True)

    pbar = tqdm(total=n_episodes, position=2, leave=True, desc='Evaluation',
                disable=(not make_plot))
    while episode_count < n_episodes:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(int(action))
        if done:
            obs = env.reset(init_outside_thresh=True)
            episode_count += 1
            pbar.update(1)
    pbar.close()

    if make_plot:
        plot_log(env, fig_title=fig_title)


def find_policy_from_q(env: gym.Env, agent) -> Tuple[np.ndarray, np.ndarray,
                                                     np.ndarray]:
    """
    Get response of the trained "Q-net" for all possible (state, action) and
    then calculate optimal policy according to learned Q values.
    """
    states_float, states_binary = env.get_all_states()

    # Convert to Torch tensor, and run it through the q-net
    states_binary = th.tensor(states_binary)
    q_values = agent.q_net(states_binary).detach().numpy()

    best_action = np.ones(len(states_float), dtype=int) * -1
    for i in range(len(states_float)):
        best_action[i] = np.argmax(q_values[i, :])
    return states_float, q_values, best_action


def calculate_policy_optimality(env: gym.Env, agent) -> float:
    """
    Metric for optimality of policy: we can do this because we know the
    optimal policy already. Measure how many of the actions are correct
    according to the Q-functions that we learned. We only judge actions
    for states outside of reward threshold (inside episode is anyway over
    after 1 step and agent has no way to learn what's best there.
    :return the performance metric
    """
    states_q, q_values, best_action = find_policy_from_q(env, agent)

    _, x, r = env.get_response()
    idx = np.where(r > env.reward_threshold)[0][-1]
    x_reward_thresh = x[idx]
    n_states_total = np.sum(
        (states_q < -x_reward_thresh) | (states_q > x_reward_thresh))

    # How many of the actions that the agent would take are actually
    # according to optimal policy? (this is environment dependent and
    # something we can do because we know the optimal policy).
    n_correct_actions = np.sum(
        (best_action == 0) & (states_q < -x_reward_thresh))
    n_correct_actions += np.sum(
        (best_action == 1) & (states_q > x_reward_thresh))

    policy_eval = 100 * n_correct_actions / float(n_states_total)
    # print(f'Optimality of policy: {policy_eval:.1f}%')

    return policy_eval
