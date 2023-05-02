import numpy as np
import matplotlib.pyplot as plt
import torch as th
from typing import Tuple

from stable_baselines3 import DQN
from tqdm import tqdm

from qbm_rl_steering.environment.env_1D_continuous import TargetSteeringEnv


def plot_response(env: TargetSteeringEnv, fig_title: str = '') -> None:
    """ Plot response of transfer line environment, i.e. BPM position and
    reward vs. dipole kick angle.
    :param env: OpenAI gym-based environment of transfer line
    :param fig_title: figure title """
    angles, x_bpm, intensities = env.get_response()

    fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=(7, 4.5))
    fig.suptitle(fig_title)

    # BPM response
    l1, = ax1.plot(1e6*angles, 1e3*x_bpm, 'tab:blue')

    # Intensity response
    ax2 = ax1.twinx()
    l2, = ax2.plot(1e6*angles, intensities, c='k')
    handles = [l1, l2]
    labels = ['BPM pos. (= state)', 'Intensity']

    simple_rewards = []
    for r in intensities:
        simple_rewards.append(env.get_reward(r))
    simple_rewards = np.array(simple_rewards)
    l11, = ax2.plot(1e6*angles,
                    (simple_rewards - np.min(simple_rewards)) /
                    (np.max(simple_rewards) - np.min(simple_rewards)),
                    c='forestgreen')
    handles.append(l11)
    labels.append('Reward')

    l22 = ax2.axhline(env.reward_threshold, c='forestgreen', ls='--')
    handles.append(l22)
    labels.append('Threshold')

    ax1.set_xlabel('MSSB angle (urad)')
    ax1.set_ylabel('BPM pos. (mm)')
    ax2.set_ylabel('Intensity')
    plt.legend(handles, labels, loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_q_net_response(env: TargetSteeringEnv, agent: DQN,
                        fig_title: str = '', save_name: str = None) -> None:
    """
    The idea is to plot the Q-net of the agent (to look inside the
    agent's brain...) versus the state and action axis.
    """
    angles, x_bpm, intensities = env.get_response()
    idx = np.where(intensities > env.reward_threshold)[0][-1]
    x_reward_thresh = x_bpm[idx]

    simple_rewards = []
    for r in intensities:
        simple_rewards.append(env.get_reward(r))
    simple_rewards = np.array(simple_rewards)

    states_float, states_binary = env.get_all_states()

    # Convert to Torch tensor, and run it through the q-net
    states_binary = th.tensor(states_binary)
    q_values = agent.q_net(states_binary).detach().numpy()

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    fig.suptitle(fig_title)

    l1, = axs[0].plot(1e3*x_bpm, intensities, c='k')
    l12 = axs[0].axhline(env.reward_threshold, c='k', ls='--')
    handles = [l1, l12]
    labels = ['Integrated intensity', 'Target ']

    ax11 = axs[0].twinx()
    l11, = ax11.plot(1e3*x_bpm, simple_rewards, c='forestgreen')
    ax11.set_ylabel('Reward')

    labels[0] = 'Integrated intensity / reward'
    handles.append(l11)
    labels.append('Reward')

    axs[0].legend(handles, labels, loc='lower left', fontsize=10)
    axs[0].set_ylabel('Integrated intensity')

    cols = ['tab:red', 'tab:blue', 'tab:green']
    for i in range(q_values.shape[1]):
        axs[1].plot(1e3*states_float, q_values[:, i], c=cols[i],
                    label=f'Action {i}')
    axs[1].axvline(1e3*x_reward_thresh, ls='--', color='k')
    axs[1].axvline(-1e3*x_reward_thresh, ls='--', color='k')

    # Run Monte Carlo to get V* values
    # if env.action_space.n == 2:
    #     mc_agent = MonteCarloAgent(env, gamma=agent.gamma)
    #     states, v_star = mc_agent.run_mc(n_iterations=5000)
    #     axs[1].plot(1e3*states, v_star, c='k', label='V* (MC)')

    axs[1].legend(loc='upper right', fontsize=10)
    axs[1].set_ylabel('Q value')
    axs[1].set_xlabel('State, BPM pos. (mm)')
    if save_name is not None:
        plt.savefig(save_name, dpi=200)
    plt.show()


def run_random_trajectories(env: TargetSteeringEnv, n_episodes: int = 20,
                            fig_title: str = '') -> None:
    """ Test the environment, create trajectories, use reset, etc. using random
    actions.
    :param env: OpenAI gym-based environment of transfer line
    :param n_episodes: number of episode to run test for
    :param fig_title: figure title """
    env.clear_log()
    env.reset()
    episode_count = 0
    while episode_count < n_episodes:
        action = np.random.randint(env.action_space.n)
        _, _, done, _ = env.step(action)
        if done:
            env.reset()
            episode_count += 1

    # Extract all data and convert binary states to floats
    data, n_steps = env.interaction_logger.extract_all_data()
    state_float = []
    for s in data['state']:
        state_float.append(env.make_binary_state_float(s))
    state_float = np.array(state_float)
    data['state_float'] = state_float

    # Plot
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(7, 6))
    fig.suptitle(fig_title)
    axs[0].plot(data['state_float'])
    axs[1].plot(data['action'])
    axs[2].plot(data['reward'])
    axs[2].axhline(env.intensity_threshold, c='k', ls='--', label='Target reward')

    for i in range(3):
        for j in n_steps:
            axs[i].axvline(j, c='red', ls='--')

    axs[0].set_ylabel('State (x pos.) (m)')
    axs[1].set_ylabel('Action')
    axs[2].set_ylabel('Reward')

    axs[-1].set_xlabel('Iterations')
    plt.tight_layout()
    plt.show()


def plot_log(env: TargetSteeringEnv, fig_title: str = '',
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
    axs[0].set_yticks([i for i in env.interaction_logger.done_reason_map.keys()])
    axs[0].set_yticklabels(
        [s for s in env.interaction_logger.done_reason_map.values()], rotation=45)
    axs[0].set_ylim(-0.5, max(env.interaction_logger.done_reason_map.keys()) + 0.5)

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
                'tab:red', ls='solid', marker='o', ms=4, label='Initial')
    axs[2].plot(episodic_data['episode_count'], episodic_data['reward_final'],
                c='tab:green', ls='solid', marker='x', ms=4, mew=1,
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

    axs[1].legend(loc='upper left', fontsize=10)
    axs[2].legend(loc='lower left', fontsize=10)
    axs[-1].set_xlabel('Episode')
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.show()


def evaluate_agent(env: TargetSteeringEnv, agent: DQN,
                   n_episodes: int = 100, make_plot: bool = False,
                   fig_title: str = 'Agent test') -> None:
    """ Run agent for a number of episodes on environment and plot log.
    :param env: OpenAI gym-based environment of transfer line
    :param agent: DQN agent (trained or untrained)
    :param n_episodes: number of episodes used to evaluate agent
    :param make_plot: flag to decide whether to show plots or not
    :param fig_title: figure title """
    episode_count = 0
    env.clear_log()
    obs = env.reset(init_outside_threshold=True)

    pbar = tqdm(total=n_episodes, position=2, leave=True, desc='Evaluation',
                disable=(not make_plot))
    while episode_count < n_episodes:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(int(action))
        if done:
            obs = env.reset(init_outside_threshold=True)
            episode_count += 1
            pbar.update(1)
    pbar.close()

    if make_plot:
        plot_log(env, fig_title=fig_title)


def calculate_performance_metric(env: TargetSteeringEnv) -> (float, float):
    """
    Define metric that characterizes performance of the agent. We count how
    many times the agent manages to reach the target without going above 'UB
    optimal behaviour' (i.e. max. number of steps required assuming optimal
    behaviour).
    :param env: OpenAI gym-based environment of transfer line
    :return metric value (as a single float)
    """
    episodic_data = env.interaction_logger.extract_episodic_data()

    upper_bound_optimal = env.get_max_n_steps_optimal_behaviour()
    msk_steps = episodic_data['episode_length'] <= upper_bound_optimal
    msk_reward = episodic_data['reward_final'] >= env.intensity_threshold
    msk_nothing_to_do = episodic_data['episode_length'] == 0

    # How many of the episodes did the agent succeed?
    n_success = np.sum(msk_steps & msk_reward & (~msk_nothing_to_do))

    return n_success / float(np.sum(~msk_nothing_to_do))


# def find_policy_from_q(env: TargetSteeringEnv, agent: DQN) ->\
#         Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Get response of the trained "Q-net" for all possible (state, action) and
#     then calculate optimal policy according to learned Q values.
#     """
#     states_float, states_binary = env.get_all_states()
#
#     # Convert to Torch tensor, and run it through the q-net
#     states_binary = th.tensor(states_binary)
#     q_values = agent.q_net(states_binary).detach().numpy()
#
#     best_action = np.ones(len(states_float), dtype=int) * -1
#     for i in range(len(states_float)):
#         best_action[i] = np.argmax(q_values[i, :])
#     return states_float, q_values, best_action

def get_q_net_response(env: TargetSteeringEnv, agent: DQN, n_samples_states: int) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate the (trained or untrained) Q net for a number of states
        sampled from the env.observation_space
        :param n_samples_states: number of samples made from the
        env.observation_space.
        :return: sampled states and corresponding q values
        """
        # n_actions = env.action_space.n
        # q_values = np.zeros((n_samples_states, n_actions))
        sampled_states = np.zeros(n_samples_states)
        for i in range(n_samples_states):
            sampled_states[i] = env.observation_space.sample()

        # Sort according to value of state
        idx_sorted = np.argsort(sampled_states)
        sampled_states = sampled_states[idx_sorted]
        sampled_states = sampled_states.reshape(-1, 1)
        # q_values = q_values[idx_sorted]

        sampled_states = th.tensor(sampled_states)
        q_values = agent.q_net(sampled_states).detach().numpy()

        best_action = np.ones(sampled_states.shape[0], dtype=int) * -1
        for i in range(sampled_states.shape[0]):
            best_action[i] = np.argmax(q_values[i, :])
        sampled_states = np.array(sampled_states)

        return sampled_states, q_values, best_action


def calculate_policy_optimality(env: TargetSteeringEnv, agent: DQN, n_samples_states=300) -> float:
    """
    Metric for optimality of policy: we can do this because we know the
    optimal policy already. Measure how many of the actions are correct
    according to the Q-functions that we learned. We only judge actions
    for states outside of reward threshold (inside episode is anyway over
    after 1 step and agent has no way to learn what's best there.
    :return the performance metric
    """
    # states_q, q_values, best_action = find_policy_from_q(env, agent)
    states_q, q_values, best_action = get_q_net_response(env, agent, n_samples_states)
    states_q = states_q.flatten()

    _, x, r = env.get_response()
    idx = np.where(r > env.intensity_threshold)[0][-1]
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

    return policy_eval
