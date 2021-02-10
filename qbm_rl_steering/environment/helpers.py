import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN

from .env_desc import TargetSteeringEnv


def plot_response(env: TargetSteeringEnv, fig_title: str = '') -> None:
    """ Plot response of transfer line environment, i.e. BPM position and
    reward vs. dipole kick angle.
    :param env: OpenAI gym-based environment of transfer line
    :param fig_title: figure title """
    angles, x_bpm, rewards = env.get_response()

    fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=(6, 5))
    fig.suptitle(fig_title)

    # BPM response
    l1, = ax1.plot(1e6*angles, 1e3*x_bpm, 'tab:blue')

    # Show margins and episode abort criteria
    ax1.axhline(1e3*(env.x_min - env.x_margin_abort_episode), color='grey')
    ax1.axhline(1e3 * (env.x_min - env.x_margin_discretization), color='black')
    l11 = ax1.axhline(1e3*(env.x_max + env.x_margin_abort_episode),
                      color='grey')
    l12 = ax1.axhline(1e3 * (env.x_max + env.x_margin_discretization),
                      color='black')

    # Reward response
    ax2 = ax1.twinx()
    l2, = ax2.plot(1e6*angles, rewards, c='tab:red')

    ax1.set_xlabel('MSSB angle (urad)')
    ax1.set_ylabel('BPM pos. (mm)')
    ax2.set_ylabel('Reward')
    plt.legend((l1, l11, l12, l2),
               ('BPM pos.', 'Episode abort', 'Discretisation', 'Reward'),
               loc='upper left', fontsize=10)
    plt.tight_layout()
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
    data, n_steps = env.logger.extract_all_data()
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
    axs[2].axhline(env.reward_threshold, c='k', ls='--', label='Target reward')
    axs[2].axhline(env.get_max_reward(), c='k', ls='-', label='Max. reward')

    for i in range(3):
        for j in n_steps:
            axs[i].axvline(j, c='red', ls='--')

    axs[0].set_ylabel('State (x pos.) (m)')
    axs[1].set_ylabel('Action')
    axs[2].set_ylabel('Reward')

    axs[-1].set_xlabel('Iterations')
    plt.tight_layout()
    plt.show()


def plot_log(env: TargetSteeringEnv, fig_title: str = '') -> None:
    """ Plot the evolution of the state, action, and reward using the data
    stored in environment logger .
    :param env: OpenAI gym-based environment of transfer line
    :param fig_title: figure title """
    episodic_data = env.logger.extract_episodic_data()

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(7, 7))
    fig.suptitle(fig_title)

    # Episode abort reason
    axs[0].plot(episodic_data['episode_count'], episodic_data['done_reason'],
                c='tab:blue', ls='None', marker='.', ms=4)
    axs[0].set_yticks([i for i in env.logger.done_reason_map.keys()])
    axs[0].set_yticklabels(
        [s for s in env.logger.done_reason_map.values()], rotation=45)
    axs[0].set_ylim(-0.5, max(env.logger.done_reason_map.keys()) + 0.5)

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
                'tab:green', label='Initial')
    axs[2].plot(episodic_data['episode_count'], episodic_data['reward_final'],
                c='tab:red', label='Final')
    axs[2].axhline(env.reward_threshold, c='k', ls='--', label='Target reward')
    axs[2].axhline(env.get_max_reward(), c='k', label='Max. reward')
    axs[2].set_ylim(-0.05, 1.05)

    axs[0].set_ylabel('Abort reason')
    axs[1].set_ylabel('# steps per episode')
    axs[2].set_ylabel('Reward')

    axs[1].legend(loc='upper left', fontsize=10)
    axs[2].legend(loc='lower left', fontsize=10)
    axs[-1].set_xlabel('Episode')
    plt.tight_layout()
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
    obs = env.reset()
    while episode_count < n_episodes:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # agent.render()
        if done:
            obs = env.reset()
            episode_count += 1
    if make_plot:
        plot_log(env, fig_title=fig_title)


def calculate_performance_metrics(env: TargetSteeringEnv) -> (float, float):
    """ Define metric that characterizes performance of the agent
    Option (I): we count how many times the agent manages to reach the target
    without going above 'UB optimal behaviour' (i.e. max. number of steps
    required assuming optimal behaviour).
    Option (II): Take difference between initial and final reward and
    divide by number of steps required.
    I think option (II) gives a bit more detail, but we implement both.
    :param env: OpenAI gym-based environment of transfer line
    :return tuple of metrics for option I and II described above. """
    episodic_data = env.logger.extract_episodic_data()

    # Option (I)
    upper_bound_optimal = env.get_max_n_steps_optimal_behaviour()
    msk_steps = episodic_data['episode_length'] <= upper_bound_optimal
    msk_reward = episodic_data['reward_final'] >= env.reward_threshold
    msk_nothing_to_do = episodic_data['episode_length'] == 0

    n_success = np.sum(msk_steps & msk_reward & (~msk_nothing_to_do))
    metric_1 = n_success / float(np.sum(~msk_nothing_to_do))

    # Option (II)
    delta_reward = (episodic_data['reward_final'] -
                    episodic_data['reward_initial'])
    metric_2 = np.mean(
        delta_reward[~msk_nothing_to_do] /
        episodic_data['episode_length'][~msk_nothing_to_do])

    return metric_1, metric_2
