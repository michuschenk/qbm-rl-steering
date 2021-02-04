import numpy as np
import matplotlib.pyplot as plt
from .env_desc import TargetSteeringEnv


def plot_response(env: TargetSteeringEnv, fig_title: str = '') -> None:
    """ Scan through angles and plot response of transfer line environment
    :param env: openAI gym-based environment of transfer line
    :param fig_title: figure title
    :return: None
    """
    # Scan through angles and plot response
    angles = np.linspace(env.mssb_angle_min, env.mssb_angle_max, 200)
    x_bpm = np.zeros_like(angles)
    rewards = np.zeros_like(angles)
    for i, ang in enumerate(angles):
        x, r = env.get_pos_at_bpm_target(total_angle=ang)
        x_bpm[i] = x
        rewards[i] = r

    fig, axs = plt.subplots(1, 1, sharex=True, figsize=(5, 4))
    fig.suptitle(fig_title)
    l1, = axs[0].plot(angles, x_bpm, 'b')

    axs[0].axhline(env.x_min - env.x_margin_abort_episode, color='grey')
    axs[0].axhline(env.x_min - env.x_margin_discretisation, color='black')

    l11 = axs[0].axhline(env.x_max + env.x_margin_abort_episode, color='grey')
    l12 = axs[0].axhline(env.x_max + env.x_margin_discretisation, color='black')

    axs[0].set_xlabel('MSSB angle (rad)')
    axs[0].set_ylabel('BPM pos. (m)')

    ax2 = axs[0].twinx()
    l2, = ax2.plot(angles, rewards, 'r')
    ax2.set_ylabel('Reward')

    plt.legend((l1, l11, l12, l2),
               ('BPM pos.', 'Margin episode abort',
                'Margin discretisation', 'Reward'),
               loc='upper left')
    plt.tight_layout()
    plt.show()


def run_random_trajectories(env: TargetSteeringEnv, n_episodes: int = 5,
                            fig_title: str = '') -> None:
    """ Test the environment, create trajectories, use reset, etc. using random
    actions.
    :param env: openAI gym environment
    :param n_episodes: number of episode to run
    :param fig_title: figure title
    :return: None
    """
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
    for i, s in enumerate(data['state']):
        data['state'][i] = env.make_binary_state_float(s)

    # Plot
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(7, 6))
    fig.suptitle(fig_title)
    axs[0].plot(data['state'])
    axs[1].plot(data['action'])
    axs[2].plot(data['reward'])

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
    stored in env.log .
    :param env: openAI gym environment
    :param fig_title: figure title
    :return: None
    """
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

    # Episode length
    axs[1].plot(episodic_data['episode_count'], episodic_data['episode_length'],
                c='tab:blue')
    axs[1].axhline(env.max_steps_per_epoch, c='k', label='Max. # steps')
    axs[1].axhline(env.get_max_n_steps_optimal_behaviour(),
                   c='k', ls='--', label='UB optimal behaviour')
    axs[1].set_ylim(0, 1.1*env.max_steps_per_epoch)

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


def test_agent(env, agent, n_epochs=100, fig_title='Agent test'):
    """ Run agent for a number of epochs on environment and plot log.
    :param env: openAI gym environment
    :param agent: agent (trained or untrained)
    :param n_epochs: number of epochs for the test
    :param fig_title: figure title of output plot
    :return: None
    """
    epoch_count = 0
    env.clear_log()
    obs = env.reset()
    while epoch_count < n_epochs:
        action, _states = agent.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # agent.render()
        if done:
            obs = env.reset()
            epoch_count += 1
    plot_log(env, fig_title=fig_title)


def calculate_performance_metrics(env: TargetSteeringEnv):
    """ Define metric that characterizes performance of the agent
    Option (I): we count how many times the agent manages to reach the target
    without going above 'UB optimal behaviour' (i.e. max. number of steps
    required assuming optimal behaviour).
    Option (II): Take difference between initial and final reward and
    divide by number of steps required.
    I think option (II) gives a bit more detail, but we implement both. """
    episodic_data = env.logger.extract_episodic_data()

    # Option (I)
    upper_bound_optimal = env.get_max_n_steps_optimal_behaviour()
    msk_steps = episodic_data['episode_length'] < upper_bound_optimal
    msk_reward = episodic_data['reward_final'] > env.reward_threshold
    msk_nothing_to_do = episodic_data['episode_length'] == 0

    n_success = np.sum(msk_steps & msk_reward & (~msk_nothing_to_do))
    performance_metric_1 = n_success / float(np.sum(~msk_nothing_to_do))

    # Option (II)
    delta_reward = (episodic_data['reward_final'] -
                    episodic_data['reward_initial'])
    performance_metric_2 = np.mean(
        delta_reward[~msk_nothing_to_do] /
        episodic_data['episode_length'][~msk_nothing_to_do])

    return {'metric_1': performance_metric_1, 'metric_2': performance_metric_2}
