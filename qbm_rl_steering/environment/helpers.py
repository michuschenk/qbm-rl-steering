import numpy as np
import matplotlib.pyplot as plt
from .env_desc import TargetSteeringEnv


def plot_response(env: TargetSteeringEnv, fig_title: str = '') -> None:
    """
    Scan through angles and plot response of transfer line environment
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

    fig = plt.figure(1)
    fig.suptitle(fig_title)
    ax1 = plt.gca()
    l1, = ax1.plot(angles, x_bpm, 'b')

    ax1.axhline(env.x_min - env.x_margin_abort_episode, color='grey')
    ax1.axhline(env.x_min - env.x_margin_discretisation, color='black')

    l11 = ax1.axhline(env.x_max + env.x_margin_abort_episode, color='grey')
    l12 = ax1.axhline(env.x_max + env.x_margin_discretisation, color='black')

    # for i in np.arange(
    #         env.x_min - env.x_margin_discretisation,
    #         env.x_max + env.x_margin_discretisation,
    #         env.x_delta):
    #     ax1.axhline(i, color='black', ls='--', lw=0.5)

    ax1.set_xlabel('MSSB angle (rad)')
    ax1.set_ylabel('BPM pos. (m)')

    ax2 = ax1.twinx()
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
    """
    Test the environment, create trajectories, use reset, etc. using random
    actions.
    :param env: openAI gym environment
    :param n_episodes: number of episode to run
    :param fig_title: figure title
    :return: None
    """
    env.reset()
    episode_count = 0
    while episode_count < n_episodes:
        action = np.random.randint(env.action_space.n)
        _, _, done, _ = env.step(action)
        if done:
            env.reset()
            episode_count += 1
    log_all = env.log_all

    # Unpack data (convert states from binary to floats)
    state = []
    action = []
    reward = []
    n = 0
    for log_ep in log_all:
        for data in log_ep:
            state.append(env._make_binary_state_float(data[0]))
            action.append(data[1])
            reward.append(data[2])
            n += 1
    state = np.array(state)
    action = np.array(action)
    reward = np.array(reward)

    # Plot
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(7, 6))
    fig.suptitle(fig_title)
    axs[0].plot(state)
    axs[1].plot(action)
    axs[2].plot(reward)

    # Ends of episodes
    n_steps = []
    for log_ep in log_all:
        n_steps.append(len(log_ep))
    n_steps = np.cumsum(np.array(n_steps))

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
    """
    Plot the evolution of the state, action, and reward using the data stored
    in env.log .
    :param env: openAI gym environment
    :param fig_title: figure title
    :return: None
    """
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(7, 7))
    fig.suptitle(fig_title)

    # Extract logging data
    # Rewards correspond to integrated intensity on target
    log_all = env.log_all
    n_episodes = len(log_all)
    episode_count = np.arange(n_episodes)
    episode_length = np.zeros(n_episodes)
    reward_init = np.zeros(n_episodes)
    reward_final = np.zeros(n_episodes)
    done_reason = np.zeros(n_episodes)
    for i, log_ep in enumerate(log_all):
        episode_length[i] = len(log_ep) - 1  # we don't count the last entry
        reward_init[i] = log_ep[0][2]
        reward_final[i] = log_ep[-1][2]
        done_reason[i] = log_ep[-1][5]

    # Episode abort reason
    axs[0].plot(episode_count, done_reason, 'tab:blue', ls='None',
                marker='.', ms=4)
    axs[0].set_yticks([i for i in env.done_reason_map.keys()])
    axs[0].set_yticklabels([s for s in env.done_reason_map.values()],
                           rotation=45)
    axs[0].set_ylim(-0.5, max(env.done_reason_map.keys()) + 0.5)

    # Episode length
    axs[1].plot(episode_count, episode_length, c='tab:blue')
    axs[1].axhline(env.max_steps_per_epoch, c='k',
                   label='Max. # steps')
    axs[1].axhline(env.get_max_n_steps_optimal_behaviour(),
                   c='k', ls='--', label='UB optimal behaviour')
    axs[1].set_ylim(0, 1.1*env.max_steps_per_epoch)

    # Reward
    axs[2].plot(episode_count, reward_init, 'tab:green', label='Initial')
    axs[2].plot(episode_count, reward_final, 'tab:red', label='Final')
    axs[2].axhline(env.reward_threshold, c='k', ls='--',
                   label='Target reward')
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
    obs = env.reset()
    while epoch_count < n_epochs:
        action, _states = agent.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # agent.render()
        if done:
            obs = env.reset()
            epoch_count += 1
    plot_log(env, fig_title=fig_title)
