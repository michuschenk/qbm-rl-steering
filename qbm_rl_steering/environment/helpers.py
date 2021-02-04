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
        x, r = env._get_pos_at_bpm_target(total_angle=ang)
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

    for i in np.arange(
            env.x_min - env.x_margin_discretisation,
            env.x_max + env.x_margin_discretisation,
            env.x_delta):
        ax1.axhline(i, color='black', ls='--', lw=0.5)

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


def run_random_trajectories(env: TargetSteeringEnv, n_epochs: int = 5,
                            n_episodes: int = 40, fig_title: str = '') -> None:
    """
    Test the environment, create trajectories, use reset, etc. using random
    actions.
    :param env: openAI gym environment
    :param n_epochs: number of epochs to run
    :param n_episodes: number of episodes per epoch
    :param fig_title: figure title
    :return: None
    """
    for i in range(n_epochs):
        for j in range(n_episodes):
            action = np.random.randint(3)
            env.step(action)
        env.reset()
    log = np.array(env.log)

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(7, 6))
    fig.suptitle(fig_title)
    labels = ('state', 'action', 'reward')
    for i in range(3):
        axs[i].plot(log[:, i])
        axs[i].set_ylabel(labels[i])
        for j in range(n_epochs + 1):
            axs[i].axvline(j * n_episodes, color='red', ls='--')
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
    log_all = np.array(env.log_all)
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(7, 7))
    fig.suptitle(fig_title)

    # Rewards correspond to intensity
    reward_init = []
    reward_final = []
    done_reason = []
    nb_steps = []
    for ep in range(len(log_all)):
        log_ep = log_all[ep]
        nb_steps.append(len(log_ep))
        reward_init.append(log_ep[0][2])
        reward_final.append(log_ep[-1][2])
        done_reason.append(log_ep[-1][5])

    episode = np.arange(len(log_all))
    reward_init = np.array(reward_init)
    reward_final = np.array(reward_final)
    nb_steps = np.array(nb_steps)
    done_reason = np.array(done_reason)

    # Episode abort reason
    axs[0].plot(episode, done_reason, 'tab:blue', ls='None', marker='.', ms=4)
    axs[0].set_yticks([i for i in env.done_reason_map.keys()])
    axs[0].set_yticklabels([s for s in env.done_reason_map.values()],
                           rotation=45)
    axs[0].set_ylim(-0.5, max(env.done_reason_map.keys()) + 0.5)

    # Episode length
    axs[1].plot(episode, nb_steps, c='tab:blue')
    axs[1].axhline(env.max_steps_per_epoch, c='k',
                   label='Max. # steps')
    optimal_upper_bound = np.ceil(
        ((env.x_max - env.x_min) / env.x_delta - 1) / 2.)
    axs[1].axhline(optimal_upper_bound, c='k', ls='--',
                   label='UB optimal behaviour')
    axs[1].set_ylim(0, 1.1*env.max_steps_per_epoch)

    # Reward
    axs[2].plot(episode, reward_init, 'tab:green', label='Initial')
    axs[2].plot(episode, reward_final, 'tab:red', label='Final')
    axs[2].axhline(env.reward_threshold, c='k', ls='--',
                   label='Target reward')
    axs[2].axhline(env._get_max_reward(), c='k', label='Max. reward')
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
