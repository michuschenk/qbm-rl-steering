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
    angles = np.linspace(env.mssb_angle_min, env.mssb_angle_max, 100)
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
    ax1.set_xlabel('MSSB angle (rad)')
    ax1.set_ylabel('BPM pos. (m)')

    ax2 = ax1.twinx()
    l2, = ax2.plot(angles, rewards, 'r')
    ax2.set_ylabel('Reward')

    plt.legend((l1, l2), ('BPM pos.', 'Reward'),
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


def plot_log(env: TargetSteeringEnv, fig_title: str = '',
             plot_epoch_end: bool = False) -> None:
    """
    Plot the evolution of the state, action, and reward using the data stored
    in env.log .
    :param env: openAI gym environment
    :param fig_title: figure title
    :param plot_epoch_end: flag to switch on/off the markers for end of epsiode
    :return: None
    """
    log = np.array(env.log)
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(7, 6))
    fig.suptitle(fig_title)
    labels = ('state', 'action', 'reward')
    for i in range(3):
        axs[i].plot(log[:, i])
        axs[i].set_ylabel(labels[i])

        if plot_epoch_end:
            epoch_ends = np.where(log[:, -1] == 1)[0]
            for ep in epoch_ends:
                axs[i].axvline(ep, color='red', ls='--')
    axs[-1].set_xlabel('Steps')
    plt.tight_layout()
    plt.show()
