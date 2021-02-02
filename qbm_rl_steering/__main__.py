from environment.env_desc import TargetSteeringEnv
from environment.helpers import plot_response, run_random_trajectories
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy

import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    # Transfer line environment
    my_env = TargetSteeringEnv()

    # To understand environment better, plot response and test random
    # action-taking
    plot_response(my_env)
    run_random_trajectories(my_env)

    # DQN: state and action is discrete: see here:
    # https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
    model = DQN(MlpPolicy, my_env, verbose=1)
    model.learn(total_timesteps=2000)

    # Plot log taking while learning DQN
    log = np.array(my_env.log)
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(7, 6))
    labels = ('state', 'action', 'reward')
    for i in range(3):
        axs[i].plot(log[:, i])
        axs[i].set_ylabel(labels[i])
        # for j in range(n_epochs+1):
        #     axs[i].axvline(j * n_episodes, color='red', ls='--')
    axs[-1].set_xlabel('Iterations')
    plt.tight_layout()
    plt.show()
