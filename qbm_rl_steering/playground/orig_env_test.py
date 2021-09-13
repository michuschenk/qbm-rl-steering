import qbm_rl_steering.environment.orig_awake_env as awake_sim
import numpy as np
import matplotlib.pyplot as plt


def plot_results(env, label):

    rewards = env.rewards
    initial_states = env.initial_conditions

    iterations = []
    finals = []
    starts = []

    for i in range(len(rewards)):
        if (len(rewards[i]) > 0):
            finals.append(rewards[i][len(rewards[i]) - 1])
            starts.append(-np.sqrt(np.mean(np.square(initial_states[i]))))
            iterations.append(len(rewards[i]))

    plot_suffix = f', number of iterations: {env.TOTAL_COUNTER}'

    fig, axs = plt.subplots(2, 1, constrained_layout=True)

    ax=axs[0]
    ax.plot(iterations)
    ax.set_title('Iterations' + plot_suffix)
    fig.suptitle(label, fontsize=12)

    ax = axs[1]
    color = 'blue'
    ax.set_ylabel('Final RMS', color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax.plot(finals, color=color)
    ax.axhline(env.threshold, ls=':',c='r')
    ax.set_xlabel('Episodes ')
    color = 'lime'
    ax.plot(starts, color=color)
    plt.show()


env = awake_sim.e_trajectory_simENV()
env.action_scale = 3e-4
env.threshold = -0.2
env.MAX_TIME =50


# plot_results(env, 'test')
