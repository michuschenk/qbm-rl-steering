import matplotlib.pyplot as plt
import numpy as np
import pickle


def plot_agent_evaluation(
        states_q: np.ndarray, q_values: np.ndarray, best_action: np.ndarray,
        visited_states: np.ndarray, x_reward_thresh: float) -> None:
    """
    Plot the evaluation of the agent after training.
    """
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(7, 7))
    fig.suptitle('QBM agent evaluation')

    # Value functions
    cols = ['tab:red', 'tab:blue']
    for i in range(q_values.shape[1]):
        axs[0].plot(1e3 * states_q, q_values[:, i], c=cols[i], label=f'Action {i}')
    axs[0].axvline(1e3*x_reward_thresh, c='grey', ls='--', label='Terminal state\nboundaries')
    axs[0].axvline(-1e3*x_reward_thresh, c='grey', ls='--')
    axs[0].axvspan(-1e3*x_reward_thresh, 1e3*x_reward_thresh, color='grey', alpha=0.3, label='Terminal states')
    axs[0].set_ylabel('Q value')
    axs[0].legend(loc='upper left', bbox_to_anchor=(1., 1.))

    # Plot policy
    for a in list(set(best_action)):
        msk_a = best_action == a
        axs[1].plot(1e3 * states_q[msk_a], best_action[msk_a],
                    marker='o', ms=3, ls='None', c=cols[a])
    axs[1].axvline(1e3*x_reward_thresh, c='grey', ls='--')
    axs[1].axvline(-1e3*x_reward_thresh, c='grey', ls='--')
    axs[1].axvspan(-1e3*x_reward_thresh, 1e3*x_reward_thresh, color='grey', alpha=0.3)
    axs[1].set_yticks([0, 1])
    axs[1].set_ylabel('Greedy policy')

    # What states have been visited and how often?
    axs[2].hist(1e3*np.array(visited_states), bins=128, color='tab:green')
    axs[2].axvline(1e3*x_reward_thresh, c='grey', ls='--')
    axs[2].axvline(-1e3*x_reward_thresh, c='grey', ls='--')
    axs[2].axvspan(-1e3*x_reward_thresh, 1e3*x_reward_thresh, color='grey', alpha=0.3)
    axs[2].set_xlabel('State (mm)')
    axs[2].set_ylabel('# state visits')

    plt.subplots_adjust(right=0.74)
    plt.savefig('paper/Qlearn_cont/004_qlearn_ferl_cont_optimality.pdf')
    plt.show()


with open('paper/Qlearn_cont/eval_ferl_optimality_cont.pkl', 'rb') as fid:
    res_ferl = pickle.load(fid)

plot_agent_evaluation(res_ferl['states_q'], res_ferl['q_values'], res_ferl['best_action'],
                      res_ferl['visited_states'], res_ferl['x_reward_thresh'])

