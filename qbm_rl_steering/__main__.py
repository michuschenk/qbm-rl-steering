import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

import environment.helpers as hlp
from environment.env_desc import TargetSteeringEnv

N_BITS_OBSERVATION_SPACE = 8


def test_environment() -> TargetSteeringEnv:
    """ To understand environment better, plot response and test random
    action-taking.
    :return env: TargetSteering environment """
    env = TargetSteeringEnv(N_BITS_OBSERVATION_SPACE)
    check_env(env)
    hlp.plot_response(env, fig_title='Env. test: response function')
    hlp.run_random_trajectories(
        env, fig_title='Env test: random trajectories', n_episodes=15)
    return env


def init_agent(env: TargetSteeringEnv, scan_params: dict = None) -> DQN:
    """ Initialize an agent for training.
    :param env: OpenAI gym environment.
    :param scan_params: dictionary with additional keyword arguments for DQN
    or arguments to overwrite (this can also be overwriting the policy_kwargs)
    :return new instance of DQN agent. """
    policy_kwargs = dict(net_arch=[128, 128])
    dqn_kwargs = dict(
        policy='MlpPolicy', env=env, verbose=0, learning_starts=0,
        policy_kwargs=policy_kwargs, exploration_initial_eps=1.0,
        exploration_final_eps=0., exploration_fraction=0.5)

    # Update dqn_kwargs dictionary by adding (or replacing) scan parameters.
    if scan_params is not None:
        dqn_kwargs.update(scan_params)

    return DQN(**dqn_kwargs)


def evaluate_performance(n_evaluations: int = 30, n_steps_train: int = 2000,
                         n_episodes_test: int = 1000,
                         max_steps_per_episode: int = 20,
                         scan_params: dict = None, make_plots: bool = False) \
        -> (np.ndarray, np.ndarray):
    """ Evaluate performance of agent for the scan params and return
    np.arrays containing the average and standard deviation of the two
    metrics defined in helpers.calculate_performance_metrics(..).
    :param n_evaluations: number of full from-scratch-trainings of the agent
    :param n_steps_train: number of training steps per evaluation
    :param n_episodes_test: number of episodes to evaluate performance
    :param max_steps_per_episode: number of steps per episode (abort criterion)
    :param scan_params: dictionary of parameters that we scan
    :param make_plots: flag to decide whether to show plots or not
    :return: average and std. dev of both performance metrics. """
    if scan_params is None:
        print('Running performance test with default parameters')

    metrics = np.zeros((2, n_evaluations))
    tqdm_pbar = tqdm(range(n_evaluations), ncols=80, position=0,
                     desc='Evaluations: ', leave=False)
    for j in tqdm_pbar:
        # Initialize environment and agent
        env = TargetSteeringEnv(
            N_BITS_OBSERVATION_SPACE,
            max_steps_per_episode=max_steps_per_episode)
        agent = init_agent(env, scan_params)

        # Evaluate agent before training
        hlp.evaluate_agent(env, agent, n_episodes=n_episodes_test,
                           make_plot=make_plots,
                           fig_title='Agent test before training')

        # Run agent training
        agent.learn(total_timesteps=n_steps_train)
        if make_plots:
            hlp.plot_log(env, fig_title='Agent training')

        agent.save('dqn_transferline')

        # Run evaluation of trained agent
        test_env = TargetSteeringEnv(
            N_BITS_OBSERVATION_SPACE,
            max_steps_per_episode=max_steps_per_episode)
        test_agent = DQN.load('dqn_transferline')
        hlp.evaluate_agent(
            test_env, test_agent, n_episodes=n_episodes_test,
            make_plot=make_plots, fig_title='Agent test after training')

        # Calculate performance metrics
        metrics[:, j] = hlp.calculate_performance_metrics(test_env)

    metrics_avg = np.mean(metrics, axis=1)
    metrics_std = np.std(metrics, axis=1) / np.sqrt(n_evaluations)

    return metrics_avg, metrics_std


if __name__ == "__main__":
    # env = test_environment()

    # Scenarios for parameter scans
    scan_scenarios = {
        'exploration_fraction': np.arange(0., 1.1, 0.1),
        'n_steps_train': np.arange(500, 40500, 10000),
        'target_update_interval': np.arange(500, 3100, 500),
        'max_steps_per_episode': np.arange(10, 45, 5),
        'gamma': np.arange(0.9, 0.991, 0.02),
        'net_arch_layer_nodes': np.array([32, 64, 128, 256]),
        'net_arch_hidden_layers': np.array([1, 2, 3]),
        'single_default': np.array([1])
    }
    scenario = 'n_steps_train'
    scan_values = scan_scenarios[scenario]

    # Run the scan (adapt the correct kwarg)
    metrics_avg = np.zeros((2, len(scan_values)))
    metrics_std = np.zeros((2, len(scan_values)))

    tqdm_scan_values = tqdm(scan_values, ncols=80, position=1, desc='Total: ')
    for i, val in enumerate(tqdm_scan_values):
        metrics_avg[:, i], metrics_std[:, i] = evaluate_performance(
            scan_params=dict(
                exploration_fraction=0.8,
                policy_kwargs=dict(net_arch=[8]*2),
                gamma=0.95),
            n_steps_train=val, max_steps_per_episode=15, make_plots=False,
            n_evaluations=5)

    # Final plot
    fig = plt.figure(1, figsize=(7, 5.5))
    fig.suptitle('Performance evaluation')
    ax1 = plt.gca()
    (h, caps, _) = ax1.errorbar(
        x=scan_values, y=metrics_avg[0, :], yerr=metrics_std[0, :],
        c='tab:red', capsize=4, elinewidth=2)

    for cap in caps:
        cap.set_color('tab:red')
        cap.set_markeredgewidth(2)

    ax1.set_xlabel(scenario)
    ax1.set_ylabel('Fraction of successes')
    ax1.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.show()
