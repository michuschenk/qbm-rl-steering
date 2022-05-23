import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

import qbm_rl_steering.utils.helpers as hlp
from environment.env_1D_continuous import TargetSteeringEnv


def test_environment() -> TargetSteeringEnv:
    """ To understand environment better, plot response and test random
    action-taking.
    :return env: TargetSteering environment """
    env = TargetSteeringEnv()
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
        exploration_final_eps=0.0, exploration_fraction=0.5, train_freq=3,
        learning_rate=5e-4, target_update_interval=100, tau=0.05)

    # Update dqn_kwargs dictionary by adding (or replacing) scan parameters.
    if scan_params is not None:
        dqn_kwargs.update(scan_params)

    return DQN(**dqn_kwargs)


def evaluate_performance(n_evaluations: int = 30, n_steps_train: int = 2000,
                         n_episodes_test: int = 300,
                         max_steps_per_episode: int = 20,
                         scan_params: dict = None, make_plots: bool = False) \
        -> (np.ndarray, np.ndarray):
    """ Evaluate performance of agent for the scan params and return
    np.arrays containing the average and standard deviation of the two
    metrics defined in helpers.calculate_performance_metric(..).
    :param n_evaluations: number of full from-scratch-trainings of the agent
    :param n_steps_train: number of training steps per evaluation
    :param n_episodes_test: number of episodes to evaluate performance
    :param max_steps_per_episode: number of steps per episode (abort criterion)
    :param scan_params: dictionary of parameters that we scan
    :param make_plots: flag to decide whether to show plots or not
    :return: average and std. dev of both performance metric. """
    if scan_params is None:
        print('Running performance test with default parameters')

    metric = np.zeros(n_evaluations)
    tqdm_pbar = tqdm(range(n_evaluations), ncols=80, position=0,
                     desc='Evaluations: ', leave=False)
    for j in tqdm_pbar:
        # Initialize environment and agent
        env = TargetSteeringEnv(
            max_steps_per_episode=max_steps_per_episode)
        agent = init_agent(env, scan_params)

        # Evaluate agent before training
        # hlp.evaluate_agent(env, agent, n_episodes=n_episodes_test,
        #                    make_plot=make_plots,
        #                    fig_title='Agent test before training')

        # Run agent training
        agent.learn(total_timesteps=n_steps_train)

        if make_plots:
            hlp.plot_log(env, fig_title='Agent training')

        agent.save('dqn_transferline')

        # Run evaluation of trained agent
        test_env = TargetSteeringEnv(
            max_steps_per_episode=max_steps_per_episode)
        test_agent = DQN.load('dqn_transferline')
        hlp.evaluate_agent(
            test_env, test_agent, n_episodes=n_episodes_test,
            make_plot=make_plots, fig_title='Agent test after training')

        # Show Q-net of a trained agent
        if make_plots:
            env = TargetSteeringEnv(
                max_steps_per_episode=max_steps_per_episode)
            hlp.plot_q_net_response(env, agent, 'Q-net response, trained agent')

        # Calculate performance metric
        metric[j] = hlp.calculate_policy_optimality(
            env=test_env, agent=test_agent)

    metric_avg = np.mean(metric)
    metric_std = np.std(metric) / np.sqrt(n_evaluations)

    return metric_avg, metric_std


def show_scan_result(scan_values: np.ndarray, metric_avg: np.ndarray,
                     metric_std: np.ndarray, scenario: str):
    """
    Plot the success metric for the scanned values.
    :param scan_values: values of the scan parameters
    :param metric_avg: performance metric, mean over all evaluations
    :param metric_std: performance metric, std. dev. over all evaluations
    :param scenario: name of the scan scenario, will be used as x-label
    :return: None
    """
    fig = plt.figure(1, figsize=(7, 5.5))
    fig.suptitle('Performance evaluation')
    ax1 = plt.gca()
    (h, caps, _) = ax1.errorbar(
        x=scan_values, y=metric_avg, yerr=metric_std,
        c='tab:red', capsize=4, elinewidth=2)

    for cap in caps:
        cap.set_color('tab:red')
        cap.set_markeredgewidth(2)

    ax1.set_xlabel(scenario)
    ax1.set_ylabel('Optimality (%)')
    ax1.set_ylim(-5., 105.)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # TODO: improve scans in a way similar to ferl.py, e.g. adding kwargs for
    #  all the parameters and just separating by scan type (single, 1d, 2d).

    # env = test_environment()

    # Scenarios for parameter scans
    scan_scenarios = {
        'exploration_fraction': np.arange(0., 1.1, 0.1),
        'n_steps_train': np.arange(500, 101500, 20000),
        # 'n_steps_train': np.array([30, 90, 180, 270, 360, 450, 540, 630, 720]),
        'target_update_interval': np.arange(500, 3100, 500),
        'max_steps_per_episode': np.arange(4, 25, 3),
        'gamma': np.arange(0.75, 1.02, 0.05),
        'net_arch_layer_nodes': np.array([8, 16, 32, 64, 96, 128]),
        'net_arch_hidden_layers': np.array([1, 2, 3]),
        'single_default': np.array([1]),
        'tau': np.linspace(0., 0.1, 6)
    }

    scenario = 'n_steps_train'
    scan_values = scan_scenarios[scenario]

    # Run the scan (adapt the correct kwarg)
    metric_avg = np.zeros(len(scan_values))
    metric_std = np.zeros(len(scan_values))

    tqdm_scan_values = tqdm(scan_values, ncols=80, position=1, desc='Total: ')
    for i, val in enumerate(tqdm_scan_values):
        scan_params = dict(
            exploration_fraction=0.989, exploration_final_eps=0.,
            policy_kwargs=dict(net_arch=[8] * 2),
            gamma=0.759, tau=0.0001295, learning_rate=0.02863,
            target_update_interval=30, train_freq=1,  # 100,  3
            batch_size=80, buffer_size=100000)

        metric_avg[i], metric_std[i] = evaluate_performance(
            scan_params=scan_params,
            n_steps_train=val, max_steps_per_episode=13,
            n_evaluations=10, make_plots=False)

    show_scan_result(scan_values, metric_avg, metric_std, scenario)

    # import optuna
    #
    # def objective(trial):
    #     lr = trial.suggest_float('lr', 1e-3, 5e-1, log=True)
    #     train_freq = trial.suggest_int("train_freq", 1, 10, step=1)
    #     batch_size = trial.suggest_int("batch_size", 8, 128, step=8)
    #     gamma = trial.suggest_float("gamma", 0.6, 0.99, log=False)
    #     # n_nodes = trial.suggest_int("n_nodes", 32, 256, step=32)
    #     tau = trial.suggest_float("tau", 0.0001, 0.005, log=True)
    #     exp_frac = trial.suggest_float("exp_frac", 0.7, 0.99, log=False)
    #     update_interv = trial.suggest_int("update_interv", 1, 200, log=True)
    #     max_steps = trial.suggest_int("max_steps", 8, 20, log=False)
    #
    #     scan_params = dict(
    #             exploration_fraction=exp_frac, exploration_final_eps=0.,
    #             policy_kwargs=dict(net_arch=[8] * 2),
    #             gamma=gamma, tau=tau, learning_rate=lr,
    #             target_update_interval=update_interv, train_freq=train_freq,
    #             batch_size=batch_size, buffer_size=100000)
    #
    #     metric, _ = evaluate_performance(
    #                 scan_params=scan_params,
    #                 n_steps_train=10000, max_steps_per_episode=max_steps,
    #                 n_evaluations=10, make_plots=False)
    #
    #     return metric
    #
    # study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=150)
