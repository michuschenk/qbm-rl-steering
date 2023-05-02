import matplotlib.pyplot as plt
import numpy as np
import dill

from qbm_rl_steering.utils.run_utils import train_and_evaluate_agent


env_type = 'discrete'

run_type = '1d_scan'
save_agents = False
agent_directory = 'trained_agents/'
n_repeats_scan = 10
total_timesteps = 20

# Environment settings
if env_type == 'continuous':
    kwargs_env = {
        'type': 'continuous',
        'n_actions': 2,
        'max_steps_per_episode': 15
    }
elif env_type == 'discrete':
    kwargs_env = {
        'type': 'discrete',
        'n_bits_observation_space': 8,
        'n_actions': 2,
        'max_steps_per_episode': 8
    }
else:
    raise ValueError("env_type must be 'discrete' or 'continuous'.")

# RL settings
kwargs_rl = {
    'learning_rate': (0.04778, 0.000201),
    'small_gamma': 0.756,
    'exploration_epsilon': (1.0, 0.),
    'exploration_fraction': 0.766,
    'replay_batch_size': 32,
    'target_update_frequency': 1,
    'soft_update_factor': 0.4008
}

# Graph config and quantum annealing settings
# Commented values are what's in the paper
kwargs_anneal = {
    'sampler_type': 'SQA',
    'kwargs_qpu': {'aws_device':
                   'arn:aws:braket:::device/qpu/d-wave/DW_2000Q_6',
                   's3_location': None},
    'n_replicas': 1,
    'n_meas_for_average': 16,
    'n_annealing_steps': 180,
    'big_gamma': (41.41, 0.),
    'beta': 0.746
}

if run_type == 'optuna':
    make_plots = False

    import optuna

    def objective(trial):
        lr_i = trial.suggest_float('lr_i', 1e-3, 5e-1, log=True)
        lr_f = trial.suggest_float('lr_f', 1e-4, 1e-2, log=True)
        max_steps = trial.suggest_int("max_steps", 8, 20, log=False)
        batch_size = trial.suggest_int("batch_size", 8, 64, step=8)
        gamma = trial.suggest_float("gamma", 0.7, 0.99, log=False)
        tau = trial.suggest_float("tau", 0.001, 0.99, log=True)
        exp_frac = trial.suggest_float("exp_frac", 0.7, 0.99, log=False)
        anneal_steps = trial.suggest_int("anneal_steps", 30, 200, step=10)
        big_gamma = trial.suggest_float("big_gamma", 5, 50, log=True)
        beta = trial.suggest_float("beta", 0.02, 5., log=True)
        n_meas_avg = trial.suggest_int("n_meas_avg", 1, 30, step=5)

        kwargs_env.update({'max_steps_per_episode': max_steps})
        kwargs_rl.update({'learning_rate': (lr_i, lr_f),  'replay_batch_size': batch_size, 'soft_update_factor': tau,
                          'small_gamma': gamma, 'exploration_fraction': exp_frac})
        kwargs_anneal.update({'n_annealing_steps': anneal_steps, 'big_gamma': (big_gamma, 0), 'beta': beta,
                              'n_meas_for_average': n_meas_avg})

        sum_opt = 0.
        for i in range(10):
            agent, optimality = train_and_evaluate_agent(
                kwargs_env=kwargs_env, kwargs_rl=kwargs_rl,
                kwargs_anneal=kwargs_anneal, total_timesteps=total_timesteps,
                make_plots=make_plots, n_samples_states=200, calc_optimality=True)
            sum_opt += optimality

        return sum_opt

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=150)

elif run_type == 'single':
    make_plots = True
    agent, optimality = train_and_evaluate_agent(
        kwargs_env=kwargs_env, kwargs_rl=kwargs_rl,
        kwargs_anneal=kwargs_anneal, total_timesteps=total_timesteps,
        make_plots=make_plots, n_samples_states=200)
    print(f'Optimality {optimality:.2f} %')

    # Plot weights
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(7, 7))
    for k, val in agent.q_function.w_hh_history.items():
        axs[0].plot(val, label=str(k))
    for k, val in agent.q_function.w_vh_history.items():
        axs[1].plot(val, label=str(k))

    axs[0].set_ylabel(r'$w_{{hh}}$, train net')
    axs[1].set_ylabel(r'$w_{{vh}}$, train net')
    axs[1].set_xlabel('Updates')

    plt.show()

    if save_agents:
        agent_path = agent_directory + 'single_run.pkl'
        with open(agent_path, 'wb') as fid:
            dill.dump(agent, fid)

elif run_type == '1d_scan':
    make_plots = False

    param_arr = np.array([3, 5, 10, 20, 30, 40, 50, 60, 70, 80])
    f_name = 'n_interactions'
    results = np.zeros((n_repeats_scan, len(param_arr)))

    tot_n_scans = len(param_arr)
    for k, val in enumerate(param_arr):
        print(f'Param. scan nb.: {k + 1}/{tot_n_scans}')

        # kwargs_anneal.update({'n_annealing_steps': val})
        # kwargs_anneal.update({'n_replicas': int(val)})
        for m in range(n_repeats_scan):
            agent, results[m, k] = train_and_evaluate_agent(
                kwargs_env=kwargs_env, kwargs_rl=kwargs_rl,
                kwargs_anneal=kwargs_anneal,
                total_timesteps=val,
                make_plots=make_plots,
                calc_optimality=True)

            if save_agents:
                agent_path = agent_directory + f_name + f'{val}_run_{m}.pkl'
                with open(agent_path, 'wb') as fid:
                    dill.dump(agent, fid)

    # Plot scan summary
    plt.figure(1, figsize=(6, 5))
    (h, caps, _) = plt.errorbar(
        param_arr, np.mean(results, axis=0),
        yerr=np.std(results, axis=0) / np.sqrt(n_repeats_scan),
        capsize=4, elinewidth=2, color='tab:red')

    for cap in caps:
        cap.set_color('tab:red')
        cap.set_markeredgewidth(2)

    plt.xlabel('n_steps_train')
    plt.ylabel('Optimality (%)')
    plt.ylim(-10, 110)
    plt.tight_layout()
    plt.show()

else:
    # Assume 2d_scan
    make_plots = False

    param_1 = np.array([5, 10, 15, 20, 25])
    f_name_1 = f'n_meas_'
    param_2 = np.array([2, 6, 10, 14, 18])
    f_name_2 = f'_n_replicas_'

    results = np.zeros((n_repeats_scan, len(param_1), len(param_2)))

    tot_n_scans = len(param_1) * len(param_2)
    for k, val_1 in enumerate(param_1):
        for l, val_2 in enumerate(param_2):
            # FOR LEARNING RATE SCAN ONLY
            # if final learning rate larger than initial, skip
            # if val_1 > val_2:
            #     continue

            print(f'Param. scan nb.: {k+l+1}/{tot_n_scans}')
            for m in range(n_repeats_scan):
                kwargs_anneal.update(
                    {'n_meas_for_average': int(val_1),
                     'n_replicas': int(val_2)})
                # kwargs_rl.update(
                #     {'learning_rate': (val_2, val_1)})
                # kwargs_anneal.update(
                #     {'big_gamma': (25., val_1), 'beta': val_2}
                # )

                agent, results[m, k, l] = train_and_evaluate_agent(
                    kwargs_env=kwargs_env, kwargs_rl=kwargs_rl,
                    kwargs_anneal=kwargs_anneal,
                    total_timesteps=total_timesteps,
                    make_plots=make_plots,
                    calc_optimality=True)

                if save_agents:
                    agent_path = (
                        agent_directory + f_name_1 + f'{val_1}' +
                        f_name_2 + f'{val_2}_run_{m}.pkl')
                    with open(agent_path, 'wb') as fid:
                        dill.dump(agent, fid)

    # Plot scan summary, mean
    plt.figure(1, figsize=(6, 5))
    plt.imshow(np.flipud(np.mean(results, axis=0).T))
    cbar = plt.colorbar()

    plt.xticks(range(len(param_1)),
               labels=[i for i in param_1])
    plt.yticks(range(len(param_2)),
               labels=[i for i in param_2[::-1]])

    plt.ylabel('n_replicas')
    plt.xlabel('n_meas_for_average')
    cbar.set_label('Mean optimality (%)')
    plt.tight_layout()
    plt.savefig('mean_res.png', dpi=300)
    plt.show()

    # Plot scan summary, std
    plt.figure(2, figsize=(6, 5))
    plt.imshow(np.flipud(np.std(results, axis=0).T/np.sqrt(n_repeats_scan)))
    cbar = plt.colorbar()

    plt.xticks(range(len(param_1)),
               labels=[i for i in param_1])
    plt.yticks(range(len(param_2)),
               labels=[i for i in param_2[::-1]])

    plt.ylabel('n_replicas')
    plt.xlabel('n_meas_for_average')
    cbar.set_label('Std. optimality (%)')
    plt.tight_layout()
    plt.savefig('std_res.png', dpi=300)
    plt.show()
