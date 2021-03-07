import matplotlib.pyplot as plt
import numpy as np
import dill

from qbm_rl_steering.agents.qbmq import train_and_evaluate_agent


run_type = 'single'
save_agents = False
agent_directory = 'trained_agents/'
n_repeats_scan = 8  # How many times to run the same parameters in scans

# Environment settings
kwargs_env = {
    'n_bits_observation_space': 8,
    'n_actions': 2,
    'simple_reward': True,
    'max_steps_per_episode': 10
}

# RL settings
kwargs_rl = {
    'learning_rate': (0.02, 0.005),
    # 'learning_rate': (0.1,),
    'lr_kwargs': {'learning_rate_schedule': 'linear',
                  'n_warmup': 0},
    'small_gamma': 0.85,  # 0.85
    'exploration_epsilon': (1.0, 0.04),
    'exploration_fraction': 0.8
}

# Graph config and quantum annealing settings
# Commented values are what's in the paper
kwargs_anneal = {
    'annealer_type': 'SQA',
    'kwargs_qpu': {'aws_device':
                   'arn:aws:braket:::device/qpu/d-wave/DW_2000Q_6',
                   's3_location': None},
    'n_graph_nodes': 16,  # nodes of Chimera graph (2 units DWAVE)
    'n_replicas': 10,  # 10
    'n_meas_for_average': 10,  # 30
    'n_annealing_steps': 140,  # 100, 300, it seems that 100 is best
    'big_gamma': (25., 0.8),  # 0.5
    'beta': 2.,  #0.5
}

# Training time steps
total_timesteps = 300  # 500
total_timesteps = 1  # 500

if run_type == 'single':
    make_plots = True
    agent, optimality = train_and_evaluate_agent(
        kwargs_env=kwargs_env, kwargs_rl=kwargs_rl,
        kwargs_anneal=kwargs_anneal, total_timesteps=total_timesteps,
        make_plots=make_plots)
    print(f'Optimality {optimality:.2f} %')

    # Plot weights
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(7, 7))
    for k, val in agent.q_function.w_hh_history.items():
        axs[0].plot(val, label=str(k))
    for k, val in agent.q_function.w_vh_history.items():
        axs[1].plot(val, label=str(k))

    axs[0].set_ylabel(r'$w_{{hh}}$')
    axs[1].set_ylabel(r'$w_{{vh}}$')
    axs[1].set_xlabel('Iteration')

    # axs[0].legend(loc='lower right')
    # axs[1].legend(loc='lower right')
    plt.show()

    if save_agents:
        agent_path = agent_directory + 'single_run.pkl'
        with open(agent_path, 'wb') as fid:
            dill.dump(agent, fid)

elif run_type == '1d_scan':
    make_plots = False

    param_arr = np.array([20, 40, 60, 80, 100, 120, 140, 160])
    f_name = 'n_annealing_steps'
    results = np.zeros((n_repeats_scan, len(param_arr)))

    tot_n_scans = len(param_arr)
    for k, val in enumerate(param_arr):
        print(f'Param. scan nb.: {k + 1}/{tot_n_scans}')

        kwargs_anneal.update({'n_annealing_steps': val})
        # kwargs_anneal.update({'n_replicas': int(val)})
        for m in range(n_repeats_scan):
            agent, results[m, k] = train_and_evaluate_agent(
                kwargs_env=kwargs_env, kwargs_rl=kwargs_rl,
                kwargs_anneal=kwargs_anneal,
                total_timesteps=total_timesteps,
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

    plt.xlabel('n_annealing_steps')
    plt.ylabel('Optimality (%)')
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
