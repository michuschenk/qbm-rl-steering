import matplotlib.pyplot as plt
import numpy as np
import dill

from qbm_rl_steering.agents.qbmq import train_and_evaluate_agent


run_type = 'single'
save_agents = False
agent_directory = 'trained_agents/'
n_repeats_scan = 6  # How many times to run the same parameters in scans

# Environment settings
kwargs_env = {
    'n_bits_observation_space': 8,
    'n_actions': 2,
    'simple_reward': True,
    'max_steps_per_episode': 25
}

# RL settings
kwargs_rl = {
    'learning_rate': (0.001, 0.0005),
    'small_gamma': 0.9,  # 0.85
    'exploration_epsilon': (1.0, 0.04),
    'exploration_fraction': 0.6
}

# Graph config and quantum annealing settings
# Commented values are what's in the paper
kwargs_anneal = {
    'annealer_type': 'SA',
    'kwargs_qpu': {'aws_device':
                   'arn:aws:braket:::device/qpu/d-wave/DW_2000Q_6',
                   's3_location': None},
    'n_graph_nodes': 16,  # nodes of Chimera graph (2 units DWAVE)
    'n_replicas': 25,  # 25
    'n_meas_for_average': 150,  # 20, 150
    'n_annealing_steps': 1000,  # 100, 300, it seems that 100 is best
    'big_gamma': 0.01,
    'beta': (0.01, 2.),
}

# Training time steps
total_timesteps = 500  # 500

if run_type == 'single':
    make_plots = True
    agent, optimality = train_and_evaluate_agent(
        kwargs_env=kwargs_env, kwargs_rl=kwargs_rl,
        kwargs_anneal=kwargs_anneal, total_timesteps=total_timesteps,
        make_plots=make_plots)
    print(f'Optimality {optimality:.2f} %')

    if save_agents:
        agent_path = agent_directory + 'single_run.pkl'
        with open(agent_path, 'wb') as fid:
            dill.dump(agent, fid)

elif run_type == '1d_scan':
    make_plots = False

    param_arr = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50])
    f_name = 'n_replicas_'
    results = np.zeros((n_repeats_scan, len(param_arr)))

    tot_n_scans = len(param_arr)
    for k, val in enumerate(param_arr):
        print(f'Param. scan nb.: {k + 1}/{tot_n_scans}')

        kwargs_anneal.update({'n_replicas': int(val)})
        for m in range(n_repeats_scan):
            agent, results[m, k] = train_and_evaluate_agent(
                kwargs_env=kwargs_env, kwargs_rl=kwargs_rl,
                kwargs_anneal=kwargs_anneal,
                total_timesteps=total_timesteps,
                make_plots=make_plots)

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

    plt.xlabel('n_replicas')
    plt.ylabel('Optimality (%)')
    plt.tight_layout()
    plt.show()

else:
    # Assume 2d_scan
    make_plots = False

    param_1 = np.array([25., 20., 15., 10.])
    f_name_1 = f'G_i_'
    param_2 = np.array([1., 0.5, 0.2, 0.1])
    f_name_2 = f'_G_f_'

    results = np.zeros((n_repeats_scan, len(param_1), len(param_2)))

    tot_n_scans = len(param_1) * len(param_2)
    for k, val_1 in enumerate(param_1):
        for l, val_2 in enumerate(param_2):
            print(f'Param. scan nb.: {k+l+1}/{tot_n_scans}')
            for m in range(n_repeats_scan):
                kwargs_anneal.update(
                    {'big_gamma': (val_1, val_2)})
                # kwargs_rl.update(
                #     {'learning_rate': (val_1, val_2)})

                agent, results[m, k, l] = train_and_evaluate_agent(
                    kwargs_env=kwargs_env, kwargs_rl=kwargs_rl,
                    kwargs_anneal=kwargs_anneal,
                    total_timesteps=total_timesteps,
                    make_plots=make_plots)

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

    plt.xticks(range(len(param_2)),
               labels=[i for i in param_2])
    plt.yticks(range(len(param_1)),
               labels=[i for i in param_1[::-1]])

    plt.xlabel('G_f')
    plt.ylabel('G_i')
    cbar.set_label('Mean optimality (%)')
    plt.tight_layout()
    plt.savefig('mean_res.png', dpi=300)
    plt.show()

    # Plot scan summary, std
    plt.figure(2, figsize=(6, 5))
    plt.imshow(np.flipud(np.std(results, axis=0).T/np.sqrt(n_repeats_scan)))
    cbar = plt.colorbar()

    plt.xticks(range(len(param_2)),
               labels=[i for i in param_2])
    plt.yticks(range(len(param_1)),
               labels=[i for i in param_1[::-1]])

    plt.xlabel('lr_f')
    plt.ylabel('lr_i')
    cbar.set_label('Std. optimality (%)')
    plt.tight_layout()
    plt.savefig('std_res.png', dpi=300)
    plt.show()
