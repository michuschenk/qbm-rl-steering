import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from ray.util.multiprocessing import Pool

import datetime
import shutil
import pickle

from functools import partial

import numpy as np


def run_full(params, process_id=None):
    from tensorflow.keras.optimizers.schedules import (ExponentialDecay,
                                                       PolynomialDecay,
                                                       PiecewiseConstantDecay)

    from qbm_rl_steering.core.ddpg_agents import ClassicalDDPG, QuantumDDPG
    from qbm_rl_steering.environment.rms_env_nd import RmsSteeringEnv
    from qbm_rl_steering.core.run_utils import (trainer, evaluator,
                                                plot_training_log,
                                                plot_evaluation_log)

    import pandas as pd


    env = RmsSteeringEnv(
        n_dims=params['env/n_dims'],
        max_steps_per_episode=params['env/max_steps_per_episode'],
        required_steps_above_reward_threshold=
        params['env/required_steps_above_reward_threshold'])

    # Learning rate schedules: lr_critic = 5e-4, lr_actor = 1e-4
    #lr_schedule_critic = ExponentialDecay(params['lr/init'],
    #                                      params['n_episodes'],
    #                                      params['lr/decay_factor'])
    #lr_schedule_actor = ExponentialDecay(params['lr/init'],
    #                                     params['n_episodes'],
    #                                     params['lr/decay_factor'])

    #lr_schedule_critic = PolynomialDecay(params['lr/init'],
    #                                     params['n_episodes'],
    #                                     end_learning_rate=params['lr/final'])

    #lr_schedule_actor = PolynomialDecay(params['lr/init'],
    #                                    params['n_episodes'],
    #                                    end_learning_rate=params['lr/final'])

    if params['quantum_ddpg']:
        agent = QuantumDDPG(state_space=env.observation_space,
                            action_space=env.action_space,
                            learning_rate_schedule_critic=lr_schedule_critic,
                            learning_rate_schedule_actor=lr_schedule_actor,
                            grad_clip_actor=1e4, grad_clip_critic=1.,
                            gamma=params['agent/gamma'],
                            tau_critic=params['agent/tau'],
                            tau_actor=params['agent/tau']
                            )
    else:
        agent = ClassicalDDPG(state_space=env.observation_space,
                              action_space=env.action_space,
                              learning_rate_critic=params['lr/init'],
                              learning_rate_actor=params['lr/init'],
                              grad_clip_actor=np.inf, grad_clip_critic=np.inf,
                              gamma=params['agent/gamma'],
                              tau_critic=params['agent/tau'],
                              tau_actor=params['agent/tau']
                              )

    # Action noise schedule
    action_noise_schedule = PolynomialDecay(
        params['action_noise/init'], params['n_episodes'],
        params['action_noise/final'])

    # Epsilon greedy schedule
    epsilon_greedy_schedule = PolynomialDecay(
        params['epsilon_greedy/init'], params['n_episodes'],
        params['epsilon_greedy/final'])

    # Schedule n_anneals
    t_transition = [int(x * params['n_episodes']) for x in
                    np.linspace(0, 1., params['anneals/n_pieces'] + 1)][1:-1]
    y_transition = [int(n) for n in np.linspace(params['anneals/init'],
                                                params['anneals/final'],
                                                params['anneals/n_pieces'])]
    n_anneals_schedule = PiecewiseConstantDecay(t_transition, y_transition)

    # PREPARE OUTPUT FOLDER
    date_time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    out_path = './runs/indiv/' + date_time_now
    if process_id is not None:
        out_path = out_path + f'_pid_{process_id}'
    os.makedirs(out_path)
    shutil.copy('./run_ddpg.py', out_path + '/run_ddpg.py')
    shutil.copy('./core/ddpg_agents.py', out_path + '/ddpg_agents.py')
    shutil.copy('./environment/rms_env_nd.py', out_path + '/rms_env_nd.py')
    with open(out_path + '/params_dict.pkl', 'wb') as fid:
        pickle.dump(params, fid)

    # AGENT TRAINING
    episode_log = trainer(
        env=env, agent=agent, action_noise_schedule=action_noise_schedule,
        epsilon_greedy_schedule=epsilon_greedy_schedule,
        n_anneals_schedule=n_anneals_schedule, n_episodes=params['n_episodes'],
        max_steps_per_episode=params['env/max_steps_per_episode'],
        batch_size=params['trainer/batch_size'],
        n_exploration_steps=params['trainer/n_exploration_steps'],
        n_episodes_early_stopping=params['trainer/n_episodes_early_stopping']
    )
    plot_training_log(env, agent, episode_log, save_path=out_path)
    df_train_log = pd.DataFrame(episode_log)
    df_train_log.to_csv(out_path + '/train_log')

    # AGENT EVALUATION
    # a) Random state inits
    env = RmsSteeringEnv(
        n_dims=params['env/n_dims'],
        max_steps_per_episode=params['env/max_steps_per_episode'],
        required_steps_above_reward_threshold=
        params['env/required_steps_above_reward_threshold'])
    eval_log_random = evaluator(env, agent, n_episodes=100, reward_scan=False)
    try:
        df_eval_log = pd.DataFrame({'rewards': eval_log_random})
    except ValueError:
        print('Issue creating eval df ... probably all evaluations '
              'used same number of steps')

        n_stp = eval_log_random.shape[1]
        res_dict = {}
        for st in range(n_stp):
            res_dict[f'step_{st}'] = eval_log_random[:, st]
        df_eval_log = pd.DataFrame(res_dict)

    df_eval_log.to_csv(out_path + '/eval_log_random')
    plot_evaluation_log(env, params['env/max_steps_per_episode'],
                        eval_log_random, save_path=out_path, type='random')

    # b) Systematic state inits
    env = RmsSteeringEnv(
        n_dims=params['env/n_dims'],
        max_steps_per_episode=params['env/max_steps_per_episode'],
        required_steps_above_reward_threshold=
        params['env/required_steps_above_reward_threshold'])
    eval_log_scan = evaluator(env, agent, n_episodes=100, reward_scan=True)
    try:
        df_eval_log = pd.DataFrame({'rewards': eval_log_scan})
    except ValueError:
        print('Issue creating eval df ... probably all evaluations used '
              'same number of steps')

        n_stp = eval_log_scan.shape[1]
        res_dict = {}
        for st in range(n_stp):
            res_dict[f'step_{st}'] = eval_log_scan[:, st]
        df_eval_log = pd.DataFrame(res_dict)

    df_eval_log.to_csv(out_path + '/eval_log_scan')
    plot_evaluation_log(env, params['env/max_steps_per_episode'],
                        eval_log_scan, save_path=out_path, type='scan')

    return eval_log_random, eval_log_scan


def run_worker(scan_param_value, default_params, scan_param_name):

    n_stats = 1

    # Copy default parameters and overwrite the corresponding scan parameter
    params = default_params.copy()
    params.update({scan_param_name: scan_param_value})

    results_worker = {
        scan_param_value: {'random': {'steps_avg': np.zeros(n_stats),
                                      'steps_max': np.zeros(n_stats)},
                           'scan': {'steps_avg': np.zeros(n_stats),
                                    'steps_max': np.zeros(n_stats)}}
    }

    process_id = os.getpid()
    for i in range(n_stats):
        print('===================================================')
        print(f'SCAN: {scan_param_name.upper()}: {scan_param_value}')
        print(f'Running stats iteration {i+1}/{n_stats}')
        print('===================================================\n')

        eval_log_random, eval_log_scan = run_full(params, process_id)

        # Count number of steps per episode (RANDOM evaluation)
        n_steps = np.array([(len(rew) - 1) for rew in eval_log_random])
        max_n_steps = np.max(n_steps)
        avg_n_steps = np.mean(n_steps)
        results_worker[scan_param_value]['random']['steps_avg'][i] = avg_n_steps
        results_worker[scan_param_value]['random']['steps_max'][i] = max_n_steps

        # Count number of steps per episode (SCAN evaluation)
        n_steps = np.array([(len(rew) - 1) for rew in eval_log_scan])
        max_n_steps = np.max(n_steps)
        avg_n_steps = np.mean(n_steps)
        results_worker[scan_param_value]['scan']['steps_avg'][i] = avg_n_steps
        results_worker[scan_param_value]['scan']['steps_max'][i] = max_n_steps

    try:
        os.makedirs('runs/bkp_res')
    except FileExistsError:
        print('Folder "runs/bkp_res" already exists.')
        pass
    fname = (f'runs/bkp_res/{scan_param_name.replace("/", "_")}_' +
             f'{scan_param_value}.pkl')
    with open(fname, 'wb') as fid:
        pickle.dump(results_worker, fid)

    return results_worker


def plot_scan_results(all_res_fname):

    import matplotlib.pyplot as plt

    with open(all_res_fname, 'rb') as fid:
        res = pickle.load(fid)

    eval_type = 'random'
    scan_param_name = fname.split('/')[-1].split('.pkl')[0]
    scan_param_vals = np.array(list(res.keys()))
    n_stats = len(res[scan_param_vals[0]][eval_type]['steps_avg'])
    scan_avg_steps = np.zeros((n_stats, len(scan_param_vals)))
    scan_max_steps = np.zeros((n_stats, len(scan_param_vals)))

    for i, sv in enumerate(scan_param_vals):
        scan_avg_steps[:, i] = res[sv][eval_type]['steps_avg'][:]
        scan_max_steps[:, i] = res[sv][eval_type]['steps_max'][:]

    mean_steps_avg = np.mean(scan_avg_steps, axis=0)
    std_steps_avg = np.std(scan_avg_steps, axis=0)
    mean_steps_max = np.mean(scan_max_steps, axis=0)
    std_steps_max = np.std(scan_max_steps, axis=0)

    plt.figure()
    plt.suptitle(f'Param. {scan_param_name.upper()}, with eval. type {eval_type.upper()}',
                 fontsize=10)
    plt.plot(scan_param_vals, mean_steps_avg, marker='x', c='tab:blue', label='Avg.')
    plt.fill_between(scan_param_vals, mean_steps_avg - std_steps_avg,
                     mean_steps_avg + std_steps_avg, color='tab:blue', alpha=0.5)
    plt.plot(scan_param_vals, mean_steps_max, marker='x', c='tab:red', label='Max')
    plt.fill_between(scan_param_vals, mean_steps_max - std_steps_max,
                     mean_steps_max + std_steps_max, color='tab:red', alpha=0.5)
    plt.xlabel(scan_param_name)
    plt.ylabel('# steps')
    plt.legend(loc='best')
    plt.savefig(f'SCAN_RES_{scan_param_name}.png', dpi=150)
    plt.close()


if __name__ == '__main__':

    default_params = {
        'quantum_ddpg': False,
        'n_episodes': 2000,
        'env/n_dims': 6,
        'env/max_steps_per_episode': 25,
        'env/required_steps_above_reward_threshold': 1,
        'trainer/batch_size': 100,
        'trainer/n_exploration_steps': 100,
        'trainer/n_episodes_early_stopping': 1000,
        'agent/gamma': 0.99,
        'agent/tau': 0.005,
        # 'agent/tau_actor': 0.005,
        'lr/init': 2e-3,
        'lr/final': 2e-3,
        'lr/decay_factor': 1.,
        #'lr_actor/init': 1e-4,
        #'lr_actor/decay_factor': 1.,
        'action_noise/init': 0.2,
        'action_noise/final': 0.2,
        'epsilon_greedy/init': 0.,
        'epsilon_greedy/final': 0.,
        'anneals/n_pieces': 2,
        'anneals/init': 1,
        'anneals/final': 2,
    }

    # TODO: implement 2D parameter scans.

    # Scan definition
    # scan_param_name = 'action_noise/max_steps_per_episode'
    # scan_param_values = np.array([0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    # scan_param_values = np.arange(0., 0.55, 0.05)
    # scan_param_name = 'epsilon_greedy/init'
    # scan_param_values = [1, 2, 5, 10, 25, 50, 75]
    # scan_param_name = 'anneals/final'
    # scan_param_name = 'lr_actor/init'
    # scan_param_values = np.array([2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5])
    # scan_param_name = 'n_episodes'
    # scan_param_values = np.array([100, 200, 300, 400, 500, 600, 700])  #, 400, 500, 600, 700])

    # scan_param_name = 'env/max_steps_per_episode'
    # scan_param_values = np.array([5, 10, 15, 20, 25, 30])
    # scan_param_name = 'trainer/batch_size'
    # scan_param_values = np.array([16, 32, 48, 64, 128, 256])

    scan_param_name = 'n_episodes'
    scan_param_values = np.array([50, 100, 200, 400, 600])

    #scan_param_name = 'agent/tau'
    #scan_param_values = np.array([0.0001, 0.0003, 0.0005, 0.0008, 0.001, 0.002])

    # Run scan
    results_all = {}
    with Pool(max(len(scan_param_values), 32)) as p:
        f = partial(run_worker, default_params=default_params,
                    scan_param_name=scan_param_name)
        res = p.map(f, scan_param_values)
        for r in res:
            results_all.update(r)

    # Save all results
    try:
        os.makedirs('runs/all_res')
    except FileExistsError:
        print('Folder "runs/all_res" already exists.')
        pass
    fname = (f'runs/all_res/{scan_param_name.replace("/", "_")}.pkl')
    with open(fname, 'wb') as fid:
        pickle.dump(results_all, fid)

    # Create overview plot
    plot_scan_results(fname)


"""
# 1D SCANS
max_steps_per_episode = [25, 50, 75]
# batch_size = [12, 24, 48]
# n_anneals = [1, 5, 25, 50]
action_noise = ...
n_steps_inside_reward_objective = ...
learning_rate_decay = ...

# 2D SCANS
# learning rates: (init) critic vs. actor = [2e-3, 1e-3, 5e-4] x [1e-3, 5e-4, 1e-4]
# tau_critic x tau_actor scan
tau_critic = [0.05, 0.1]
tau_actor = [0.05, 0.1]
tau_results = {
    'steps_avg': np.zeros((n_runs_stats, len(tau_critic), len(tau_actor))),
    'steps_max': np.zeros((n_runs_stats, len(tau_critic), len(tau_actor)))}

params = default_params.copy()
for i, tc in enumerate(tau_critic):
    for ii, ta in enumerate(tau_actor):
        params.update(
            {'agent': {'gamma': 0.99, 'tau_critic': tc, 'tau_actor': ta}})

        for j in range(n_runs_stats):
            print('==================================')
            print(f'SCAN IV: TAU_CRITIC: {tc}, TAU_ACTOR: {ta}')
            print(f'Running stats iteration {j+1}/{n_runs_stats}')
            print('==================================\n')
            eval_log = run_full(params)

            # count number of steps per episode
            n_steps = np.array([(len(r) - 1) for r in eval_log])
            max_n_steps = np.max(n_steps)
            avg_n_steps = np.mean(n_steps)

            tau_results['steps_avg'][j, i, ii] = avg_n_steps
            tau_results['steps_max'][j, i, ii] = max_n_steps

# Show results
mean_steps_avg = np.mean(tau_results['steps_avg'], axis=0)
std_steps_avg = np.std(tau_results['steps_avg'], axis=0)
mean_steps_max = np.mean(tau_results['steps_max'], axis=0)
std_steps_max = np.std(tau_results['steps_max'], axis=0)

plt.figure()
pc = plt.pcolormesh(tau_critic, tau_actor, tau_results['steps_avg'],
                    shading='auto',
                    cmap=plt.get_cmap('plasma'),
                    vmin=np.min(tau_results['steps_avg']),
                    vmax=np.max(tau_results['steps_avg']))
plt.colorbar(pc)
plt.xlabel('tau_critic')
plt.ylabel('tau_actor')
plt.savefig('tau_avg_steps.png', dpi=150)
plt.close()

plt.figure()
pc = plt.pcolormesh(tau_critic, tau_actor, tau_results['steps_max'],
                    shading='auto',
                    cmap=plt.get_cmap('plasma'),
                    vmin=np.min(tau_results['steps_max']),
                    vmax=np.max(tau_results['steps_max']))
plt.colorbar(pc)
plt.xlabel('tau_critic')
plt.ylabel('tau_actor')
plt.savefig('tau_max_steps.png', dpi=150)
"""
