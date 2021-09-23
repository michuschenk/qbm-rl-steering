import datetime
import os
import shutil
import pickle

import multiprocessing as mp
from functools import partial

import pandas as pd
import numpy as np

from tensorflow.keras.optimizers.schedules import (ExponentialDecay,
                                                   PolynomialDecay,
                                                   PiecewiseConstantDecay)

from qbm_rl_steering.core.ddpg_agents import ClassicalDDPG, QuantumDDPG
from qbm_rl_steering.environment.rms_env_nd import RmsSteeringEnv
from qbm_rl_steering.core.run_utils import (trainer, evaluator,
                                            plot_training_log,
                                            plot_evaluation_log)


def run_full(params, worker_id=None):
    env = RmsSteeringEnv(
        n_dims=params['env/n_dims'],
        max_steps_per_episode=params['env/max_steps_per_episode'],
        required_steps_above_reward_threshold=
        params['env/required_steps_above_reward_threshold'])

    # Learning rate schedules: lr_critic = 5e-4, lr_actor = 1e-4
    lr_schedule_critic = ExponentialDecay(params['lr_critic/init'],
                                          params['n_episodes'],
                                          params['lr_critic/decay_factor'])
    lr_schedule_actor = ExponentialDecay(params['lr_actor/init'],
                                         params['n_episodes'],
                                         params['lr_actor/decay_factor'])

    if params['quantum_ddpg']:
        agent = QuantumDDPG(state_space=env.observation_space,
                            action_space=env.action_space,
                            learning_rate_schedule_critic=lr_schedule_critic,
                            learning_rate_schedule_actor=lr_schedule_actor,
                            grad_clip_actor=1e4, grad_clip_critic=1.,
                            gamma=params['agent/gamma'],
                            tau_critic=params['agent/tau_critic'],
                            tau_actor=params['agent/tau_actor']
                            )
    else:
        agent = ClassicalDDPG(state_space=env.observation_space,
                              action_space=env.action_space,
                              learning_rate_schedule_critic=lr_schedule_critic,
                              learning_rate_schedule_actor=lr_schedule_actor,
                              gamma=params['agent/gamma'],
                              tau_critic=params['agent/tau_critic'],
                              tau_actor=params['agent/tau_actor']
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
    out_path = './runs/' + date_time_now
    if worker_id is not None:
        out_path = out_path + f'_worker{worker_id}'
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


# TODO: parameters to consider: 1) epsilon (init and final). 2) gradient
#  calculation epsilon. 3) max_steps_per_episode. 4) batch size. 5) n_episodes.
#  6) tau_critic / tau_actor. 7) learning rates. 8) action noise. 9) n_anneals.
#  10) more than 1 step inside reward obj. It's endless.

# TODO: don't need to specifically run an n-D scan, but rather n 1-D scans
#  should be enough ... focus on n_dims = 4, I) epsilon_init = [0., 0.1, 0.2,
#  0.4], II) max_steps_per_episode = [25, 50, 75], III) batch_size = [12, 24,
#  48], IV) tau_critic vs. tau_actor = [0.05, 0.1] x [0.05, 0.1],
#  V) learning_rates (init) critic vs. actor = [2e-3, 1e-3, 5e-4] x [1e-3,
#  5e-4, 1e-4], VI) n_anneals (final) = [1, 5, 25, 50].
#  Maybe run each one 3 times. Evaluation metric: avg./max #steps in
#  evaluation.

# !!!!!!!!!!!!!!!!!!!!!!!!!!!
# TODO: LEARNING RATES MISSING
# !!!!!!!!!!!!!!!!!!!!!!!!!!!

def run_worker(scan_param_value, default_params, scan_param_name):

    n_stats = 2

    # Copy default parameters and overwrite the corresponding scan parameter
    params = default_params.copy()
    params.update({scan_param_name: scan_param_value})

    results_worker = {
        scan_param_value: {'random': {'steps_avg': np.zeros(n_stats),
                                      'steps_max': np.zeros(n_stats)},
                           'scan': {'steps_avg': np.zeros(n_stats),
                                    'steps_max': np.zeros(n_stats)}}
    }

    worker_id = mp.current_process().name.split('-')[-1]
    for i in range(n_stats):
        print('===================================================')
        print(f'SCAN: {scan_param_name.upper()}: {scan_param_value}')
        print(f'Running stats iteration {i+1}/{n_stats}')
        print('===================================================\n')

        eval_log_random, eval_log_scan = run_full(params, worker_id)

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

    with open(f'bkp_res_{scan_param_name}_{scan_param_value}.pkl', 'wb') as fid:
        pickle.dump(results_worker, fid)

    return results_worker


if __name__ == '__main__':
    default_params = {
        'quantum_ddpg': True,
        'n_episodes': 10,
        'env/n_dims': 4,
        'env/max_steps_per_episode': 10,
        'env/required_steps_above_reward_threshold': 1,
        'trainer/batch_size': 16,
        'trainer/n_exploration_steps': 5,
        'trainer/n_episodes_early_stopping': 15,
        'agent/gamma': 0.99,
        'agent/tau_critic': 0.1,
        'agent/tau_actor': 0.1,
        'lr_critic/init': 5e-4,
        'lr_critic/decay_factor': 0.95,
        'lr_actor/init': 1e-4,
        'lr_actor/decay_factor': 0.95,
        'action_noise/init': 0.1,
        'action_noise/final': 0.,
        'epsilon_greedy/init': 0.3,
        'epsilon_greedy/final': 0.,
        'anneals/n_pieces': 2,
        'anneals/init': 1,
        'anneals/final': 2,
    }

    scan_param_values = [0.1, 0.2, 0.3, 0.4]
    scan_param_name = 'epsilon_greedy/init'

    results_all = {}
    with mp.Pool(max(len(scan_param_values), 8)) as p:
        f = partial(run_worker, default_params=default_params,
                    scan_param_name=scan_param_name)
        res = p.map(f, scan_param_values)
        for r in res:
            results_all.update(r)

    with open(f'scan_res_all_{scan_param_name}.pkl', 'wb') as fid:
        pickle.dump(results_all, fid)


"""
n_runs_stats = 3

# I) epsilon_init scan
epsilon_init = [0., 0.1, 0.2, 0.4]
epsilon_results = {'steps_avg': np.zeros((n_runs_stats, len(epsilon_init))),
                   'steps_max': np.zeros((n_runs_stats, len(epsilon_init)))}

params = default_params.copy()
for i, eps in enumerate(epsilon_init):
    params.update({'epsilon_greedy': {'init': eps, 'final': 0.}})

    for j in range(n_runs_stats):
        print('==================================')
        print(f'SCAN I: EPSILON_INIT: {eps}')
        print(f'Running stats iteration {j+1}/{n_runs_stats}')
        print('==================================\n')
        eval_log = run_full(params)

        # count number of steps per episode
        n_steps = np.array([(len(r) - 1) for r in eval_log])
        max_n_steps = np.max(n_steps)
        avg_n_steps = np.mean(n_steps)

        epsilon_results['steps_avg'][j, i] = avg_n_steps
        epsilon_results['steps_max'][j, i] = max_n_steps

# Show results
mean_steps_avg = np.mean(epsilon_results['steps_avg'], axis=0)
std_steps_avg = np.std(epsilon_results['steps_avg'], axis=0)
mean_steps_max = np.mean(epsilon_results['steps_max'], axis=0)
std_steps_max = np.std(epsilon_results['steps_max'], axis=0)

plt.figure()
plt.plot(epsilon_init, mean_steps_avg, marker='x', c='tab:blue', label='Avg.')
plt.fill_between(epsilon_init, mean_steps_avg - std_steps_avg,
                 mean_steps_avg + std_steps_avg, color='tab:blue', alpha=0.5)
plt.plot(epsilon_init, mean_steps_max, marker='x', c='tab:red', label='Max')
plt.fill_between(epsilon_init, mean_steps_max - std_steps_max,
                 mean_steps_max + std_steps_max, color='tab:red', alpha=0.5)
plt.xlabel('Init epsilon greedy')
plt.ylabel('# steps')
plt.legend(loc='best')
plt.savefig('epsilon_scan.png', dpi=150)
plt.close()

print('FINISHED SCAN I.')
print('**********************************\n')


# II) max_steps_per_episode scan
max_steps_per_episode = [25, 50, 75]
max_steps_results = {
    'steps_avg': np.zeros((n_runs_stats, len(max_steps_per_episode))),
    'steps_max': np.zeros((n_runs_stats, len(max_steps_per_episode)))}

params = default_params.copy()
for i, nsteps in enumerate(max_steps_per_episode):
    params.update(
        {'env': {'n_dims': 4, 'max_steps_per_episode': nsteps,
                 'required_steps_above_reward_threshold': 1}})

    for j in range(n_runs_stats):
        print('==================================')
        print(f'SCAN II: MAX_STEPS: {nsteps}')
        print(f'Running stats iteration {j+1}/{n_runs_stats}')
        print('==================================\n')
        eval_log = run_full(params)

        # count number of steps per episode
        n_steps = np.array([(len(r) - 1) for r in eval_log])
        max_n_steps = np.max(n_steps)
        avg_n_steps = np.mean(n_steps)

        max_steps_results['steps_avg'][j, i] = avg_n_steps
        max_steps_results['steps_max'][j, i] = max_n_steps

# Show results
mean_steps_avg = np.mean(max_steps_results['steps_avg'], axis=0)
std_steps_avg = np.std(max_steps_results['steps_avg'], axis=0)
mean_steps_max = np.mean(max_steps_results['steps_max'], axis=0)
std_steps_max = np.std(max_steps_results['steps_max'], axis=0)

plt.figure()
plt.plot(max_steps_per_episode, mean_steps_avg, marker='x', c='tab:blue',
         label='Avg.')
plt.fill_between(max_steps_per_episode, mean_steps_avg - std_steps_avg,
                 mean_steps_avg + std_steps_avg, color='tab:blue', alpha=0.5)
plt.plot(max_steps_per_episode, mean_steps_max, marker='x', c='tab:red',
         label='Max')
plt.fill_between(max_steps_per_episode, mean_steps_max - std_steps_max,
                 mean_steps_max + std_steps_max, color='tab:red', alpha=0.5)
plt.xlabel('Max steps per episode')
plt.ylabel('# steps')
plt.legend(loc='best')
plt.savefig('max_steps_per_episode_scan.png', dpi=150)
plt.close()

print('FINISHED SCAN II.')
print('**********************************\n')


# III) batch_size scan
batch_size = [12, 24, 48]
batch_size_results = {
    'steps_avg': np.zeros((n_runs_stats, len(batch_size))),
    'steps_max': np.zeros((n_runs_stats, len(batch_size)))}

params = default_params.copy()
for i, bs in enumerate(batch_size):
    params.update(
        {'trainer': {'batch_size': bs,
                     'n_exploration_steps': 10,
                     'n_episodes_early_stopping': 15}})

    for j in range(n_runs_stats):
        print('==================================')
        print(f'SCAN III: BATCH_SIZE: {bs}')
        print(f'Running stats iteration {j+1}/{n_runs_stats}')
        print('==================================\n')
        eval_log = run_full(params)

        # count number of steps per episode
        n_steps = np.array([(len(r) - 1) for r in eval_log])
        max_n_steps = np.max(n_steps)
        avg_n_steps = np.mean(n_steps)

        batch_size_results['steps_avg'][j, i] = avg_n_steps
        batch_size_results['steps_max'][j, i] = max_n_steps

# Show results
mean_steps_avg = np.mean(batch_size_results['steps_avg'], axis=0)
std_steps_avg = np.std(batch_size_results['steps_avg'], axis=0)
mean_steps_max = np.mean(batch_size_results['steps_max'], axis=0)
std_steps_max = np.std(batch_size_results['steps_max'], axis=0)

plt.figure()
plt.plot(batch_size, mean_steps_avg, marker='x', c='tab:blue',
         label='Avg.')
plt.fill_between(batch_size, mean_steps_avg - std_steps_avg,
                 mean_steps_avg + std_steps_avg, color='tab:blue', alpha=0.5)
plt.plot(batch_size, mean_steps_max, marker='x', c='tab:red',
         label='Max')
plt.fill_between(batch_size, mean_steps_max - std_steps_max,
                 mean_steps_max + std_steps_max, color='tab:red', alpha=0.5)
plt.xlabel('Batch size')
plt.ylabel('# steps')
plt.legend(loc='best')
plt.savefig('batch_size_scan.png', dpi=150)
plt.close()

print('FINISHED SCAN III.')
print('**********************************\n')


# VI) n_anneals scan
n_anneals = [1, 5, 25, 50]
n_anneals_results = {
    'steps_avg': np.zeros((n_runs_stats, len(n_anneals))),
    'steps_max': np.zeros((n_runs_stats, len(n_anneals)))}

params = default_params.copy()
for i, n_ann in enumerate(n_anneals):
    params.update(
        {'anneals': {'n_pieces': 2, 'init': 1, 'final': n_ann}})

    for j in range(n_runs_stats):
        print('==================================')
        print(f'SCAN VI: N_ANNEALS: {n_ann}')
        print(f'Running stats iteration {j+1}/{n_runs_stats}')
        print('==================================\n')
        eval_log = run_full(params)

        # count number of steps per episode
        n_steps = np.array([(len(r) - 1) for r in eval_log])
        max_n_steps = np.max(n_steps)
        avg_n_steps = np.mean(n_steps)

        n_anneals_results['steps_avg'][j, i] = avg_n_steps
        n_anneals_results['steps_max'][j, i] = max_n_steps

# Show results
mean_steps_avg = np.mean(n_anneals_results['steps_avg'], axis=0)
std_steps_avg = np.std(n_anneals_results['steps_avg'], axis=0)
mean_steps_max = np.mean(n_anneals_results['steps_max'], axis=0)
std_steps_max = np.std(n_anneals_results['steps_max'], axis=0)

plt.figure()
plt.plot(n_anneals, mean_steps_avg, marker='x', c='tab:blue',
         label='Avg.')
plt.fill_between(n_anneals, mean_steps_avg - std_steps_avg,
                 mean_steps_avg + std_steps_avg, color='tab:blue', alpha=0.5)
plt.plot(n_anneals, mean_steps_max, marker='x', c='tab:red',
         label='Max')
plt.fill_between(n_anneals, mean_steps_max - std_steps_max,
                 mean_steps_max + std_steps_max, color='tab:red', alpha=0.5)
plt.xlabel('# anneals final')
plt.ylabel('# steps')
plt.legend(loc='best')
plt.savefig('n_anneals_scan.png', dpi=150)
plt.close()

print('FINISHED SCAN VI.')
print('**********************************\n')


# IV) tau_critic x tau_actor scan
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

print('FINISHED SCAN IV.')
print('**********************************\n')
"""
