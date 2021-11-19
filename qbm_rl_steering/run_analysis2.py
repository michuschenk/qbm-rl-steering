import os
import json

import datetime
import shutil
import pickle

import numpy as np
import pandas as pd

import glob


def run_analysis(scan_param_name, folder):

    try:
        df_ = pd.read_csv(folder + '/eval_log_random')
        eval_log_random = df_['rewards'].values

        df_ = pd.read_csv(folder + '/eval_log_scan')
        eval_log_scan = df_['rewards'].values

        df_train = pd.read_csv(folder + '/train_log')
    except Exception as err:
        print(f'Problem with folder {folder}')
        print(err)
        return None, None, None, None, None, None, None, None

    with open(folder + '/params_dict.pkl', 'rb') as fid:
        params = pickle.load(fid)
    param_value = params[scan_param_name]

    # Count number of steps per episode (RANDOM evaluation)
    n_steps = np.array([(len(json.loads(rew)) - 1) for rew in eval_log_random])
    init_rewards = np.array([json.loads(rew)[0] for rew in eval_log_random])
    final_rewards = np.array([json.loads(rew)[-1] for rew in eval_log_random])
    max_n_steps = np.max(n_steps)
    avg_n_steps = np.mean(n_steps)
    res_random_avg = avg_n_steps
    res_random_max = max_n_steps

    # Count number of steps per episode (SCAN evaluation)
    n_steps = np.array([(len(json.loads(rew)) - 1) for rew in eval_log_scan])
    init_rewards = np.array([json.loads(rew)[0] for rew in eval_log_scan])
    final_rewards = np.array([json.loads(rew)[-1] for rew in eval_log_scan])
    max_n_steps = np.max(n_steps)
    avg_n_steps = np.mean(n_steps)
    res_scan_avg = avg_n_steps
    res_scan_max = max_n_steps

    # Train logs
    total_updates = np.sum(df_train['n_total_steps'])
    total_random_steps = np.sum(df_train['n_random_steps'])
    n_episodes = len(df_train)

    return (param_value, res_random_avg, res_random_max, res_scan_avg, res_scan_max, n_episodes, total_updates, total_random_steps)


def plot_scan_results(scan_param_name, results_all):

    import matplotlib.pyplot as plt

    eval_type = 'scan'
    scan_param_vals = np.array(sorted(list(results_all.keys())))
    # n_stats = len(res[scan_param_vals[0]][eval_type]['steps_avg'])
    # scan_avg_steps = np.zeros((n_stats, len(scan_param_vals)))
    # scan_max_steps = np.zeros((n_stats, len(scan_param_vals)))

    mean_steps_avg = np.zeros(len(scan_param_vals))
    std_steps_avg = np.zeros(len(scan_param_vals))
    mean_steps_max = np.zeros(len(scan_param_vals))
    std_steps_max = np.zeros(len(scan_param_vals))
    mean_train_episodes = np.zeros(len(scan_param_vals))
    std_train_episodes = np.zeros(len(scan_param_vals))
    mean_train_updates = np.zeros(len(scan_param_vals))
    std_train_updates = np.zeros(len(scan_param_vals))
    mean_train_updates_rand = np.zeros(len(scan_param_vals))
    std_train_updates_rand = np.zeros(len(scan_param_vals))

    for i, sv in enumerate(scan_param_vals):
        mean_steps_avg[i] = np.mean(results_all[sv][eval_type]['steps_avg'])
        std_steps_avg[i] = np.std(results_all[sv][eval_type]['steps_avg'])
        mean_steps_max[i] = np.mean(results_all[sv][eval_type]['steps_max'])
        std_steps_max[i] = np.std(results_all[sv][eval_type]['steps_max'])
        mean_train_episodes[i] = np.mean(results_all[sv]['train']['n_episodes'])
        std_train_episodes[i] = np.std(results_all[sv]['train']['n_episodes'])
        mean_train_updates[i] = np.mean(results_all[sv]['train']['total_updates'])
        std_train_updates[i] = np.std(results_all[sv]['train']['total_updates'])
        mean_train_updates_rand[i] = np.mean(results_all[sv]['train']['random_steps'])
        std_train_updates_rand[i] = np.std(results_all[sv]['train']['random_steps'])

    plt.figure(figsize=(6, 5))
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
    plt.savefig(f'SCAN_RES_{eval_type}_{scan_param_name.replace("/", "_")}.png', dpi=150)
    plt.close()

    fig = plt.figure(figsize=(6, 5))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312, sharex=ax1)
    ax3 = fig.add_subplot(313, sharex=ax1)
    plt.suptitle(f'Param. {scan_param_name.upper()}, eval vs. train', fontsize=10)
    ax1.plot(scan_param_vals, mean_train_episodes, marker='x', c='tab:blue')
    ax1.fill_between(scan_param_vals, mean_train_episodes - std_train_episodes,
                     mean_train_episodes + std_train_episodes, color='tab:blue', alpha=0.5)

    ax2.plot(scan_param_vals, mean_train_updates, marker='x', c='tab:blue', label='Total')
    ax2.fill_between(scan_param_vals, mean_train_updates - std_train_updates,
                     mean_train_updates + std_train_updates, color='tab:blue', alpha=0.5)
    ax2.plot(scan_param_vals, mean_train_updates_rand, marker='x', c='tab:red', label='Random')
    ax2.fill_between(scan_param_vals, mean_train_updates_rand - std_train_updates_rand,
                     mean_train_updates_rand + std_train_updates_rand, color='tab:red', alpha=0.5)

    ax3.plot(scan_param_vals, mean_steps_avg, marker='x', c='tab:blue')
    ax3.fill_between(scan_param_vals, mean_steps_avg - std_steps_avg,
                     mean_steps_avg + std_steps_avg, color='tab:blue', alpha=0.5)

    ax1.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    ax2.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    ax3.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

    ax2.legend()
    ax3.set_xlabel(scan_param_name)
    ax1.set_ylabel('# episodes\n(TRAIN)')
    ax2.set_ylabel('# updates\n(TRAIN)')
    ax3.set_ylabel('# steps avg.\n(EVAL)')

    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax3.set_ylim(bottom=0)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(f'TRAIN_vs_EVAL_{scan_param_name.replace("/", "_")}.png', dpi=150)
    plt.close()


if __name__ == '__main__':

    folders = glob.glob('runs/indiv/*')
    scan_param_name = 'n_episodes'
    res_random_avg_all = []
    res_scan_avg_all = []
    res_random_max_all = []
    res_scan_max_all = []
    param_value_all = []

    train_total_updates = []
    train_episodes = []
    train_random_steps = []

    for f in folders:
        param_value, res_rand_avg, res_rand_max, res_scan_avg, res_scan_max, n_episodes, total_updates, total_random_steps = run_analysis(scan_param_name, f)
        if param_value is None:
            continue
        res_random_avg_all.append(res_rand_avg)
        res_random_max_all.append(res_rand_max)
        res_scan_avg_all.append(res_scan_avg)
        res_scan_max_all.append(res_scan_max)
        param_value_all.append(param_value)
        train_total_updates.append(total_updates)
        train_episodes.append(n_episodes)
        train_random_steps.append(total_random_steps)

    param_value_unique = set(param_value_all)
    param_value_all = np.array(param_value_all)
    res_random_avg_all = np.array(res_random_avg_all)
    res_random_max_all = np.array(res_random_max_all)
    res_scan_avg_all = np.array(res_scan_avg_all)
    res_scan_max_all = np.array(res_scan_max_all)
    train_total_updates = np.array(train_total_updates)
    train_episodes = np.array(train_episodes)
    train_random_steps = np.array(train_random_steps)

    results_all = {}
    for pv in param_value_unique:
        msk = param_value_all == pv

        res_indiv = {'random': {'steps_avg': res_random_avg_all[msk],
                                'steps_max': res_random_max_all[msk]},
                     'scan': {'steps_avg': res_scan_avg_all[msk],
                              'steps_max': res_scan_max_all[msk]},
                     'train': {'n_episodes': train_episodes[msk],
                               'total_updates': train_total_updates[msk],
                               'random_steps': train_random_steps[msk]}}
        results_all[np.round(pv, 6)] = res_indiv

    # Create overview plot
    plot_scan_results(scan_param_name, results_all)
