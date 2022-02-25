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
    except Exception as err:
        print(f'Problem with folder {folder}')
        print(err)
        return None, None, None, None, None

    with open(folder + '/params_dict.pkl', 'rb') as fid:
        params = pickle.load(fid)
    param_value = params[scan_param_name]

    # Count number of steps per episode (RANDOM evaluation)
    n_steps = np.array([(len(json.loads(rew)) - 1) for rew in eval_log_random])
    max_n_steps = np.max(n_steps)
    avg_n_steps = np.mean(n_steps)
    res_random_avg = avg_n_steps
    res_random_max = max_n_steps

    # Count number of steps per episode (SCAN evaluation)
    n_steps = np.array([(len(json.loads(rew)) - 1) for rew in eval_log_scan])
    max_n_steps = np.max(n_steps)
    avg_n_steps = np.mean(n_steps)
    res_scan_avg = avg_n_steps
    res_scan_max = max_n_steps

    return param_value, res_random_avg, res_random_max, res_scan_avg, res_scan_max


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
    for i, sv in enumerate(scan_param_vals):
        mean_steps_avg[i] = np.mean(results_all[sv][eval_type]['steps_avg'])
        std_steps_avg[i] = np.std(results_all[sv][eval_type]['steps_avg'])
        mean_steps_max[i] = np.mean(results_all[sv][eval_type]['steps_max'])
        std_steps_max[i] = np.std(results_all[sv][eval_type]['steps_max'])

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
    plt.savefig(f'SCAN_RES_{eval_type}_{scan_param_name.replace("/", "_")}.png', dpi=150)
    plt.close()


if __name__ == '__main__':

    folders = glob.glob('runs/indiv/*')
    scan_param_name = 'env/max_steps_per_episode'
    res_random_avg_all = []
    res_scan_avg_all = []
    res_random_max_all = []
    res_scan_max_all = []
    param_value_all = []

    for f in folders:
        param_value, res_rand_avg, res_rand_max, res_scan_avg, res_scan_max = run_analysis(
            scan_param_name, f)
        if param_value is None:
            continue
        res_random_avg_all.append(res_rand_avg)
        res_random_max_all.append(res_rand_max)
        res_scan_avg_all.append(res_scan_avg)
        res_scan_max_all.append(res_scan_max)
        param_value_all.append(param_value)

    param_value_unique = set(param_value_all)
    param_value_all = np.array(param_value_all)
    res_random_avg_all = np.array(res_random_avg_all)
    res_random_max_all = np.array(res_random_max_all)
    res_scan_avg_all = np.array(res_scan_avg_all)
    res_scan_max_all = np.array(res_scan_max_all)


    results_all = {}
    for pv in param_value_unique:
        msk = param_value_all == pv

        res_indiv = {'random': {'steps_avg': res_random_avg_all[msk],
                                'steps_max': res_random_max_all[msk]},
                     'scan': {'steps_avg': res_scan_avg_all[msk],
                              'steps_max': res_scan_max_all[msk]}}
        results_all[np.round(pv, 6)] = res_indiv

    # Create overview plot
    plot_scan_results(scan_param_name, results_all)
