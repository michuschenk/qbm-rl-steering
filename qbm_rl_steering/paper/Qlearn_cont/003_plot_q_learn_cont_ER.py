import matplotlib.pyplot as plt
import numpy as np
import pickle


# with open('paper/Qlearn_cont/res_scan_conv_150222_ferl_wER_combined.pkl', 'rb') as fid:
with open('paper/Qlearn_cont/res_scan_conv_040322_FERL_wER.pkl', 'rb') as fid:
    res_ferl = pickle.load(fid)

# with open('paper/Qlearn_cont/res_scan_conv_150222_class.pkl', 'rb') as fid:
with open('paper/Qlearn_cont/res_scan_conv_040322_class_wER.pkl', 'rb') as fid:
    res_dqn = pickle.load(fid)


# Plot scan summaries
plt.figure(1, figsize=(6, 5.3))
plt.suptitle('Q-learning, cont, with ER')
ax1 = plt.gca()

# FERL
(h, caps, _) = ax1.errorbar(
    res_ferl['param_arr'][:-5], np.mean(res_ferl['results'], axis=0)[:-5],
    yerr=np.std(res_ferl['results'], axis=0)[:-5] / np.sqrt(res_ferl['n_repeats_scan']),
    capsize=4, elinewidth=2, color='tab:blue', ls='--', label='FERL')

for cap in caps:
    cap.set_color('tab:blue')
    cap.set_markeredgewidth(2)

# DQN
(h, caps, _) = ax1.errorbar(
    x=res_dqn['scan_values'][:-3], y=res_dqn['metric_avg'][:-3], yerr=res_dqn['metric_std'][:-3],
    c='tab:red', capsize=4, elinewidth=2, ls='--', label='DQN')

for cap in caps:
    cap.set_color('tab:red')
    cap.set_markeredgewidth(2)

ax1.set_xlabel('# training steps')
# ax2.set_xlabel('# training steps (DQN)')
ax1.set_xlim(0, 500)
# ax2.set_xlim(0, 600)

ax1.set_ylabel('Optimality (%)')
ax1.set_ylim(40, 102)
ax1.set_xticks([0, 100, 200, 300, 400, 500])
# ax2.set_xticks([0, 200, 400, 600])

h1, l1 = ax1.get_legend_handles_labels()
plt.legend(h1, l1, loc=(0.8, 0.8))

# INSET ZOOM AXES
axins = ax1.inset_axes([0.56, 0.08, 0.4, 0.45])

(h, caps, _) = axins.errorbar(
    res_ferl['param_arr'][:-5], np.mean(res_ferl['results'], axis=0)[:-5],
    yerr=np.std(res_ferl['results'], axis=0)[:-5] / np.sqrt(res_ferl['n_repeats_scan']),
    capsize=3, elinewidth=1.5, color='tab:blue', ls='--', label='FERL')

for cap in caps:
    cap.set_color('tab:blue')
    cap.set_markeredgewidth(1.5)


(h, caps, _) = axins.errorbar(
    x=res_dqn['scan_values'][:2], y=res_dqn['metric_avg'][:2], yerr=res_dqn['metric_std'][:2],
    c='tab:red', capsize=3, elinewidth=1.5, ls='--', label='DQN')

for cap in caps:
    cap.set_color('tab:red')
    cap.set_markeredgewidth(1.5)

# sub region of the original image
x1, x2, y1, y2 = 0., 35., 40., 102.
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
# axins.set_xticklabels([])
# axins.set_yticklabels([])

ax1.indicate_inset_zoom(axins, edgecolor="black", ls='-', alpha=0.2)

plt.tight_layout()
plt.savefig('paper/Qlearn_cont/003_qlearn_cont_comp.pdf')
plt.show()
