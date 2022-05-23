import matplotlib.pyplot as plt
import numpy as np
import pickle

with open('paper/Qlearn_cont/size2x8_qnet_scan_dqn_steps.pkl', 'rb') as fid:
    res = pickle.load(fid)

# Plot scan summaries
plt.figure(1, figsize=(6, 5))
plt.suptitle('Q-learning, cont, with ER')

(h, caps, _) = plt.errorbar(
    x=res['scan_values'][:], y=res['metric_avg'][:], yerr=res['metric_std'][:],
    c='tab:red', capsize=4, elinewidth=2, ls='--', label="DQN")

for cap in caps:
    cap.set_color('tab:red')
    cap.set_markeredgewidth(2)

plt.xlim(left=0)
plt.ylim(40, 102)
plt.xlabel('# training steps')
plt.ylabel('Optimality (%)')

# plt.legend(loc='lower right')

plt.tight_layout()
plt.savefig('paper/Qlearn_cont/005b_cont_ER_qNetSize2x8.pdf')
plt.show()
