import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob

flist = sorted(glob.glob('paper/Qlearn_cont/qnet_size*.pkl'))[::2]

res_list = []
for f in flist:
    with open(f, 'rb') as fid:
        res_list.append(pickle.load(fid))

# Plot scan summaries
plt.figure(1, figsize=(6, 5.3))
plt.suptitle('Q-learning, cont, with ER')

cmap = plt.get_cmap('tab10')
for i, res in enumerate(res_list):
    (h, caps, _) = plt.errorbar(
        x=res['scan_values'][:], y=res['metric_avg'][:], yerr=res['metric_std'][:],
        c=cmap(i), capsize=4, elinewidth=2, ls='--', label=f"{res['train_steps']} training steps")

    for cap in caps:
        cap.set_color(cmap(i))
        cap.set_markeredgewidth(2)

plt.xlim(left=0)
plt.ylim(40, 102)
plt.xlabel('# nodes per hidden layer')
plt.ylabel('Optimality (%)')

plt.legend(loc='lower right')

plt.tight_layout()
plt.savefig('paper/Qlearn_cont/005_cont_ER_qNetSize.pdf')
plt.show()
