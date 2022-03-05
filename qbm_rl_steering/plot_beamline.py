import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import patches

header = ['NAME', 'KEYWORD', 'S', 'L', 'SIGMA_X', 'SIGMA_Y', 'BETX', 'BETY', 'DX', 'DPX', 'DY', 'DPY', 'ALFX', 'ALFY',
          'X', 'Y', 'K1L', 'MUX', 'MUY', 'ENV_X', 'ENV_Y']
df = pd.read_csv('environment/utils/electron_tt43.out', skiprows=47, header=None, delim_whitespace=True)
df.columns = header

fig = plt.figure(1, figsize=(7, 5))
ax = plt.gca()
ax.axhline(0, ls='-', lw=0.5, color='k')

first_k = True
first_mon = True
first_bend = True
first_quad_f = True
first_quad_d = True
for s, typ, l, k1l in zip(df['S'], df['KEYWORD'], df['L'], df['K1L']):
    if typ == 'KICKER':
        if first_k:
            label = "Corrector dipole"
            first_k = False
        else:
            label = None

        ax.add_patch(patches.FancyArrowPatch(
            (s, 0.75),
            (s, -0.75),
            arrowstyle=patches.ArrowStyle("-"),
            label=label,
            edgecolor="tab:green",
            facecolor="tab:green",
        ))
    elif typ == 'MONITOR':
        if first_mon:
            label = "Beam position monitor"
            first_mon = False
        else:
            label = None

        ax.add_patch(patches.FancyArrowPatch(
            (s, 0.75),
            (s, -0.75),
            arrowstyle=patches.ArrowStyle("-"),
            label=label,
            edgecolor="tab:red",
            facecolor="tab:red",
        ))
    elif typ == 'QUADRUPOLE':
        if k1l < 0:
            if first_quad_d:
                label = "Defocusing quadrupole"
                first_quad_d = False
            else:
                label = None
            colour = "tab:cyan"
        else:
            if first_quad_f:
                label = "Focusing quadrupole"
                first_quad_f = False
            else:
                label = None
            colour = "tab:olive"

        ax.add_patch(
            patches.Rectangle((s, -0.5), l, 1, facecolor=colour, alpha=0.7, label=label))
    elif typ == "RBEND":
        if first_bend:
            label = "Main dipole"
            first_bend = False
        else:
            label = None

        ax.add_patch(
            patches.Rectangle(
                (s, -0.5), l, 1., facecolor="grey", label=label, alpha=0.4))


plt.legend(ncol=3, loc='lower right')
plt.yticks([], [])
plt.xlabel('s (m)')
plt.xlim(0, 15)
plt.ylim(-1.2, 1.2)

plt.tight_layout()
plt.savefig('awake_beamline.pdf')
plt.show()

# twiss_H, _ = readAWAKEelectronTwiss()
#

#
# s_pos = []
# for e in twiss_H.elements:
#     if e.name.startswith("DRIFT"):
#         continue
#     # if e.name.startswith("BPM"):
#     #     ax.axvline(e.s, color='tab:red')
#     if e.name.startswith("MC"):
#         # kicker
#         patches.Rectangle(
#             (e.s, -0.75), e..length, 1.5, facecolor="lightcoral", label="Dipole"
#     def _get_patch(self, s: float) -> patches.Patch:
#         return
#         )
# ax.add_patch(patch)
#
# plt.show()
#
# # fig, ax = plt.subplots(1, figsize=(7, 5))
# # colors = ["black", "mediumblue"]
# # for seq, color, label in zip(sequences, colors, "H"):
# #     s_list = [elem.s for elem in seq]
# #     beta_list = [elem.beta for elem in seq]
# #     d_list = [elem.d for elem in seq]
# #     ax.plot(s_list, beta_list, "-", color=color, label=label)
# # ax.set_xlabel("s [m]")
# # ax.set_ylabel("beta [m]")
# # ax.legend()
# # plt.subplots_adjust(top=0.9, left=0.1, right=0.95)
# # plt.show()
