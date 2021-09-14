import numpy as np
from qbm_rl_steering.core._not_maintained_qac import QuantumActorCritic
from cern_awake_env.simulation import SimulationEnv
import matplotlib.pyplot as plt


gamma_rl = 0.95
n_epochs = 20
max_episode_length = 25
initial_exploration_steps = 10
initial_reward = 0
batch_size = 19

# env = TargetSteeringEnv(max_steps_per_episode=max_episode_length)
env = SimulationEnv(plane='H', remove_singular_devices=True)
agent = QuantumActorCritic(env, gamma_rl=gamma_rl, batch_size=batch_size)

# Derivative, numerical for one specific action
n_act = env.action_space.shape[0]
fixed_state = np.random.uniform(-1, 1, env.observation_space.shape[0])
fixed_action = np.random.uniform(-1, 1, n_act)

# scan one of the actions, keeping all the other actions fixed
i = 2
action_i = np.linspace(-1, 1, batch_size)
action_space = np.array(list(fixed_action) * len(action_i)).reshape(
    len(action_i), -1)
action_space[:, i] = action_i
state_space = np.array(list(fixed_state) * len(action_i)).reshape(
    len(action_i), -1)
# plt.pcolormesh(action_space)
# plt.pcolormesh(state_space)
# plt.show()

# dq_da = np.zeros((1, action_space.shape[0]))
# dqds = np.zeros((len(s), len(a)))
# for i, s_ in enumerate(state_space):
q, _, _ = agent.critic.calculate_q_value_on_batch(state_space, action_space)
dq_da_ = agent.get_action_derivative(state_space, action_space, epsilon=0.4)
dq_da_analytical_ = agent.get_action_derivative_analytical(
    state_space, action_space)

# dq_da_2_ = agent.get_action_derivative(state_space, action_space, epsilon=0.45)
# dq_da_3_ = agent.get_action_derivative(state_space, action_space, epsilon=0.55)
# dq_da_ = (dq_da_ + dq_da_2_ + dq_da_3_) / 3.
# dq_da_5pt_ = agent.get_action_derivative_5point(
#     state_space, action_space, epsilon=0.5)

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 10))
imq = axs[0].plot(action_space[:, i], q)
axs[0].set_title('Q')
axs[0].set_ylabel('action')
#
# imdqda = axs[1].pcolormesh(state_space[:, 0], action_space[:, i], dq_da_.T,
#                            shading='auto')
axs[1].plot(action_space[:, i], dq_da_.T[:, i], c='tab:blue')
axs[2].plot(action_space[:, i], dq_da_analytical_.T[:, i], c='tab:blue')
# axs[1].plot(action_space[:, i], dq_da_5pt_.T[:, i], c='tab:red')

# axs[1].axvline(s[6], c='red')
# # fig.colorbar(imdqda, ax=axs[1])
# # axs[1].set_title('dq / da')
# # axs[1].set_ylabel('action')
plt.show()

# imdqds = axs[2].pcolormesh(s, a, dqds.T, shading='auto')
# axs[2].axhline(a[5], c='red')
# fig.colorbar(imdqds, ax=axs[2])
# axs[2].set_title('dq / ds')
# axs[2].set_xlabel('state')
# axs[2].set_ylabel('action')
# plt.show()

# plt.figure()
# plt.suptitle('q vs dqda')
# plt.plot(a, q[6, :], label='Q')
# plt.plot(a, dqda[6, :], label='dQ/da')
# plt.legend()
# plt.xlabel('action')
# plt.ylabel('Q and dq/da resp.')
# plt.show()
#
# plt.figure()
# plt.suptitle('q vs dqds')
# plt.plot(s, q[:, 5], label='Q')
# plt.plot(s, dqds[:, 5], label='dQ/ds')
# plt.legend()
# plt.xlabel('state')
# plt.ylabel('Q and dq/ds resp.')
# plt.show()