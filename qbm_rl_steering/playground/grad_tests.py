from qbm_rl_steering.core.ddpg_agents import DDPGAgent
from qbm_rl_steering.run_ddpg import trainer
from cern_awake_env.simulation import SimulationEnv
import tensorflow as tf
import numpy as np

max_steps = 2
env = SimulationEnv(plane='H', remove_singular_devices=True)

max_episodes = 1
batch_size = 1

gamma = 0.99
tau = 5e-2
buffer_maxlen = 100000
critic_lr = 5e-4
actor_lr = 1e-2

agent = DDPGAgent(env, gamma, tau, buffer_maxlen, critic_lr, actor_lr,
                  use_qbm=True)
init_rewards, final_rewards, episode_length = trainer(
    env, agent, max_episodes, max_steps, batch_size, action_noise=0.05)

X, A, R, X2, D = agent.replay_buffer.sample(1)
with tf.GradientTape() as tape2:
    A_mu = agent.main_actor_net(X)

q_before = []
for i in range(10):
    qbe, _, _ = agent.main_critic_net.calculate_q_value_on_batch(X, A_mu)
    q_before.append(qbe)
print('q_before', np.array(q_before).mean())

jacobi_mu_wrt_muTheta = tape2.jacobian(
    A_mu, agent.main_actor_net.trainable_variables)
grad_Q_wrt_a = agent.get_action_derivative(X, A_mu, batch_size=1)

grads_mu = []
for i in range(len(jacobi_mu_wrt_muTheta)):
    on_batch = tf.tensordot(
        jacobi_mu_wrt_muTheta[i], grad_Q_wrt_a,
        axes=((0, 1), (0, 1)))
    on_batch /= batch_size
    grads_mu.append(on_batch)

agent.actor_optimizer.apply_gradients(
    zip(grads_mu, agent.main_actor_net.trainable_variables))

A_mu_after = agent.main_actor_net(X)

q_after = []
for i in range(10):
    qaft, _, _ = agent.main_critic_net.calculate_q_value_on_batch(X, A_mu_after)
    q_after.append(qaft)
print('q_after', np.array(q_after).mean())


# # GRADIENTS dQ/da
# dq_da = agent.get_action_derivative(X, A_mu, batch_size=1)
# print('dq_da', dq_da)
# q, _, _ = agent.critic.calculate_q_value_on_batch(X, A_mu)
# print('q', q)
# for i in range(agent.action_dim):
#     Aeps = A_mu[0].numpy()
#     Aeps[i] += 0.5
#     Aeps = np.atleast_2d(Aeps)
#     q, _, _ = agent.critic.calculate_q_value_on_batch(X, Aeps)
#     print('q, eps i', i, q)
