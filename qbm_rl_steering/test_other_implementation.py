# https://deeplearningcourses.com/c/cutting-edge-artificial-intelligence
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from models import Critic_gen, Actor_gen
from collections import deque
from sys import exit
from buffer import BasicBuffer_a, BasicBuffer_b
import random


from qbm_rl_steering.core.qbm import QFunction


# np.random.seed(0)
# tf.random.set_seed(0)


class DDPGAgent:

    def __init__(self, env, gamma, tau_critic, tau_actor, buffer_maxlen,
                 critic_learning_rate, actor_learning_rate, use_qbm=True):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_max = env.action_space.high[0]

        self.use_qbm = use_qbm
        # self.action_max = 1

        # hyperparameters
        self.gamma = gamma
        self.tau_critic = tau_critic
        self.tau_actor = tau_actor
        self.critic_learning_rate = critic_learning_rate
        self.actor_learning_rate = actor_learning_rate

        # Network layers
        actor_layer = [512, 200, 128]
        # critic_layer = [1024, 512, 300, 1]
        critic_layer = [200, 100, 1]

        self.grads_mu_all_mean = []
        self.grads_mu_all_min = []
        self.grads_mu_all_max = []

        # Main network outputs
        self.mu = Actor_gen(len(self.env.observation_space.high),
                            len(self.env.action_space.high),
                            actor_layer, self.action_max)

        # QBM Q-function parameters
        kwargs_q_func = dict(
            sampler_type='SQA',
            state_space=self.env.observation_space,
            action_space=self.env.action_space,
            small_gamma=self.gamma,
            n_replicas=1,
            big_gamma=(20., 0.), beta=2,
            n_annealing_steps=200,
            n_meas_for_average=1,  # 2500,   # 5000,
            kwargs_qpu={})

        if not self.use_qbm:
            self.q_mu = Critic_gen(len(self.env.observation_space.high),
                                   len(self.env.action_space.high),
                                   critic_layer)
        else:
            self.critic = QFunction(**kwargs_q_func)

        # Target networks
        self.mu_target = Actor_gen(len(self.env.observation_space.high),
                                   len(self.env.action_space.high),
                                   actor_layer, self.action_max)

        if not self.use_qbm:
            self.q_mu_target = Critic_gen(len(self.env.observation_space.high),
                                          len(self.env.action_space.high),
                                          critic_layer)
        else:
            # QBM
            self.critic_target = QFunction(**kwargs_q_func)

        # Copying weights in,
        self.mu_target.set_weights(self.mu.get_weights())

        if not self.use_qbm:
            self.q_mu_target.set_weights(self.q_mu.get_weights())
        else:
            # For QBM
            for k in self.critic.w_hh.keys():
                self.critic_target.w_hh[k] = self.critic.w_hh[k]
            for k in self.critic_target.w_vh.keys():
                self.critic_target.w_vh[k] = self.critic.w_vh[k]

        # optimizers
        self.mu_optimizer = tf.keras.optimizers.Adam(
            learning_rate=actor_learning_rate)

        if not self.use_qbm:
            self.q_mu_optimizer = tf.keras.optimizers.Adam(
                learning_rate=critic_learning_rate)

        # This is done in the QBM definition and I'll have to call it
        # explicitly below.
        self.replay_buffer = BasicBuffer_a(size=buffer_maxlen,
                                           obs_dim=self.obs_dim,
                                           act_dim=self.action_dim)

        self.q_losses = []
        self.mu_losses = []

        self.q_before = []
        self.q_after = []

    def get_action(self, s, noise_scale):
        a = self.mu.predict(s.reshape(1, -1))[0]
        a += noise_scale * np.random.randn(self.action_dim)
        return np.clip(a, -self.action_max, self.action_max)

    def update(self, batch_size):
        X, A, R, X2, D = self.replay_buffer.sample(batch_size)
        X = np.asarray(X, dtype=np.float32)
        A = np.asarray(A, dtype=np.float32)
        R = np.asarray(R, dtype=np.float32)
        X2 = np.asarray(X2, dtype=np.float32)

        # Updating Critic
        if not self.use_qbm:
            with tf.GradientTape() as tape:
                A2 = self.mu_target(X2)
                q_target = R + self.gamma * self.q_mu_target([X2, A2])
                qvals = self.q_mu([X, A])
                q_loss = tf.reduce_mean((qvals - q_target) ** 2)
            # print('critic now', qvals)
            # print('critic target', q_target)
            grads_q = tape.gradient(q_loss, self.q_mu.trainable_variables)
            self.q_mu_optimizer.apply_gradients(
                zip(grads_q, self.q_mu.trainable_variables))
            # print('critic after', self.q_mu([X, A]))
            self.q_losses.append(q_loss)
        else:
            # Update critic QBM
            A2 = self.mu_target(X2)
            for jj in np.arange(len(X)):
                # Recalculate q_value of (sample.state, sample.action) pair
                q_value, spin_configs, visible_nodes = (
                    self.critic.calculate_q_value(X[jj], A[jj]))
                # print('critic now', q_value)

                # Now calculate the next_q_value of the greedy action, without
                # actually taking the action (to take actions in env.,
                # we don't follow purely greedy action).
                # next_action = self.get_target_actions(next_states[jj])
                # next_action = next_action.flatten()
                next_q_value, spin_configurations, visible_nodes = (
                    self.critic_target.calculate_q_value(
                        state=X2[jj], action=A2[jj]))
                # print('critic target', next_q_value)

                # Update weights and target Q-function if needed
                self.critic.update_weights(
                    spin_configs, visible_nodes, q_value, next_q_value,
                    R[jj], learning_rate=self.critic_learning_rate)

                q_value, spin_configs, visible_nodes = (
                    self.critic.calculate_q_value(X[jj], A[jj]))
                # print('critic after', q_value)

        # Updating ZE Actor
        if not self.use_qbm:
            # with tf.GradientTape() as tape2:
            #     A_mu = self.mu(X)
            #     Q_mu = self.q_mu([X, A_mu])
            #     print('q before train step', Q_mu)
            #     self.q_before.append(Q_mu[0])
            #     mu_loss = -tf.reduce_mean(Q_mu)
            # grads_mu = tape2.gradient(mu_loss, self.mu.trainable_variables)
            # self.mu_losses.append(mu_loss)
            # print('grads_mu[0]_orig', grads_mu[0])

            # TRY MANUALLY AND COMPARE, LOOKED FINE BUT A MINUS SIGN ...
            with tf.GradientTape() as tape2:
                A_mu = self.mu(X)
            jacobi_mu_wrt_muTheta = tape2.jacobian(
                A_mu, self.mu.trainable_variables
            )
            # print('jacobi shape', jacobi_mu_wrt_muTheta[0].shape)

            with tf.GradientTape() as tape3:
                tape3.watch(A_mu)
                Q_mu = self.q_mu([X, A_mu])
                self.q_before.append(np.mean(Q_mu))

            grad_Q_wrt_a = tape3.gradient(Q_mu, A_mu)
            # print('grad_Q_wrt_a.shape', grad_Q_wrt_a.shape)

            grads_mu = []
            for i in range(len(jacobi_mu_wrt_muTheta)):
                on_batch = tf.tensordot(
                    jacobi_mu_wrt_muTheta[i], grad_Q_wrt_a,
                    axes=((0, 1), (0, 1)))
                on_batch /= batch_size
                grads_mu.append(-on_batch)
            # print('grads_mu[0])_manual', grads_mu[0])
        else:
            with tf.GradientTape() as tape2:
                A_mu = self.mu(X)
                # print('state', X)
                # print('action before train step', A_mu)

            q, _, _ = self.critic.calculate_q_value_on_batch(X, A_mu)
            # print('q before train step', np.mean(q))
            self.q_before.append(np.mean(q))

            # Apply chain-rule manually here:
            jacobi_mu_wrt_muTheta = tape2.jacobian(
                A_mu, self.mu.trainable_variables)
            # print('mean_jacob[0]', np.mean(jacobi_mu_wrt_muTheta[0]))
            grad_Q_wrt_a = self.get_action_derivative(X, A_mu, batch_size)
            # grad_Q_wrt_a /= np.max(np.abs(grad_Q_wrt_a))
            # grad_Q_wrt_a *= 1e-5
            # print('mean_grad_Q_wrt_a', np.mean(grad_Q_wrt_a))
            # print('grad_Q_wrt_a', grad_Q_wrt_a)

            grads_mu = []
            for i in range(len(jacobi_mu_wrt_muTheta)):
                on_batch = tf.tensordot(
                    jacobi_mu_wrt_muTheta[i], grad_Q_wrt_a,
                    axes=((0, 1), (0, 1)))
                on_batch /= batch_size
                grads_mu.append(-on_batch)
            # for i in range(len(jacobi_mu_wrt_muTheta)):
            #     on_batch = []
            #     for batch in range(batch_size):
            #         outp = np.dot(tf.transpose(
            #             jacobi_mu_wrt_muTheta[i][batch]),
            #             grad_Q_wrt_a[batch])
            #         on_batch.append(outp)
            #     on_batch = -np.mean(np.asarray(on_batch, dtype=np.float32),
            #                         axis=0).T
            #     grads_mu.append(on_batch)
            mean_ = 0.
            min_ = 1E39
            max_ = -1E39
            for grd in grads_mu:
                mean_ += np.mean(grd)
                min_ = min(np.min(grd), min_)
                max_ = max(np.max(grd), max_)
            mean_ /= len(grads_mu)

            self.grads_mu_all_min.append(min_)
            self.grads_mu_all_max.append(max_)
            self.grads_mu_all_mean.append(mean_)

        self.mu_optimizer.apply_gradients(
            zip(grads_mu, self.mu.trainable_variables))

        # Evaluate Q value for new actor (providing same state, should now
        # give higher Q)
        if not self.use_qbm:
            A_mu_after = self.mu(X)
            Q_mu_after = self.q_mu([X, A_mu_after])
            # print('q after training step', np.mean(Q_mu_after))
            self.q_after.append(np.mean(Q_mu_after))
        else:
            A_mu_after = self.mu(X)
            q_after, _, _ = self.critic.calculate_q_value_on_batch(X,
                                                                   A_mu_after)
            # print('action after training step', A_mu_after)
            # print('q after training step', np.mean(q_after))
            self.q_after.append(np.mean(q_after))

        # Update target networks
        if not self.use_qbm:
            temp1 = np.array(self.q_mu_target.get_weights())
            temp2 = np.array(self.q_mu.get_weights())
            temp3 = self.tau_critic * temp2 + (1 - self.tau_critic) * temp1
            self.q_mu_target.set_weights(temp3)
        else:
            # Critic soft update
            for k in self.critic.w_hh.keys():
                self.critic_target.w_hh[k] = (
                        self.tau_critic * self.critic.w_hh[k] +
                        (1. - self.tau_critic) * self.critic_target.w_hh[k])
            for k in self.critic.w_vh.keys():
                self.critic_target.w_vh[k] = (
                        self.tau_critic * self.critic.w_vh[k] +
                        (1. - self.tau_critic) * self.critic_target.w_vh[k])

        # Update mu network
        temp1 = np.array(self.mu_target.get_weights())
        temp2 = np.array(self.mu.get_weights())
        temp3 = self.tau_actor * temp2 + (1 - self.tau_actor) * temp1
        self.mu_target.set_weights(temp3)

    def get_action_derivative(self, states, actions, batch_size,
                              epsilon=0.25):
        # Need to take derivative for each action separately
        # e.g. if we have batch size of 5, and 10 actions, we expect an
        # output for dQ / da of shape (5, 10).
        # grads_mean = np.zeros((batch_size, self.action_dim))

        n_avgs = 1
        grads = np.zeros((n_avgs, batch_size, self.action_dim))
        for j in range(n_avgs):
            for i in range(self.action_dim):
                actions_tmp1 = np.array(actions).copy()
                actions_tmp1[:, i] += epsilon
                qeps_plus, _, _ = self.critic.calculate_q_value_on_batch(
                    states, actions_tmp1)

                actions_tmp2 = np.array(actions).copy()
                actions_tmp2[:, i] -= epsilon
                qeps_minus, _, _ = self.critic.calculate_q_value_on_batch(
                    states, actions_tmp2)

                grads[j, :, i] = np.atleast_1d(
                    np.float_((qeps_plus - qeps_minus) / (2 * epsilon)))

        grads = np.asarray(grads, dtype=np.float32)
        grads = np.mean(grads, axis=0)
        # print('grads', grads.shape)
        return grads
