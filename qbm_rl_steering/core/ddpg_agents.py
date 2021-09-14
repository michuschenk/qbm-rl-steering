# Loosely based on:
# https://deeplearningcourses.com/c/cutting-edge-artificial-intelligence
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from qbm_rl_steering.core.utils import (generate_classical_critic,
                                        generate_classical_actor)
from qbm_rl_steering.core.utils import ReplayBuffer
from qbm_rl_steering.core.qbm import QFunction


# TODO: implement common base class for classical and quantum ddpg
# TODO: needs further cleaning: get rid of n_steps_estimate and implement the
#  learning rate schedules properly.


class ClassicalDDPG:
    def __init__(self, state_space, action_space, gamma,
                 tau_critic, tau_actor, buffer_size,
                 critic_init_learning_rate, actor_init_learning_rate,
                 n_steps_estimate=1000):
        """ Implements the classical DDPG agent where both actor and critic
        are represented by classical neural networks.
        :param state_space: openAI gym env state space
        :param action_space: openAI gym env action space
        :param gamma: reward discount factor
        :param tau_critic: soft update factor for critic target network
        :param tau_actor: soft update factor for actor target network
        :param buffer_size: max. size of replay buffer, not to be confused
        with batch_size.
        :param critic_init_learning_rate: initial learning rate of critic,
        may decay over time.
        :param actor_init_learning_rate: initial learning rate of actor,
        may decay over time.
        :param n_steps_estimate: estimated total number of training steps,
        needed for learning rate schedules (would like to get rid of this).
        """
        self.n_dims_state_space = len(state_space.high)
        self.n_dims_action_space = len(action_space.high)

        self.step = 0

        # Some main hyperparameters
        self.gamma = gamma
        self.tau_critic = tau_critic
        self.tau_actor = tau_actor

        self.lr_schedule_critic = ExponentialDecay(
            critic_init_learning_rate, n_steps_estimate, 1.)
        self.lr_schedule_actor = ExponentialDecay(
            actor_init_learning_rate, n_steps_estimate, 1.)

        # Main and target actor network initialization
        # ACTOR
        actor_hidden_layers = [512, 200, 128]
        self.main_actor_net = generate_classical_actor(
            self.n_dims_state_space, self.n_dims_action_space,
            actor_hidden_layers)
        self.target_actor_net = generate_classical_actor(
            self.n_dims_state_space, self.n_dims_action_space,
            actor_hidden_layers)

        # CRITIC
        critic_hidden_layers = [200, 100, 1]
        self.main_critic_net = generate_classical_critic(
            self.n_dims_state_space, self.n_dims_action_space,
            critic_hidden_layers)
        self.target_critic_net = generate_classical_critic(
            self.n_dims_state_space, self.n_dims_action_space,
            critic_hidden_layers)

        # Copy weights from main to target nets
        self.target_actor_net.set_weights(self.main_actor_net.get_weights())
        self.target_critic_net.set_weights(self.main_critic_net.get_weights())

        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.lr_schedule_actor)
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.lr_schedule_critic)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            size=buffer_size, obs_dim=self.n_dims_state_space,
            act_dim=self.n_dims_action_space)

        # For logging
        # TODO: add critic gradient statistics as well?
        # self.critic_grads_log = {'mean': [], 'min': [], 'max': []}
        self.actor_grads_log = {'mean': [], 'min': [], 'max': []}
        self.losses_log = {'Q': [], 'Mu': []}
        self.q_log = {'before': [], 'after': []}

    def get_proposed_action(self, state, noise_scale):
        """ Use main actor network to obtain proposed action for en input
        state. Add potentially some Gaussian noise for exploration. Action is
        clipped at [-1, 1] in any case. """
        action = self.main_actor_net.predict(state.reshape(1, -1))[0]
        action += noise_scale * np.random.randn(self.n_dims_action_space)
        return np.clip(action, -1., 1.)

    def update(self, batch_size):
        """ Calculate and apply the updates of the critic and actor
        networks based on batch of samples from experience replay buffer. """
        s, a, r, s2, d = self.replay_buffer.sample(batch_size)
        s = np.asarray(s, dtype=np.float32)
        a = np.asarray(a, dtype=np.float32)
        r = np.asarray(r, dtype=np.float32)
        s2 = np.asarray(s2, dtype=np.float32)

        self._update_critic(s, a, r, s2)
        self._update_actor(s, batch_size)

        # This is for debugging only, i.e. not required for algorithm
        # to work: evaluate Q value for actor after update: providing same
        # state, should now evaluate to higher Q.
        a_mu_after = self.main_actor_net(s)
        q_mu_after = self.main_critic_net([s, a_mu_after])
        self.q_log['after'].append(np.mean(q_mu_after))

        # Apply Polyak updates
        self._update_target_networks()

        self.step += 1

    def _update_critic(self, state, action, reward, next_state):
        """ Update the main critic network based on given batch of input
        states. """
        with tf.GradientTape() as tape:
            next_action = self.target_actor_net(next_state)
            q_target = (reward + self.gamma *
                        self.target_critic_net([next_state, next_action]))
            q_vals = self.main_critic_net([state, action])
            q_loss = tf.reduce_mean((q_vals - q_target) ** 2)

        grads_q = tape.gradient(
            q_loss, self.main_critic_net.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(grads_q, self.main_critic_net.trainable_variables))
        self.losses_log['Q'].append(q_loss)

    def _update_actor(self, states, batch_size):
        """ Update the main actor network based on given batch of input
        states. """
        # with tf.GradientTape() as tape2:
        #     A_mu = self.mu(X)
        #     Q_mu = self.q_mu([X, A_mu])
        #     self.q_before.append(Q_mu[0])
        #     mu_loss = -tf.reduce_mean(Q_mu)
        # grads_mu = tape2.gradient(mu_loss, self.mu.trainable_variables)
        # self.mu_losses.append(mu_loss)

        # The same thing as above implemented manually to be as close to
        # QuantumDDPG implementation as possible
        # TODO: find out where the tensorflow warning comes from and try to
        #  speed up the code ...
        with tf.GradientTape() as tape2:
            a_mu = self.main_actor_net(states)
        jacobi_mu_wrt_mu_theta = tape2.jacobian(
            a_mu, self.main_actor_net.trainable_variables
        )

        with tf.GradientTape() as tape3:
            tape3.watch(a_mu)
            q_mu = self.main_critic_net([states, a_mu])
            self.q_log['before'].append(np.mean(q_mu))

        grad_q_wrt_a = tape3.gradient(q_mu, a_mu)

        grads_mu = []
        for i in range(len(jacobi_mu_wrt_mu_theta)):
            on_batch = tf.tensordot(
                jacobi_mu_wrt_mu_theta[i], grad_q_wrt_a,
                axes=((0, 1), (0, 1)))
            on_batch /= batch_size
            grads_mu.append(-on_batch)

        self.actor_optimizer.apply_gradients(
            zip(grads_mu, self.main_actor_net.trainable_variables))

        # Just for logging
        mean_ = 0.
        min_ = 1E39
        max_ = -1E39
        for grd in grads_mu:
            mean_ += np.mean(grd)
            min_ = min(np.min(grd), min_)
            max_ = max(np.max(grd), max_)
        mean_ /= len(grads_mu)

        self.actor_grads_log['min'].append(min_)
        self.actor_grads_log['max'].append(max_)
        self.actor_grads_log['mean'].append(mean_)

    def _update_target_networks(self):
        """ Apply Polyak update to both target networks. """
        # CRITIC
        target_weights = np.array(self.target_critic_net.get_weights())
        main_weights = np.array(self.main_critic_net.get_weights())
        new_target_weights = (self.tau_critic * main_weights +
                              (1 - self.tau_critic) * target_weights)
        self.target_critic_net.set_weights(new_target_weights)

        # ACTOR
        target_weights = np.array(self.target_actor_net.get_weights())
        main_weights = np.array(self.main_actor_net.get_weights())
        new_target_weights = (self.tau_actor * main_weights +
                              (1 - self.tau_actor) * target_weights)
        self.target_actor_net.set_weights(new_target_weights)


class QuantumDDPG:
    def __init__(self, state_space, action_space, gamma,
                 tau_critic, tau_actor, buffer_size,
                 critic_init_learning_rate, actor_init_learning_rate,
                 grad_clip_actor=20, grad_clip_critic=1.,
                 n_steps_estimate=1000):
        """ Implements quantum DDPG where actor and critic networks are
        represented by quantum Boltzmann machines and classical neural
        networks, respectively.
        :param state_space: openAI gym env state space
        :param action_space: openAI gym env action space
        :param gamma: reward discount factor
        :param tau_critic: soft update factor for critic target network
        :param tau_actor: soft update factor for actor target network
        :param buffer_size: max. size of replay buffer, not to be confused
        with batch_size.
        :param critic_init_learning_rate: initial learning rate of critic,
        may decay over time.
        :param actor_init_learning_rate: initial learning rate of actor,
        may decay over time.
        :param grad_clip_actor: limits to which actor gradients are clipped
        :param grad_clip_critic: limits to which critic gradients are clipped
        :param n_steps_estimate: estimated total number of training steps,
        needed for learning rate schedules (would like to get rid of this).
        """
        self.n_dims_state_space = len(state_space.high)
        self.n_dims_action_space = len(action_space.high)

        self.step = 0

        # Main hyperparameters
        self.gamma = gamma
        self.tau_critic = tau_critic
        self.tau_actor = tau_actor

        self.lr_schedule_critic = ExponentialDecay(
            critic_init_learning_rate, n_steps_estimate, 1.)
        self.lr_schedule_actor = ExponentialDecay(
            actor_init_learning_rate, n_steps_estimate, 1.)

        kwargs_q_func = dict(
            sampler_type='SQA',
            state_space=state_space,
            action_space=action_space,
            small_gamma=self.gamma,
            n_replicas=1,
            big_gamma=(20., 0.), beta=2,
            n_annealing_steps=100,
            n_meas_for_average=1,
            kwargs_qpu={})

        # Gradient clipping limits
        self.grad_clip_actor = grad_clip_actor
        self.grad_clip_critic = grad_clip_critic

        # Main and target actor network initialization
        # ACTOR
        actor_hidden_layers = [512, 200, 128]
        self.main_actor_net = generate_classical_actor(
            self.n_dims_state_space, self.n_dims_action_space,
            actor_hidden_layers)
        self.target_actor_net = generate_classical_actor(
            self.n_dims_state_space, self.n_dims_action_space,
            actor_hidden_layers)

        # CRITIC
        self.main_critic_net = QFunction(**kwargs_q_func)
        self.target_critic_net = QFunction(**kwargs_q_func)

        # Copy weights of main networks onto targets
        self.target_actor_net.set_weights(self.main_actor_net.get_weights())
        for k in self.main_critic_net.w_hh.keys():
            self.target_critic_net.w_hh[k] = self.main_critic_net.w_hh[k]
        for k in self.target_critic_net.w_vh.keys():
            self.target_critic_net.w_vh[k] = self.main_critic_net.w_vh[k]

        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.lr_schedule_actor)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            size=buffer_size, obs_dim=self.n_dims_state_space,
            act_dim=self.n_dims_action_space)

        # For logging
        # TODO: add critic gradient statistics as well?
        self.actor_grads_log = {'mean': [], 'min': [], 'max': []}
        self.losses_log = {'Q': [], 'Mu': []}
        self.q_log = {'before': [], 'after': []}

    def get_proposed_action(self, state, noise_scale):
        """ Use main actor network to obtain proposed action for en input
        state. Add potentially some Gaussian noise for exploration. Action is
        clipped at [-1, 1] in any case. """
        action = self.main_actor_net.predict(state.reshape(1, -1))[0]
        action += noise_scale * np.random.randn(self.n_dims_action_space)
        return np.clip(action, -1., 1.)

    def update(self, batch_size):
        """ Calculate and apply the updates of the critic and actor
        networks based on batch of samples from experience replay buffer. """
        s, a, r, s2, d = self.replay_buffer.sample(batch_size)
        s = np.asarray(s, dtype=np.float32)
        a = np.asarray(a, dtype=np.float32)
        r = np.asarray(r, dtype=np.float32)
        s2 = np.asarray(s2, dtype=np.float32)

        self._update_critic(s, a, r, s2)
        self._update_actor(s, batch_size)

        # Evaluate Q value for new actor (providing same state, should now
        # give higher Q)
        a_mu_after = self.main_actor_net(s)
        q_after, _, _ = self.main_critic_net.calculate_q_value_on_batch(
            s, a_mu_after)
        self.q_log['after'].append(np.mean(q_after))

        # Apply Polyak updates
        self._update_target_networks()

        self.step += 1

    def _update_critic(self, state, action, reward, next_state):
        """ Update the main critic network based on given batch of input
        states. """
        next_action = self.target_actor_net(next_state)

        # Loop over batch of states, actions, rewards, next_states
        for jj in np.arange(len(state)):
            # Recalculate q_value of (sample.state, sample.action) pair
            q_value, spin_configs, visible_nodes = (
                self.main_critic_net.calculate_q_value(state[jj], action[jj]))

            # Now calculate the next_q_value of the greedy action, without
            # actually taking the action (to take actions in env.,
            # we don't follow purely greedy action).
            next_q_value, spin_configurations, visible_nodes = (
                self.target_critic_net.calculate_q_value(
                    state=next_state[jj], action=next_action[jj]))

            # Update weights and target Q-function if needed
            # Note that clipping is also done inside QBM update_weights
            # method ...
            self.main_critic_net.update_weights(
                spin_configs, visible_nodes, q_value, next_q_value,
                reward[jj],
                learning_rate=self.lr_schedule_critic(self.step).numpy(),
                grad_clip=self.grad_clip_critic)

    def _update_actor(self, state, batch_size):
        """ Update the main actor network based on given batch of input
        states. """
        with tf.GradientTape() as tape2:
            a_mu = self.main_actor_net(state)

        q, _, _ = self.main_critic_net.calculate_q_value_on_batch(state, a_mu)
        self.q_log['before'].append(np.mean(q))

        # Apply chain-rule manually here:
        jacobi_mu_wrt_mu_theta = tape2.jacobian(
            a_mu, self.main_actor_net.trainable_variables)
        grad_q_wrt_a = self.get_action_derivative(state, a_mu, batch_size)

        grads_mu = []
        for i in range(len(jacobi_mu_wrt_mu_theta)):
            on_batch = tf.tensordot(
                jacobi_mu_wrt_mu_theta[i], grad_q_wrt_a,
                axes=((0, 1), (0, 1)))
            on_batch /= batch_size

            on_batch = np.clip(on_batch,
                               a_min=-self.grad_clip_actor,
                               a_max=self.grad_clip_actor)
            grads_mu.append(-on_batch)

        self.actor_optimizer.apply_gradients(
            zip(grads_mu, self.main_actor_net.trainable_variables))

        # For logging
        mean_ = 0.
        min_ = 1E39
        max_ = -1E39
        for grd in grads_mu:
            mean_ += np.mean(grd)
            min_ = min(np.min(grd), min_)
            max_ = max(np.max(grd), max_)
        mean_ /= len(grads_mu)

        self.actor_grads_log['min'].append(min_)
        self.actor_grads_log['max'].append(max_)
        self.actor_grads_log['mean'].append(mean_)

    def _update_target_networks(self):
        """ Apply Polyak update to both target networks. """
        # CRITIC
        for k in self.main_critic_net.w_hh.keys():
            self.target_critic_net.w_hh[k] = (
                    self.tau_critic * self.main_critic_net.w_hh[k] +
                    (1. - self.tau_critic) * self.target_critic_net.w_hh[k])
        for k in self.main_critic_net.w_vh.keys():
            self.target_critic_net.w_vh[k] = (
                    self.tau_critic * self.main_critic_net.w_vh[k] +
                    (1. - self.tau_critic) * self.target_critic_net.w_vh[k])

        # ACTOR
        target_weights = np.array(self.target_actor_net.get_weights())
        main_weights = np.array(self.main_actor_net.get_weights())
        new_target_weights = (self.tau_actor * main_weights +
                              (1 - self.tau_actor) * target_weights)
        self.target_actor_net.set_weights(new_target_weights)

    def get_action_derivative(
            self, states: list, actions: list, batch_size: int,
            epsilon: float = 0.2):
        """ Calculate numerical derivative of QBM Q values with respect to
        action.
        :param states: batch of states from replay buffer.
        :param actions: corresponding actions.
        :param batch_size: number of samples drawn from replay buffer.
        :param epsilon: step size to take numerical derivative using shift
        rule.
        :return np.ndarray of gradients of shape (batch_size,
        n_dims_action_space). """
        # Need to take derivative for each action separately
        # e.g. if we have batch size of 5, and 10 actions, we expect an
        # output for dQ / da of shape (5, 10).
        # grads_mean = np.zeros((batch_size, self.action_dim))

        n_avg = 1
        grads = np.zeros((n_avg, batch_size, self.n_dims_action_space))
        for j in range(n_avg):
            for i in range(self.n_dims_action_space):
                actions_tmp1 = np.array(actions).copy()
                actions_tmp1[:, i] += epsilon
                qeps_plus, _, _ = (
                    self.main_critic_net.calculate_q_value_on_batch(
                        states, actions_tmp1)
                )

                actions_tmp2 = np.array(actions).copy()
                actions_tmp2[:, i] -= epsilon
                qeps_minus, _, _ = (
                    self.main_critic_net.calculate_q_value_on_batch(
                        states, actions_tmp2)
                )

                grads[j, :, i] = np.atleast_1d(
                    np.float_((qeps_plus - qeps_minus) / (2 * epsilon)))

        grads = np.asarray(grads, dtype=np.float32)
        grads = np.mean(grads, axis=0)
        return grads
