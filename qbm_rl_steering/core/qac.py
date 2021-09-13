import gym
import numpy as np

import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras as K
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

from qbm_rl_steering.core.qbm import QFunction
from qbm_rl_steering.core.utils import Memory


class QuantumActorCritic:
    def __init__(self, env: gym.Env, gamma_rl: float, batch_size: int):
        # env intel
        self.env = env
        self.action_n = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape

        # constants
        self.action_limit = max(env.action_space.high)
        self.gamma_rl = gamma_rl  # discounted reward factor
        self.tau_soft_update = 0.05  # soft update factor
        self.replay_buffer_size = int(1e5)
        self.replay_batch_size = batch_size  # training batch size.
        self.action_noise_scale = 0.1

        # Critic-related input
        self.n_annealing_steps = 70  # 150
        self.n_annealings_for_average = 60  # 200
        self.learning_rate_critic = 3e-3

        # Actor-related input
        self.init_learning_rate_actor = 3e-3

        # create networks
        self.dummy_Q_target_prediction_input = np.zeros(
            (self.replay_batch_size, 1))
        self.dummy_dones_input = np.zeros((self.replay_batch_size, 1))

        self.critic = self._generate_critic_net()
        self.critic_target = self._generate_critic_net()
        self.actor = self._generate_actor_net()
        self.actor_target = self._generate_actor_net()

        # Initialize replay buffer
        self.replay_memory = Memory(
            self.state_dim[0], self.action_n, self.replay_buffer_size)

    def _generate_actor_net(self):
        """ Create classical actor neural network. """
        state_input = KL.Input(shape=self.state_dim)
        dense = KL.Dense(128, activation='relu')(state_input)
        dense = KL.Dense(128, activation='relu')(dense)
        out = KL.Dense(self.action_n, activation='tanh')(dense)
        model = K.Model(inputs=state_input, outputs=out)
        model.compile(optimizer=K.optimizers.Adam(
            learning_rate=self.init_learning_rate_actor),
            loss=self._ddpg_actor_loss)
        model.summary()
        return model

    def get_action(self, states, noise=None, episode=1):
        """ Get batch of proposed actions from the local actor network based
        on batch of input states. Also add noise during training phase. """

        if noise is None:
            noise = self.action_noise_scale
        if len(states.shape) == 1:
            states = states.reshape(1, -1)
        action = self.actor.predict_on_batch(states)
        if noise != 0:
            action += noise / episode * np.random.randn(self.action_n)
            # action += noise * np.random.randn(self.action_n)
            action = np.clip(action, -self.action_limit, self.action_limit)
        return action

    def get_target_actions(self, states):
        """ Get batch of proposed actions from the target actor network based
        on batch of input states. """
        # states = np.atleast_2d(states)
        return self.actor_target.predict_on_batch(states)

    def train_actor(self, states, actions):
        # print('before training actor', self.actor.get_weights())
        self.actor.train_on_batch(states, states)
        # print('after training actor', self.actor.get_weights())

    def _generate_critic_net(self):
        """ Initialize QBM with random weights / couplings. Here we also
        already fix the spin configuration sampler to be simulated quantum
        annealing. Well working default parameters are set based on past
        experience with Q learning QBM. """
        # TODO: fix the interface here ...
        # Define QBM q-function parameters
        kwargs_q_func = dict(
            sampler_type='SQA',
            state_space=self.env.observation_space,
            action_space=self.env.action_space,
            small_gamma=self.gamma_rl,
            n_replicas=1,
            big_gamma=(20., 0.), beta=2.,
            n_annealing_steps=self.n_annealing_steps,
            n_meas_for_average=self.n_annealings_for_average,
            kwargs_qpu={})

        return QFunction(**kwargs_q_func)

    def _ddpg_actor_loss(self, y_true, y_pred):
        # y_pred is the action from the actor net
        # y_true is the state, we maximise q
        q = self.q_custom_gradient(y_true, y_pred)
        return -K.backend.mean(q)

    @tf.custom_gradient
    def q_custom_gradient(self, y_true, y_pred):
        def get_q_value(y_true, y_pred):
            q_value, _, _ = (
                self.critic.calculate_q_value_on_batch(y_true, y_pred))
            q_value = np.array(q_value)
            # print('q_value', q_value)

            dq_over_dstate = self.get_state_derivative(y_true, y_pred)
            # print('dq_over_dstate', dq_over_dstate)

            dq_over_daction = self.get_action_derivative(y_true, y_pred)
            # print('dq_over_daction', dq_over_daction)

            return (np.float32(q_value), np.float32(dq_over_dstate),
                    np.float32(dq_over_daction))

        z, dz_over_dstate, dz_over_daction = tf.numpy_function(
            get_q_value, [y_true, y_pred], [tf.float32, tf.float32, tf.float32])

        def grad(dy):
            return (tf.dtypes.cast(dy * dz_over_dstate, dtype=tf.float32),
                    tf.dtypes.cast(dy * dz_over_daction, dtype=tf.float32))

        return z, grad

    # def get_state_derivative(self, y_true, y_pred, epsilon=0.2):
    #     # print('state deriv, plus epsilon')
    #     qeps_plus, _, _ = self.critic.calculate_q_value_on_batch(
    #         y_true + epsilon, y_pred)
    #
    #     # print('state deriv, minus epsilon')
    #     qeps_minus, _, _ = self.critic.calculate_q_value_on_batch(
    #         y_true - epsilon, y_pred)
    #     return np.atleast_1d(
    #         np.float_((qeps_plus - qeps_minus) / (2 * epsilon)))

    def get_state_derivative(self, states, actions, epsilon=0.15):
        # Need to take derivative for each action separately
        # e.g. if we have batch size of 5, and 10 actions, we expect an
        # output for dQ / da of shape (5, 10).
        grads = np.zeros((self.replay_batch_size, self.state_dim[0]))

        for i in range(self.state_dim[0]):
            states_tmp1 = np.array(states).copy()
            states_tmp1[:, i] += epsilon
            qeps_plus, _, _ = self.critic.calculate_q_value_on_batch(
                states_tmp1, actions)

            states_tmp2 = np.array(states).copy()
            states_tmp2[:, i] -= epsilon
            qeps_minus, _, _ = self.critic.calculate_q_value_on_batch(
                states_tmp2, actions)

            grads[:, i] = np.atleast_1d(
                np.float_((qeps_plus - qeps_minus) / (2 * epsilon)))
        grads = np.asarray(grads, dtype=np.float32)
        return grads.T
    # def get_action_derivative(self, y_true, y_pred, epsilon=0.2):
        # # print('action deriv, plus epsilon')
        # qeps_plus, _, _ = self.critic.calculate_q_value_on_batch(
        #     y_true, y_pred + epsilon)
        #
        # # print('action deriv, minus epsilon')
        # qeps_minus, _, _ = self.critic.calculate_q_value_on_batch(
        #     y_true, y_pred - epsilon)
        # return np.atleast_1d(
        #     np.float_((qeps_plus - qeps_minus) / (2 * epsilon)))

    def get_action_derivative(self, states, actions, epsilon=0.8):
        # Need to take derivative for each action separately
        # e.g. if we have batch size of 5, and 10 actions, we expect an
        # output for dQ / da of shape (5, 10).
        grads = np.zeros((self.replay_batch_size, self.action_n))
        n_avg = 1

        for i in range(self.action_n):
            actions_tmp1 = np.array(actions).copy()
            actions_tmp1[:, i] += epsilon
            qeps_plus = np.zeros((n_avg, self.replay_batch_size))
            for k in range(n_avg):
                qeps_plus[k, :], _, _ = self.critic.calculate_q_value_on_batch(
                    states, actions_tmp1)
            qeps_plus_mean = np.mean(qeps_plus, axis=0)

            actions_tmp2 = np.array(actions).copy()
            actions_tmp2[:, i] -= epsilon
            qeps_minus = np.zeros((n_avg, self.replay_batch_size))
            for k in range(n_avg):
                qeps_minus[k, :], _, _ = self.critic.calculate_q_value_on_batch(
                    states, actions_tmp2)
            qeps_minus_mean = np.mean(qeps_minus, axis=0)

            grads[:, i] = np.atleast_1d(
                np.float_((qeps_plus_mean - qeps_minus_mean) / (2 * epsilon)))

        grads = np.asarray(grads, dtype=np.float32)
        print('GRADS wrt ACTION num.', grads)

        return grads.T

    def get_action_derivative_analytical(self, states, actions):
        # grads_analytical = np.zeros((self.replay_batch_size, self.action_n))
        _, _, _, _, grads_analytical = self.critic.calculate_q_value_on_batch(
            states, actions, calc_derivative=True)

        # We need to flip the sign since this is the derivative wrt F,
        # but we want derivative wrt. Q, where Q = -F
        grads_analytical *= -1

        print('GRADS wrt ACTION analytical', grads_analytical)
        # _, _, _, _, grads_analytical2 = self.critic.calculate_q_value(
        #     states[0], actions[0], calc_derivative=True)
        # print('grads analytical 1', grads_analytical)
        # print('grads analytical 2', grads_analytical2)
        return grads_analytical.T

    def train_critic(self, states, next_states, actions, rewards, dones):
        # Training the QBM
        # Use experiences found in replay_buffer to update weights
        next_actions = self.get_target_actions(next_states)
        for jj in np.arange(len(states)):
            # print('self.actor.get_weights()', self.actor.get_weights())
            # print('self.critic.w_hh', self.critic.w_hh)

            # Act only greedily here: should be OK to do that always since we
            # collect our experiences according to epsilon-greedy policy.

            # Recalculate q_value of (sample.state, sample.action) pair
            q_value, spin_configs, visible_nodes = (
                self.critic_target.calculate_q_value(states[jj], actions[jj]))

            # Now calculate the next_q_value of the greedy action, without
            # actually taking the action (to take actions in env.,
            # we don't follow purely greedy action).
            # next_action = self.get_target_actions(next_states[jj])
            # next_action = next_action.flatten()
            next_q_value, spin_configurations, visible_nodes = (
                self.critic_target.calculate_q_value(
                    state=next_states[jj], action=next_actions[jj]))

            # Update weights and target Q-function if needed
            self.critic.update_weights(
                spin_configs, visible_nodes, q_value, next_q_value,
                rewards[jj], learning_rate=self.learning_rate_critic)

    def _soft_update_actor_and_critic(self):
        """ Perform update of target actor and critic network weights using
        Polyak average. """

        # Critic soft update
        for k in self.critic.w_hh.keys():
            self.critic_target.w_hh[k] = (
                    self.tau_soft_update * self.critic.w_hh[k] +
                    (1.0 - self.tau_soft_update) * self.critic_target.w_hh[k])
        for k in self.critic.w_vh.keys():
            self.critic_target.w_vh[k] = (
                    self.tau_soft_update * self.critic.w_vh[k] +
                    (1.0 - self.tau_soft_update) * self.critic_target.w_vh[k])

        # Actor soft update
        weights_actor_local = np.array(self.actor.get_weights())
        weights_actor_target = np.array(self.actor_target.get_weights())
        self.actor_target.set_weights(
            self.tau_soft_update * weights_actor_local +
            (1.0 - self.tau_soft_update) * weights_actor_target)

    def train(self):
        """ Load a number of experiences from replay buffer and train agent's
        local actor and critic networks. This includes running Polyak update
        of the target actor and critic weights.
        """
        states, actions, rewards, next_states, dones = (
            self.replay_memory.get_sample(batch_size=self.replay_batch_size))
        # print('train, states.shape, actions.shape, rewards.shape, '
        #       'next_states.shape, dones.shape', states.shape, actions.shape,
        #       rewards.shape, next_states.shape, dones.shape)
        # rewards *= 100
        self.train_critic(states, next_states, actions, rewards, dones)
        self.train_actor(states, actions)
        self._soft_update_actor_and_critic()
