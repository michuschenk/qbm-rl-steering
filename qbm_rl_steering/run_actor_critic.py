import numpy as np
import gym

import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras as K
from tensorflow.python.framework.ops import disable_eager_execution

from qbm_rl_steering.core.visualization import plot_log
from qbm_rl_steering.environment.target_steering_1d import TargetSteeringEnv
from qbm_rl_steering.core.qbm import QFunction
from qbm_rl_steering.core.utils import Memory

try:
    import matplotlib

    matplotlib.use('qt5agg')
except ImportError as err:
    print(err)

disable_eager_execution()


class QuantumActorCritic:
    def __init__(self, env: gym.Env, gamma_rl: float):
        # env intel
        self.env = env
        self.action_n = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape

        # constants
        self.action_limit = max(env.action_space.high)
        self.gamma_rl = gamma_rl  # discounted reward factor
        self.tau_soft_update = 0.1  # soft update factor
        self.replay_buffer_size = int(1e5)
        self.replay_batch_size = 10  # training batch size.
        self.action_noise_scale = 0.1

        # Critic-related input
        self.n_annealing_steps = 80
        self.n_annealings_for_average = 40
        self.learning_rate_critic = 4e-3

        # Actor-related input
        self.init_learning_rate_actor = 1e-3

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
            action = np.clip(action, -self.action_limit, self.action_limit)
        return action

    def get_target_actions(self, states):
        """ Get batch of proposed actions from the target actor network based
        on batch of input states. """
        return self.actor_target.predict_on_batch(states)

    def train_actor(self, states, actions):
        self.actor.train_on_batch(states, states)

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
            n_graph_nodes=16,
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
            dq_over_dstate = self.get_state_derivative(y_true, y_pred)
            dq_over_daction = self.get_action_derivative(y_true, y_pred)

            return (np.float32(q_value), np.float32(dq_over_dstate),
                    np.float32(dq_over_daction))

        z, dz_over_dstate, dz_over_daction = tf.numpy_function(
            get_q_value, [y_true, y_pred], [tf.float32, tf.float32, tf.float32])

        def grad(dy):
            return (tf.dtypes.cast(dy * dz_over_dstate, dtype=tf.float32),
                    tf.dtypes.cast(dy * dz_over_daction, dtype=tf.float32))

        return z, grad

    def get_state_derivative(self, y_true, y_pred, epsilon=0.2):
        qeps_plus, _, _ = self.critic.calculate_q_value_on_batch(
            y_true + epsilon, y_pred)
        qeps_minus, _, _ = self.critic.calculate_q_value_on_batch(
            y_true - epsilon, y_pred)
        return np.float_((qeps_plus - qeps_minus) / (2 * epsilon))

    def get_action_derivative(self, y_true, y_pred, epsilon=0.2):
        qeps_plus, _, _ = self.critic.calculate_q_value_on_batch(
            y_true, y_pred + epsilon)
        qeps_minus, _, _ = self.critic.calculate_q_value_on_batch(
            y_true, y_pred - epsilon)
        return np.float_((qeps_plus - qeps_minus) / (2 * epsilon))

    def train_critic(self, states, next_states, actions, rewards, dones):
        # Training the QBM
        # Use experiences found in replay_buffer to update weights
        for jj in np.arange(len(states)):
            # Act only greedily here: should be OK to do that always since we
            # collect our experiences according to epsilon-greedy policy.

            # Recalculate q_value of (sample.state, sample.action) pair
            q_value, spin_configs, visible_nodes = (
                self.critic_target.calculate_q_value(states[jj], actions[jj]))

            # Now calculate the next_q_value of the greedy action, without
            # actually taking the action (to take actions in env.,
            # we don't follow purely greedy action).
            next_action = self.get_target_actions(next_states[jj])
            # print('next action', next_action)
            next_q_value, spin_configurations, visible_nodes = (
                self.critic_target.calculate_q_value(
                    state=next_states[jj], action=next_action))

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
        self.train_critic(states, next_states, actions, rewards, dones)
        self.train_actor(states, actions)
        self._soft_update_actor_and_critic()


if __name__ == "__main__":

    gamma_rl = 0.85
    n_epochs = 16
    max_episode_length = 10
    initial_exploration_steps = 10
    initial_reward = 0

    env = TargetSteeringEnv(max_steps_per_episode=max_episode_length)
    agent = QuantumActorCritic(env, gamma_rl=gamma_rl)

    state, reward, done, ep_rew, ep_len, ep_cnt = env.reset(), initial_reward, \
                                                  False, [[]], 0, 0

    # Calculate reward in current state
    _, intensity = env.get_pos_at_bpm_target(env.mssb_angle)
    ep_rew[-1].append(env.get_reward(intensity))
    total_steps = max_episode_length * n_epochs

    # Main loop: collect experience in env and update/log each epoch
    to_exploitation = False
    for t in range(total_steps):
        if t > initial_exploration_steps:
            action = agent.get_action(state, episode=1)
            action = np.squeeze(action)
        else:
            action = env.action_space.sample()

        # Step the env
        next_state, reward, done, _ = env.step(action)
        ep_rew[-1].append(reward)  # keep adding to the last element till done
        ep_len += 1

        done = False if ep_len == max_episode_length else done

        # Store experience to replay buffer
        agent.replay_memory.store(state, action, reward, next_state, done)

        state = next_state

        if done or (ep_len == max_episode_length):
            ep_cnt += 1
            if True:
                print(f"Episode: {len(ep_rew) - 1}, Reward: {ep_rew[-1][-1]}, "
                      f"Length: {len(ep_rew[-1])}")
            ep_rew.append([])

            for _ in range(ep_len):
                agent.train()

            state, reward, done, ep_ret, ep_len = (
                env.reset(), initial_reward, False, 0, 0)

            _, intensity = env.get_pos_at_bpm_target(env.mssb_angle)
            ep_rew[-1].append(env.get_reward(intensity))

    init_rewards = []
    rewards = []
    reward_lengths = []
    for episode in ep_rew[:-1]:
        if (len(episode) > 0):
            rewards.append(episode[-1])
            init_rewards.append(episode[0])
            reward_lengths.append(len(episode) - 1)
    print('Total number of interactions:', np.sum(reward_lengths))

    plot_log(env, fig_title='Training')

    # Agent evaluation
    n_episodes_eval = 50
    episode_counter = 0
    env = TargetSteeringEnv(max_steps_per_episode=max_episode_length)
    while episode_counter < n_episodes_eval:
        state = env.reset(init_outside_threshold=True)
        state = np.atleast_2d(state)
        while True:
            a = agent.get_action(state, noise=0)
            a = np.squeeze(a)
            state, reward, done, _ = env.step(a)
            if done:
                episode_counter += 1
                break

    plot_log(env, fig_title='Evaluation')
