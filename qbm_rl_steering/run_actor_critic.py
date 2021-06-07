import numpy as np

import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras as K
from tensorflow.python.framework.ops import disable_eager_execution

from qbm_rl_steering.utils.helpers import plot_log
from qbm_rl_steering.environment.env_desc import TargetSteeringEnv
from qbm_rl_steering.agents.qbm_core import QFunction

try:
    import matplotlib
    matplotlib.use('qt5agg')
except ImportError as err:
    print(err)

disable_eager_execution()

class Memory:
    """ A FIFO experience replay buffer.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.states = np.zeros([size, obs_dim], dtype=np.float32)
        self.actions = np.zeros([size, act_dim], dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.next_states = np.zeros([size, obs_dim], dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def get_sample(self, batch_size=32):
        if self.size < batch_size:
            idxs = np.random.randint(0, self.size, size=self.size)
        else:
            idxs = np.random.randint(0, self.size, size=batch_size)
        return self.states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs], self.dones[idxs]

class ClassicACAgent(object):
    def __init__(self, GAMMA, env):
        # env intel
        self.env = env
        self.action_n = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape
        self.state_n = self.state_dim[0]
        # constants
        self.ACT_LIMIT = max(env.action_space.high)  # required for clipping prediciton aciton
        self.GAMMA = GAMMA  # discounted reward factor
        self.TAU = 0.1  # soft update factor
        self.BUFFER_SIZE = int(1e6)
        self.BATCH_SIZE = 10  # training batch size.
        self.ACT_NOISE_SCALE = 0.2

        # QBM related stuff
        self.n_annealing_steps = 100
        self.n_meas_for_average = 50
        self.learning_rate = 1e-3

        # create networks
        self.dummy_Q_target_prediction_input = np.zeros((self.BATCH_SIZE, 1))
        self.dummy_dones_input = np.zeros((self.BATCH_SIZE, 1))

        self.critic = self._gen_critic_network()
        self.critic_target = self._gen_critic_network()
        self.actor = self._gen_actor_network()  # the local actor wich is trained on.
        self.actor_target = self._gen_actor_network()  # the target actor which is slowly updated toward optimum

        self.memory = Memory(self.state_n, self.action_n, self.BUFFER_SIZE)

    def _gen_actor_network(self):
        state_input = KL.Input(shape=self.state_dim)
        dense = KL.Dense(128, activation='relu')(state_input)
        dense = KL.Dense(128, activation='relu')(dense)
        out = KL.Dense(self.action_n, activation='tanh')(dense)
        model = K.Model(inputs=state_input, outputs=out)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.001,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True)
        model.compile(optimizer=K.optimizers.Adam(learning_rate=0.001),
                      loss=self._ddpg_actor_loss)
        model.summary()
        return model

    def get_action(self, states, noise=None, episode=1):
        if noise is None: noise = self.ACT_NOISE_SCALE
        if len(states.shape) == 1: states = states.reshape(1, -1)
        action = self.actor.predict_on_batch(states)
        if noise != 0:
            action += noise/episode * np.random.randn(self.action_n)
            action = np.clip(action, -self.ACT_LIMIT, self.ACT_LIMIT)
        return action

    def get_target_action(self, states):
        return self.actor_target.predict_on_batch(states)

    def train_actor(self, states, actions):
        self.actor.train_on_batch(states, states) # Q_predictions)

    def _gen_critic_network(self):
        # Define Q functions and their updates
        kwargs_q_func = dict(
            sampler_type='SQA',
            state_space=self.env.observation_space,
            action_space=self.env.action_space,
            small_gamma=self.GAMMA,
            n_graph_nodes=16,
            n_replicas=1,
            big_gamma=(20., 0.), beta=2.,
            n_annealing_steps=self.n_annealing_steps,
            n_meas_for_average=self.n_meas_for_average,
            kwargs_qpu={})

        q_function = QFunction(**kwargs_q_func)
        return q_function

    def _ddpg_actor_loss(self, y_true, y_pred):
        # y_pred is the action from the actor net. y_true is the state, we maximise the q
        q = self.q_custom_gradient(y_true, y_pred)
        return -K.backend.mean(q)

    @tf.custom_gradient
    def q_custom_gradient(self, y_true, y_pred):
        def get_q_value(y_true, y_pred):
            q_value, _, _ = (
                self.critic.calculate_q_value_on_batch(y_true, y_pred))
            dq_over_dstate = self.get_state_derivative(y_true, y_pred)
            dq_over_daction = self.get_action_derivative(y_true, y_pred)

            return np.float32(q_value), np.float32(dq_over_dstate),\
                   np.float32(dq_over_daction)
            # first is function, second is gradient

        z, dz_over_dstate, dz_over_daction = tf.numpy_function(
            get_q_value, [y_true, y_pred], [tf.float32, tf.float32, tf.float32])

        def grad(dy):
            return (tf.dtypes.cast(dy * dz_over_dstate, dtype=tf.float32),
                    tf.dtypes.cast(dy * dz_over_daction, dtype=tf.float32))
        return z, grad

    def get_state_derivative(self, y_true, y_pred, epsilon=0.04):
        # q0, _, _ = self.critic.calculate_q_value(y_true, y_pred)
        qeps_plus, _, _ = self.critic.calculate_q_value_on_batch(
            y_true + epsilon, y_pred)
        qeps_minus, _, _ = self.critic.calculate_q_value_on_batch(
            y_true - epsilon, y_pred)
        return np.float_((qeps_plus - qeps_minus) / (2*epsilon))

    def get_action_derivative(self, y_true, y_pred, epsilon=0.04):
        # q0, _, _ = self.critic.calculate_q_value(y_true, y_pred)
        qeps_plus, _, _ = self.critic.calculate_q_value_on_batch(
            y_true, y_pred + epsilon)
        qeps_minus, _, _ = self.critic.calculate_q_value_on_batch(
            y_true, y_pred - epsilon)
        return np.float_((qeps_plus - qeps_minus) / (2*epsilon))

    def train_critic(self, states, next_states, actions, rewards, dones):
        # Training the QBM
        # Use experiences in replay_buffer to update weights
        # n_replay_batch = self.replay_batch_size
        # if len(self.replay_buffer) < self.replay_batch_size:
        #     n_replay_batch = len(self.replay_buffer)
        # replay_samples = random.sample(self.replay_buffer, n_replay_batch)

        for jj in np.arange(len(states)):
            # Act only greedily here: should be OK to do that always
            # because we collect our experiences according to an
            # epsilon-greedy policy

            # Recalculate the q_value of the (sample.state, sample.action)
            # pair
            q_value, spin_configs, visible_nodes = (
                self.critic_target.calculate_q_value(states[jj], actions[jj]))

            # Now calculate the next_q_value of the greedy action, without
            # actually taking the action (to take actions in env.,
            # we don't follow purely greedy action).
            next_action = self.get_target_action(next_states[jj])
            # print('next action', next_action)
            next_q_value, spin_configurations, visible_nodes = (
                self.critic_target.calculate_q_value(
                    state=next_states[jj], action=next_action))

            # Update weights and update target Q-function if needed
            # TODO: change learning rate to fixed value...
            self.critic.update_weights(
                spin_configs, visible_nodes, q_value, next_q_value,
                rewards[jj], learning_rate=5e-3)

    def _soft_update_actor_and_critic(self):
        # Critic soft update:
        # TODO: check here if something doesn't work ...
        for k in self.critic.w_hh.keys():
            self.critic_target.w_hh[k] = (
                    self.TAU * self.critic.w_hh[k] +
                    (1.0 - self.TAU) * self.critic_target.w_hh[k])
        for k in self.critic.w_vh.keys():
            self.critic_target.w_vh[k] = (
                    self.TAU * self.critic.w_vh[k] +
                    (1.0 - self.TAU) * self.critic_target.w_vh[k])

        # Actor soft update
        weights_actor_local = np.array(self.actor.get_weights())
        weights_actor_target = np.array(self.actor_target.get_weights())
        self.actor_target.set_weights(
            self.TAU * weights_actor_local +
            (1.0 - self.TAU) * weights_actor_target)

    def store(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

    def train(self):
        """Trains the networks of the agent (local actor and critic) and soft-updates  target.
        """
        states, actions, rewards, next_states, dones = self.memory.get_sample(
            batch_size=self.BATCH_SIZE)
        self.train_critic(states, next_states, actions, rewards, dones)
        # print('states', states)
        # print('actions', actions)
        # print('memory buffer', states.shape)
        # print('states in memory', states)
        self.train_actor(states, actions)
        self._soft_update_actor_and_critic()


if __name__ == "__main__":

    GAMMA = 0.85
    EPOCHS = 16
    MAX_EPISODE_LENGTH = 10
    START_STEPS = 15
    INITIAL_REW = 0

    env = TargetSteeringEnv(max_steps_per_episode=MAX_EPISODE_LENGTH)
    agent = ClassicACAgent(GAMMA, env)

    s = np.linspace(-1, 1, 20)
    a = np.linspace(-1, 1, 20)
    q = np.zeros((len(s), len(a)))
    dqda = np.zeros((len(s), len(a)))
    dqds = np.zeros((len(s), len(a)))
    for i, s_ in enumerate(s):
        for j, a_ in enumerate(a):
            a_ = np.atleast_1d(a_)
            s_ = np.atleast_1d(s_)
            q[i, j], _, _ = agent.critic.calculate_q_value(s_, a_)
            dqda[i, j] = agent.get_action_derivative(s_, a_, epsilon=0.5)
            dqds[i, j] = agent.get_state_derivative(s_, a_, epsilon=0.5)

    # fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(8, 10))
    # imq = axs[0].pcolormesh(s, a, q.T, shading='auto')
    # fig.colorbar(imq, ax=axs[0])
    # axs[0].set_title('Q')
    # axs[0].set_ylabel('action')
    #
    # imdqda = axs[1].pcolormesh(s, a, dqda.T, shading='auto')
    # axs[1].axvline(s[6], c='red')
    # fig.colorbar(imdqda, ax=axs[1])
    # axs[1].set_title('dq / da')
    # axs[1].set_ylabel('action')
    #
    # imdqds = axs[2].pcolormesh(s, a, dqds.T, shading='auto')
    # axs[2].axhline(a[5], c='red')
    # fig.colorbar(imdqds, ax=axs[2])
    # axs[2].set_title('dq / ds')
    # axs[2].set_xlabel('state')
    # axs[2].set_ylabel('action')
    # plt.show()
    #
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

    state, reward, done, ep_rew, ep_len, ep_cnt = env.reset(), INITIAL_REW, \
                                                  False, [[]], 0, 0

    # Calculate reward in current state
    _, intensity = env.get_pos_at_bpm_target(env.mssb_angle)
    ep_rew[-1].append(env.get_reward(intensity))
    total_steps = MAX_EPISODE_LENGTH * EPOCHS

    # Main loop: collect experience in env and update/log each epoch
    to_exploitation = False
    for t in range(total_steps):

        # print('actor weights:', agent.actor.get_weights())
        # print('n nans actor weights:', np.sum(np.isnan(np.array(
        #       agent.actor.get_weights()).flatten())))

        if t > START_STEPS:
            # print('\n\n\n!!!!!!!! END OF RANDOM SAMPLING !!!!!!!!\n\n\n')
            if not to_exploitation:
                print('Now exploiting ...')
                to_exploitation = True
            action = agent.get_action(state, episode=1)
            action = np.squeeze(action)
        else:
            # print('\n!!!!!!!! USING RANDOM SAMPLING !!!!!!!!\n')
            action = env.action_space.sample()

        # Step the env
        # print('action before env.step', action)
        next_state, reward, done, _ = env.step(action)
        #print("reward ",reward,done)
        ep_rew[-1].append(reward) #keep adding to the last element till done
        ep_len += 1

        #print(done)
        done = False if ep_len==MAX_EPISODE_LENGTH else done

        # Store experience to replay buffer
        agent.store(state, action, reward, next_state, done)

        state = next_state

        if done or (ep_len == MAX_EPISODE_LENGTH):
            ep_cnt += 1
            if True: #ep_cnt % RENDER_EVERY == 0:
                print(f"Episode: {len(ep_rew)-1}, Reward: {ep_rew[-1][-1]}, "
                      f"Length: {len(ep_rew[-1])}")
            ep_rew.append([])

            for _ in range(ep_len):
                agent.train()

            state, reward, done, ep_ret, ep_len = (
                env.reset(), INITIAL_REW, False, 0, 0)

            _, intensity = env.get_pos_at_bpm_target(env.mssb_angle)
            ep_rew[-1].append(env.get_reward(intensity))

    init_rewards = []
    rewards = []
    reward_lengths = []
    for episode in ep_rew[:-1]:
        if(len(episode) > 0):
            rewards.append(episode[-1])
            init_rewards.append(episode[0])
            reward_lengths.append(len(episode)-1)
    print('Total number of interactions:', np.sum(reward_lengths))

    plot_log(env, fig_title='Training')
    # fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(8, 6))
    # axs[0].plot(reward_lengths)
    # axs[0].axhline(env.max_steps_per_episode, c='k', ls='-',
    #                label='Max. # steps')
    # axs[0].set_ylabel('# steps per episode')
    # axs[0].set_ylim(0, env.max_steps_per_episode + 0.5)
    # axs[0].legend(loc='upper right')
    #
    # axs[1].plot(init_rewards, c='r', marker='.', label='initial')
    # axs[1].plot(rewards, c='forestgreen', marker='x', label='final')
    # axs[1].legend(loc='lower right')
    # axs[1].set_xlabel('Episode')
    # axs[1].set_ylabel('Reward')
    # plt.show()


    # Agent evaluation
    n_episodes_eval = 50
    episode_counter = 0

    env = TargetSteeringEnv(max_steps_per_episode=MAX_EPISODE_LENGTH)
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
