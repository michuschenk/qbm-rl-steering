import numpy as np
import matplotlib.pyplot as plt
import sys

import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.optimizers as KO
import tensorflow.keras as K
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

import matplotlib
matplotlib.use('qt5agg')

class Memory:
    """A FIFO experiene replay buffer.
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
        idxs = np.random.randint(0, self.size, size=batch_size)
        return self.states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs], self.dones[idxs]

class ClassicACAgent(object):
    def __init__(self, state_dim, action_n, act_limit):
        # env intel
        self.action_n = action_n
        self.state_dim = state_dim
        self.state_n = state_dim[0]
        # constants
        self.ACT_LIMIT = act_limit  # requiered for clipping prediciton aciton
        self.GAMMA = 0.99  # discounted reward factor
        self.TAU = 1  # soft update factor
        self.BUFFER_SIZE = int(1e6)
        self.BATCH_SIZE = 50  # training batch size.
        self.ACT_NOISE_SCALE = 0.2
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
        model.compile(optimizer=K.optimizers.Adam(learning_rate=0.001), loss=self._ddpg_actor_loss)
        model.summary()
        return model

    def _ddpg_actor_loss(self, y_true, y_pred):
        # y_pred is the action from the actor net. y_true is the state, we maximise the q
        q =self.critic([y_true,y_pred, self.dummy_Q_target_prediction_input, self.dummy_dones_input])
        return -K.backend.mean(q)

    def get_action(self, states, noise=None,episode = 1):

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

        self.actor.train_on_batch(states, states)#Q_predictions)



    def _gen_critic_network(self):
        # Inputs to network. Most of them are for the loss function, not for the feed forward
        state_input = KL.Input(shape=self.state_dim, name='state_input')
        action_input = KL.Input(shape=(self.action_n,), name='action_input')
        Q_target_prediction_input = KL.Input(shape=(1,), name='Q_target_prediction_input')
        dones_input = KL.Input(shape=(1,), name='dones_input')
        # define network structure
        concat_state_action = KL.concatenate([state_input, action_input])
        dense = KL.Dense(128, activation='relu')(concat_state_action)
        dense = KL.Dense(128, activation='relu')(dense)
        out = KL.Dense(1, activation='linear')(dense)
        model = K.Model(inputs=[state_input, action_input, Q_target_prediction_input, dones_input], outputs=out)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.001,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True)
        model.compile(optimizer=K.optimizers.Adam(learning_rate=0.001), loss=self._ddpg_critic_loss(Q_target_prediction_input, dones_input))
        model.summary()
        return model

    def _ddpg_critic_loss(self, Q_target_prediction_input, dones_input):
        def loss(y_true, y_pred):
            #  y_true = rewards ; y_pred = Q
            target_Q = y_true + (self.GAMMA * Q_target_prediction_input * (1 - dones_input))
            mse = K.losses.mse(target_Q, y_pred)
            return mse

        return loss

    def train_critic(self, states, next_states, actions, rewards, dones):

        next_actions = self.get_target_action(next_states)
        Q_target_prediction = self.get_target_Q(next_states, next_actions)
        self.critic.train_on_batch([states, actions, Q_target_prediction, dones], rewards)

    def get_Q(self, states, actions):
        return self.critic.predict([states, actions, self.dummy_Q_target_prediction_input, self.dummy_dones_input])

    def get_target_Q(self, states, actions):
        return self.critic_target.predict_on_batch(
            [states, actions, self.dummy_Q_target_prediction_input, self.dummy_dones_input])

    def _soft_update_actor_and_critic(self):

        # Critic soft update:
        weights_critic_local = np.array(self.critic.get_weights())
        weights_critic_target = np.array(self.critic_target.get_weights())
        self.critic.set_weights(self.TAU * weights_critic_local + (1.0 - self.TAU) * weights_critic_target)
        # Actor soft update
        weights_actor_local = np.array(self.actor.get_weights())
        weights_actor_target = np.array(self.actor_target.get_weights())
        self.actor_target.set_weights(self.TAU * weights_actor_local + (1.0 - self.TAU) * weights_actor_target)

    def store(self, state, action, reward, next_state, done):

        self.memory.store(state, action, reward, next_state, done)

    def train(self):
        """Trains the networks of the agent (local actor and critic) and soft-updates  target.
        """
        states, actions, rewards, next_states, dones = self.memory.get_sample(batch_size=self.BATCH_SIZE)
        self.train_critic(states, next_states, actions, rewards, dones)
        self.train_actor(states, actions)
        self._soft_update_actor_and_critic()


import gym
#from cern_awake_env.simulation import SimulationEnv
from envs import env_awake_steering_simulated as awake_sim
if __name__ == "__main__":

    GAMMA = 0.99
    EPOCHS = 10
    MAX_EPISODE_LENGTH = 50
    START_STEPS = 10
    INITIAL_REW = 0

    env =  awake_sim.e_trajectory_simENV()
    env.action_scale = 3e-4
    env.threshold = -0.16


    agent = ClassicACAgent(env.observation_space.shape,env.action_space.shape[0],max(env.action_space.high))

    state, reward, done, ep_rew, ep_len, ep_cnt = env.reset(),INITIAL_REW, False, [[]], 0, 0
    ep_rew[-1].append((-1.)*np.sqrt(np.mean(np.square(state))))
    total_steps = MAX_EPISODE_LENGTH * EPOCHS

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        if t > START_STEPS:
            action = agent.get_action(state,episode=1)#len(ep_rew))
            action = np.squeeze(action)
        else:
            action = env.action_space.sample()

        # Step the env
        next_state, reward, done, _ = env.step(action)
        #print("reward ",reward,done)
        ep_rew[-1].append(reward) #keep adding to the last element till done
        ep_len += 1


        #print(done)
        done = False if ep_len==MAX_EPISODE_LENGTH else done

        # Store experience to replay buffer
        agent.store(state,action,reward,next_state,done)


        state = next_state

        if done or (ep_len == MAX_EPISODE_LENGTH):
            ep_cnt += 1
            if True: #ep_cnt % RENDER_EVERY == 0:
                print(f"Episode: {len(ep_rew)-1}, Reward: {ep_rew[-1][-1]}, Length: {len(ep_rew[-1])}")
            ep_rew.append([])

            for _ in range(ep_len):
                agent.train()

            state, reward, done, ep_ret, ep_len = env.reset(), INITIAL_REW, False, 0, 0
            ep_rew[-1].append((-1.)*np.sqrt(np.mean(np.square(state))))

    init_rewards = []
    rewards = []
    reward_lengths = []
    for episode in ep_rew:
        if(len(episode)>0):
            rewards.append(episode[-1])
            init_rewards.append(episode[0])
            reward_lengths.append(len(episode)-1)


    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    ax = axs[0]
    ax.plot(reward_lengths)

    ax = axs[1]
    ax.plot(init_rewards)
    ax.plot(rewards)
    plt.show()
