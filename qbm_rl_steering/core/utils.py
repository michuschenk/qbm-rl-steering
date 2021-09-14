import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate


class ReplayBuffer:
    """ Implements simple replay buffer for experience replay. """

    def __init__(self, size, obs_dim, act_dim):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def push(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = np.asarray([rew])
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        temp_dict = dict(s=self.obs1_buf[idxs],
                         s2=self.obs2_buf[idxs],
                         a=self.acts_buf[idxs],
                         r=self.rews_buf[idxs],
                         d=self.done_buf[idxs])
        return (temp_dict['s'], temp_dict['a'], temp_dict['r'].reshape(-1, 1),
                temp_dict['s2'], temp_dict['d'])


# Generator functions for classical actor and critic models
def generate_classical_critic(n_dims_state_space: int, n_dims_action_space: int,
                              hidden_layers: list):
    """ Initializes DDPG critic network represented by classical neural
    network.
    :param n_dims_state_space: number of dimensions of state space.
    :param n_dims_action_space: number of dimensions of action space.
    :param hidden_layers: list of number of nodes per hidden layer.
    :return: keras dense feed-forward network model. """
    input_state = Input(shape=n_dims_state_space)
    input_action = Input(shape=n_dims_action_space)
    x = input_state
    for i, j in enumerate(hidden_layers[:-1]):
        if i == 1:
            x = concatenate([x, input_action], axis=-1)
        x = Dense(j, activation='relu')(x)
    x = Dense(hidden_layers[-1])(x)

    return tf.keras.Model([input_state, input_action], x)


def generate_classical_actor(n_dims_state_space: int, n_dims_action_space: int,
                             hidden_layers: list):
    """ Initializes DDPG actor network represented by classical neural
    network.
    :param n_dims_state_space: number of dimensions of state space.
    :param n_dims_action_space: number of dimensions of action space.
    :param hidden_layers: list of number of nodes per hidden layer.
    :return: keras dense feed-forward network model. """
    input_state = Input(shape=n_dims_state_space)
    x = input_state
    for i in hidden_layers:
        x = Dense(i, activation='relu')(x)
    x = Dense(n_dims_action_space, activation='tanh')(x)
    return tf.keras.Model(input_state, x)
