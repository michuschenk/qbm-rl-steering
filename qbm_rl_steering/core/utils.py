import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate, \
    BatchNormalization, Activation, Add, Lambda

from tensorflow.keras.initializers import glorot_normal
from tensorflow import random_normal_initializer

# from collections import deque
# import random

# BUFFER_UNBALANCE_GAP = 0.5

# class ReplayBuffer:
#     def __init__(self, buffer_size):
#         self.buffer = deque(
#             maxlen=int(buffer_size))  # with format of (s,a,r,s')
#
#         # constant sizes to use
#         # self.batch_size = batch_size
#
#         # temp variables
#         self.p_indices = [BUFFER_UNBALANCE_GAP / 2]
#
#     def append(self, state, action, r, sn, d):
#         self.buffer.append(
#             [state, action, np.expand_dims(r, -1), sn, np.expand_dims(d, -1)])
#
#     def get_batch(self, batch_size, unbalance_p=True):
#         # unbalance indices
#         p_indices = None
#         if random.random() < unbalance_p:
#             self.p_indices.extend(
#                 (np.arange(len(self.buffer) - len(self.p_indices)) + 1)
#                 * BUFFER_UNBALANCE_GAP + self.p_indices[-1])
#             p_indices = self.p_indices / np.sum(self.p_indices)
#
#         chosen_indices = np.random.choice(len(self.buffer),
#                                           size=min(batch_size,
#                                                    len(self.buffer)),
#                                           replace=False,
#                                           p=p_indices)
#
#         buffer = [self.buffer[chosen_index] for chosen_index in chosen_indices]
#
#         return buffer


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

# def generate_classical_critic(n_dims_state_space, n_dims_action_space):
#     last_init = random_normal_initializer(stddev=0.00005)
#
#     # State as input
#     state_input = tf.keras.layers.Input(shape=(n_dims_state_space),
#                                         dtype=tf.float32)
#     state_out = tf.keras.layers.Dense(60, activation=tf.nn.leaky_relu,
#         kernel_initializer=glorot_normal())(state_input)
#     state_out = tf.keras.layers.BatchNormalization()(state_out)
#     state_out = tf.keras.layers.Dense(30, activation=tf.nn.leaky_relu,
#         kernel_initializer=glorot_normal())(state_out)
#
#     # Action as input
#     action_input = tf.keras.layers.Input(shape=(n_dims_action_space),
#                                          dtype=tf.float32)
#     action_out = tf.keras.layers.Dense(30, activation=tf.nn.leaky_relu,
#         kernel_initializer=glorot_normal())(action_input)
#
#     # Both are passed through separate layer before concatenating
#     added = tf.keras.layers.Add()([state_out, action_out])
#
#     added = tf.keras.layers.BatchNormalization()(added)
#     outs = tf.keras.layers.Dense(15, activation=tf.nn.leaky_relu,
#         kernel_initializer=glorot_normal())(added)
#     outs = tf.keras.layers.BatchNormalization()(outs)
#     outputs = tf.keras.layers.Dense(1, kernel_initializer=last_init)(outs)
#
#     # Outputs single value for give state-action
#     model = tf.keras.Model([state_input, action_input], outputs)
#
#     return model


# def generate_classical_critic(n_dims_state_space, n_dims_action_space,
#                               fcl1_size=200, fcl2_size=300):
#     """
#     Builds the model,
#     non-sequential, state and action as inputs:
#     two state fully connected layers and one action fully connected layer.
#     Action introduced after the second state layer, as specified in the paper
#     """
#     state_dims = n_dims_state_space
#     action_dims = n_dims_action_space
#
#     # -- state input --
#     state_input_layer = Input(shape=state_dims)
#     # -- action input --
#     action_input_layer = Input(shape=action_dims)
#     # -- hidden fully connected layers --
#     f1 = 1. / np.sqrt(fcl1_size)
#     fcl1 = Dense(fcl1_size, kernel_initializer=RandomUniform(-f1, f1),
#                  bias_initializer=RandomUniform(-f1, f1))(state_input_layer)
#     fcl1 = BatchNormalization()(fcl1)
#     # activation applied after batchnorm
#     fcl1 = Activation("relu")(fcl1)
#     f2 = 1. / np.sqrt(fcl2_size)
#     fcl2 = Dense(fcl2_size, kernel_initializer=RandomUniform(-f2, f2),
#                  bias_initializer=RandomUniform(-f2, f2))(fcl1)
#     fcl2 = BatchNormalization()(fcl2)
#     # activation applied after batchnorm
#     # fcl2 = Activation("linear")(fcl2)
#     # Introduce action after the second layer
#     action_layer = Dense(fcl2_size,
#                          kernel_initializer=RandomUniform(-f2, f2),
#                          bias_initializer=RandomUniform(-f2, f2))(
#         action_input_layer)
#     action_layer = Activation("relu")(action_layer)
#     concat = Add()([fcl2, action_layer])
#     concat = Activation("relu")(concat)
#     # Outputs single value for give state-action
#     f3 = 0.003
#     output = Dense(1, kernel_initializer=RandomUniform(-f3, f3),
#                    bias_initializer=RandomUniform(-f3, f3),
#                    kernel_regularizer=tf.keras.regularizers.l2(0.01))(concat)
#
#     model = tf.keras.Model([state_input_layer, action_input_layer], output)
#     return model


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
        x = Dense(i, activation=tf.nn.leaky_relu,
                  kernel_initializer=glorot_normal())(x)
    x = Dense(n_dims_action_space, activation='tanh',
              kernel_initializer=random_normal_initializer(stddev=0.0005))(x)
    return tf.keras.Model(input_state, x)

# def generate_classical_actor(n_dims_state_space, n_dims_action_space,
#                              fcl1_size=200, fcl2_size=300):
#     """
#     Builds the model. Consists of two fully connected layers with batch norm.
#     """
#     state_dims = n_dims_state_space
#     action_dims = n_dims_action_space
#     # upper_bound = 1.
#
#     # -- input layer --
#     input_layer = Input(shape=state_dims)
#     # -- first fully connected layer --
#     fcl1 = Dense(fcl1_size)(input_layer)
#     fcl1 = BatchNormalization()(fcl1)
#     # activation applied after batchnorm
#     fcl1 = Activation("relu")(fcl1)
#     # -- second fully connected layer --
#     fcl2 = Dense(fcl2_size)(fcl1)
#     fcl2 = BatchNormalization()(fcl2)
#     # activation applied after batchnorm
#     fcl2 = Activation("relu")(fcl2)
#     # -- output layer --
#     f3 = 0.003
#     output_layer = Dense(action_dims, activation="tanh",
#                          kernel_initializer=RandomUniform(-f3, f3),
#                          bias_initializer=RandomUniform(-f3, f3),
#                          kernel_regularizer=tf.keras.regularizers.l2(0.01))(fcl2)
#     # scale the output
#     # output_layer = Lambda(lambda i: i * upper_bound)(output_layer)
#     # output_layer =  output_layer * self.upper_bound
#     model = tf.keras.Model(input_layer, output_layer)
#     return model
