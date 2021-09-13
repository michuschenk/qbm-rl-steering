import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate


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
