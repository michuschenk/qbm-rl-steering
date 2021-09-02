import numpy as np


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
        # if self.size < batch_size:
        #     idxs = np.random.randint(0, self.size, size=self.size)
        # else:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (self.states[idxs], self.actions[idxs], self.rewards[idxs],
                self.next_states[idxs], self.dones[idxs])
