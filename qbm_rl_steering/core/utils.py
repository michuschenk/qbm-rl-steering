import numpy as np


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


# import numpy as np
#
#
# class Memory:
#     """ A FIFO experience replay buffer.
#     """
#
#     def __init__(self, obs_dim, act_dim, size):
#         self.states = np.zeros([size, obs_dim], dtype=np.float32)
#         self.actions = np.zeros([size, act_dim], dtype=np.float32)
#         self.rewards = np.zeros(size, dtype=np.float32)
#         self.next_states = np.zeros([size, obs_dim], dtype=np.float32)
#         self.dones = np.zeros(size, dtype=np.float32)
#         self.ptr, self.size, self.max_size = 0, 0, size
#
#     def store(self, state, action, reward, next_state, done):
#         self.states[self.ptr] = state
#         self.next_states[self.ptr] = next_state
#         self.actions[self.ptr] = action
#         self.rewards[self.ptr] = reward
#         self.dones[self.ptr] = done
#         self.ptr = (self.ptr + 1) % self.max_size
#         self.size = min(self.size + 1, self.max_size)
#
#     def get_sample(self, batch_size=32):
#         # if self.size < batch_size:
#         #     idxs = np.random.randint(0, self.size, size=self.size)
#         # else:
#         idxs = np.random.randint(0, self.size, size=batch_size)
#         return (self.states[idxs], self.actions[idxs], self.rewards[idxs],
#                 self.next_states[idxs], self.dones[idxs])
