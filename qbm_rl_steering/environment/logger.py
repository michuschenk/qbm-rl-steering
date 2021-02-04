import numpy as np


class Logger:
    """ Class to log events taking place in an openAI gym environment. """
    def __init__(self):
        self.log_all = []
        self.log_episode = []

        self.done_reason_map = {
            -1: '',
            0: 'Max. # steps',
            1: 'Reward thresh.',
            2: 'State OOB'}

    def episode_reset(self):
        """ Clear only episodic log. """
        self.log_episode = []

    def clear_all(self):
        """ Clear all logs. """
        self.log_episode = []
        self.log_all = []

    def extract_episodic_data(self):
        """ Function to extract episodic data into a dictionary """
        n_episodes = len(self.log_all)

        data = {}
        keys = ['episode_count', 'episode_length', 'reward_initial',
                'reward_final', 'done_reason']
        for k in keys:
            data[k] = np.zeros(n_episodes)

        for i, log_ep in enumerate(self.log_all):
            # We don't count the last entry
            data['episode_length'][i] = len(log_ep) - 1
            data['reward_initial'][i] = log_ep[0][2]
            data['reward_final'][i] = log_ep[-1][2]
            data['done_reason'][i] = log_ep[-1][5]
            data['episode_count'][i] = i
        return data

    def extract_all_data(self):
        """ Does not only extract episodic data, but all the steps, in a flat
        structure. Returns data dictionary and array of n_steps that
        describes lengths of episodes. """
        # Unpack data (convert states from binary to floats)
        data = {'state': [], 'action': [], 'reward': []}

        for log_ep in self.log_all:
            for data in log_ep:
                data['state'].append(data[0])
                data['action'].append(data[1])
                data['reward'].append(data[2])

        for k in data.keys():
            data[k] = np.array(data[k])

        # Ends of episodes
        n_steps = []
        for log_ep in self.log_all:
            n_steps.append(len(log_ep))
        n_steps = np.cumsum(np.array(n_steps))

        return data, n_steps
