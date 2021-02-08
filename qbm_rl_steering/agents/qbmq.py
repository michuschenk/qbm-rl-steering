import gym
import random
from qbm_rl_steering.environment.env_desc import TargetSteeringEnv
from qbm_rl_steering.agents.qbmq_utils import *

class Q_Function(object):
    def __init__(self,n_observations,n_actions):
        self.Qhh = dict()
        self.Qvh = dict()
        self.n_observations = n_observations
        self.n_actions = self.n_actions
        self._initalise_weights(self.n_observations,self.n_actions)


    #TODO: define epsilon, that can decay over time. predict returns random or Q value

    def _initalise_weights(self,n_observations,n_actions):
        for i, ii in zip(tuple(range(4)), tuple(range(8, 12))):
            for j, jj in zip(tuple(range(4, 8)), tuple(range(12, 16))):
                self.Q_hh[(i, j)] = 2 * random.random() - 1
                self.Q_hh[(ii, jj)] = 2 * random.random() - 1
        for i, j in zip(tuple(range(4, 8)), tuple(range(12, 16))):
            self.Q_hh[(i, j)] = 2 * random.random() - 1

        # Fully connection between state and blue nodes
        for j in (tuple(range(4)) + tuple(range(12, 16))):
            for i in range(n_observations):
                self.Q_vh[(i, j,)] = 2 * random.random() - 1
            # Fully connection between action and red nodes
        for j in (tuple(range(4, 8)) + tuple(range(8, 12))):
            for i in range(n_observations, n_observations+n_actions):
                self.Q_vh[(i, j,)] = 2 * random.random() - 1


    #to be implemented:
    def predict(self,state,all_possible_actions):
        #returns action with max Q

        #requires state and actions as tuple with -1,1
        # vis_iterable = current_state[1] + available_actions_list[action_index]
        #
        # general_Q = create_general_Q_from(
        #     self.Q_hh,
        #     self.Q_vh,
        #     vis_iterable
        # )
        #
        # samples = list(SimulatedAnnealingSampler().sample_qubo(
        #     general_Q,
        #     num_reads=sample_count
        # ).samples())
        #
        # random.shuffle(samples)
        #
        # current_F = get_free_energy(
        #     get_3d_hamiltonian_average_value(
        #         samples,
        #         general_Q,
        #         replica_count,
        #         average_size,
        #         0.5,
        #         2
        #     ),
        #     samples,
        #     replica_count,
        #     2,
        # )

        # if max_tuple is None or max_tuple[0] < current_F:
        #     max_tuple = (current_F, action_index, samples, vis_iterable)


        return action

    def update_weights(self):
        pass





class QBMQN(object):
    def __init__(self, env: gym.Env,equivalent_n_actions:int):
        self.env = env
        n_observations = env.observation_space.shape
        self.n_actions = equivalent_n_actions
        self.q_function = Q_Function(n_observations[0],self.n_actions)


    def learn(self,n_iterations):
        state_0 = self.env.reset()
        all_possible_actions = env.something
        for i in range(n_iterations):
            action = self.q_function.predict(state_0,all_possible_actions)
            state_1,reward, done,_ = env.step(action)
            #check what is required for weight update
            #check how to get handle on current and future F.










if __name__ == "__main__":
    env = TargetSteeringEnv()
    equivalent_n_actions = 2
    qbmqn = QBMQN(env,equivalent_n_actions)






