from qbm_rl_steering.environment.env_desc import TargetSteeringEnv
import qbm_rl_steering.utils.qbmq_utils as utl
import qbm_rl_steering.environment.helpers as hlp
from qbm_rl_steering.agents.mc_agent import MonteCarloAgent

import matplotlib.pyplot as plt
import numpy as np
import random
import math
import tqdm

from typing import Tuple, List

# TODO: group arguments using dicts...
# TODO: implement double Q
# TODO: implement replay buffer
# TODO: parallel agents
# TODO: implement save and load weights


class QBMQN(object):
    def __init__(self, env: TargetSteeringEnv,
                 n_graph_nodes: int,
                 n_replicas: int,
                 n_meas_for_average: int,
                 n_annealing_steps: int,
                 big_gamma: Tuple[float, float] = (20., 0.5),
                 beta: float = 2.0,
                 learning_rate: Tuple[float, float] = (1e-3, 1e-3),
                 small_gamma: float = 0.99,
                 exploration_fraction: float = 0.8,
                 exploration_epsilon: Tuple[float, float] = (1., 0.05))\
            -> None:
        """
        Implementation of the QBM-RL Q-learning agent, following the paper:
        https://arxiv.org/pdf/1706.00074.pdf
        :param env: OpenAI gym environment
        :param n_graph_nodes: number of nodes of the graph structure. E.g. for
        2 unit cells of the DWAVE-2000 chip, it's 16 nodes (8 per unit).
        :param n_replicas: number of replicas in the 3D extension of the Ising
        model, see Fig. 1 in paper. (aka Trotter slices).
        :param n_meas_for_average: number of 'independent spin configuration
        measurements' that will be used for the SQA and eventually to
        calculate the average effective Hamiltonian of the system.
        :param n_annealing_steps: number of steps that one annealing
        process should take (~annealing time).
        :param big_gamma: strength of the transverse field
        (virtual, average value), decaying from the first to the second value
        in the tuple over the course of the quantum annealing process,
        see paper for details and SQA class definition.
        :param beta: inverse temperature (note that this parameter is kept
        constant in SQA other than in SA).
        :param learning_rate: RL. parameter, learning rate for update of
        coupling weights of the Chimera graph. First and second value in
        tuple correspond to initial and final learning rate resp. (for use
        with learning rate schedule).
        :param small_gamma: RL parameter, discount factor cumulative rewards.
        :param exploration_fraction: RL param., fraction of total number of
        time steps over which epsilon-greedy parameter decays.
        :param exploration_epsilon: RL parameter, initial and final (
        at time defined by exploration_fraction) epsilon for epsilon-greedy
        decay.
        """
        self.env = env

        # RL parameters
        self.learning_rate = learning_rate
        self.small_gamma = small_gamma
        self.exploration_fraction = exploration_fraction
        self.exploration_epsilon_init = exploration_epsilon[0]
        self.exploration_epsilon_final = exploration_epsilon[1]

        # Q function approximation (RL state-action value function)
        n_bits_observation_space = env.n_bits_observation_space
        n_bits_action_space = math.ceil(math.log2(env.action_space.n))
        # This is in case where env.action_space.n == 1 (does not make much
        # sense from RL point-of-view)
        if env.action_space.n < 2:
            n_bits_action_space = 1

        self.possible_actions = [act for act in range(env.action_space.n)]
        self.q_function = utl.QFunction(
            n_bits_observation_space=n_bits_observation_space,
            n_bits_action_space=n_bits_action_space,
            small_gamma=small_gamma,
            n_graph_nodes=n_graph_nodes,
            n_replicas=n_replicas,
            big_gamma=big_gamma, beta=beta,
            n_annealing_steps=n_annealing_steps,
            n_meas_for_average=n_meas_for_average)

    def _initialise_training_episode(self) -> Tuple[int, float, np.ndarray,
                                                    np.ndarray]:
        """
        This is a method used in the training loop to reinitialise an
        episode, either at the very beginning, or whenever an episode ends.
        :return: action index, q_value, QUBO samples, state-action values of
        visible_nodes.
        """
        state = self.env.reset()
        action = self.env.action_space.sample()
        q_value, samples, visible_nodes = (
            self.q_function.calculate_q_value(state, action))
        return action, q_value, samples, visible_nodes

    def get_q_net_response(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the (trained or untrained) Q net for all the possible
        state-action vectors.
        :return: states converted to float for easier plotting, q values
        [dim. (#states, #actions)]
        """
        states_float, states_binary = self.env.get_all_states()
        n_states = len(states_binary)
        n_actions = self.env.action_space.n

        q_values = np.zeros((n_states, n_actions))
        for j, s in enumerate(states_binary):
            for a in range(n_actions):
                q, _, _ = self.q_function.calculate_q_value(s, a)
                q_values[j, a] = q

        return states_float, q_values

    def learn(self, total_timesteps: int) -> None:
        """
        Train the agent for the specified number of iterations.
        :param total_timesteps: number of training steps
        """
        # TODO: there is still a slight 'inconsistency' because the
        #  visible_nodes vector actually already contains the state and
        #  action. But they are encoded in a different form than what the
        #  environment expects, so we keep all instances of action, state,
        #  and visible_nodes up to date simultaneously.

        # TODO: I think they are not actually playing out the episodes, but are
        #  rather sweeping through the states (s1, a1) (either randomly or
        #  systematically). Implement that.

        # Epsilon decay schedule for epsilon-greedy policy
        epsilon = self._get_epsilon_schedule(total_timesteps)

        # learning_rate decay schedule
        learning_rate = self._get_learning_rate_schedule(total_timesteps)

        # This is to trigger the initialization of a new episode at the
        # beginning of the training loop
        done = True

        # TRAINING LOOP
        for it in tqdm.trange(total_timesteps):
            if done:
                # Reinitialize the episode after previous one has ended
                action, q_value, samples, visible_nodes = (
                    self._initialise_training_episode())

            # Take the step in the environment
            next_state, reward, done, _ = env.step(action=action)

            # Choose next_action following the epsilon-greedy policy
            next_action, next_q_value, next_samples, next_visible_nodes = (
                self.follow_policy(next_state, epsilon[it]))

            # Update weights
            self.q_function.update_weights(samples, visible_nodes, q_value,
                                           next_q_value, reward,
                                           learning_rate[it])

            # Note that environment is already in next_state, so this line
            # should not be necessary
            # state = next_state
            action = next_action
            q_value = next_q_value
            samples = next_samples
            visible_nodes = next_visible_nodes

    def follow_policy(self, state: np.ndarray, epsilon: float) -> \
            Tuple[int, float, np.ndarray, np.ndarray]:
        """
        Follow the epsilon-greedy policy to get the next action. With
        probability epsilon we pick a random action, and with probability
        (1 - epsilon) we pick the action greedily, i.e. such that
        a = argmax_a Q(s, a).
        :param state: state that the environment is in (binary vector, directly
        obtained from either env.reset(), or env.step()).
        :param epsilon: probability for choosing random action.
        :return: chosen action index, q_value, spin_configurations, state-action
        values of visible_nodes
        """
        if np.random.random() < epsilon:
            # Pick action randomly
            action = random.choice(self.possible_actions)
            q_value, spin_configurations, visible_nodes = (
                self.q_function.calculate_q_value(state=state, action=action))
            return action, q_value, spin_configurations, visible_nodes
        else:
            # Pick action greedily
            # Since the Chimera graph (QBM) does not offer an input ->
            # output layer structure in the classical Q-net sense, we have to
            # loop through all the actions to calculate the Q value for every
            # action to then pick the argmax_a Q
            max_dict = {'q_value': float('-inf'), 'action': None,
                        'spin_configurations': None, 'visible_nodes': None}

            for action in self.possible_actions:
                q_value, spin_configurations, visible_nodes = (
                    self.q_function.calculate_q_value(
                        state=state, action=action))

                # TODO: what needs to be done is to pick action randomly in
                #  case there are several actions with the same Q values.
                if max_dict['q_value'] < q_value:
                    max_dict['q_value'] = q_value
                    max_dict['action'] = action
                    max_dict['spin_configurations'] = spin_configurations
                    max_dict['visible_nodes'] = visible_nodes

            return (max_dict['action'], max_dict['q_value'],
                    max_dict['spin_configurations'], max_dict['visible_nodes'])

    def predict(self, state, deterministic):
        """
        Based on the given state, we pick the best action (here we always
        pick the action greedily, i.e. epsilon = 0., as we are assuming that
        the agent has been trained). This method is required to
        evaluate the trained agent.
        :param state: state encoded as binary-encoded vector as obtained from
        environment .reset() or .step()
        :param deterministic: dummy argument to satisfy stable-baselines3
        interface
        :return next action, None: need to fulfill the stable-baselines3
        interface
        """
        action, q_value, samples, visible_nodes = (
            self.follow_policy(state=state, epsilon=0.))

        return action, None

    def _get_epsilon_schedule(self, total_timesteps: int) -> np.ndarray:
        """
        Define epsilon schedule as linear decay between time step 0 and
        time step exploration_fraction * total_timesteps, starting from
        exploration_initial_eps and ending at exploration_final_eps.
        :param total_timesteps: total number of training steps
        :return epsilon array including decay.
        """
        n_steps_decay = int(self.exploration_fraction * total_timesteps)
        eps_step = (
            (self.exploration_epsilon_final - self.exploration_epsilon_init) /
            n_steps_decay)
        eps_decay = np.arange(
            self.exploration_epsilon_init, self.exploration_epsilon_final,
            eps_step)
        eps = np.ones(total_timesteps) * self.exploration_epsilon_final
        eps[:n_steps_decay] = eps_decay
        return eps

    def _get_learning_rate_schedule(self, total_timesteps: int) -> np.ndarray:
        """
        Calculates the linear decay schedule for the learning_rate
        :param total_timesteps: total number of timesteps for the RL training.
        :return: np array of learning_rate as a function of time step
        """
        learning_decay = np.linspace(
            self.learning_rate[0], self.learning_rate[1], total_timesteps)
        return learning_decay


if __name__ == "__main__":

    # Environment settings
    N_BITS_OBSERVATION_SPACE = 8
    n_actions = 2
    simple_reward = True

    # RL settings
    learning_rate = (1e-3, 1e-3)
    small_gamma = 0.8
    exploration_epsilon = (1.0, 0.04)
    exploration_fraction = 0.6

    # Graph config and quantum annealing settings
    # Commented values are what's in the paper
    n_graph_nodes = 16
    n_replicas = 25  # 25
    n_meas_for_average = 20  # 150
    n_annealing_steps = 100  # 300
    big_gamma = (20., 0.5)
    beta = 2.

    # Init. environment
    env = TargetSteeringEnv(n_bits_observation_space=N_BITS_OBSERVATION_SPACE,
                            simple_reward=simple_reward, n_actions=n_actions)

    # Initialize agent
    agent = QBMQN(env=env, n_graph_nodes=n_graph_nodes, n_replicas=n_replicas,
                  n_meas_for_average=n_meas_for_average,
                  n_annealing_steps=n_annealing_steps, big_gamma=big_gamma,
                  beta=beta, learning_rate=learning_rate,
                  small_gamma=small_gamma,
                  exploration_fraction=exploration_fraction,
                  exploration_epsilon=exploration_epsilon)

    # Train agent
    total_timesteps = 100  # 500
    agent.learn(total_timesteps=total_timesteps)

    # Plot learning evolution
    hlp.plot_log(env, fig_title='Agent training')

    # Evaluate agent on random initial states
    # env = TargetSteeringEnv(n_bits_observation_space=N_BITS_OBSERVATION_SPACE,
    #                         simple_reward=simple_reward, n_actions=n_actions)
    # hlp.evaluate_agent(env, agent, n_episodes=10,
    #                    make_plot=True, fig_title='Agent evaluation')

    # Get response of the trained "Q-net" for all possible (state, action) pairs
    states, q_values = agent.get_q_net_response()

    fig = plt.figure(1, figsize=(7, 5))
    ax = plt.gca()
    cols = ['tab:red', 'tab:blue']
    for i in range(q_values.shape[1]):
        ax.plot(1e3*states, q_values[:, i], c=cols[i], label=f'Action {i}')

    # MC agent
    mc_agent = MonteCarloAgent(env, gamma=small_gamma)
    states_v, v_star = mc_agent.run_mc(200)
    ax.plot(1e3*states_v, v_star, c='k', label='V* (MC)')

    ax.set_xlabel('State, BPM pos. (mm)')
    ax.set_ylabel('Q value')
    ax.legend(loc='upper right')
    plt.show()
