from qbm_rl_steering.environment.env_desc import TargetSteeringEnv
import qbm_rl_steering.agents.qbmq_utils as utl
import qbm_rl_steering.environment.helpers as hlp

import numpy as np
import math
import tqdm

from typing import Tuple


class QBMQN(object):
    def __init__(self, env: TargetSteeringEnv, n_replicas: int,
                 n_meas_for_average: int, big_gamma: float = 0.5,
                 beta: float = 2.0, learning_rate: float = 5e-4,
                 exploration_fraction: float = 0.8,
                 exploration_initial_eps: float = 1.0,
                 exploration_final_eps: float = 0.05,
                 small_gamma: float = 0.99) -> None:
        """
        Implementation of the QBM Q learning agent, following the paper:
        https://arxiv.org/pdf/1706.00074.pdf
        :param env: OpenAI gym environment
        :param n_replicas: number of replicas in the 3D extension of the Ising
        model, see Fig. 1 in paper.
        :param n_meas_for_average: number of 'independent measurements' that
        will be used to QUBO sample and eventually to calculate the average
        effective Hamiltonian of the system.
        :param big_gamma: hyperparameter; strength of the transverse field
        (virtual, average value), see paper for details.
        :param beta: hyperparameter; inverse temperature used for simulated
        annealing, see paper for details.
        :param learning_rate: RL. parameter, learning rate for update of
        coupling weights of the Chimera graph.
        :param exploration_fraction: RL param., fraction of total number of
        time steps over which epsilon-greedy parameter decays.
        :param exploration_initial_eps: RL parameter, initial epsilon for
        epsilon-greedy decay.
        :param exploration_final_eps: RL parameter, final epsilon for
        epsilon-greedy decay.
        :param small_gamma: RL parameter, discount factor cumulative rewards.
        """
        self.env = env

        # RL parameters
        self.learning_rate = learning_rate
        self.small_gamma = small_gamma
        self.exploration_fraction = exploration_fraction
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps

        # Q function approximation (RL state-action value function)
        n_bits_observation_space = env.n_bits_observation_space
        n_bits_action_space = math.ceil(math.log2(env.action_space.n))
        # This is in case where env.action_space.n == 1 (does not make much
        # sense from RL point-of-view)
        if env.action_space.n < 2:
            n_bits_action_space = 1

        possible_actions = [i for i in range(env.action_space.n)]
        self.q_function = utl.QFunction(
            n_bits_observation_space, n_bits_action_space, possible_actions,
            learning_rate, small_gamma, n_replicas, n_meas_for_average,
            big_gamma, beta)

    def _initialise_training_episode(self) -> \
            Tuple[int, float, np.ndarray, np.ndarray]:
        """
        This is a method used in the training loop to reinitialise an
        episode, either at the very beginning, or whenever an episode ends.
        :return: action index, q_value, QUBO samples, state-action values of
        visible_nodes.
        """
        state = self.env.reset()
        action = self.env.action_space.sample()
        q_value, samples, visible_nodes = self.q_function.calculate_q_value(
            state, action)
        return action, q_value, samples, visible_nodes

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

        # Epsilon decay schedule for epsilon-greedy policy
        epsilon = self._get_epsilon_schedule(total_timesteps)

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
                self.q_function.follow_policy(next_state, epsilon[it]))

            # Update weights
            self.q_function.update_weights(samples, visible_nodes, q_value,
                                           next_q_value, reward)

            # Note that environment is already in next_state, so this line
            # should not be necessary
            # state = next_state
            action = next_action
            q_value = next_q_value
            samples = next_samples
            visible_nodes = next_visible_nodes

    def predict(self, state, deterministic):
        """
        Based on the given state, we pick the best action (here we always
        pick the action greedily, i.e. epsilon = 0., as we are assuming that
        the agent has been trained). This method is required to evaluate the
        trained agent.
        :param state: state encoded as binary-encoded vector as obtained from
        environment .reset() or .step()
        :param deterministic: dummy argument to satisfy stable-baselines3
        interface
        :return next action, None: need to fulfill the stable-baselines3
        interface
        """
        action, q_value, samples, visible_nodes = (
            self.q_function.follow_policy(state=state, epsilon=0.))

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
            (self.exploration_final_eps - self.exploration_initial_eps) /
            n_steps_decay)
        eps_decay = np.arange(
            self.exploration_initial_eps, self.exploration_final_eps, eps_step)
        eps = np.ones(total_timesteps) * self.exploration_final_eps
        eps[:n_steps_decay] = eps_decay
        return eps


if __name__ == "__main__":
    N_BITS_OBSERVATION_SPACE = 8
    simple_reward = True
    n_actions = 2

    env = TargetSteeringEnv(n_bits_observation_space=N_BITS_OBSERVATION_SPACE,
                            simple_reward=simple_reward, n_actions=n_actions)
    agent = QBMQN(env, n_replicas=10, n_meas_for_average=1000,
                  big_gamma=0.5, beta=2., exploration_fraction=0.8,
                  exploration_initial_eps=1.0, exploration_final_eps=0.04,
                  small_gamma=0.99, learning_rate=5e-4)

    total_timesteps = 500
    agent.learn(total_timesteps=total_timesteps)
    hlp.plot_log(env, fig_title='Agent training')

    # Agent evaluation
    env = TargetSteeringEnv(n_bits_observation_space=N_BITS_OBSERVATION_SPACE,
                            simple_reward=simple_reward, n_actions=n_actions)
    hlp.evaluate_agent(env, agent, n_episodes=10,
                       make_plot=True, fig_title='Agent evaluation')
