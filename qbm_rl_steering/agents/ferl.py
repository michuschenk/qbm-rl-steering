import gym

from qbm_rl_steering.environment.env_1D_continuous import TargetSteeringEnv
import qbm_rl_steering.core.qbm as qbmc

import numpy as np
import random
import tqdm

from typing import Tuple, List, Dict
from collections import namedtuple


class QBMQ(object):
    def __init__(self, env: TargetSteeringEnv,
                 sampler_type: str,
                 n_replicas: int,
                 n_meas_for_average: int,
                 n_annealing_steps: int,
                 big_gamma: Tuple[float, float] = (20., 0.5),
                 beta: float = 2.0,
                 learning_rate: Tuple[float, float] = (1e-3, 1e-3),
                 small_gamma: float = 0.99,
                 exploration_fraction: float = 0.8,
                 exploration_epsilon: Tuple[float, float] = (1., 0.05),
                 replay_batch_size: int = 1,
                 target_update_frequency: int = 100,
                 soft_update_factor: float = 1.,
                 n_rows_qbm: int = 1,
                 n_columns_qbm: int = 2,
                 kwargs_qpu: Dict = {}) -> None:
        """Implements the QBM-RL Q-learning agent, following paper:
        https://arxiv.org/pdf/1706.00074.pdf
        :param env: OpenAI gym compatible environment
        :param sampler_type: Choose your sampler, either 'SQA', 'QPU', or 'QAOA'.
        :param n_replicas: number of replicas in the 3D extension of the Ising
        spin model, see Fig. 1 in paper. (aka Trotter slices).
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
        :param learning_rate: RL parameter, learning rate for update of
        coupling weights of the Chimera graph. First and second value in
        tuple correspond to initial and final learning rate resp. (for use
        with learning rate schedule).
        :param small_gamma: RL parameter, discount factor cumulative rewards.
        :param exploration_fraction: RL parameter, fraction of total number
        of time steps over which epsilon-greedy parameter decays.
        :param exploration_epsilon: RL parameter, initial and final (at time
        defined by exploration_fraction) epsilon for epsilon-greedy decay.
        :param replay_batch_size: number of experiences that we base our
        training on in every training iteration
        :param target_update_frequency: number of iterations after which we
        update the target Q-function following soft-update rule
        :param soft_update_factor: factor to consider when updating the
        target Q-function (Polyak update).
        :param n_rows_qbm: number of unit cells along x-direction (Chimera
        topology).
        :param n_columns_qbm: number of unit cells along y-direction
        (Chimera topology).
        :param kwargs_qpu: additional keyword arguments required for the
        initialization of the D-Wave QPU on Amazon Braket."""
        self.env = env

        # RL parameters
        self.learning_rate = learning_rate
        self.small_gamma = small_gamma
        self.exploration_fraction = exploration_fraction
        self.exploration_epsilon_init = exploration_epsilon[0]
        self.exploration_epsilon_final = exploration_epsilon[1]

        # Q function approximation (RL state-action value function)
        self.possible_actions = [
            act for act in range(self.env.action_space.n)]

        # Define Q functions and their updates
        kwargs_q_func = dict(
            sampler_type=sampler_type,
            state_space=env.observation_space,
            action_space=env.action_space,
            small_gamma=small_gamma,
            n_replicas=n_replicas,
            big_gamma=big_gamma, beta=beta,
            n_annealing_steps=n_annealing_steps,
            n_meas_for_average=n_meas_for_average,
            n_rows_qbm=n_rows_qbm,
            n_columns_qbm=n_columns_qbm,
            kwargs_qpu=kwargs_qpu)

        self.q_function = qbmc.QFunction(**kwargs_q_func)
        self.q_function_target = qbmc.QFunction(**kwargs_q_func)

        self.soft_update_factor = soft_update_factor
        self.target_update_frequency = target_update_frequency

        # Experience replay
        self.ReplayMemory = namedtuple(
            'ReplayMemory', ['state', 'action', 'reward', 'next_state', 'done'])
        self.replay_batch_size = replay_batch_size
        self.replay_buffer = []

    def _initialise_training_episode(self, epsilon: float) \
            -> Tuple[int, float, np.ndarray, np.ndarray]:
        """Reinitialises an episode, either at the very beginning, or
        whenever an episode ends. This is following the epsilon-greedy
        policy.
        :param epsilon: epsilon-greedy parameter
        :returns action, q_value, QUBO samples, state-action values of
        visible nodes."""
        state = self.env.reset(init_outside_threshold=False)
        action, q_value, samples, visible_nodes = self.follow_policy(
            state, epsilon)
        return action, q_value, samples, visible_nodes

    def _update_target_q_function(self) -> None:
        """Updates the target Q-function with the weights from the training
        Q-function. Allows for Polyak update, i.e. soft update, controlled by
        parameter self.soft_update_factor.
        :returns None."""
        for k in self.q_function_target.w_hh.keys():
            self.q_function_target.w_hh[k] *= (1. - self.soft_update_factor)
            self.q_function_target.w_hh[k] += (
                    self.soft_update_factor * self.q_function.w_hh[k])

        for k in self.q_function_target.w_vh.keys():
            self.q_function_target.w_vh[k] *= (1. - self.soft_update_factor)
            self.q_function_target.w_vh[k] += (
                    self.soft_update_factor * self.q_function.w_vh[k])

    # TODO: move to helpers ...
    def get_q_net_response(self, n_samples_states: int) ->\
            Tuple[np.ndarray, np.ndarray]:
        """Evaluates Q-net for a number of states sampled from the
        env.observation_space.
        :param n_samples_states: number of samples taken from the
        env.observation_space. In case of discrete, binary state
        space, all states can be evaluated and n_samples_states
        is overwrruled by that number.
        :returns sampled states and corresponding q values."""
        obs_space = self.env.observation_space
        states = []
        if isinstance(obs_space, gym.spaces.MultiBinary):
            states_float, states = self.env.get_all_states()
            n_samples_states = len(states)
        elif isinstance(obs_space, gym.spaces.Box):
            for i in range(n_samples_states):
                states.append(self.env.observation_space.sample())
            states = np.sort(np.array(states))
        else:
            raise TypeError("State space must be of type Box or MultiBinary.")
        n_actions = self.env.action_space.n

        q_values = np.zeros((n_samples_states, n_actions))
        for j, s in enumerate(states):
            for a in range(n_actions):
                q, _, _ = self.q_function_target.calculate_q_value(s, a)
                q_values[j, a] = q

        if isinstance(obs_space, gym.spaces.MultiBinary):
            return states_float, q_values
        else:
            return states, q_values

    def learn(self, total_timesteps: int) -> List:
        """Trains the agent for the specified number of iterations.
        :param total_timesteps: number of training steps.
        :returns list of visited states during the training."""
        # This is just for debugging: keep track which of the states have been
        # visited
        visited_states = []

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
                # Random state, epsilon-greedy action
                action, q_value, spin_configs, visible_nodes = (
                    self._initialise_training_episode(epsilon=epsilon[it]))

            state = self.env.state.copy()
            if isinstance(self.env.observation_space, gym.spaces.MultiBinary):
                state = self.env.make_binary_state_float(state)
            visited_states.append(state)

            # Take the step in the environment
            next_state, reward, done, _ = self.env.step(action=action)

            # Choose next_action following the epsilon-greedy policy
            next_action, next_q_value, next_samples, next_visible_nodes = (
                self.follow_policy(next_state, epsilon[it]))

            # Add memory to replay buffer
            self.replay_buffer.append(
                self.ReplayMemory(state, action, reward, next_state, done))

            # For next iteration on 'exploration policy'
            action = next_action

            # Use experiences in replay_buffer to update weights
            n_replay_batch = self.replay_batch_size
            if len(self.replay_buffer) < self.replay_batch_size:
                n_replay_batch = len(self.replay_buffer)
            replay_samples = random.sample(self.replay_buffer, n_replay_batch)

            for sample in replay_samples:
                # Act only greedily here: should be OK to do that always
                # because we collect our experiences according to an
                # epsilon-greedy policy

                # Recalculate the q_value of the (sample.state, sample.action)
                # pair
                q_value, spin_configs, visible_nodes = (
                    self.q_function_target.calculate_q_value(
                        sample.state, sample.action))

                # Now calculate the next_q_value of the greedy action, without
                # actually taking the action (to take actions in env.,
                # we don't follow purely greedy action).
                _, next_q_value, _, _ = (self.follow_policy(
                    sample.next_state, epsilon=0.))

                # Update weights and update target Q-function if needed
                self.q_function.update_weights(
                    spin_configs, visible_nodes, q_value, next_q_value,
                    sample.reward, learning_rate[it])

            if it % self.target_update_frequency == 0:
                self._update_target_q_function()

        return visited_states

    def follow_policy(self, state: np.ndarray, epsilon: float) -> \
            Tuple[int, float, np.ndarray, np.ndarray]:
        """Follows epsilon-greedy policy to get the next action. With
        probability epsilon we pick a random action, and with probability
        (1 - epsilon) we pick the action greedily, i.e. such that
        a = argmax_a Q(s, a).
        :param state: state that the environment is in directly obtained
        from either env.reset(), or env.step()).
        :param epsilon: probability for choosing random action.
        :returns action, q_value, spin_configurations, state-action values
        of visible_nodes."""
        if np.random.random() < epsilon:
            # Pick action randomly
            action = random.choice(self.possible_actions)
            q_value, spin_configurations, visible_nodes = (
                self.q_function_target.calculate_q_value(
                    state=state, action=action))
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
                    self.q_function_target.calculate_q_value(
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
        """Based on given state pick best action (here always pick action
        greedily, i.e. epsilon = 0., as we are assuming that agent has been
        trained). This method is required to evaluate the trained agent.
        :param state: state of environment as obtained with env.reset() or
        env.step()
        :param deterministic: dummy argument to satisfy stable-baselines3
        interface
        :returns next action, None: need to fulfill the stable-baselines3
        interface."""
        action, q_value, samples, visible_nodes = (
            self.follow_policy(state=state, epsilon=0.))

        return action, None

    def _get_epsilon_schedule(self, total_timesteps: int) -> np.ndarray:
        """Defines epsilon schedule as linear decay between time step 0 and
        time step exploration_fraction * total_timesteps, starting from
        exploration_initial_eps and ending at exploration_final_eps.
        :param total_timesteps: total number of training steps
        :returns epsilon array including decay."""
        n_steps_decay = int(self.exploration_fraction * total_timesteps)
        eps_decay = np.linspace(
            self.exploration_epsilon_init, self.exploration_epsilon_final,
            n_steps_decay)
        eps = np.ones(total_timesteps) * self.exploration_epsilon_final
        eps[:n_steps_decay] = eps_decay
        return eps

    def _get_learning_rate_schedule(self, total_timesteps: int) -> np.ndarray:
        """Calculates the linear decay schedule for the learning rate.
        :param total_timesteps: total number of timesteps for the RL training.
        :returns array of learning rates as a function of time step."""
        return np.linspace(
            self.learning_rate[0], self.learning_rate[1], total_timesteps)
