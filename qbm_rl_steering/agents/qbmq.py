from qbm_rl_steering.environment.env_desc import TargetSteeringEnv
import qbm_rl_steering.utils.qbmq_utils as utl
import qbm_rl_steering.environment.helpers as hlp
from qbm_rl_steering.agents.mc_agent import MonteCarloAgent

import matplotlib.pyplot as plt
import numpy as np
import dill
import random
import math
import tqdm

from typing import Tuple, List, Dict

# TODO: group arguments using dicts...
# TODO: implement double Q
# TODO: implement replay buffer
# TODO: parallel agents


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
        n_bits_observation_space = self.env.n_bits_observation_space
        n_bits_action_space = math.ceil(math.log2(self.env.action_space.n))
        # This is in case where env.action_space.n == 1 (does not make much
        # sense from RL point-of-view)
        if self.env.action_space.n < 2:
            n_bits_action_space = 1

        self.possible_actions = [
            act for act in range(self.env.action_space.n)]
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

    def learn_systematic(self, total_timesteps: int, mode: str = 'sweep') \
            -> List:
        """
        Follow more what they do in the paper (at least according to my
        understanding). Pick one state and action pair (s1, a1) and just
        do one step from there, i.e. don't play out the episodes as we
        usually do in episodic RL. We will have the possibility to pick
        the pair (s1, a1) randomly (mode = 'random'), or go through all
        the state-action pairs systematically (mode = 'sweep').
        :param total_timesteps: number of training steps
        :param mode: decides how we pick the first state-action pair
        (s1, a1)
        return: list of visited states during the training
        """
        # This is just for debugging: keep track which of the states have been
        # visited
        visited_states = []

        # learning_rate decay schedule
        learning_rate = self._get_learning_rate_schedule(total_timesteps)

        all_states_float, all_states_binary = self.env.get_all_states()
        if total_timesteps < len(all_states_binary):
            print('****** WARNING: total_timesteps < number of states')

        if mode == 'sweep':

            pbar = tqdm.tqdm(total=total_timesteps, position=0, leave=True)
            it = 0
            while it < total_timesteps:
                for j, s1 in enumerate(all_states_binary):
                    a1 = np.random.choice(self.env.action_space.n)

                    # Put environment in that state (need to use float)
                    _ = self.env.reset(init_state=all_states_float[j])
                    visited_states.append(
                        self.env.make_binary_state_float(self.env.state))

                    # Calc. q value of the (s1, a1) pairß
                    q_s1_a1, spin_configs, vis_nodes = (
                        self.q_function.calculate_q_value(
                            state=s1, action=a1))

                    # Take the step in the environment
                    s2, reward, done, _ = self.env.step(action=a1)

                    # Choose next_action following greedy policy (?)
                    # It looks like it's always argmax Q in the paper (alg. 3)
                    a2, q_s2_a2, next_spin_configs, next_vis_nodes = (
                        self.follow_policy(s2, epsilon=0.))

                    # Update weights
                    self.q_function.update_weights(
                        spin_configs, vis_nodes, q_s1_a1,
                        q_s2_a2, reward, learning_rate[it])

                    it += 1
                    pbar.update(1)
                    if it >= total_timesteps:
                        break
            pbar.close()
        else:
            raise NotImplementedError("Only mode 'sweep' is allowed " +
                                      "at the moment")
        return visited_states

    def learn(self, total_timesteps: int,
              play_out_episode: bool = True) -> List:
        """
        Train the agent for the specified number of iterations.
        :param total_timesteps: number of training steps
        :param play_out_episode: if True the episodes will be played until
        the end, otherwise only 1 step is performed in every iteration.
        :return list of visited states during the training
        """
        # TODO: there is still a slight 'inconsistency' because the
        #  visible_nodes vector actually already contains the state and
        #  action. But they are encoded in a different form than what the
        #  environment expects, so we keep all instances of action, state,
        #  and visible_nodes up to date simultaneously.

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
                # Random initialization
                action, q_value, spin_configs, visible_nodes = (
                    self._initialise_training_episode())

            visited_states.append(
                self.env.make_binary_state_float(self.env.state))

            # Take the step in the environment
            next_state, reward, done, _ = self.env.step(action=action)

            # Choose next_action following the epsilon-greedy policy
            next_action, next_q_value, next_samples, next_visible_nodes = (
                self.follow_policy(next_state, epsilon[it]))

            # Update weights
            self.q_function.update_weights(
                spin_configs, visible_nodes, q_value,
                next_q_value, reward, learning_rate[it])

            # Note that environment is already in next_state, so this line
            # should not be necessary
            # state = next_state
            action = next_action
            q_value = next_q_value
            spin_configs = next_samples
            visible_nodes = next_visible_nodes

            if not play_out_episode:
                done = True

        return visited_states

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


def find_policy_from_q(agent: QBMQN) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
    """
    Get response of the trained "Q-net" for all possible (state, action) and
    then calculate optimal policy according to learned Q values.
    """
    states, q_values = agent.get_q_net_response()

    best_action = np.ones(len(states), dtype=int) * -1
    for i in range(len(states)):
        best_action[i] = np.argmax(q_values[i, :])
    return states, q_values, best_action


def plot_agent_evaluation(
        states_q: np.ndarray, q_values: np.ndarray, best_action: np.ndarray,
        states_v: np.ndarray, v_star_values: np.ndarray,
        visited_states: np.ndarray) -> None:
    """
    Plot the evaluation of the agent after training.
    """
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 7))
    fig.suptitle('QBM agent evaluation')

    # Value functions
    cols = ['tab:red', 'tab:blue']
    for i in range(q_values.shape[1]):
        axs[0].plot(1e3 * states_q, q_values[:, i], c=cols[i],
                    label=f'Action {i}')

    # MC agent results
    axs[0].plot(1e3 * states_v, v_star_values, c='k', label='V* (MC)')
    axs[0].set_ylabel('Q value')
    axs[0].legend(loc='upper right')

    # Plot policy
    for a in list(set(best_action)):
        msk_a = best_action == a
        axs[1].plot(1e3 * states_q[msk_a], best_action[msk_a],
                    marker='o', ms=3, ls='None', c=cols[a])
    axs[1].set_ylabel('Best action')

    # What states have been visited and how often?
    axs[2].hist(1e3 * np.array(visited_states), bins=100)
    axs[2].set_xlabel('State, BPM pos. (mm)')
    axs[2].set_ylabel('# visits')

    plt.show()


def calculate_policy_optimality(env: TargetSteeringEnv, states: np.ndarray,
                                best_action: np.ndarray) \
        -> Tuple[QBMQN, np.ndarray]:
    """
    Metric for optimality of policy: we can do this because we know the
    optimal policy already. Measure how many of the actions are correct
    according to the Q-functions that we learned. We only judge actions
    for states outside of reward threshold (inside episode is anyway over
    after 1 step and agent has no way to learn what's best there.
    :return the agent object and the performance metric
    """
    _, x, r = env.get_response()
    idx = np.where(r > env.reward_threshold)[0][-1]
    x_reward_thresh = x[idx]
    n_states_total = np.sum(
        (states < -x_reward_thresh) | (states > x_reward_thresh))

    # How many of the actions that the agent would take are actually
    # according to optimal policy? (this is environment dependent and
    # something we can do because we know the optimal policy).
    n_correct_actions = np.sum(
        (best_action == 0) & (states < -x_reward_thresh))
    n_correct_actions += np.sum(
        (best_action == 1) & (states > x_reward_thresh))

    policy_eval = 100 * n_correct_actions / float(n_states_total)
    print(f'Optimality of policy: {policy_eval:.1f}%')
    return policy_eval


def train_and_evaluate_agent(
        kwargs_env: Dict, kwargs_rl: Dict, kwargs_anneal: Dict,
        total_timesteps: int, make_plots: bool = True) -> float:
    """

    """
    # Initialize environment
    env = TargetSteeringEnv(**kwargs_env)

    # Initialize agent and train
    agent = QBMQN(env=env, **kwargs_anneal, **kwargs_rl)
    visited_states = agent.learn(total_timesteps=total_timesteps,
                                 play_out_episode=True)
    # When using learn_systematic it's best to make sure you sweep through
    # all states at least once.
    # visited_states = agent.learn_systematic(total_timesteps=total_timesteps)

    # Run Monte Carlo agent
    mc_agent = MonteCarloAgent(env, gamma=kwargs_rl['small_gamma'])
    states_v, v_star_values = mc_agent.run_mc(200)

    # Plot learning evolution (note that this does not work when we either
    # set play_out_episode to True or when using learn_systematic.
    # hlp.plot_log(env, fig_title='Agent training')

    # Evaluate the agent
    # Evaluate agent on random initial states
    # env = TargetSteeringEnv(**kwargs_env)
    # hlp.evaluate_agent(env, agent, n_episodes=10, make_plot=True,
    #                    fig_title = 'Agent evaluation')

    states_q, q_values, best_action = find_policy_from_q(agent)
    if make_plots:
        plot_agent_evaluation(
            states_q, q_values, best_action, states_v, v_star_values,
            visited_states)

    policy_optimality = calculate_policy_optimality(env, states_q, best_action)
    return agent, policy_optimality


if __name__ == "__main__":

    run_type = '2d_scan'
    save_agents = True
    agent_directory = 'trained_agents/'
    n_repeats_scan = 5  # How many times to run the same parameters in scans

    # Environment settings
    kwargs_env = {
        'n_bits_observation_space': 8,
        'n_actions': 2,
        'simple_reward': True,
        'max_steps_per_episode': 25
    }

    # RL settings
    kwargs_rl = {
        'learning_rate': (2e-2, 6e-4),
        'small_gamma': 0.8,
        'exploration_epsilon': (1.0, 0.04),
        'exploration_fraction': 0.7
    }

    # Graph config and quantum annealing settings
    # Commented values are what's in the paper
    kwargs_anneal = {
        'n_graph_nodes': 16,  # nodes of Chimera graph (2 units DWAVE)
        'n_replicas': 25,  # 25
        'n_meas_for_average': 50,  # 150
        'n_annealing_steps': 100,  # 300, it seems that 100 is best
        'big_gamma': (20., 0.5),
        'beta': 1.
    }

    # Training time steps
    total_timesteps = 1000  # 500

    if run_type == 'single':
        make_plots = True
        agent, optimality = train_and_evaluate_agent(
            kwargs_env=kwargs_env, kwargs_rl=kwargs_rl,
            kwargs_anneal=kwargs_anneal, total_timesteps=total_timesteps,
            make_plots=make_plots)
        print(f'Optimality {optimality:.2f} %')

        if save_agents:
            agent_path = agent_directory + 'single_run.pkl'
            with open(agent_path, 'wb') as fid:
                dill.dump(agent, fid)

    elif run_type == '1d_scan':
        make_plots = False

        param_arr = np.array([10, 20, 40, 80, 100, 150])
        f_name = 'n_meas_for_average_'
        results = np.zeros((n_repeats_scan, len(param_arr)))

        tot_n_scans = len(param_arr)
        for k, val in enumerate(param_arr):
            print(f'Param. scan nb.: {k + 1}/{tot_n_scans}')

            kwargs_anneal.update({'n_meas_for_average': val})
            for m in range(n_repeats_scan):
                agent, results[m, k] = train_and_evaluate_agent(
                    kwargs_env=kwargs_env, kwargs_rl=kwargs_rl,
                    kwargs_anneal=kwargs_anneal,
                    total_timesteps=total_timesteps,
                    make_plots=make_plots)

                if save_agents:
                    agent_path = agent_directory + f_name + f'{val}_run_{m}.pkl'
                    with open(agent_path, 'wb') as fid:
                        dill.dump(agent, fid)

        # Plot scan summary
        plt.figure(1, figsize=(6, 5))
        (h, caps, _) = plt.errorbar(
            param_arr, np.mean(results, axis=0),
            yerr=np.std(results, axis=0) / np.sqrt(n_repeats_scan),
            capsize=4, elinewidth=2, color='tab:red')

        for cap in caps:
            cap.set_color('tab:red')
            cap.set_markeredgewidth(2)

        plt.xlabel('n_meas_for_average')
        plt.ylabel('Optimality (%)')
        plt.tight_layout()
        plt.show()

    else:
        # Assume 2d_scan
        make_plots = False

        param_1 = np.array([0.1, 0.2, 0.5, 1.])
        f_name_1 = f'big_gamma_f_'
        param_2 = np.array([0.5, 1., 2., 3.])
        f_name_2 = f'_beta_'

        results = np.zeros((n_repeats_scan, len(param_1), len(param_2)))

        tot_n_scans = len(param_1) * len(param_2)
        for k, val_1 in enumerate(param_1):
            for l, val_2 in enumerate(param_2):
                print(f'Param. scan nb.: {k+l+1}/{tot_n_scans}')
                for m in range(n_repeats_scan):
                    kwargs_anneal.update(
                        {'big_gamma': (20., val_1), 'beta': val_2})

                    agent, results[m, k, l] = train_and_evaluate_agent(
                        kwargs_env=kwargs_env, kwargs_rl=kwargs_rl,
                        kwargs_anneal=kwargs_anneal,
                        total_timesteps=total_timesteps,
                        make_plots=make_plots)

                    if save_agents:
                        agent_path = (
                            agent_directory + f_name_1 + f'{val_1}' +
                            f_name_2 + f'{val_2}_run_{m}.pkl')
                        with open(agent_path, 'wb') as fid:
                            dill.dump(agent, fid)

        # Plot scan summary, mean
        plt.figure(1, figsize=(6, 5))
        plt.imshow(np.flipud(np.mean(results.T, axis=0)))
        cbar = plt.colorbar()

        plt.xticks(range(len(param_2)),
                   labels=[i for i in param_2])
        plt.yticks(range(len(param_1)),
                   labels=[i for i in param_1[::-1]])

        plt.xlabel('beta')
        plt.ylabel('big_gamma_f')
        cbar.set_label('Mean optimality (%)')
        plt.tight_layout()
        plt.savefig('mean_res.png', dpi=300)
        plt.show()

        # Plot scan summary, std
        plt.figure(2, figsize=(6, 5))
        plt.imshow(np.flipud(np.std(results.T, axis=0)/np.sqrt(n_repeats_scan)))
        cbar = plt.colorbar()

        plt.xticks(range(len(param_2)),
                   labels=[i for i in param_2])
        plt.yticks(range(len(param_1)),
                   labels=[i for i in param_1[::-1]])

        plt.xlabel('beta')
        plt.ylabel('big_gamma_f')
        cbar.set_label('Std. optimality (%)')
        plt.tight_layout()
        plt.savefig('std_res.png', dpi=300)
        plt.show()
