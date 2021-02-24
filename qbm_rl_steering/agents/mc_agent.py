import numpy as np
from typing import List, Tuple, Dict


class MonteCarloAgent:
    def __init__(self, env, gamma=1.) -> None:
        """
        Monte Carlo agent to calculate the V* values of the open AI gym
        environment. Note that you have to replace the optimal policy and
        adapt it to the environment you are evaluating.
        :param env: Open AI - based gym environment
        :param gamma: discount factor
        """
        self.env = env
        self.gamma = gamma

        self.V = {}
        self.returns = self._initialize_returns()

    def _initialize_returns(self) -> Dict:
        """
        Initializes a dict where the keys are the states of the environment
        and the values will be the returns from these states assuming the
        optimal policy.
        :return dictionary of lists that will be filled with returns
        """
        returns = {}
        all_states = self._get_all_states()
        for s in all_states:
            returns[s] = []
        return returns

    def _optimal_policy(self, state: Tuple) -> int:
        """
        Here we define the optimal policy (if known). If state is below pos. 0
        => go up, i.e. action 0, otherwise take action 1. This is the optimal
        policy (we know it)
        :return: the action index under the optimal policy
        """
        state_float = self.env.make_binary_state_float(np.array(state))

        # This is the optimal policy for the problem at hand
        if state_float < 0:
            return 0
        return 1

    def _get_all_states(self) -> List:
        """
        Creates all the different states of the environment we are working
        with and returns them as a list
        :return: list of all possible states of the environment
        """
        all_states = range(0, 2 ** self.env.n_bits_observation_space, 1)
        all_states_binary = []
        for state in all_states:
            # Convert to corresponding binary state as used in env.
            state_binary = tuple(self.env.make_binary(state))
            all_states_binary.append(tuple(state_binary))
        return all_states_binary

    def _extract_v_star(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the states and corresponding values V* (i.e. optimal policy
        values)
        :return: np arrays of states and v_star values
        """
        states = np.zeros(len(self.V.keys()))
        v_star = np.zeros(len(self.V.keys()))
        for i, (s, v) in enumerate(self.V.items()):
            states[i] = self.env.make_binary_state_float(np.array(s))
            v_star[i] = v

        sort_idx = np.argsort(states)
        states = states[sort_idx]
        v_star = v_star[sort_idx]

        return states, v_star

    def _run_episode(self) -> List:
        """
        Runs one episode of the environment until the done flag is true and
        collects states and rewards of all the steps in a list that's returned.
        :return: list of states and rewards visited
        """
        s = tuple(self.env.reset())
        states_and_rewards = [(s, 0)]

        done = False
        while not done:
            a = self._optimal_policy(s)
            s, r, done, _ = self.env.step(a)
            s = tuple(s)
            states_and_rewards.append((s, r))
        return states_and_rewards

    def run_mc(self, n_iterations=200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Monte Carlo to find states and corresponding values according to
        the optimal policy.
        :return: tuple of states and corresponding values V*
        """

        for t in range(n_iterations):
            states_and_rewards = self._run_episode()

            # Calculate the returns by working backwards from terminal state
            G = 0.
            states_and_returns = []
            first = True
            for s, r in reversed(states_and_rewards):
                if first:
                    first = False
                else:
                    states_and_returns.append((s, G))
                G = r + self.gamma * G
            states_and_returns.reverse()

            seen_states = set()
            for s, G in states_and_returns:
                # Check if we have already seen s (first-visit MC policy
                # evaluation)
                if s not in seen_states:
                    self.returns[s].append(G)
                    self.V[s] = np.mean(self.returns[s])
                    seen_states.add(s)
        states, v_star = self._extract_v_star()
        return states, v_star
