from environment.env_desc import TargetSteeringEnv
import environment.helpers as hlp

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env


if __name__ == "__main__":
    n_bits_observation_space = 8

    # Transfer line environment
    my_env = TargetSteeringEnv(n_bits_observation_space)

    # To understand environment better, plot response and test random
    # action-taking
    # hlp.plot_response(my_env, fig_title='Env. test: response function')
    # hlp.run_random_trajectories(
    #   my_env, fig_title='Env test: random trajectories')

    # DQN: state and action is discrete: see here:
    # https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
    policy_kwargs = dict(net_arch=[128, 128])
    agent = DQN('MlpPolicy', my_env, verbose=1, learning_starts=0,
                policy_kwargs=policy_kwargs, exploration_fraction=0.5,
                exploration_initial_eps=1.0, exploration_final_eps=0.)

    # Agent test before learning
    hlp.test_agent(
        my_env, agent, n_epochs=200, fig_title='Agent test before training')

    # Run RL and plot log
    agent.learn(total_timesteps=2000)
    hlp.plot_log(my_env, fig_title='Agent training')
    agent.save('dqn_transferline')
    del agent  # remove to demonstrate saving and loading

    # Reload and test the trained agent; recreate environment
    my_env = TargetSteeringEnv(n_bits_observation_space)
    agent = DQN.load('dqn_transferline')
    hlp.test_agent(
        my_env, agent, n_epochs=500, fig_title='Agent test after training')

    # Define metric that characterizes performance of the agent
    # Option (I): we count how many times the agent manages to reach the target
    # without going above 'UB optimal behaviour' (i.e. max. number of steps
    # required assuming optimal behaviour).
    # Option (II): Take difference between initial and final reward and
    # divide by number of steps required.
    # I think option (II) gives a bit more detail, but we implement both.

    # Implement log as an object with some useful methods
    import numpy as np
    n_episodes = len(my_env.log_all)
    episode_length = np.zeros(n_episodes)
    reward_init = np.zeros_like(episode_length)
    reward_final = np.zeros_like(episode_length)
    for ep, log_ep in enumerate(my_env.log_all):
        episode_length[ep] = len(log_ep) - 1
        reward_init[ep] = log_ep[0][2]
        reward_final[ep] = log_ep[-1][2]

    # Option (I): count how many times the agent succeeded.
    upper_bound_optimal = my_env.get_max_n_steps_optimal_behaviour()
    msk_steps = episode_length < upper_bound_optimal
    msk_reward = reward_final > my_env.reward_threshold
    msk_nothing_to_do = episode_length == 0

    n_success = np.sum(msk_steps & msk_reward & (~msk_nothing_to_do))
    success_metric_1 = n_success / float(np.sum(~msk_nothing_to_do))
    print('success_metric_1', success_metric_1)

    # Option (II):
    delta_reward = reward_final - reward_init
    success_metric_2 = np.mean(
        delta_reward[~msk_nothing_to_do] / episode_length[~msk_nothing_to_do])
    print('success_metric_2', success_metric_2)
