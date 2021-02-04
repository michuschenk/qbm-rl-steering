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
        my_env, agent, n_epochs=200, fig_title='Agent test after training')
