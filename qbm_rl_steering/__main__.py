from environment.env_desc import TargetSteeringEnv
from environment.helpers import plot_response, run_random_trajectories
from environment.helpers import plot_log

from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.common.env_checker import check_env


if __name__ == "__main__":
    n_bits_observation_space = 8

    # Transfer line environment
    my_env = TargetSteeringEnv(n_bits_observation_space)

    # To understand environment better, plot response and test random
    # action-taking
    # plot_response(my_env, fig_title='Env. test: response function')
    # run_random_trajectories(my_env, fig_title='Env test: random trajectories')

    # DQN: state and action is discrete: see here:
    # https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
    agent = DQN(MlpPolicy, my_env, verbose=1, learning_starts=500)
    agent.learn(total_timesteps=5000)

    plot_log(my_env, fig_title='Agent training')

    # Save agent and delete
    agent.save("dqn_transferline")
    del agent  # remove to demonstrate saving and loading

    # Reload and test the trained agent; recreate environment
    my_env = TargetSteeringEnv(n_bits_observation_space)
    obs = my_env.reset()
    agent = DQN.load("dqn_transferline")

    n_epochs_test = 100
    epoch_count = 0
    while epoch_count < n_epochs_test:
        action, _states = agent.predict(obs, deterministic=True)
        obs, reward, done, info = my_env.step(action)
        # agent.render()
        if done:
            obs = my_env.reset()
            epoch_count += 1

    plot_log(my_env, fig_title='Agent test after training')
