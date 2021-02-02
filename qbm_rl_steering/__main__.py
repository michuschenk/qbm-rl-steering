from environment.env_desc import TargetSteeringEnv
from environment.helpers import plot_response, run_random_trajectories, plot_log
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
# from stable_baselines.deepq.dqn import DQN
# from stable_baselines.deepq.policies import MlpPolicy


if __name__ == "__main__":
    # Transfer line environment
    my_env = TargetSteeringEnv()

    # To understand environment better, plot response and test random
    # action-taking
    # plot_response(my_env)
    # run_random_trajectories(my_env)

    # DQN: state and action is discrete: see here:
    # https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
    agent = DQN(MlpPolicy, my_env, verbose=1)
    agent.learn(total_timesteps=100000)

    plot_log(my_env, title='Agent training', plot_epoch_end=False)

    # Save agent and delete
    agent.save("dqn_transferline")
    del agent  # remove to demonstrate saving and loading

    # Reload and test the trained agent; recreate environment
    my_env = TargetSteeringEnv()
    obs = my_env.reset()
    agent = DQN.load("dqn_transferline")

    n_epochs_test = 10
    epoch_count = 0
    while epoch_count < n_epochs_test:
        action, _states = agent.predict(obs, deterministic=True)
        obs, reward, done, info = my_env.step(action)
        # agent.render()
        if done:
            obs = my_env.reset()
            epoch_count += 1

    plot_log(my_env, title='Agent test', plot_epoch_end=True)
