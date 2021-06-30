import numpy as np
from tqdm import tqdm

from qbm_rl_steering.core.qac import QuantumActorCritic
from cern_awake_env.simulation import SimulationEnv
# from qbm_rl_steering.environment.target_steering_1d import TargetSteeringEnv
# from qbm_rl_steering.environment.target_steering_2d import TargetSteeringEnv2D

import gym

try:
    import matplotlib
    matplotlib.use('qt5agg')
except ImportError as err:
    print(err)

import matplotlib.pyplot as plt


def trainer(env: gym.Env, agent: QuantumActorCritic, n_episodes: int,
            max_episode_length: int, exploration_steps: int):
    """ Run agent-environment interactions for agent training.
    :param env: openAI gym environment
    :param agent: RL agent (DDPG), either QuantumActorCritic or classical DDPG
    :param n_episodes: total number of training episodes
    :param max_episode_length: maximum number of steps per episode
    :param exploration_steps: initial number of interactions with env for which
    agent follows a random policy. After that period, it will follow the
    current actor policy (including some action noise to keep some level of
    exploration).
    :return all_rewards: list of lists containing rewards for all episodes of
    training.
    """
    all_rewards = []
    episode_count = 0
    total_step_count = 0

    reset_env = True
    while episode_count < n_episodes:
        # Beginning new episode
        if reset_env:
            episode_rewards = []
            state = env.reset()
            step_count = 0
            reset_env = False

        # Calculate reward in current state
        try:
            episode_rewards.append(
                env.compute_reward(state, goal=None, info={}))
        except:
            _, intensity = env.get_pos_at_bpm_target(
                env.mssb_angle, env.mbb_angle)
            episode_rewards.append(env.get_reward(intensity))

        if total_step_count >= exploration_steps:
            action = agent.get_action(state, episode=1)
            action = np.squeeze(action)
            if total_step_count == exploration_steps:
                print('END OF EXPLORATION PERIOD: following policy action')
        else:
            action = env.action_space.sample()

        # Step the env
        next_state, reward, done, _ = env.step(action)
        episode_rewards.append(reward)
        step_count += 1
        total_step_count += 1

        done = False if (step_count == max_episode_length) else done

        # Store experience to replay buffer
        agent.replay_memory.store(state, action, reward, next_state, done)

        state = next_state

        # Bookkeeping and training on replay buffer
        if done or (step_count == max_episode_length):
            episode_count += 1
            all_rewards.append(np.array(episode_rewards))
            print(f"Episode: {episode_count}, "
                  f"Reward initial: {episode_rewards[0]}, "
                  f"Reward final: {episode_rewards[-1]}, "
                  f"Number of steps: {step_count}")

            # Train loop
            for _ in tqdm(range(step_count)):
                agent.train()
            reset_env = True

    return np.array(all_rewards)


def evaluator(env: gym.Env, agent: QuantumActorCritic, n_episodes: int,
              max_episode_length: int):
    """ Run evaluation for a number of episodes, typically using a trained
    agent.
    :param env: openAI gym environment
    :param agent: RL agent (DDPG), either QuantumActorCritic or classical DDPG
    :param n_episodes: number of episodes for evaluation
    :param max_episode_length: maximum number of steps per episode
    :return all_rewards: list of lists containing rewards for all episodes of
    training.
    """
    episode_count = 0

    all_rewards = []
    while episode_count < n_episodes:
        episode_rewards = []
        state = env.reset()

        # Get reward of the initial state
        try:
            episode_rewards.append(env.compute_reward(state, goal=None, info={}))
        except:
            _, intensity = env.get_pos_at_bpm_target(
                env.mssb_angle, env.mbb_angle)
            episode_rewards.append(env.get_reward(intensity))

        step_count = 0
        while True:
            a = agent.get_action(state, noise=0)
            a = np.squeeze(a)
            state, reward, done, _ = env.step(a)
            episode_rewards.append(reward)
            if done or step_count > max_episode_length:
                all_rewards.append(np.array(episode_rewards))
                episode_count += 1
                break
            step_count += 1
    return np.array(all_rewards)


def plot_log(all_rewards, plot_title=''):
    """ Extract initial and final rewards as well as episode lengths and plot
    them. """
    initial_rewards = np.zeros(len(all_rewards))
    final_rewards = np.zeros(len(all_rewards))
    episode_lengths = np.zeros(len(all_rewards))
    for i in range(len(all_rewards)):
        initial_rewards[i] = all_rewards[i][0]
        final_rewards[i] = all_rewards[i][-1]
        episode_lengths[i] = len(all_rewards[i]) - 1
    print(f'Total number of interactions: {np.sum(episode_lengths)}')

    fig, axs = plt.subplots(2, 1, sharex=True)
    plt.suptitle(plot_title, fontsize=12)
    axs[0].plot(episode_lengths)
    axs[1].plot(initial_rewards, c='r', label='Initial')
    axs[1].plot(final_rewards, c='g', label='Final')
    axs[1].legend(loc='lower right')
    axs[0].set_ylabel('Episode length')
    axs[1].set_ylabel('Reward')
    axs[1].set_xlabel('Episode')
    plt.show()


# TODO: fix 'issue' that env does not stop immediately when we are already
#  within threshold?

n_trainings = 10

# To save performance data
# i.e. rewards evolution
train_stats = []
eval_stats = []

# Env / train and eval
n_episodes_train = 100
max_episode_length = 20
exploration_steps = 50

n_episodes_evaluation = 100

# Agent
gamma_rl = 0.95
batch_size = 16
for i in range(n_trainings):

    # ENVIRONMENT
    # env = TargetSteeringEnv(max_steps_per_episode=max_episode_length)
    # env = TargetSteeringEnv2D(max_steps_per_episode=max_episode_length)
    env = SimulationEnv(plane='H', remove_singular_devices=True)

    # AGENT
    agent = QuantumActorCritic(env, gamma_rl=gamma_rl, batch_size=batch_size)

    # RUN TRAINING LOOP
    training_rewards = trainer(
        env=env, agent=agent, n_episodes=n_episodes_train,
        max_episode_length=max_episode_length,
        exploration_steps=exploration_steps)

    # PLOT TRAINING EVOLUTION
    plot_log(training_rewards, plot_title=f'Training {i}')

    # RUN EVALUATION
    env = SimulationEnv(plane='H', remove_singular_devices=True)
    # env = TargetSteeringEnv2D(max_steps_per_episode=max_episode_length)
    # env = TargetSteeringEnv(max_steps_per_episode=max_episode_length)

    evaluation_rewards = evaluator(
        env=env, agent=agent, n_episodes=n_episodes_evaluation,
        max_episode_length=max_episode_length)

    plot_log(evaluation_rewards, plot_title=f'Evaluation {i}')

    train_stats.append(training_rewards)
    eval_stats.append(evaluation_rewards)

np.save('data_train', np.array(train_stats))
np.save('data_eval', np.array(eval_stats))
