import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers.schedules import (ExponentialDecay,
                                                   PolynomialDecay,
                                                   PiecewiseConstantDecay)

from qbm_rl_steering.core.ddpg_agents import ClassicalDDPG, QuantumDDPG
from qbm_rl_steering.environment.rms_env_nd import RmsSteeringEnv


def trainer(env, agent, max_episodes, max_steps_per_episode, batch_size,
            action_noise_schedule, epsilon_greedy_schedule, n_anneals_schedule,
            n_exploration_steps=30, n_episodes_early_stopping=30):
    """ Convenience function to run training with DDPG.
    :param env: openAI gym environment instance
    :param agent: ddpg instance (ClassicalDDPG or QuantumDDPG)
    :param max_episodes: max. number of episodes that training will run
    :param max_steps_per_episode: max. number of steps allowed per episode
    :param batch_size: number of samples drawn from experience replay buffer
    at every step.
    :param action_noise_schedule: action noise decay over time.
    :param epsilon_greedy_schedule: epsilon-greedy schedule. Decides what
    fraction of actions will be purely random.
    :param n_anneals_schedule: number of anneals as training progresses.
    :param n_exploration_steps: number of initial random steps in env.
    :param n_episodes_early_stopping: number of consecutive episodes with
    certain number of steps (< 4) to count towards early stopping.
    :return tuple of init, final rewards, number of steps, and random steps
    (all per episode).
    """
    episode_log = {
        'initial_rewards': [], 'final_rewards': [], 'n_total_steps': [],
        'n_random_steps': []
    }

    total_step_count = 0
    early_stopping_count = 0

    for episode in range(max_episodes):
        if early_stopping_count >= n_episodes_early_stopping:
            break

        n_count_random_steps = 0
        state = env.reset(init_outside_threshold=True)
        episode_log['initial_rewards'].append(env.calculate_reward(
            env.calculate_state(env.kick_angles)))

        # Apply n_anneals_schedule
        n_anneals = int(n_anneals_schedule(episode))
        agent.main_critic_net.n_meas_for_average = n_anneals
        agent.target_critic_net.n_meas_for_average = n_anneals

        # Episode loop
        epsilon = epsilon_greedy_schedule(episode)

        # Action noise decay
        action_noise = action_noise_schedule(episode)

        for step in range(max_steps_per_episode):
            eps_sample = np.random.uniform(0, 1, 1)
            if ((total_step_count < n_exploration_steps) or
                    (eps_sample <= epsilon)):
                action = env.action_space.sample()
                n_count_random_steps += 1
            else:
                action = agent.get_proposed_action(state, action_noise)

            total_step_count += 1
            next_state, reward, done, _ = env.step(action)
            d_store = False if step == max_steps_per_episode - 1 else done
            agent.replay_buffer.push(state, action, reward, next_state, d_store)

            if agent.replay_buffer.size > batch_size:
                agent.update(batch_size, episode)
            else:
                agent.update(agent.replay_buffer.size, episode)

            if done or step == max_steps_per_episode - 1:
                episode_log['final_rewards'].append(reward)
                print("*****************************************************")
                print(f"Ep {episode}: "
                      f"init. rew.: "
                      f"{round(episode_log['initial_rewards'][-1], 2)} .. "
                      f"final rew.: "
                      f"{round(episode_log['final_rewards'][-1], 2)} .. "
                      f"steps: {step + 1}, of which random:"
                      f" {n_count_random_steps}")
                print("*****************************************************\n")
                episode_log['n_total_steps'].append(step + 1)
                episode_log['n_random_steps'].append(n_count_random_steps)
                if (step < 3) and (reward > env.reward_threshold):
                    early_stopping_count += 1
                    print(f"Early stopping count: "
                          f"{early_stopping_count}/"
                          f"{n_episodes_early_stopping}")
                else:
                    early_stopping_count = 0
                break

            state = next_state

    return episode_log


def evaluator(env, agent, n_episodes, reward_scan=True):
    """ Run trained agent for a number of episodes.
    :param env: openAI gym based environment
    :param agent: trained agent
    :param n_episodes: number of episodes for evaluation.
    :param reward_scan: if False, init episodes randomly. If True, run scan
    in specific range of rewards.
    :return ndarray with all rewards. """
    all_rewards = []

    episode = 0

    while episode < n_episodes:
        if reward_scan:
            target_init_rewards = np.linspace(
                env.reward_threshold * 4, env.reward_threshold, n_episodes)
            state = env.reset(
                init_specific_reward_state=target_init_rewards[episode])
        else:
            state = env.reset(init_outside_threshold=True)

        rewards = [env.calculate_reward(
            env.calculate_state(env.kick_angles))]

        n_steps_eps = 0
        while True:
            a = agent.get_proposed_action(state, noise_scale=0)
            state, reward, done, _ = env.step(a)
            rewards.append(reward)
            if done or n_steps_eps == (max_steps_per_episode - 1):
                episode += 1
                all_rewards.append(rewards)
                break
            n_steps_eps += 1

    return np.array(all_rewards)


def plot_training_log(env, agent, data):
    """ Plot the log data from the training. """
    # a) Training log
    fig1, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(data['n_total_steps'], label='Total steps')
    axs[0].plot(data['n_random_steps'], '--', c='k',
                label='Random steps')
    axs[1].plot(data['initial_rewards'], c='r', label='Initial')
    axs[1].plot(data['final_rewards'], c='g', label='Final')
    axs[1].axhline(env.reward_threshold, color='k', ls='--')
    axs[0].set_ylabel('Number of steps')
    axs[1].set_ylabel('Reward (um)')
    axs[1].set_xlabel('Episodes')
    axs[0].legend(loc='upper right')
    axs[1].legend(loc='lower left')
    plt.tight_layout()
    plt.show()

    # b) AgentL q before vs after
    fig2 = plt.figure()
    plt.plot(np.array(agent.q_log['before']), c='r', label='q before')
    plt.plot(np.array(agent.q_log['after']), c='g', label='q after')
    plt.legend()
    plt.ylabel('Q value')
    plt.xlabel('Steps')
    plt.tight_layout()
    plt.show()

    # c) Agent: q_after minus q_before
    fig3 = plt.figure()
    plt.plot(np.array(agent.q_log['after']) - np.array(agent.q_log['before']))
    plt.ylabel(r'$Q_{{f}} - Q_{{i}}$')
    plt.xlabel('Steps')
    plt.tight_layout()
    plt.show()

    # d) Agent: gradients evolution
    fig4 = plt.figure()
    plt.plot(agent.actor_grads_log['mean'], label='mean')
    plt.plot(agent.actor_grads_log['min'], label='min')
    plt.plot(agent.actor_grads_log['max'], label='max')
    plt.ylabel('Grads')
    plt.xlabel('Steps')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_evaluation_log(env, max_steps_per_episode, data):
    """ Use rewards returned by evaluator function and create plots. """
    # a) Evaluation log
    fig5, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot([(len(r) - 1) for r in data])
    axs[1].plot([r[0] for r in data], c='r', label='Initial')
    axs[1].plot([r[-1] for r in data], c='g', label='Final')
    axs[1].axhline(env.reward_threshold, color='k', ls='--')
    axs[0].set_ylabel('Number of steps')
    axs[1].set_ylabel('Reward (um)')
    axs[1].set_xlabel('Episodes')
    axs[1].legend(loc='lower left')
    plt.tight_layout()
    plt.show()

    # b) Extract and plot all intermediate rewards
    all_rewards = np.zeros((len(data), max_steps_per_episode + 1))
    cmap = plt.get_cmap("magma")
    fig6, axs = plt.subplots(2, 1, sharex=True)
    max_steps = 0
    for i in range(len(data)):
        all_rewards[i, :len(data[i])] = data[i]
        if len(data[i]) > max_steps:
            max_steps = len(data[i])
    axs[0].plot([(len(r) - 1) for r in data])
    for j in range(max_steps):
        axs[1].plot(all_rewards[:, j], c=cmap(j / max_steps), alpha=0.7)
    axs[1].plot([r[0] for r in data], c='r', label='Initial')
    axs[1].plot([r[-1] for r in data], c='g', label='Final')
    axs[1].axhline(env.reward_threshold, color='k', ls='--')
    axs[0].set_ylabel('Number of steps')
    axs[1].set_ylabel('Reward (um)')
    axs[1].set_xlabel('Episodes')
    axs[1].legend(loc='lower left')
    plt.tight_layout()
    plt.show()


quantum_ddpg = True

n_dims = 3
max_steps_per_episode = 50
env = RmsSteeringEnv(n_dims=n_dims, max_steps_per_episode=max_steps_per_episode)

n_episodes = 200
batch_size = 32
n_exploration_steps = 70
n_episodes_early_stopping = 25

gamma = 0.99
tau_critic = 0.1
tau_actor = 0.1

# Learning rate schedules
# lr_critic: 5e-4, lr_actor: 1e-4
lr_schedule_critic = ExponentialDecay(1e-3, n_episodes, 0.95)
lr_schedule_actor = ExponentialDecay(1e-3, n_episodes, 0.95)

if quantum_ddpg:
    agent = QuantumDDPG(state_space=env.observation_space,
                        action_space=env.action_space, gamma=gamma,
                        tau_critic=tau_critic, tau_actor=tau_actor,
                        learning_rate_schedule_critic=lr_schedule_critic,
                        learning_rate_schedule_actor=lr_schedule_actor,
                        grad_clip_actor=1e4, grad_clip_critic=1.)
else:
    agent = ClassicalDDPG(state_space=env.observation_space,
                          action_space=env.action_space, gamma=gamma,
                          tau_critic=tau_critic, tau_actor=tau_actor,
                          learning_rate_schedule_critic=lr_schedule_critic,
                          learning_rate_schedule_actor=lr_schedule_actor)

# Action noise schedule
action_noise_schedule = PolynomialDecay(0.1, n_episodes, 0.01)

# Number of anneals schedule (first 50%: 1, 50-70%: 20, 70+%: 50 anneals)
n_anneals_schedule = PiecewiseConstantDecay(
    [0.5 * n_episodes, 0.7 * n_episodes],
    [1, 20, 50])

# Epsilon greedy schedule
epsilon_greedy_schedule = PolynomialDecay(0.3, n_episodes, 0.01)

# AGENT TRAINING
episode_log = trainer(env=env, agent=agent, max_episodes=n_episodes,
                      max_steps_per_episode=max_steps_per_episode,
                      batch_size=batch_size,
                      action_noise_schedule=action_noise_schedule,
                      epsilon_greedy_schedule=epsilon_greedy_schedule,
                      n_anneals_schedule=n_anneals_schedule,
                      n_exploration_steps=n_exploration_steps,
                      n_episodes_early_stopping=n_episodes_early_stopping)
plot_training_log(env, agent, episode_log)

# AGENT EVALUATION
# a) Random state inits
env = RmsSteeringEnv(n_dims=n_dims, max_steps_per_episode=max_steps_per_episode)
episode_log = evaluator(env, agent, n_episodes=100, reward_scan=False)
plot_evaluation_log(env, max_steps_per_episode, episode_log)

# b) Systematic state inits
env = RmsSteeringEnv(n_dims=n_dims, max_steps_per_episode=max_steps_per_episode)
episode_log = evaluator(env, agent, n_episodes=100, reward_scan=True)
plot_evaluation_log(env, max_steps_per_episode, episode_log)
