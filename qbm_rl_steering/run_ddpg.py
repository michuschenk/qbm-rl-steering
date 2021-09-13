import numpy as np
import matplotlib.pyplot as plt

from qbm_rl_steering.core.ddpg_agents import ClassicalDDPG, QuantumDDPG
from qbm_rl_steering.environment.target_steering_1d import TargetSteeringEnv
from qbm_rl_steering.environment.rms_env_nd import RmsSteeringEnv


# TODO: implement early stopping, epsilon decay, learning schedule more
#  properly
# TODO: Implement return values of trainer as dict.
# TODO: make trainer so generic that it can deal with different envs without
#  changing few lines every time.
# TODO: implement n_anneals schedule over time properly.
def trainer(env, agent, max_episodes, max_steps_per_episode, batch_size,
            init_action_noise, n_exploration_steps=30,
            action_noise_decay=False, epsilon=0.3,
            early_stopping_consecutive=30):
    """ Convenience function to run training with DDPG.
    :param env: openAI gym environment instance
    :param agent: ddpg instance (ClassicalDDPG or QuantumDDPG)
    :param max_episodes: max. number of episodes that training will run
    :param max_steps_per_episode: max. number of steps allowed per episode
    :param batch_size: number of samples drawn from experience replay buffer
    at every step.
    :param init_action_noise: initial scale of action noise
    :param n_exploration_steps: number of initial random steps in env.
    :param action_noise_decay: flag stating whether or not action noise
    should decay over time.
    :param epsilon: epsilon-greedy parameter: what fraction of actions will
    be purely random.
    :param early_stopping_consecutive: number of consecutive episodes with
    certain number of steps (< 4) to count towards early stopping.
    :return tuple of init, final rewards, number of steps, and random steps
    (all per episode).
    """
    episode_init_rewards = []
    episode_final_rewards = []
    episode_length = []
    episode_random_steps = []
    total_step_count = 0
    early_stopping_count = 0

    for episode in range(max_episodes):
        if early_stopping_count >= early_stopping_consecutive:
            print('STOPPING EARLY...')
            break
        n_count_random_steps = 0
        state = env.reset(init_outside_threshold=True)
        episode_init_rewards.append(env.calculate_reward(
            env.calculate_state(env.kick_angles)))

        # Linear decay of action noise
        if action_noise_decay:
            action_noise = (1. - episode / max_episodes) * init_action_noise
        else:
            action_noise = init_action_noise

        # n_meas_for_average. Increase for last 30% of episodes
        if (episode / float(max_episodes)) > 0.7:
            print('CHANGING TO 50 AVERAGES')
            agent.main_critic_net.n_meas_for_average = 50
            agent.target_critic_net.n_meas_for_average = 50

        # Episode loop
        epsilon -= epsilon / max_episodes
        for step in range(max_steps_per_episode):
            eps_sample = np.random.uniform(0, 1, 1)
            if ((total_step_count < n_exploration_steps) or
                    (eps_sample <= epsilon)):
                action = env.action_space.sample()
                # print('Sampling randomly')
                n_count_random_steps += 1
            else:
                action = agent.get_proposed_action(state, action_noise)
                # print('Following actor')

            total_step_count += 1
            next_state, reward, done, _ = env.step(action)
            d_store = False if step == max_steps_per_episode - 1 else done
            agent.replay_buffer.push(state, action, reward, next_state, d_store)

            if agent.replay_buffer.size > batch_size:
                agent.update(batch_size)
            else:
                agent.update(agent.replay_buffer.size)

            if done or step == max_steps_per_episode - 1:
                episode_final_rewards.append(reward)
                print("*****************************************************")
                print(f"Episode {episode}: init rew: "
                      f"{round(episode_init_rewards[-1], 2)} .. final rew: " +
                      f"{round(episode_final_rewards[-1], 2)} .. steps: "
                      f"{step + 1} .. of which random: {n_count_random_steps}")
                print("*****************************************************\n")
                episode_length.append(step + 1)
                episode_random_steps.append(n_count_random_steps)
                if (step < 3) and (reward > env.reward_threshold):
                    early_stopping_count += 1
                    print(f'COUNTING TOWARDS EARLY STOPPING, CURRENTLY AT: '
                          f'{early_stopping_count}/{early_stopping_consecutive}')
                else:
                    early_stopping_count = 0
                break

            state = next_state

    return (episode_init_rewards, episode_final_rewards, episode_length,
            episode_random_steps)


# TODO: all this needs cleaning up: incl. plots. Also allow to plot
#  intermediate steps.
# TODO: implement easy switch between classical and quantum DDPG.
n_dims = 10
max_steps_per_episode = 50
# thresh = -0.08
env = RmsSteeringEnv(n_dims=n_dims, max_steps_per_episode=max_steps_per_episode)
# env = awake_sim.e_trajectory_simENV()
# env.action_scale = 3e-4
# env.threshold = thresh
# env.MAX_TIME = max_steps

n_episodes = 200  # 80
batch_size = 24  # 15
n_exploration_steps = 50  # 30

gamma = 0.99  # 0.95
tau_critic = 0.1  # 1
tau_actor = 0.1  # 1
buffer_maxlen = 10000
critic_lr = 5e-4  # 1e-3
actor_lr = 1e-4  # 5e-4

agent = QuantumDDPG(env.observation_space, env.action_space, gamma,
                    tau_critic, tau_actor, buffer_maxlen, critic_lr, actor_lr,
                    grad_clip_actor=10000., grad_clip_critic=1.,
                    n_steps_estimate=int(n_episodes * max_steps_per_episode / 2.))

init_rewards, final_rewards, episode_length, episode_random_steps = trainer(
    env, agent, n_episodes, max_steps_per_episode, batch_size, init_action_noise=0.1,
    # best value so far: 0.1
    n_exploration_steps=n_exploration_steps, action_noise_decay=True)

fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(episode_length, label='Total steps')
axs[0].plot(episode_random_steps, '--', c='k', label='Random steps')
axs[1].plot(init_rewards, c='r', label='Initial')
axs[1].plot(final_rewards, c='g', label='Final')
axs[0].set_ylabel('Number of steps')
axs[1].set_ylabel('Reward')
axs[1].set_xlabel('Episodes')
axs[0].legend(loc='upper right')
axs[1].legend(loc='lower left')
plt.tight_layout()
plt.show()

plt.plot(np.array(agent.q_log['before']), c='r', label='q before')
plt.plot(np.array(agent.q_log['after']), c='g', label='q after')
plt.legend()
plt.ylabel('Q value')
plt.xlabel('Steps')
plt.tight_layout()
plt.show()

plt.plot(np.array(agent.q_log['after']) - np.array(agent.q_log['before']))
plt.ylabel(r'$Q_{{f}} - Q_{{i}}$')
plt.xlabel('Steps')
plt.tight_layout()
plt.show()

# Agent evaluation
n_episodes_eval = 80
episode_counter = 0

rewards_eval = []
env = RmsSteeringEnv(n_dims=n_dims, max_steps_per_episode=max_steps_per_episode)
# env = awake_sim.e_trajectory_simENV()
# env.action_scale = 3e-4
# env.threshold = thresh
# env.MAX_TIME = max_steps

while episode_counter < n_episodes_eval:
    state = env.reset()
    state = np.atleast_2d(state)
    reward_ep = []
    reward_ep.append(env.calculate_reward(
        env.calculate_state(env.kick_angles)))

    n_steps_eps = 0
    while True:
        a = agent.get_proposed_action(state, noise_scale=0)
        a = np.squeeze(a)
        state, reward, done, _ = env.step(a)
        reward_ep.append(reward)
        if done or n_steps_eps > max_steps_per_episode:
            episode_counter += 1
            rewards_eval.append(reward_ep)
            break
        n_steps_eps += 1

rewards_eval = np.array(rewards_eval)

fig, axs = plt.subplots(2, 1, sharex=True)
init_rew = np.zeros(len(rewards_eval))
final_rew = np.zeros(len(rewards_eval))
length = np.zeros(len(rewards_eval))
for i in range(len(rewards_eval)):
    init_rew[i] = rewards_eval[i][0]
    final_rew[i] = rewards_eval[i][-1]
    length[i] = len(rewards_eval[i]) - 1

axs[1].plot(init_rew, c='r', label='Initial')
axs[1].plot(final_rew, c='g', label='Final')
axs[0].plot(length)
axs[0].set_ylabel('Number of steps')
axs[1].set_ylabel('Reward')
axs[1].set_xlabel('Episodes')
axs[1].legend(loc='lower left')
plt.tight_layout()
plt.show()

# PLOTTING ALL REWARDS
cmap = plt.get_cmap("magma")
fig, axs = plt.subplots(2, 1, sharex=True)
init_rew = np.zeros(len(rewards_eval))
final_rew = np.zeros(len(rewards_eval))
length = np.zeros(len(rewards_eval))
all_rewards = np.zeros((len(rewards_eval), max_steps_per_episode))

# max_required = 0
# for i in range(len(rewards_eval)):
#     init_rew[i] = rewards_eval[i][0]
#     final_rew[i] = rewards_eval[i][-1]
#     length[i] = len(rewards_eval[i]) - 1
#     all_rewards[i, :len(rewards_eval[i])] = rewards_eval[i]
#
#     if len(rewards_eval[i]) > max_required:
#         max_required = len(rewards_eval[i])
#
# for j in range(max_required):
#     # msk = all_rewards[:, j] == 0
#     axs[1].plot(all_rewards[:, j], c=cmap(j/max_required), alpha=0.7)

axs[1].plot(init_rew, c='r', label='Initial')
axs[1].plot(final_rew, c='g', label='Final')
axs[0].plot(length)
axs[0].set_ylabel('Number of steps')
axs[1].set_ylabel('Reward')
axs[1].set_xlabel('Episodes')
axs[1].legend(loc='lower left')
# axs[1].set_ylim(top=-1)
plt.tight_layout()
plt.show()

# GRADIENTS
plt.figure()
plt.plot(agent.actor_grads_log['mean'], label='mean')
plt.plot(agent.actor_grads_log['min'], label='min')
plt.plot(agent.actor_grads_log['max'], label='max')
plt.ylabel('Grads')
plt.xlabel('Steps')
plt.legend()
plt.tight_layout()
plt.show()
