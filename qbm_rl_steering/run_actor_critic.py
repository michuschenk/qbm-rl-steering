import numpy as np

from qbm_rl_steering.core.visualization import plot_log
from qbm_rl_steering.core.qac import QuantumActorCritic

from cern_awake_env.simulation import SimulationEnv

from qbm_rl_steering.environment.target_steering_1d import TargetSteeringEnv
from qbm_rl_steering.environment.target_steering_2d import TargetSteeringEnv2D

try:
    import matplotlib

    matplotlib.use('qt5agg')
except ImportError as err:
    print(err)

import matplotlib.pyplot as plt


gamma_rl = 0.95
n_epochs = 15
max_episode_length = 20  #12
initial_exploration_steps = 50
initial_reward = 0
reward_scale = 1

# env = TargetSteeringEnv(max_steps_per_episode=max_episode_length)
env = SimulationEnv(plane='H', remove_singular_devices=True)
# env = TargetSteeringEnv2D(max_steps_per_episode=max_episode_length)
agent = QuantumActorCritic(env, gamma_rl=gamma_rl, batch_size=16)

state, reward, done, ep_rew, ep_len, ep_cnt = (
    env.reset(), initial_reward, False, [[]], 0, 0)
reward *= reward_scale

# Calculate reward in current state
# _, intensity = env.get_pos_at_bpm_target(env.mssb_angle, env.mbb_angle)
# ep_rew[-1].append(env.get_reward(intensity))
ep_rew[-1].append(reward_scale * env.compute_reward(
            state, goal=None, info={}))

total_steps = max_episode_length * n_epochs

# Main loop: collect experience in env and update/log each epoch
random_action = None
for t in range(total_steps):
    if t > initial_exploration_steps:
        action = agent.get_action(state, episode=1)
        action = np.squeeze(action)
        random_action = False
    else:
        action = env.action_space.sample()
        random_action = True
        print('sampling random action')

    # Step the env
    print('action', action)
    next_state, reward, done, _ = env.step(action)
    reward *= reward_scale
    ep_rew[-1].append(reward)  # keep adding to the last element till done
    ep_len += 1

    done = False if ep_len == max_episode_length else done

    # Store experience to replay buffer
    agent.replay_memory.store(state, action, reward, next_state, done)

    state = next_state

    if done or (ep_len == max_episode_length):
        ep_cnt += 1
        if True:
            print(f"Episode: {len(ep_rew) - 1}, Reward initial: "
                  f"{ep_rew[-1][0]}, "
                  f"Reward final:"
                  f" {ep_rew[-1][-1]}, "
                  f"Length: {len(ep_rew[-1])}")
        ep_rew.append([])

        for _ in range(ep_len):
            # if random_action:
            #     # Train only QBM Q net for the random action phase
            #     states, actions, rewards, next_states, dones = (
            #         agent.replay_memory.get_sample(
            #             batch_size=agent.replay_batch_size))
            #     agent.train_critic(states, next_states, actions, rewards,
            #                        dones, random_phase=True)
            # else:
            #     # Once we are out of the random action phase, train actor and
            #     # critic. Idea is that Q net already has been trained in the
            #     # right direction.
            agent.train()

        state, reward, done, ep_ret, ep_len = (
            env.reset(), initial_reward, False, 0, 0)
        reward *= reward_scale

        # _, intensity = env.get_pos_at_bpm_target(env.mssb_angle, env.mbb_angle)
        # ep_rew[-1].append(env.get_reward(intensity))
        ep_rew[-1].append(reward_scale * env.compute_reward(
            state, goal=None, info={}))

# Extract logging data
init_rewards = []
rewards = []
reward_lengths = []
for episode in ep_rew[:-1]:
    if len(episode) > 0:
        rewards.append(episode[-1])
        init_rewards.append(episode[0])
        reward_lengths.append(len(episode) - 1)
print('Total number of interactions:', np.sum(reward_lengths))

fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(reward_lengths)
axs[1].plot(init_rewards, c='r')
axs[1].plot(rewards, c='g')
plt.show()

# Agent evaluation
n_episodes_eval = 80
episode_counter = 0

rewards_eval = []
env = SimulationEnv(plane='H', remove_singular_devices=True)
# env = TargetSteeringEnv2D(max_steps_per_episode=max_episode_length)
# env = TargetSteeringEnv(max_steps_per_episode=max_episode_length)
while episode_counter < n_episodes_eval:
    state = env.reset()
    state = np.atleast_2d(state)
    reward_ep = []

    # _, intensity = env.get_pos_at_bpm_target(env.mssb_angle, env.mbb_angle)
    # if env.get_reward(intensity) > -15:
    #     continue
    # print('init reward', env.get_reward(intensity))
    #
    # reward_ep.append(env.get_reward(intensity))
    reward_ep.append(reward_scale * env.compute_reward(state, goal=None,
                                                       info={}))
    n_steps_eps = 0
    while True:
        a = agent.get_action(state, noise=0)
        a = np.squeeze(a)
        state, reward, done, _ = env.step(a)
        reward *= reward_scale
        reward_ep.append(reward)
        if done or n_steps_eps > max_episode_length:
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

axs[1].plot(init_rew, c='r')
axs[1].plot(final_rew, c='g')
axs[0].plot(length)
plt.show()
