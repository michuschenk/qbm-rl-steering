from test_other_implementation import DDPGAgent
import numpy as np
import gym
# from qbm_rl_steering.environment.target_steering_1d import TargetSteeringEnv
from qbm_rl_steering.environment.target_steering_2d import TargetSteeringEnv2D
from cern_awake_env.simulation import SimulationEnv
import matplotlib.pyplot as plt
import matplotlib

def smooth(x):
  # last 100
  n = len(x)
  y = np.zeros(n)
  for i in range(n):
    start = max(0, i - 99)
    y[i] = float(x[start:(i+1)].sum()) / (i - start + 1)
  return y


def trainer(env, agent, max_episodes, max_steps, batch_size, action_noise,
            n_exploration_steps=30):
    episode_init_rewards = []
    episode_final_rewards = []
    episode_length = []
    total_step_count = 0

    # plt.ion()
    # plt.figure()
    # plt.show()

    for episode in range(max_episodes):
        state = env.reset()
        episode_init_rewards.append(env.get_reward(
            env.get_pos_at_bpm_target(env.mssb_angle, env.mbb_angle)[1]))

        for step in range(max_steps):
            if total_step_count < n_exploration_steps:
                action = env.action_space.sample()
            else:
                action = agent.get_action(state, action_noise)

            total_step_count += 1
            next_state, reward, done, _ = env.step(action)
            d_store = False if step == max_steps-1 else done
            agent.replay_buffer.push(state, action, reward, next_state, d_store)
            # if step == 0:
            #     episode_init_rewards.append(reward)

            if agent.replay_buffer.size > batch_size:
                agent.update(batch_size)

            if done or step == max_steps-1:
                episode_final_rewards.append(reward)
                print("*******************************")
                print("Episode " + str(episode) + ": init rew: " +
                      str(episode_init_rewards[-1]) + " .. final rew: " +
                      str(episode_final_rewards[-1]))
                print("*******************************\n")
                episode_length.append(step)
                # plt.plot(episode_init_rewards, 'r')
                # plt.plot(episode_final_rewards, 'g')
                # plt.draw()
                break

            state = next_state

    return episode_init_rewards, episode_final_rewards, episode_length


# env = gym.make("Pendulum-v0")
max_steps = 25  # 10
# env = TargetSteeringEnv(max_steps_per_episode=max_steps)
env = TargetSteeringEnv2D(max_steps_per_episode=max_steps)
# env = SimulationEnv(plane='H', remove_singular_devices=True,
#                     twissfile='electron_tt43_4d.madx.out')

max_episodes = 50  # 100
batch_size = 15  # 10
n_exploration_steps = 30

gamma = 0.95  # 0.85
tau_critic = 1.  # 5e-3
tau_actor = 1.
buffer_maxlen = 10000
critic_lr = 1e-3  # 1e-2  # 1e-3
actor_lr = 5e-4  # 5e-4

agent = DDPGAgent(env, gamma, tau_critic, tau_actor, buffer_maxlen, critic_lr,
                  actor_lr, use_qbm=True)
init_rewards, final_rewards, episode_length = trainer(
    env, agent, max_episodes, max_steps, batch_size, action_noise=0.1,
    n_exploration_steps=n_exploration_steps)

fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(episode_length)
axs[1].plot(init_rewards, c='r')
axs[1].plot(final_rewards, c='g')
plt.show()

plt.plot(np.array(agent.q_before), c='r', label='q before')
plt.plot(np.array(agent.q_after), c='g', label='q after')
plt.legend()
plt.show()

plt.plot(np.array(agent.q_after)-np.array(agent.q_before))
plt.ylabel('Qafter - Qbefore')
plt.show()


# Agent evaluation
n_episodes_eval = 80
episode_counter = 0

rewards_eval = []
# env = SimulationEnv(plane='H', remove_singular_devices=True,
#                     twissfile='electron_tt43_4d.madx.out')
env = TargetSteeringEnv2D(max_steps_per_episode=max_steps)
while episode_counter < n_episodes_eval:
    state = env.reset()
    state = np.atleast_2d(state)
    reward_ep = []
    # reward_ep.append(env.compute_reward(state, goal=None, info={}))
    reward_ep.append(env.get_reward(
        env.get_pos_at_bpm_target(env.mssb_angle, env.mbb_angle)[1]))
    n_steps_eps = 0
    while True:
        a = agent.get_action(state, noise_scale=0)
        a = np.squeeze(a)
        state, reward, done, _ = env.step(a)
        reward_ep.append(reward)
        if done or n_steps_eps > max_steps:
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

plt.figure()
plt.plot(agent.q_losses)
plt.ylabel('Q losses')
plt.show()


plt.figure()
plt.plot(agent.grads_mu_all_mean, label='mean')
plt.plot(agent.grads_mu_all_min, label='min')
plt.plot(agent.grads_mu_all_max, label='max')
plt.ylabel('Grads')
plt.legend()
plt.show()
