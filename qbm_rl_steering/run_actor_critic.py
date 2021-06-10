import numpy as np

from qbm_rl_steering.core.visualization import plot_log
from qbm_rl_steering.core.qac import QuantumActorCritic
# from qbm_rl_steering.environment.target_steering_1d import TargetSteeringEnv

from cern_awake_env.simulation import SimulationEnv

try:
    import matplotlib

    matplotlib.use('qt5agg')
except ImportError as err:
    print(err)


gamma_rl = 0.85
n_epochs = 16
max_episode_length = 10
initial_exploration_steps = 10
initial_reward = 0

# env = TargetSteeringEnv(max_steps_per_episode=max_episode_length)
env = SimulationEnv(plane='H', remove_singular_devices=True)
agent = QuantumActorCritic(env, gamma_rl=gamma_rl)

state, reward, done, ep_rew, ep_len, ep_cnt = (
    env.reset(), initial_reward, False, [[]], 0, 0)

# Calculate reward in current state
_, intensity = env.get_pos_at_bpm_target(env.mssb_angle)
ep_rew[-1].append(env.get_reward(intensity))

total_steps = max_episode_length * n_epochs

# Main loop: collect experience in env and update/log each epoch
to_exploitation = False
for t in range(total_steps):
    if t > initial_exploration_steps:
        action = agent.get_action(state, episode=1)
        action = np.squeeze(action)
    else:
        action = env.action_space.sample()

    # Step the env
    next_state, reward, done, _ = env.step(action)
    ep_rew[-1].append(reward)  # keep adding to the last element till done
    ep_len += 1

    done = False if ep_len == max_episode_length else done

    # Store experience to replay buffer
    agent.replay_memory.store(state, action, reward, next_state, done)

    state = next_state

    if done or (ep_len == max_episode_length):
        ep_cnt += 1
        if True:
            print(f"Episode: {len(ep_rew) - 1}, Reward: {ep_rew[-1][-1]}, "
                  f"Length: {len(ep_rew[-1])}")
        ep_rew.append([])

        for _ in range(ep_len):
            agent.train()

        state, reward, done, ep_ret, ep_len = (
            env.reset(), initial_reward, False, 0, 0)

        _, intensity = env.get_pos_at_bpm_target(env.mssb_angle)
        ep_rew[-1].append(env.get_reward(intensity))

init_rewards = []
rewards = []
reward_lengths = []
for episode in ep_rew[:-1]:
    if (len(episode) > 0):
        rewards.append(episode[-1])
        init_rewards.append(episode[0])
        reward_lengths.append(len(episode) - 1)
print('Total number of interactions:', np.sum(reward_lengths))

plot_log(env, fig_title='Training')

# Agent evaluation
n_episodes_eval = 50
episode_counter = 0
env = TargetSteeringEnv(max_steps_per_episode=max_episode_length)
while episode_counter < n_episodes_eval:
    state = env.reset(init_outside_threshold=True)
    state = np.atleast_2d(state)
    while True:
        a = agent.get_action(state, noise=0)
        a = np.squeeze(a)
        state, reward, done, _ = env.step(a)
        if done:
            episode_counter += 1
            break

plot_log(env, fig_title='Evaluation')
