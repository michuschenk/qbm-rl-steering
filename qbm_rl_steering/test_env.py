from cern_awake_env.simulation import SimulationEnv
from qbm_rl_steering.environment.rms_env_nd import RmsSteeringEnv
from importlib.resources import open_text as _open_text


env = RmsSteeringEnv(n_dims=3)

# env = SimulationEnv(plane='H',
#                     remove_singular_devices=True)

# env.reset(init_outside_threshold=True)
env.reset()

rews = []
states = []
acts = []
dones = []
for i in range(10):
    act = env.action_space.sample()
    state, rew, done, _ = env.step(act)
    print('act', act)
    print('state', state)
    print('rew', rew)

    # env.render()
    # dones.append(done)
    # rews.append(rew)
    # states.append(state)
    # acts.append(act)
    if done:
        print('stopped at step', i)
        break
