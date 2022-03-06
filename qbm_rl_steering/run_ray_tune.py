import time

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.optuna import OptunaSearch

from qbm_rl_steering.core.ddpg_agents import QuantumDDPG
from qbm_rl_steering.core.run_utils import trainer, evaluator

import numpy as np

from tensorflow.keras.optimizers.schedules import PolynomialDecay, PiecewiseConstantDecay
from qbm_rl_steering.environment.orig_awake_env import e_trajectory_simENV


def get_val(params):

    # Learning rate schedules: lr_critic = 5e-4, lr_actor = 1e-4
    lr_schedule_critic = PolynomialDecay(params['lr_critic/init'],
                                         params['n_steps'],
                                         end_learning_rate=params['lr/final'])
    lr_schedule_actor = PolynomialDecay(params['lr_actor/init'],
                                        params['n_steps'],
                                        end_learning_rate=params['lr/final'])

    env = e_trajectory_simENV()

    agentMy = QuantumDDPG(state_space=env.observation_space,
                          action_space=env.action_space,
                          learning_rate_schedule_critic=lr_schedule_critic,
                          learning_rate_schedule_actor=lr_schedule_actor,
                          grad_clip_actor=1e4, grad_clip_critic=1.,
                          # grad_clip_actor=np.inf, grad_clip_critic=np.inf,
                          gamma=params['agent/gamma'],
                          tau_critic=params['agent/tau_critic'],
                          tau_actor=params['agent/tau_actor'],
                          )

    # Action noise schedule
    action_noise_schedule = PolynomialDecay(
        params['action_noise/init'], params['n_steps'],
        params['action_noise/final'])

    # Epsilon greedy schedule
    epsilon_greedy_schedule = PolynomialDecay(
        params['epsilon_greedy/init'], params['n_steps'],
        params['epsilon_greedy/final'])

    # Schedule n_anneals
    t_transition = [int(x * params['n_steps']) for x in
                    np.linspace(0, 1., params['anneals/n_pieces'] + 1)][1:-1]
    y_transition = [int(n) for n in np.linspace(params['anneals/init'],
                                                params['anneals/final'],
                                                params['anneals/n_pieces'])]
    n_anneals_schedule = PiecewiseConstantDecay(t_transition, y_transition)

    # AGENT TRAINING
    trainer(
        env=env, agent=agentMy, action_noise_schedule=action_noise_schedule,
        epsilon_greedy_schedule=epsilon_greedy_schedule,
        n_anneals_schedule=n_anneals_schedule, n_steps=params['n_steps'],
        max_steps_per_episode=params['env/max_steps_per_episode'],
        batch_size=params['trainer/batch_size'],
        n_exploration_steps=params['trainer/n_exploration_steps'],
        n_episodes_early_stopping=params['trainer/n_episodes_early_stopping']
    )

    env = e_trajectory_simENV()
    eval_log_scan = evaluator(env, agentMy, n_episodes=200, reward_scan=False)  # reward_scan=True
    n_tot_eval_steps = np.sum(np.array([(len(r) - 1) for r in eval_log_scan]))

    return n_tot_eval_steps


def objective(config):
    # Hyperparameters
    params = {
        'quantum_ddpg': True,  # False
        'n_steps': 260,  # 500
        'env/n_dims': 10,
        'env/max_steps_per_episode': config["max_steps"],  # 50,
        'env/required_steps_above_reward_threshold': 1,
        'trainer/batch_size': 32,  # 32
        'trainer/n_exploration_steps': 250,  # 150    400: works well, too...  , 500,  # 100,
        'trainer/n_episodes_early_stopping': 20,
        'agent/gamma': 0.99,
        'agent/tau_critic': 0.01,  # 0.001,
        'agent/tau_actor': 0.01,  # 0.001,
        'lr_critic/init': 1e-3,  # 1e-3
        'lr_critic/decay_factor': 1.,
        'lr_actor/init': 1e-3,
        'lr_actor/decay_factor': 1.,
        'lr/final': 5e-5,  # 5e-5,
        'action_noise/init': 0.2,  # 0.2,
        'action_noise/final': 0.,
        'epsilon_greedy/init': 0.,  # 0.1
        'epsilon_greedy/final': 0.,
        'anneals/n_pieces': 2,
        'anneals/init': 1,
        'anneals/final': 1,
    }

    val = 0
    for i in range(2):
        val += get_val(params)

    tune.report(n_total_steps_eval=val)
    time.sleep(0.1)


ray.init(configure_logging=False)

algo = OptunaSearch()
algo = ConcurrencyLimiter(algo, max_concurrent=4)
scheduler = AsyncHyperBandScheduler()
analysis = tune.run(
    objective,
    metric="n_total_steps_eval",
    mode="min",
    search_alg=algo,
    scheduler=scheduler,
    num_samples=5,
    config={
        "max_steps": tune.randint(5, 12)
    },
)

print("Best hyperparameters found were: ", analysis.best_config)
