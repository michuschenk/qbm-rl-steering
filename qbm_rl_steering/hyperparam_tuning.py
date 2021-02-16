import optuna

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from qbm_rl_steering.environment.env_desc import TargetSteeringEnv


def optimize_dqn(trial):
    """
    Learning hyperparameters we want to optimise
    """
    return {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 5e-3),
        'target_update_interval': trial.suggest_int(
            'target_update_interval', 10, 10000),
        'tau': trial.suggest_uniform('tau', 0., 1.),
        'exploration_fraction': trial.suggest_uniform(
            'exploration_fraction', 0.4, 1.),
        'train_freq': trial.suggest_int('train_freq', 1, 20),
        'exploration_final_eps': trial.suggest_uniform(
            'exploration_final_eps', 0, 0.1)
    }


def optimize_agent(trial):
    """
    Train the model and optimize
    Optuna maximises the negative log likelihood, so we need to negate the
    reward here
    """
    dqn_kwargs = optimize_dqn(trial)
    env = make_vec_env(lambda: TargetSteeringEnv(), n_envs=1, seed=0)

    model = DQN('MlpPolicy', env=env, verbose=0, learning_starts=0,
                exploration_initial_eps=1.0, policy_kwargs=dict(net_arch=[8]*2),
                **dqn_kwargs)

    model.learn(20000)
    mean_reward, _ = evaluate_policy(model, TargetSteeringEnv(),
                                     n_eval_episodes=40)
    return -1 * mean_reward


if __name__ == '__main__':
    study = optuna.create_study()
    try:
        study.optimize(optimize_agent, n_trials=500, n_jobs=1)
    except KeyboardInterrupt:
        print('Interrupted by keyboard.')
