import optuna
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

from qbm_rl_steering.run_ddpg_func import get_val

def get_objective(parent_run_id):
    # get an objective function for optuna that creates nested MLFlow runs
    def objective(trial):
        trial_run = client.create_run(
            experiment_id=experiment,
            tags={
                MLFLOW_PARENT_RUN_ID: parent_run_id
            }
        )
        # x = trial.suggest_float("x", -10.0, 10.0)
        max_steps = trial.suggest_int("max_steps", 5, 60, step=3)
        client.log_param(trial_run.info.run_id, "max_steps", max_steps)

        params = {
            'quantum_ddpg': True,  # False
            'n_steps': 260,  #      500
            'env/n_dims': 10,
            'env/max_steps_per_episode': max_steps,  # 50,
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

        val = get_val(params)
        client.log_metric(trial_run.info.run_id, "n_steps_eval", val)
        return val
    
    return objective

client = MlflowClient()
experiment_name = "min_steps_eval"

try:
    experiment = client.create_experiment(experiment_name)
except:
    experiment = client.get_experiment_by_name(experiment_name).experiment_id

study_run = client.create_run(experiment_id=experiment)
study_run_id = study_run.info.run_id
study = optuna.create_study(direction="minimize")
study.optimize(get_objective(study_run_id), n_trials=2, n_jobs=2)

client.log_param(study_run_id, "best_max_steps", study.best_trial.params["max_steps"])
client.log_metric(study_run_id, "best_n_steps_eval", study.best_value)

