import numpy as np
import optuna
import joblib
import matplotlib.pyplot as plt


fname = 'paper/Qlearn_cont/dqn_optuna_study.pkl'
study = joblib.load(fname)

param_importance = optuna.importance.get_param_importances(study)
print(param_importance)
print('best_params', study.best_params)

plt.figure(figsize=(7, 5))
plt.suptitle('DQN: optuna parameter importance')
xlabel = []
x = []
y = []

for i, (key, val) in enumerate(param_importance.items()):
    xlabel.append(key)
    x.append(i)
    y.append(val)

x = np.array(x)
y = np.array(y)
xlabel = np.array(xlabel)

plt.bar(x, 100*y)
plt.xticks(x, xlabel, rotation=60)
plt.ylabel('Relative importance (%)')
plt.xlabel('Parameter')
plt.tight_layout()
plt.savefig('paper/Qlearn_cont/006_dqn_cont_optuna_study.pdf')
plt.show()
