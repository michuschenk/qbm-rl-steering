from sklearn.ensemble import ExtraTreesRegressor
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'text.usetex': True,
                     'text.latex.preamble': r'\usepackage{amsmath}',
                     'font.size': 15})

from ray.tune import ExperimentAnalysis

feature_map = {
    'config/beta': r'$\beta$',
    'config/big_gamma_i': r'$\Gamma_i$',
    'config/target_upd_freq': r'$N_\text{target}$',
    'config/lr_i': r'$\lambda_i$',
    'config/lr_f': r'$\lambda_f$',
    'config/n_annealing_steps': r'$N_\text{steps, anneal}$',
    'config/batch_size': r'$N_\text{batch}$',
    'config/gamma': r'$\gamma$',
    'config/n_trotter': r'$N_\text{Trotter}$',
    'config/n_meas_avg': r'$N_\text{shots}$',
    'config/max_steps': r'$N_\text{it, episode}$',
    'config/epsilon_fraction': r'$r_\varepsilon$',
    'config/epsilon_i': r'$\varepsilon_i$',
    'config/tau': r'$\tau$'
}


df = pd.read_csv('first_it_cont_state.csv')
train_feat_names = [feat for feat in
                    df.keys() if feat.startswith('config/')]
X = df[train_feat_names]
y = df['val_metric']

model = ExtraTreesRegressor()
model.fit(X, y)

# idx = [feat.split('config/')[-1] for feat in X.columns]
feat_names = [feature_map[col_name] for col_name in X.columns]
feat_imp = pd.Series(model.feature_importances_, index=feat_names)

plt.figure(figsize=(7.5,5))
feat_imp.nlargest(10).plot(kind='barh')
plt.gca().invert_yaxis()
plt.gca().set_xlabel('Relative importance')
plt.tight_layout()
plt.savefig('feat_import_cont_state.pdf')
plt.show()

"""
...

n_pcs = 5
model = PCA(n_components=n_pcs).fit(X)
X_pc = model.transform(X)

most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]
most_important_names = [train_feat_names[most_important[i]] for i in range(n_pcs)]

dic = {f'PC{i}': most_important_names[i] for i in range(n_pcs)}

df_pc = pd.DataFrame(dic.items())
"""


