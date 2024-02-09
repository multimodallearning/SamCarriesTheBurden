from joblib import load
import matplotlib.pyplot as plt
import optuna.visualization.matplotlib as optuna_vis

model_id = '0427c1de20c140c5bff7284c7a4ae614'
study_type = [
    'grid_search_seg_proc',
    'hpo_sam_refinement',
    'grid_search_sam_refinement'
][-1]

study = load(f"seg_preprocessing/{study_type}_{model_id}.pkl")
print(f'{study.best_params=} with {study.best_value=} DSC improvement')
print(study.sampler)
df = study.trials_dataframe()
if 'value_dsc_diff_score' in df.columns:
    df['value'] = df['value_dsc_diff_score']
    df = df.drop(columns='value_dsc_diff_score')
df = df.sort_values('value', ascending=False)
print(df.filter(regex='params*|value').head(10))

optuna_vis.plot_optimization_history(study)
plt.tight_layout()

optuna_vis.plot_parallel_coordinate(study)
plt.tight_layout()

optuna_vis.plot_contour(study)

optuna_vis.plot_slice(study)

optuna_vis.plot_param_importances(study)

plt.tight_layout()
plt.show()
