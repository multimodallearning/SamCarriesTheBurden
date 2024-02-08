from joblib import load
import matplotlib.pyplot as plt
import optuna.visualization.matplotlib as optuna_vis

model_id = '0427c1de20c140c5bff7284c7a4ae614'
study = load(f"seg_preprocessing/hpo_seg_proc_{model_id}.pkl")
print(f'{study.best_params=} with {study.best_value=} DSC improvement')

optuna_vis.plot_optimization_history(study)
plt.tight_layout()

optuna_vis.plot_parallel_coordinate(study)
plt.tight_layout()

optuna_vis.plot_contour(study)

optuna_vis.plot_slice(study)

optuna_vis.plot_param_importances(study)

plt.tight_layout()
plt.show()
