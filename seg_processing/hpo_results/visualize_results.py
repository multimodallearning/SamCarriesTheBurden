from pathlib import Path

import matplotlib.pyplot as plt
import optuna.visualization.matplotlib as vis
from joblib import load

model_id = '2bd2f4be80b9446286416993ba6a87c1'
hpo_study = ['hpo_rnd_wlk_refine', 'grid_search_sam_refine'][0]
study_file = Path('seg_processing/hpo_results') / model_id / (hpo_study + '.pkl')

# Load the study
study = load(study_file)
print(study.best_params, study.best_value)
df = study.trials_dataframe().sort_values('value', ascending=False)

# Visualize
vis.plot_optimization_history(study)
plt.tight_layout()

vis.plot_contour(study)
plt.tight_layout()

vis.plot_slice(study)
plt.tight_layout()

plt.show()
