import seaborn as sns
import pandas as pd
from pathlib import Path

sns.set_theme(style='ticks')
# load data by concatenating the csv files
csv_path = Path('evaluation/csv_results')
csv_files = list(csv_path.glob('*.csv'))
df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

sns.boxplot(data=df, x='num_train', y='dsc', hue='method', fill=False, linewidth=1.5, legend=False)
sns.pointplot(data=df, x='num_train', y='dsc', hue='method', errorbar=None)