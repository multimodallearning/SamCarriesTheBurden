import seaborn as sns
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

sns.set_theme(style='ticks')
# load data by concatenating the csv files
csv_path = Path('evaluation/csv_results')
csv_files = list(csv_path.glob('*.csv'))
df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
# drop case with unrealistic rotation
df = df[df['file_stem'] != '0172_0304693626_01_WRI-R1_F014']

plt.figure()
sns.barplot(data=df, x='file_stem', y='dsc', hue='method')
# rotate x-axis labels
plt.xticks(rotation=-90)

plt.figure()
sns.boxplot(data=df, x='num_train', y='dsc', hue='method', fill=False, linewidth=1.5)

plt.figure()
sns.pointplot(data=df, x='num_train', y='dsc', hue='method', errorbar=None)
plt.show()
