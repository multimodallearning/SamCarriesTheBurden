# messy script to plot the results of the evaluation

from pathlib import Path

import pandas as pd

# load data by concatenating the csv files
csv_path = Path('evaluation/csv_results/dental')
csv_files = list(csv_path.glob('*.csv'))
df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
# drop case with unrealistic rotation
df = df[df['File stem'] != '0172_0304693626_01_WRI-R1_F014']

# results table
df_mean = df.groupby(['Method'])['DSC mean'].mean() * 100
df_std = df.groupby(['Method'])['DSC mean'].std() * 100
df_table = df_mean.round(2).astype(str) + ' ± ' + df_std.round(2).astype(str)
# print(df_table.to_latex(escape=False))
print(df_table)
