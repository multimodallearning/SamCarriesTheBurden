# messy script to plot the results of the evaluation

import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

sns.set_theme(style='ticks', font_scale=1.2)

# load data by concatenating the csv files
csv_path = Path('evaluation/csv_results/wrist')
csv_files = list(csv_path.glob('*.csv'))
df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
# drop case with unrealistic rotation
df = df[df['File stem'] != '0172_0304693626_01_WRI-R1_F014']

# rename methods
df['Method'] = df['Method'].replace({
    'nnUNet': 'nnU-Net BCE',
    'SAM_LRASPP': 'SAM ViT + LRASPP',
    'UNet_raw': 'U-Net',
    'UNet_SAM': 'U-Net + SAM',
    'UNet_MedSAM': 'U-Net + MedSAM',
    'UNet_RndWalk': 'U-Net + Random Walk',
    'UNet_pseudo_lbl_raw': 'ours w/o SAM',
    'UNet_pseudo_lbl_sam': 'ours w/ SAM',
    'UNet_mean_teacher': 'Mean Teacher'
})

method_order = [
    'nnUNet BCE',
    'SAM ViT + LRASPP',
    'U-Net',
    'U-Net + SAM',
    'U-Net + MedSAM',
    'U-Net + Random Walk',
    'ours w/o SAM',
    'ours w/ SAM',
    'Mean Teacher'
]

# results table
df_mean = df.groupby(['Method', 'Number training samples'])['DSC mean'].mean().unstack() * 100
df_std = df.groupby(['Method', 'Number training samples'])['DSC mean'].std().unstack() * 100
df_table = '$' + df_mean.round(2).astype(str) + ' \pm ' + df_std.round(2).astype(str) + '$'
# drop number training samples
df_table = df_table[[5, 20, 43]]
df_table = df_table.reindex(method_order)
print(df_table.to_latex(escape=False))

# collapse all dice scores into a single column
df_distribution = df.melt(id_vars=['Method', 'Number training samples', 'File stem'], value_vars=df.columns[4:].values)
df_distribution = df_distribution.dropna()
# create new dataframe with method, number training samples and dice scores as list
df_distribution = df_distribution.groupby(['Method', 'Number training samples'])['value'].apply(list).reset_index()

method_selection = ['nnU-Net BCE', 'SAM ViT + LRASPP', 'ours w/ SAM', 'Mean Teacher']
# plt.figure()
# for method in method_selection:
#     data = df_distribution[
#         (df_distribution['Method'] == method) & (df_distribution['Number training samples'] == 43)].value.values[0]
#     data = np.array(data) * 100
#     plt.plot(np.linspace(0, 1, len(data)), np.sort(data), label=method)
#     plt.ylabel('Dice')
#     plt.xlabel('Cumulative distribution')
#     plt.legend()

# save figure
# plt.savefig(f'/home/ron/Desktop/plot_curves/cdf_dice {' '.join(method_selection)}.pdf', bbox_inches='tight',
#             pad_inches=0)

df_melted = df.melt(id_vars=['Method', 'Number training samples', 'File stem'], value_vars=df.columns[4:].values)
df_melted = df_melted.rename(columns={'variable': 'Anatomy', 'value': 'Dice'})
df_melted['Dice'] *= 100
df_melted = df_melted.groupby(['Method', 'Number training samples', 'File stem'])['Dice'].mean().reset_index()
df_selected_methods = pd.concat([df_melted[df_melted['Method'] == method] for method in method_selection],
                                ignore_index=True)

# plt.figure()
# sns.barplot(data=df_selected_methods, x='File stem', y='DSC', hue='Method')
# # rotate x-axis labels
# plt.xticks(rotation=-90)

# filter to number of training samples 5, 15, 25, 35
df_selected_methods_few_samples = df_selected_methods[
    df_selected_methods['Number training samples'].isin([5, 15, 25, 35])]

plt.figure(figsize=(10, 4))
sns.pointplot(data=df_selected_methods, x='Number training samples', y='Dice', hue='Method', alpha=0, legend=False)
boxplot_alpha = 0.7
sns.boxplot(data=df_selected_methods_few_samples, x='Number training samples', y='Dice', hue='Method', fill=False,
            linewidth=1.5, legend=False, notch=False, boxprops=dict(alpha=boxplot_alpha), whiskerprops=dict(alpha=boxplot_alpha),
            flierprops=dict(alpha=boxplot_alpha, marker='x'))
ax = sns.pointplot(data=df_selected_methods, x='Number training samples', y='Dice', hue='Method', errorbar=None, dodge=True)
sns.move_legend(ax, "lower center", bbox_to_anchor=(.475, 1), ncol=4, title=None, frameon=False)
plt.ylabel('Dice Score [%]')
plt.xlabel('n (number of training samples)')
plt.savefig('/home/ron/Desktop/lineplot_dice.pdf', bbox_inches='tight',
            pad_inches=0)

# plt.figure()
# sns.pointplot(data=df_selected_methods, x='Number training samples', y='DSC', hue='Method', dodge=.4)
#
#
# plt.figure()
# sns.pointplot(data=df_selected_methods, x='Number training samples', y='DSC', hue='Method', dodge=.4)

plt.figure()
sns.lineplot(data=df_selected_methods, x='Number training samples', y='Dice', hue='Method', errorbar='ci')
# removing title of the legend
plt.legend(title=None)

plt.savefig(f'/home/ron/Desktop/lineplot_dice {' '.join(method_selection)}.pdf', bbox_inches='tight',
            pad_inches=0)

plt.show()
