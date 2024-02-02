from itertools import combinations

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from scripts.seg_grazpedwri_dataset import LightSegGrazPedWriDataset


ds = LightSegGrazPedWriDataset('train')

# all possible combinations of labels
n_labels = len(ds.BONE_LABEL)
lbl_combinations = list(combinations(range(n_labels), 2))
lbl_collision_matrix = np.zeros((n_labels, n_labels), dtype=bool)

for _, y, _ in tqdm(ds, unit='img', desc=f'Searching for overlapping labels'):
    lbl = y.bool()
    # check for overlapping labels
    for i, j in lbl_combinations:
        if (lbl[i] & lbl[j]).any():
            lbl_collision_matrix[i, j] = True
            lbl_collision_matrix[j, i] = True

# calculate disjunction of overlapping labels
G = nx.from_numpy_array(lbl_collision_matrix)
uncovered_nodes = set(range(n_labels))
independent_sets = []
while len(uncovered_nodes) > 0:
    # find maximal independent set
    ind_set = nx.approximation.maximum_independent_set(G.subgraph(uncovered_nodes))
    independent_sets.append(ind_set)
    uncovered_nodes -= ind_set

# add all independent nodes to each independent set
disjuncted_splits = list(map(lambda ind_set: nx.maximal_independent_set(G, ind_set), independent_sets))
print('possible disjuncted splits:')
for split in disjuncted_splits:
    print(sorted(split))
    print(sorted([LightSegGrazPedWriDataset.BONE_LABEL[i] for i in split]), '\n')

# plot graph
nx.draw(G, with_labels=True)
plt.show()
