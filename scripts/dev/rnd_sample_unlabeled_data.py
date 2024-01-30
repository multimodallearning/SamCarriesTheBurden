from utils.cvat_parser import CVATParser
from pathlib import Path
import random
import pandas as pd

n_samples = 500

xml_files = list(Path('data/cvat_annotation_xml').glob(f'annotations_*.xml'))
parser = CVATParser(xml_files, True, False, True)
annotated_files = parser.available_file_names

all_files = list(Path('data/img_only_front_all_left').glob('*.png'))
available_files = list(filter(lambda x: x.stem not in annotated_files, all_files))
assert len(set(available_files).intersection(set(annotated_files))) == 0, 'Files should be disjoint'

random.seed(42)
sampled_files = random.sample(available_files, n_samples)

# save sampled files to csv
df = pd.DataFrame({'filestem': list(map(lambda x: x.stem, sampled_files))})
df.to_csv(f'data/{n_samples}unlabeled_sample.csv', index=True)
