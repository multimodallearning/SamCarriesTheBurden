# helper class to load nnUNet predictions

import numpy as np
import torch
from pathlib import Path


class NNUNetPredictionLoader:
    def __init__(self, num_train_samples: int = 43):
        self.path2predictions = Path(f'data/bce_nnunet_predictions/result_{100 + num_train_samples}')
        self.available_files = list(map(lambda f: f.stem, self.path2predictions.glob('*.npz')))

    def __len__(self):
        return len(self.available_files)

    def __getitem__(self, file_name: str) -> torch.Tensor:
        assert file_name in self.available_files, f'File {file_name} not found in {self.path2predictions}'
        p_hat = np.load(self.path2predictions.joinpath(file_name).with_suffix('.npz'))['probabilities']
        return torch.from_numpy(p_hat).squeeze()


if __name__ == '__main__':
    loader = NNUNetPredictionLoader()
    print(loader.available_files)
    p = loader[loader.available_files[0]]
    pass
