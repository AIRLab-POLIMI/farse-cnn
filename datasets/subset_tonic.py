import tonic
import numpy as np

class Subset(tonic.dataset.Dataset):
    def __init__(self, dataset, subset_value, subset_mode='percentage', subset_seed=None, exclude_subset=None):

        self.dataset = dataset

        if subset_mode == 'percentage' and subset_value > 0.0 and subset_value < 1.0:
            self.length = np.ceil(len(self.dataset) * subset_value).astype('int')
        elif subset_mode == 'count' and subset_value > 0:
            self.length = min(subset_value, len(self.dataset))
        else:
            raise ValueError()

        if subset_seed is not None:
            np.random.seed(subset_seed)

        idx = np.arange(len(self.dataset))
        if exclude_subset is not None:
            idx = np.setdiff1d(idx, exclude_subset)

        self.subset_idx = np.random.choice(idx, int(self.length), replace=False)


    def __getitem__(self, index):
        return self.dataset.__getitem__(self.subset_idx[index])

    def __len__(self):
        return self.length

    @property
    def classes(self):
        return self.dataset.classes

    @property
    def num_classes(self):
        return len(self.classes)