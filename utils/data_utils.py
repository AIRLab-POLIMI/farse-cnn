import numpy as np
import tonic
from torch.utils.data import Dataset, Subset, ConcatDataset
from datasets.dataset_windows_tonic import DatasetWindows
from datasets.ncaltech101_tonic import NCaltech101
from datasets.ncars_tonic import NCars
from datasets.gen1_tonic import Gen1

def get_dataset(dataset_name, val_split=False, train_transform=None, val_transform=None, test_transform=None):
    """
    Returns instances of train, validation and test datasets of the supported datasets, specified by name.
    """
    if dataset_name == 'DVSGesture':
        # Event format: <xypt>
        train_dataset = tonic.datasets.DVSGesture(save_to='data', train=True)
        val_dataset = train_dataset
        test_dataset = tonic.datasets.DVSGesture(save_to='data', train=False)
    elif dataset_name == 'DVSGesture100ms':
        # Event format: <xypt>
        dvsg_train = tonic.datasets.DVSGesture(save_to='data', train=True)
        dvsg_test = tonic.datasets.DVSGesture(save_to='data', train=False)
        if val_split:
            train_dataset = DatasetWindows(dvsg_train, split_path="data/dvsgesture-split-train-w01-s005-subset08.json", transform=train_transform)
            val_dataset = DatasetWindows(dvsg_train, split_path="data/dvsgesture-split-train-w01-s005-subset02.json", transform=val_transform)
        else:
            train_dataset = DatasetWindows(dvsg_train, split_path="data/dvsgesture-split-train-w01-s005.json", transform=train_transform)
            val_dataset = None
        test_dataset = DatasetWindows(dvsg_test, split_path="data/dvsgesture-split-test-w01-s005.json", transform=test_transform)
    elif dataset_name == 'NCaltech101':
        # Event format: <xytp>
        train_dataset = NCaltech101(root='data/N-Caltech101/training', transform=train_transform)
        if val_split:
            val_dataset = NCaltech101(root='data/N-Caltech101/validation', transform=val_transform)
        else:
            ncaltech_val = NCaltech101(root='data/N-Caltech101/validation', transform=train_transform) # train_transform!
            train_dataset = ConcatDataset([train_dataset,ncaltech_val])
            val_dataset = None
        test_dataset = NCaltech101(root='data/N-Caltech101/testing', transform=test_transform)
    elif dataset_name == 'NCaltech101_100ms':
        # Event format: <xytp>
        train_dataset = DatasetWindows(NCaltech101(root='data/N-Caltech101/training'),
                                       split_path='data/ncaltech101-split-train-w01.json', transform=train_transform)
        if val_split:
            val_dataset = DatasetWindows(NCaltech101(root='data/N-Caltech101/validation'),
                                         split_path='data/ncaltech101-split-val-w01.json', transform=val_transform)
        else:
            ncaltech_val = DatasetWindows(NCaltech101(root='data/N-Caltech101/validation'),
                                                split_path='data/ncaltech101-split-val-w01.json', transform=train_transform) # train_transform!
            train_dataset = ConcatDataset([train_dataset, ncaltech_val])
            val_dataset = None
        test_dataset = DatasetWindows(NCaltech101(root='data/N-Caltech101/testing'),
                                            split_path='data/ncaltech101-split-test-w01.json', transform=test_transform)
    elif dataset_name == 'NCars':
        # Event format: <txyp>
        train_dataset = NCars(root='data/ncars_original/train', transform=train_transform)
        if val_split:
            ncars_val = NCars(root='data/ncars_original/train', transform=val_transform)
            train_dataset, val_dataset = random_split_disjoint(train_dataset, ncars_val, 0.8)
        else:
            val_dataset = None
        test_dataset = NCars(root='data/ncars_original/test', transform=test_transform)
    elif dataset_name == 'Gen1':
        # Event format: <txyp>
        train_dataset = Gen1('data/gen1/detection_dataset_duration_60s_ratio_1.0/train', transform=train_transform,
                             window_size=0.1, valid_idx_path='data/gen1/idx_train_01.json')
        if val_split:
            val_dataset = Gen1('data/gen1/detection_dataset_duration_60s_ratio_1.0/val', transform=val_transform,
                               window_size=0.1, valid_idx_path='data/gen1/idx_val_01.json')
        else:
            val_dataset = Gen1('data/gen1/detection_dataset_duration_60s_ratio_1.0/val', transform=train_transform,
                               window_size=0.1, valid_idx_path='data/gen1/idx_val_01.json')
            train_dataset = ConcatDataset([train_dataset, val_dataset])
            val_dataset = None
        test_dataset = Gen1('data/gen1/detection_dataset_duration_60s_ratio_1.0/test', transform=test_transform,
                            window_size=0.1, valid_idx_path='data/gen1/idx_test_01.json')
    else:
        raise ValueError(f'Dataset %s is not supported.' % dataset_name)
    return train_dataset, val_dataset, test_dataset


def random_split_disjoint(d1: Dataset, d2: Dataset, split_size):
    """
    Splits the two datasets into disjoint indices subsets.
    :param d1: first dataset
    :param d2: second dataset
    :param split_size: size of the split for the first dataset, as a percentage between 0 and 1
    """
    if split_size < 0 or split_size > 1:
        raise ValueError()

    split_len = int(split_size * len(d1))
    idx = np.arange(len(d1))
    idx_sub1 = np.random.choice(idx, split_len, replace=False)
    idx_sub2 = np.setdiff1d(idx, idx_sub1)

    return Subset(d1, idx_sub1), Subset(d2, idx_sub2)
