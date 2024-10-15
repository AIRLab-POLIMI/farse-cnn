import os
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

class DatasetWindows(Dataset):
    def __init__(self, dataset, split_path=None,
                 split_windows_size=None, split_windows_stride=None,
                 split_exclude_last=True, split_mode="seconds",
                 transform=None):

        self.dataset = dataset

        # We must apply transforms separately
        self.transform = transform

        # Ensures the same ordering regardless of the OS
        self.dataset.data = np.array(self.dataset.data)
        self.dataset.targets = np.array(self.dataset.targets)
        idx_sorted = np.argsort(self.dataset.data)
        self.dataset.data = self.dataset.data[idx_sorted]
        self.dataset.targets = self.dataset.targets[idx_sorted]

        self.split_windows_size = split_windows_size
        self.split_windows_stride = split_windows_stride
        self.split_exclude_last = split_exclude_last
        self.split_mode = split_mode
        self.split_path = split_path

        if split_path:
            if os.path.exists(self.split_path):
                with open(self.split_path, "r") as fp:
                    deserialized_data = json.load(fp)

                self.splits = deserialized_data["splits"]
            else:
                self.splits = self._compute_temporal_windows()

                data_to_serialize = {"splits": self.splits}

                with open(self.split_path, "w") as fp:
                    json.dump(data_to_serialize, fp)
        else:
            self.splits = self._compute_temporal_windows()
            print("No split_path provided, splits will not be stored.")

    @property
    def classes(self):
        return self.dataset.classes

    @property
    def num_classes(self):
        return len(self.classes)

    def _compute_temporal_windows(self):
        return DatasetWindows.compute_windows_from_dataset(
            self.dataset, self.split_windows_size, self.split_windows_stride,
            None, self.split_mode, self.split_exclude_last)

    def __getitem__(self, item):
        split_data = self.splits[item]

        events, target = self.dataset.__getitem__(split_data['item'])

        idx_first = split_data['idx_first']
        idx_last = split_data['idx_last']

        events = events[idx_first:idx_last+1]

        if self.transform:
            events = self.transform(events)

        return events, target

    def __len__(self):
        return len(self.splits)

    '''
    Static function to compute temporal windows from an explicitly passed dataset
    '''
    @staticmethod
    def compute_windows_from_dataset(dataset, split_windows_size, split_windows_stride,
                                     item_idx=None, split_mode='seconds', split_exclude_last=True):
        splits = []
        if item_idx is None:
            item_idx = range(dataset.__len__())

        for item in tqdm(item_idx, desc="Computing splits"):
            events, target = dataset.__getitem__(item)

            num_events = events.shape[0]
            x, y, p, t = events['x'], events['y'], events['p'], events['t']

            if split_mode == 'count':
                indexing = np.arange(num_events)
                start_value = 0
                eps = 0
            elif split_mode == 'seconds':
                indexing = t / 1e6
                start_value = indexing[0]
                eps = 0.01  # 10ms
            else:
                raise ValueError()

            while True:
                end_value = start_value + split_windows_size
                inside_mask = np.logical_and(start_value <= indexing,
                                             indexing < end_value)
                inside_idx = indexing[inside_mask]

                # Check if there are events left
                if inside_idx.size == 0:
                    break

                # Check if the window is full
                inside_range = inside_idx[-1] - inside_idx[0] + eps
                if split_exclude_last and inside_range <= split_windows_size:
                    break

                idx_first = np.argmax(inside_mask)
                idx_last = num_events - 1 - np.argmax(inside_mask[::-1])

                splits.append({'item': int(item),
                               'target': int(target),
                               'idx_first': int(idx_first),
                               'idx_last': int(idx_last)
                               })


                if hasattr(dataset, 'data'):
                    splits[-1]['path'] = dataset.data[item]

                start_value += split_windows_stride

        return splits