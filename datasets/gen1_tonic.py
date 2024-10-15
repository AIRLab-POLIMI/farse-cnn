import os
import tonic
import json
import numpy as np
from tqdm import tqdm
from utils.io.psee_loader import PSEELoader

class Gen1(tonic.dataset.Dataset):
    def __init__(self, root, window_size=0.05, transform=None, target_transform=None, valid_idx_path=None):
        super(Gen1, self).__init__(root, transform=transform, target_transform=target_transform)
        self.location_on_system = root

        self.classes = ['car', "pedestrian"]

        self.splits = self.split_by_bbox()
        self.window_size = window_size * 1e6 # expects window_size in seconds, converts in microseconds

        if valid_idx_path:
            if os.path.exists(valid_idx_path):
                with open(valid_idx_path, "r") as fp:
                    deserialized_data = json.load(fp)

                valid_idx = deserialized_data["valid_idx"]
            else:
                valid_idx = self.filter_invalid_samples()
                data_to_serialize = {"valid_idx": valid_idx, "window_size":self.window_size}

                with open(valid_idx_path, "w") as fp:
                    json.dump(data_to_serialize, fp)
        else:
            print("No valid_idx_path provided, valid indices will not be stored.")
            valid_idx = self.filter_invalid_samples()

        self.splits = [self.splits[i] for i in valid_idx]



    def split_by_bbox(self):
        bbox_path = os.path.join(self.location_on_system, 'bbox_files')
        bbox_files = sorted(os.listdir(bbox_path))
        dat_path = os.path.join(self.location_on_system, 'dat_files')
        dat_files = sorted(os.listdir(dat_path))

        splits = []

        for bbox_file, dat_file in zip(bbox_files, dat_files):
            bbox = np.load(os.path.join(bbox_path, bbox_file))
            unique_ts, unique_indices = np.unique(bbox['ts'], return_index=True)
            unique_indices = np.append(unique_indices, bbox.shape[0])

            for i, ts in enumerate(unique_ts):
                target = bbox[unique_indices[i]:unique_indices[i + 1]][['x', 'y', 'w', 'h', 'class_id']]
                target = list(map(tuple, target))  # convert to list of tuples
                splits.append({'path': os.path.join(dat_path,dat_file),
                               'ts': ts,
                               'target': target})

        return splits

    def filter_invalid_samples(self):
        """
        Removes splits with empty sequences of events, and sequences with unordered timestamps
        """
        invalid_idx = []

        for i in tqdm(range(len(self.splits))):
            ev, _ = self.__getitem__(i)
            if ev.shape[0] == 0:
                invalid_idx.append(i)
            t = ev['t']
            if (t != sorted(t)).any():
                invalid_idx.append(i)

        return [i for i in range(len(self.splits)) if i not in invalid_idx]



    def __getitem__(self, idx):
        """
        returns events and label
        :param idx:
        :return: t,x,y,p,  label
        """
        split_data = self.splits[idx]

        f = split_data['path']
        loader = PSEELoader(f)

        # load a window of length window_size before the bounding box timestamp
        loader.seek_time(split_data['ts'] - self.window_size)
        events = loader.load_delta_t(self.window_size + 1) # increment by 1 so the exact timestamp is loaded too

        target = split_data['target']

        if self.transform:
            events = self.transform(events)

        return events, target


    def __len__(self):
        return len(self.splits)

    @property
    def num_classes(self):
        return len(self.classes)