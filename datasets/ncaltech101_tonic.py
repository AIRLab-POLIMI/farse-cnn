import os
import tonic
import numpy as np

class NCaltech101(tonic.dataset.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        super(NCaltech101, self).__init__(save_to=root, transform=transform, target_transform=target_transform)
        self.classes = sorted(os.listdir(root))

        self.data = []
        self.targets = []

        for i, c in enumerate(self.classes):
            new_files = [os.path.join(root, c, f) for f in sorted(os.listdir(os.path.join(root, c)))]
            self.data += new_files
            self.targets += [i] * len(new_files)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        label = self.targets[idx]
        f = self.data[idx]

        dtype = np.dtype([("x", np.float32), ("y", np.float32), ("t", np.float32), ("p", np.float32)])
        events = np.load(f).view(dtype=dtype).squeeze(axis=1)

        events['t'] = events['t'] * 1e6 # convert times from seconds to microseconds
        events['p'] = events['p']/2 + 0.5 # convert polarities to range (0,1)

        events = events.astype([('x', '<i4'), ('y', '<i4'), ('t', '<i4'), ('p', '<i4')])

        if self.transform is not None:
            events = self.transform(events)

        return events, label

    @property
    def num_classes(self):
        return len(self.classes)

