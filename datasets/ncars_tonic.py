import os
import tonic
import numpy as np
from utils.io.psee_loader import PSEELoader

class NCars(tonic.dataset.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        super(NCars, self).__init__(save_to=root, transform=transform, target_transform=target_transform)
        self.classes = sorted(os.listdir(root))

        self.data = []
        self.targets = []

        for i, c in enumerate(self.classes):
            new_files = [os.path.join(root, c, f) for f in os.listdir(os.path.join(root, c))]
            self.data += new_files
            self.targets += [i] * len(new_files)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        returns events and label
        :param idx:
        :return: t,x,y,p,  label
        """
        label = self.targets[idx]
        f = self.data[idx]

        loader = PSEELoader(f)
        events = loader.load_n_events(loader.event_count())

        if self.transform is not None:
            events = self.transform(events)

        return events, label

    @property
    def num_classes(self):
        return len(self.classes)