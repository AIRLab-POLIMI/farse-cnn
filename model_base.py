import torch
import numpy as np
import matplotlib.pyplot as plt
from lightning.pytorch.core.module import LightningModule
from lightning.pytorch.utilities import grad_norm
from utils.data_utils import get_dataset

class LitBaseModel(LightningModule):
    def __init__(self, bs=1, log_mode=None, dataset='DVSGesture', start_lr=1e-3, val_split=False, train_transforms=None, val_transforms=None, test_transforms=None, frame_size=None):
        super().__init__()
        self.bs = bs
        self.log_mode = log_mode  # can be: 'neptune'
        self.dataset = dataset
        self.start_lr = start_lr
        self.val_split = val_split
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms

        if 'DVSGesture' in self.dataset:
            self.frame_size = [128, 128]
            self.num_classes = 11
        elif 'NCaltech101' in self.dataset:
            self.frame_size = [180, 240]
            self.num_classes = 101
        elif 'NCars' in self.dataset:
            self.frame_size = [100, 120]
            self.num_classes = 2
        elif 'Gen1' in self.dataset:
            self.frame_size = [240, 304]
            self.num_classes = 2

        if frame_size:
            # override frame_size if explictly provided
            self.frame_size = frame_size

        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []


    def setup(self, stage):
        self.train_dataset, self.val_dataset, self.test_dataset = get_dataset(self.dataset, val_split=self.val_split,
                                                                              train_transform=self.train_transforms,val_transform=self.val_transforms,test_transform=self.test_transforms)
        self.classes = self.test_dataset.classes


    def train_dataloader(self):
        # Shape of each batch is (bs, max_length, 4) where max_length is the maximum number of events in the samples of the batch
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.bs, shuffle=True, num_workers=torch.get_num_threads(),
                                           collate_fn=self.pad_batches)

    def val_dataloader(self):
        # Shape of each batch is (bs, max_length, 4) where max_length is the maximum number of events in the samples of the batch
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.bs, shuffle=False, num_workers=torch.get_num_threads(),
                                           collate_fn=self.pad_batches)

    def test_dataloader(self):
        # Shape of each batch is (bs, max_length, 4) where max_length is the maximum number of events in the samples of the batch
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.bs, shuffle=False, num_workers=torch.get_num_threads(),
                                           collate_fn=self.pad_batches)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.start_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
        return {"optimizer":optimizer, "lr_scheduler": scheduler, "monitor":"val/loss"}

    def on_before_optimizer_step(self, optimizer):
        self.log_dict({'train/'+k:v for k,v in grad_norm(self, norm_type=2).items()})

    def log_metrics(self, metrics):
        #if self.log_mode == 'neptune':
        self.log_dict(metrics)

    def log_figures(self, figures, close_after_log=True):
        if self.log_mode == 'neptune':
            for name,figure in figures.items():
                self.logger.experiment[name].append(figure)
                if close_after_log:
                    plt.close(figure)


    def pad_batches(self, data):
        """
        data: is a list of len "batch_size" containing sample tuples (input,label)
        """
        events = [np.stack([sample[0][f].astype(np.float32) for f in sample[0].dtype.names], axis=-1) for sample in data]
        labels = [sample[1] for sample in data]
        events_lens = [len(ev) for ev in events]
        max_len = max(events_lens)
        events = [np.pad(ev, ((0, max_len - ln), (0, 0)), mode='constant', constant_values=0) for
                  ln, ev in zip(events_lens, events)]
        events = torch.as_tensor(np.stack(events, axis=0))

        if 'DVSGesture' in self.dataset:
            # permute to xytp
            events = events[:,:,[0,1,3,2]]
        elif 'NCars' in self.dataset or 'Gen1' in self.dataset:
            # permute to xytp
            events = events[:,:,[1,2,0,3]]

        # convert p to [-1,+1]
        events[:,:,-1] = events[:,:,-1] * 2 - 1

        events_lens = torch.as_tensor(events_lens)
        labels = torch.as_tensor(labels)
        batch = {"events": events, "labels": labels, "lengths": events_lens}
        return batch