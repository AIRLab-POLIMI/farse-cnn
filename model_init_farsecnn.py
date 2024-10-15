from model_farsecnn import LitFARSECNN
import torch
import numpy as np
from layers.SubmanifoldFARSEConv import SubmanifoldFARSEConv
from layers.BranchBlock import BranchBlock

class LitInitFARSECNN(LitFARSECNN):
    def __init__(self, *args, states_per_class = 2, match_labels = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.states_per_class = states_per_class
        self.match_labels = match_labels
        self.auxiliary_loader = None

    def on_train_epoch_start(self):
        self.save_states()

    def on_validation_epoch_start(self):
        self.save_states()

    def on_test_epoch_start(self):
        self.save_states()

    def save_states(self):
        if not self.auxiliary_loader:
            self.auxiliary_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=1, shuffle=True, collate_fn=self.pad_batches)

        self.saved_states = [[] for i in range(self.num_classes)]
        flags = [False] * len(self.saved_states)
        for i, batch in enumerate(self.auxiliary_loader):
            if all(flags):
                break

            label = batch["labels"][0]

            if len(self.saved_states[label]) >= self.states_per_class:
                flags[label] = True
                continue

            with torch.no_grad():
                state = self.state_forward(batch)

            self.saved_states[label].append(state)

    def state_forward(self, batch):
        """
        Get the final states for all stateful modules in the network, computed on the given batch
        """
        events = batch["events"].to(self.device) # xytp format
        lengths = batch["lengths"].to(self.device)
        states = []

        x = self.preprocess_inputs(events, lengths)

        for l in self.farsecnn:
            if isinstance(l, SubmanifoldFARSEConv) or isinstance(l, BranchBlock):
                x = l(x, return_state=True)
            else:
                x = l(x)
            states.append(x.pop('state', None))

        return states


    def forward(self, batch):
        events = batch["events"] # xytp format
        lengths = batch["lengths"]

        # selection of states to use as initialization
        # --------
        bs = events.shape[0]
        if self.match_labels:
            labels_idx = batch["labels"].clone()
        else:
            labels_idx = np.random.randint(0, self.num_classes, size=bs, dtype='int')

        uninit_idx = np.random.choice(bs, np.ceil(bs * 0.2).astype('int'), replace=False)

        labels_idx[uninit_idx] = -1 # flag random samples to remain uninitialized

        none_state = [None] * len(self.farsecnn)
        saved_states = [[none_state]] + self.saved_states
        states = [saved_states[i+1][np.random.choice(len(saved_states[i + 1]))] for i in labels_idx]
        #states is a list of model states to be used as initialization for every sample of the batch
        #each item is a list of layer states

        layers_states = []
        for l, f in enumerate(self.farsecnn):
            l_state = [s[l] for s in states]

            if isinstance(f, BranchBlock):
                l_state_b1 = [s[0] if s else [None]*len(f.branch_1) for s in l_state]
                l_state_b1 = list(zip(*l_state_b1))
                l_state_b1 = [self.batch_states(ls) for ls in l_state_b1]

                l_state_b2 = [s[1] if s else [None]*len(f.branch_2) for s in l_state]
                l_state_b2 = list(zip(*l_state_b2))
                l_state_b2 = [self.batch_states(ls) for ls in l_state_b2]

                layers_states.append((l_state_b1, l_state_b2))
            else:
                l_state = self.batch_states(l_state) # convert list of states into batched dictionary requested by FARSEConv
                layers_states.append(l_state)
        # --------

        with torch.no_grad():
            x = self.preprocess_inputs(events, lengths)

        for l,init_state in zip(self.farsecnn[:-1], layers_states[:-1]):
            if init_state:
                x = l(x, init_state=init_state)
            else:
                x = l(x)

        l = self.farsecnn[-1]
        if layers_states[-1]:
            x = l(x, init_state=layers_states[-1])
        else:
            x = l(x)

        logits = self.classifier(x["events"])

        x = {"logits": logits, "lengths": x["lengths"], "time": x["time"]}
        return x


    def batch_states(self, states):
        if not any(states):
            return None
        hidden = torch.cat([x['hidden'] for x in states if x], dim=1)
        cell = torch.cat([x['cell'] for x in states if x], dim=1)
        h = torch.cat([x['h'] for x in states if x], dim=0)
        w = torch.cat([x['w'] for x in states if x], dim=0)
        batch_id = torch.cat([x['batch_id'] + i for i,x in enumerate(states) if x], dim=0)
        return {"hidden": hidden, "cell": cell, "h": h, "w": w, "batch_id": batch_id}