import yaml
import torch
import seaborn as sn
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.metrics import confusion_matrix
from torch.nn import functional as F
from model_base import LitBaseModel
from layers.FARSEConv import FARSEConv
from layers.SubmanifoldFARSEConv import SubmanifoldFARSEConv
from layers.TemporalDropout import TemporalDropout
from layers.SparsePool import SparseMaxPool, SparseAvgPool, SparseAdaptiveMaxPool, SparseAdaptiveAvgPool
from layers.BranchBlock import BranchBlock
from utils.farsecnn_utils import normalize_range

def accuracy(pred, target):
    return (pred == target).sum().item() / target.shape[0]

def get_asyncsparsemodule(module_data, frame_size, channel_size):
    module_name = module_data['name']
    if module_name == 'FARSEConv':
        kernel_size = module_data["kernel_size"]
        kernel_size = [min(kernel_size, frame_size[0]), min(kernel_size, frame_size[1])]
        stride = [module_data["stride"]] * 2
        hidden_size = module_data["hidden_size"]
        module = FARSEConv(channel_size, hidden_size, frame_size, kernel_size, stride)
    elif module_name == 'SubmanifoldFARSEConv':
        kernel_size = module_data["kernel_size"]
        kernel_size = [min(kernel_size, frame_size[0]), min(kernel_size, frame_size[1])]
        hidden_size = module_data["hidden_size"]
        module = SubmanifoldFARSEConv(channel_size, hidden_size, frame_size, kernel_size)
    elif module_name == 'SparseMaxPool':
        kernel_size = [module_data["kernel_size"]] * 2
        module = SparseMaxPool(frame_size, kernel_size=kernel_size)
    elif module_name == 'SparseAvgPool':
        kernel_size = [module_data["kernel_size"]] * 2
        module = SparseAvgPool(frame_size, kernel_size=kernel_size)
    elif module_name == 'SparseAdaptiveMaxPool':
        output_size = [module_data["output_size"]] * 2
        module = SparseAdaptiveMaxPool(frame_size, output_size)
    elif module_name == 'SparseAdaptiveAvgPool':
        output_size = [module_data["output_size"]] * 2
        module = SparseAdaptiveAvgPool(frame_size, output_size)
    elif module_name == 'TemporalDropout':
        window_size = module_data['window_size']
        module = TemporalDropout(window_size, frame_size)
    elif module_name == 'BranchBlock':
        merge_func = module_data['merge_func']
        branch_1_data = module_data.get('branch_1')
        if branch_1_data:
            branch_1, _, _ = get_modulelist(branch_1_data, frame_size, channel_size)
        else:
            branch_1 = None
        branch_2_data = module_data.get('branch_2')
        if branch_2_data:
            branch_2, _, _ = get_modulelist(branch_2_data, frame_size, channel_size)
        else:
            branch_2 = None
        module = BranchBlock(frame_size, channel_size, branch_1, branch_2, merge_func)
    else:
        raise ValueError('Requested module does not exist: ',module_name)
    return module

def get_modulelist(layers_data, frame_size, channel_size):
    ml = []
    for l in layers_data:
        m = get_asyncsparsemodule(l, frame_size, channel_size)
        ml.append(m)
        frame_size = m.frame_output_size
        channel_size = getattr(m, "hidden_size", channel_size)
    modulelist = torch.nn.ModuleList(ml)
    return modulelist, frame_size, channel_size


class LitFARSECNN(LitBaseModel):
    def __init__(self, config_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(config_path, "r") as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        feature_mode = config["feature_mode"]
        if feature_mode not in ['t', 'p', 'd', 'dp']:
            raise ValueError("Config error: feature_mode = %s is not valid!" % feature_mode)
        channel_size = len(feature_mode)
        self.feature_mode = feature_mode

        farsecnn_layers_data = config["farsecnn_layers"]

        self.farsecnn, frame_size, channel_size = get_modulelist(farsecnn_layers_data, self.frame_size, channel_size)

        if frame_size[0] != 1 or frame_size[1] != 1:
            self.farsecnn.append(SparseAdaptiveAvgPool(frame_size, [1, 1]))

        self.classifier = torch.nn.Linear(channel_size, self.num_classes)

    def forward(self, batch):
        events = batch["events"] # xytp format
        lengths = batch["lengths"]
        with torch.no_grad():
            x = self.preprocess_inputs(events, lengths)

        for l in self.farsecnn:
            x = l(x)
        logits = self.classifier(x["events"])
        x = {"logits":logits, "lengths":x["lengths"], "time":x["time"]}
        return x


    def preprocess_inputs(self, events, lengths):
        if 't' in self.feature_mode or 'd' in self.feature_mode:
            # duplicate time to be used also as feature, since grouping operation expects event tuples to contain <x,y,t,features>
            repeats = torch.ones(events.shape[-1], device=events.device, dtype=torch.int64)
            repeats[2] = 2
            events = events.repeat_interleave(repeats, dim=-1)

            if self.feature_mode == 't':
                # normalize time feature
                time_feature = events[:, :, 3]
                normalize_range(time_feature, pad_start_idx=lengths)
        if 'p' not in self.feature_mode:
            # exclude polarity feature
            events = events[:, :, :-1]

        with torch.no_grad():
            grouped_input = self.farsecnn[0].group_events((events, lengths))  # <n_flat_ev, n_feature>

        if 'd' in self.feature_mode:
            # convert time feature to delay (relative to pixel)
            t_feature = grouped_input["events"][..., 0]
            start_ids = grouped_input["lengths"].cumsum(dim=0)
            start_ids = start_ids.roll(1)
            start_ids[0] = 0
            t_feature[1:] = t_feature[1:] - t_feature[:-1]
            t_feature[:] = torch.clamp(t_feature[:] / 1e5, max=1) # max delay is 100ms normalized to 1
            t_feature[start_ids] = 1 # initial event set to max delay
        return grouped_input

    def _select_backpropagation_steps(self, res, target, num_steps=4):
        logits = res["logits"]
        lengths = res["lengths"]

        idx = (lengths - 1).repeat_interleave(num_steps).div(float(num_steps))
        idx *= (torch.arange(num_steps, device=lengths.device) + 1).repeat(lengths.shape[0])
        idx = idx.round().long()
        start = lengths.cumsum(dim=0)
        start = start.roll(1)
        start[0] = 0
        idx += start.repeat_interleave(num_steps)
        logits = logits[idx]

        target = target.repeat_interleave(num_steps, dim=0)
        return logits, target

    def get_mode_prediction(self, res, time_threshold=None):
        logits = res['logits']
        lengths = res['lengths']
        time = res['time']
        bs = lengths.shape[0]

        b_id = torch.arange(bs, device=logits.device, dtype=torch.int64).repeat_interleave(lengths)
        pred = F.softmax(logits, dim=1).argmax(1)
        scatter_idx = b_id*self.num_classes + pred

        if time_threshold:
            # consider only the outputs arrived in the final time_threshold microseconds before the last timestamp
            last_idx = lengths.cumsum(dim=0) - 1
            last_time = time[last_idx]
            last_time = last_time.repeat_interleave(lengths)
            time_mask = time > (last_time - time_threshold)

            scatter_idx = scatter_idx[time_mask]

        counts = torch.zeros([bs * self.num_classes], device=logits.device, dtype=torch.int64)
        counts.scatter_add_(0, scatter_idx, torch.ones_like(scatter_idx))
        counts = counts.view([bs, self.num_classes])

        mode_pred = torch.argmax(counts, dim=1)
        return mode_pred


    def shared_step(self, batch, batch_idx):
        NUM_BACKPROPAGATION_STEPS = 4
        res = self(batch)
        target = batch["labels"]
        mode_pred = self.get_mode_prediction(res) # get mode prediction before selecting steps
        logits, target = self._select_backpropagation_steps(res, target, num_steps=NUM_BACKPROPAGATION_STEPS)
        loss = F.cross_entropy(logits, target, reduction='none')
        last_loss = loss[NUM_BACKPROPAGATION_STEPS-1::NUM_BACKPROPAGATION_STEPS]
        loss, last_loss = loss.mean(), last_loss.mean()
        pred = F.softmax(logits, dim=1).argmax(1)
        last_pred = pred[NUM_BACKPROPAGATION_STEPS-1::NUM_BACKPROPAGATION_STEPS]
        out = {"loss": loss, "last_loss":last_loss, "prediction": pred, "last_prediction":last_pred, "mode_prediction":mode_pred, "target": target, "last_target": batch["labels"]}
        return out


    def shared_epoch_end(self, output, compute_figures=False):
        mean_loss = torch.stack([x['loss'] for x in output]).mean()
        pred = torch.cat([x["prediction"] for x in output], dim=0)
        target = torch.cat([x["target"] for x in output], dim=0)
        mean_acc = accuracy(pred, target)
        metrics = {"loss": mean_loss, "accuracy": mean_acc}

        mean_last_loss = torch.stack([x['last_loss'] for x in output]).mean()
        last_pred = torch.cat([x["last_prediction"] for x in output], dim=0)
        mode_pred = torch.cat([x["mode_prediction"] for x in output], dim=0)
        last_target = torch.cat([x["last_target"] for x in output], dim=0)
        mean_last_acc = accuracy(last_pred, last_target)
        mean_mode_acc = accuracy(mode_pred, last_target)
        metrics.update({"last_loss":mean_last_loss, "last_accuracy":mean_last_acc, "mode_accuracy":mean_mode_acc})

        if not compute_figures:
            return metrics
        else:
            fig = plt.figure(figsize=[10, 7])
            conf_mat = confusion_matrix(target.cpu(), pred.cpu(), labels=list(range(self.num_classes)),
                                        normalize='true')
            df_conf_mat = DataFrame(conf_mat, index=self.classes, columns=self.classes)
            sn.heatmap(df_conf_mat, annot_kws={"size": 'small'}, annot=True, square=True)
            figures = {"confusion_matrix": fig}
            return metrics, figures

    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch, batch_idx)
        self.train_step_outputs.append(output)
        return output

    def on_train_epoch_end(self):
        train_metrics = self.shared_epoch_end(self.train_step_outputs)
        self.train_step_outputs.clear()
        self.log_metrics({'train/' + k: v for k, v in train_metrics.items()})
        return {'train/loss': train_metrics['loss'], "train/accuracy": train_metrics['accuracy']}

    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch, batch_idx)
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        val_metrics, val_figures = self.shared_epoch_end(self.validation_step_outputs, compute_figures=True)
        self.validation_step_outputs.clear()
        self.log_metrics({'val/' + k: v for k, v in val_metrics.items()})
        self.log_figures({'val/' + k: v for k, v in val_figures.items()})
        return {'val/loss': val_metrics['loss'], "val/accuracy": val_metrics['accuracy']}

    def test_step(self, batch, batch_idx):
        output = self.shared_step(batch, batch_idx)
        self.test_step_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        test_metrics, test_figures = self.shared_epoch_end(self.test_step_outputs, compute_figures=True)
        self.test_step_outputs.clear()
        self.log_metrics({'test/' + k: v for k, v in test_metrics.items()})
        self.log_figures({'test/' + k: v for k, v in test_figures.items()})
        return {'test/loss': test_metrics['loss'], "test/accuracy": test_metrics['accuracy']}