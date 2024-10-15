from model_farsecnn import LitFARSECNN
import torch
import numpy as np
from utils.detection_utils import iou, bbox_center_dim_to_topleft_botright

class LitDetectFARSECNN(LitFARSECNN):
    def __init__(self, *args, B=2, **kwargs):
        super(LitDetectFARSECNN, self).__init__(*args, **kwargs)

        self.grid_size = self.farsecnn[-1].frame_size
        cell_size = [self.frame_size[1] / self.grid_size[1], self.frame_size[0] / self.grid_size[0]]  # permuted for <x,y> order
        cell_size = torch.as_tensor(cell_size)
        self.register_buffer('cell_size', cell_size)

        x_offset = torch.arange(self.grid_size[1]).unsqueeze(0).expand(self.grid_size[0], -1)
        y_offset = torch.arange(self.grid_size[0]).unsqueeze(1).expand(-1, self.grid_size[1])
        coord_offset = torch.stack([x_offset, y_offset], dim=-1)
        coord_offset = coord_offset * cell_size
        self.register_buffer('coord_offset', coord_offset)

        self.B = B
        output_size = self.grid_size[0] * self.grid_size[1] * (self.num_classes + 5 * B)

        # replace classifier with detection head
        input_size = self.classifier.in_features
        #channel_size = 4*5*self.farsecnn[-1].hidden_size
        self.classifier = torch.nn.Linear(input_size, output_size)

    def forward(self, batch):
        x = super(LitDetectFARSECNN, self).forward(batch)
        l = x['logits']
        x['logits'] = l.view(l.shape[0], self.grid_size[0], self.grid_size[1], self.num_classes + 5 * self.B)
        # grid <bs, h, w, 5*b + num_classes>
        return x

    def shared_step(self, batch, batch_idx):
        gt_bbox = batch['labels']
        gt_bbox_lengths = batch['labels_lengths']
        pred = self(batch)

        NUM_BACKPROPAGATION_STEPS = 4
        pred, gt_bbox = self._select_backpropagation_steps(pred, gt_bbox, num_steps=NUM_BACKPROPAGATION_STEPS)
        gt_bbox_lengths = gt_bbox_lengths.repeat_interleave(NUM_BACKPROPAGATION_STEPS, dim=0)

        bbox_pred, class_logits = self.postprocess_grid(pred)
        gt_bbox, gt_bbox_idx = self.convert_gt_bbox(gt_bbox, gt_bbox_lengths)

        loss = self.compute_yolo_loss(bbox_pred, class_logits, gt_bbox, gt_bbox_idx)

        out = {"loss": loss}

        with torch.no_grad():
            # compute metrics, backpropagation is performed only on loss total
            accuracy, mean_best_iou = self.compute_yole_accuracy(bbox_pred, class_logits, gt_bbox, gt_bbox_idx)
            out.update({"accuracy":accuracy, "mean_best_iou":mean_best_iou})

            last_bbox_pred = bbox_pred[NUM_BACKPROPAGATION_STEPS-1::NUM_BACKPROPAGATION_STEPS]
            last_class_logits = class_logits[NUM_BACKPROPAGATION_STEPS-1::NUM_BACKPROPAGATION_STEPS]
            last_gt_bbox = batch['labels']
            last_gt_bbox_lengths = batch['labels_lengths']
            last_gt_bbox, last_gt_bbox_idx = self.convert_gt_bbox(last_gt_bbox, last_gt_bbox_lengths)

            last_loss = self.compute_yolo_loss(last_bbox_pred, last_class_logits, last_gt_bbox, last_gt_bbox_idx)
            last_accuracy, last_mean_best_iou = self.compute_yole_accuracy(last_bbox_pred, last_class_logits, last_gt_bbox, last_gt_bbox_idx)
            out.update({"last_loss":last_loss, "last_accuracy":last_accuracy, "last_mean_best_iou":last_mean_best_iou})

        return out

    def shared_epoch_end(self, output, compute_figures=False):
        mean_loss = torch.stack([x['loss'] for x in output]).mean()
        mean_acc = [x['accuracy'] for x in output]
        mean_acc = sum(mean_acc)/len(mean_acc)
        mean_iou = [x['mean_best_iou'] for x in output]
        mean_iou = sum(mean_iou)/len(mean_iou)
        metrics = {"loss": mean_loss, "accuracy": mean_acc, "mean_best_iou":mean_iou}

        mean_last_loss = torch.stack([x['last_loss'] for x in output]).mean()
        mean_last_acc = [x['last_accuracy'] for x in output]
        mean_last_acc = sum(mean_last_acc)/len(mean_last_acc)
        mean_last_iou = [x['last_mean_best_iou'] for x in output]
        mean_last_iou = sum(mean_last_iou)/len(mean_last_iou)
        metrics.update({"last_loss":mean_last_loss, "last_accuracy":mean_last_acc, "last_mean_best_iou":mean_last_iou})
        if compute_figures:
            return metrics, {}
        else:
            return metrics

    def postprocess_grid(self, pred):
        """
        Splits the prediction grid into its bbox and class parts, applies activation functions.
        :param
        pred: prediction grid tensor produced by the forward pass. <bs, h, w, 5*b + num_classes>
        :return:
        bbox_pred: prediction grid for bbox coordinates and confidence. <bs, h, w, b, 5>
        class_logits: prediction grid for class logits. <bs, h, w, num_classes>
        """
        bbox_pred = pred[..., :-self.num_classes]
        class_logits = pred[..., -self.num_classes:]

        bbox_pred = torch.sigmoid(bbox_pred)

        bbox_pred = bbox_pred.view(bbox_pred.shape[0], bbox_pred.shape[1], bbox_pred.shape[2], self.B, 5)

        return bbox_pred, class_logits

    def parse_prediction_bbox(self, bbox_pred):
        """
        bbox_pred: <*, 5>
        parses bbox predictions of detection head from <x_center_norm, y_center_norm, sqrt_w, sqrt_h, cf>
        to <x_center, y_center, w, h, cf>
        """
        x = bbox_pred[..., 0]
        y = bbox_pred[..., 1]
        sqrt_w = bbox_pred[..., 2]
        sqrt_h = bbox_pred[..., 3]
        cf = bbox_pred[..., 4]

        coord_offset = self.coord_offset
        coord_offset = coord_offset.unsqueeze(0).expand(bbox_pred.shape[0], -1, -1, -1).unsqueeze(3)

        x = x * self.cell_size[1] + coord_offset[..., 0]
        y = y * self.cell_size[0] + coord_offset[..., 1]

        w = sqrt_w * sqrt_w * self.frame_size[1]
        h = sqrt_h * sqrt_h * self.frame_size[0]

        bbox_pred = torch.stack([x, y, w, h, cf], dim=-1)
        return bbox_pred

    def convert_gt_bbox(self, gt_bbox, gt_bbox_lengths):
        mask = torch.arange(gt_bbox.shape[1], device=gt_bbox.device).unsqueeze(0).expand([gt_bbox.shape[0], -1]) < gt_bbox_lengths.unsqueeze(1)
        gt_bbox = gt_bbox[mask]  # discard "padding" labels
        gt_bbox[:, [0, 1]] = gt_bbox[:, [0, 1]] + gt_bbox[:, [2, 3]] / 2  # convert to center coordinates

        b_id = torch.arange(gt_bbox_lengths.shape[0], device=gt_bbox.device).repeat_interleave(gt_bbox_lengths)
        gt_bbox_idx = (gt_bbox[:, [0, 1]] / self.cell_size).floor().long()
        gt_bbox_idx = torch.cat([b_id.unsqueeze(1), gt_bbox_idx], dim=1)
        # indices in the grid for each gt bbox, <batch_id, x, y>
        return gt_bbox, gt_bbox_idx

    def compute_yolo_loss(self, bbox_pred, class_logits, gt_bbox, gt_bbox_idx):
        lambda_coord = 20
        lambda_noobj = 1
        lambda_obj = 0.1

        bbox_pred_parsed = self.parse_prediction_bbox(bbox_pred)

        pred_per_bbox = bbox_pred[gt_bbox_idx[:, 0], gt_bbox_idx[:, 2], gt_bbox_idx[:, 1]]  # <x_norm, y_norm, sqrt(w), sqrt(h), cf>
        pred_per_bbox_parsed = bbox_pred_parsed[gt_bbox_idx[:, 0], gt_bbox_idx[:, 2], gt_bbox_idx[:, 1]]  # <x_center, y_center, w, h, cf>
        # <num_gt_bbox, B, 5>, for each gt bbox the B predictions of the corresponding grid cell

        # convert coordinates from <center,dim> to <topleft,botright>
        b1 = bbox_center_dim_to_topleft_botright(pred_per_bbox_parsed[..., :4])
        b2 = bbox_center_dim_to_topleft_botright(gt_bbox[..., :4])
        b2 = b2.unsqueeze(1).expand(-1, self.B, 4)

        bbox_iou = iou(b1, b2)
        max_iou, max_idx = torch.max(bbox_iou, dim=1)
        max_iou = max_iou.detach() # detach gradient
        pred_per_bbox = pred_per_bbox[torch.arange(pred_per_bbox.shape[0], device=pred_per_bbox.device), max_idx]
        # <num_gt_bbox, 5> for each gt bbox, the predictor "responsible" for that bbox

        has_obj = torch.zeros([bbox_pred.shape[0], bbox_pred.shape[1], bbox_pred.shape[2]], device=bbox_pred.device,dtype=torch.bool)
        has_obj[gt_bbox_idx[:, 0], gt_bbox_idx[:, 2], gt_bbox_idx[:, 1]] = True
        pred_noobj = bbox_pred[~has_obj]
        cf_noobj = pred_noobj[..., -1]
        # predicted confidence for cells with no object

        # cardinality of the 1^obj set
        one_obj_card = gt_bbox.shape[0]

        # COMPUTE LOSS:
        # x,y coordinates
        o = self.coord_offset[gt_bbox_idx[:, 2], gt_bbox_idx[:, 1]]
        gt_xy_norm = (gt_bbox[:, [0, 1]] - o) / self.cell_size
        xy_loss = torch.nn.functional.mse_loss(pred_per_bbox[:, [0, 1]], gt_xy_norm, reduction='sum')

        # w,h dimensions
        f = torch.as_tensor([self.frame_size[1], self.frame_size[0]], device=gt_bbox.device)
        gt_sqrt_wh = (gt_bbox[:, [2, 3]] / f).sqrt()
        wh_loss = torch.nn.functional.mse_loss(pred_per_bbox[:, [2, 3]], gt_sqrt_wh, reduction='sum')

        # confidence
        cf_obj_loss = torch.nn.functional.binary_cross_entropy(pred_per_bbox[:, 4], max_iou, reduction='mean')
        cf_noobj_loss = torch.nn.functional.binary_cross_entropy(cf_noobj, torch.zeros_like(cf_noobj), reduction='mean')

        # class
        gt_class = gt_bbox[:, -1].long()
        class_logits = class_logits[gt_bbox_idx[:, 0], gt_bbox_idx[:, 2], gt_bbox_idx[:, 1]]
        class_loss = torch.nn.functional.cross_entropy(class_logits, gt_class)

        return lambda_coord * (xy_loss + wh_loss)/one_obj_card \
                    + lambda_obj * cf_obj_loss + lambda_noobj*cf_noobj_loss \
                        + class_loss


    def compute_yole_accuracy(self, bbox_pred, class_logits, gt_bbox, gt_bbox_idx):
        """
        Computes the same accuracy metric used by YOLE, "computed by matching every ground truth bounding box with the predicted one having the highest intersection over union (IOU)".
        Additionally, this also returns the mean iou of the bounding boxes that was matched with each ground truth.
        """
        bbox_pred = self.parse_prediction_bbox(bbox_pred)

        gt_batch_id = gt_bbox_idx[:, 0]
        gt_class = gt_bbox[:,-1]

        # select, for each gt bbox, the corresponding grid in the batch
        bbox_pred = bbox_pred[gt_batch_id]
        class_logits = class_logits[gt_batch_id]

        # reshape gt bbox for broadcasting iou computation
        gt_bbox = gt_bbox.view(gt_bbox.shape[0], 1, 1, 1, gt_bbox.shape[1])
        gt_bbox = gt_bbox.expand(-1, self.grid_size[0], self.grid_size[1], self.B, -1)

        bb1 = bbox_center_dim_to_topleft_botright(bbox_pred)
        bb2 = bbox_center_dim_to_topleft_botright(gt_bbox)
        bbox_iou = iou(bb1, bb2)
        bbox_iou, _ = torch.max(bbox_iou, dim=-1)
        bbox_iou, max_x = torch.max(bbox_iou, dim=-1)
        bbox_iou, max_y = torch.max(bbox_iou, dim=-1)

        max_x = max_x[torch.arange(max_x.shape[0], device=max_x.device), max_y]

        class_logits = class_logits[torch.arange(class_logits.shape[0], device=class_logits.device), max_y, max_x]
        class_pred_id = class_logits.softmax(dim=1).argmax(dim=-1)
        accuracy = (class_pred_id == gt_class).sum().item() / gt_class.shape[0]
        iou_mean = bbox_iou.mean()
        return accuracy, iou_mean


    def detect(self, batch, confidence_threshold=0.5):
        """
        Performs detection on a batch of inputs
        :param batch: batched inputs
        :param confidence_threshold: confidence threshold for what the net considers as a detection
        :return: detection: tensor of detections <batch_id, x_center, y_center, w, h, cf, class_id, timestamp>.
        """

        pred = self.forward(batch)

        time = pred['time']
        lengths = pred['lengths']
        bs = pred['lengths'].shape[0]
        pred = pred['logits']

        bbox_pred, class_logits = self.postprocess_grid(pred)
        bbox_pred = self.parse_prediction_bbox(bbox_pred)

        cf_pred = bbox_pred[..., -1] #predicted confidence scores
        threshold_mask = cf_pred > confidence_threshold

        bbox_det = bbox_pred[threshold_mask]
        class_det = class_logits.unsqueeze(-2).expand(-1,-1,-1,2,-1)[threshold_mask]
        class_det = class_det.softmax(dim=-1).argmax(dim=-1, keepdim=True).float()

        batch_id = torch.arange(bs, device=pred.device, dtype=torch.float)
        batch_id = batch_id.repeat_interleave(lengths, dim=-1)
        batch_id = batch_id.view(-1,1,1,1).expand([-1, self.grid_size[0], self.grid_size[1], 2])
        batch_id = batch_id[threshold_mask].unsqueeze(-1)

        detection = torch.cat([batch_id, bbox_det, class_det], dim=-1)

        # add timestap of each detection
        time = time.view(-1,1,1,1).expand([-1, self.grid_size[0], self.grid_size[1], 2])
        time = time[threshold_mask].unsqueeze(-1).float()
        detection = torch.cat([detection, time], dim=-1)

        return detection



    def pad_batches(self, data):
        """
        data: is a list of len "batch_size" containing sample tuples (input,label)
        """

        events = [np.stack([sample[0][f].astype(np.float32) for f in sample[0].dtype.names], axis=-1) for sample in data]
        events_lens = [len(ev) for ev in events]
        max_len = max(events_lens)
        events = [np.pad(ev, ((0, max_len - ln), (0, 0)), mode='constant', constant_values=0) for
                  ln, ev in zip(events_lens, events)]

        labels = [sample[1] for sample in data]
        labels_lens = [len(l) for l in labels]
        max_len = max(labels_lens)
        labels = [np.pad(lb, ((0, max_len - ln), (0, 0)), mode='constant', constant_values=0) for
                  ln, lb in zip(labels_lens, labels)]

        events = torch.as_tensor(np.stack(events, axis=0))
        labels = torch.as_tensor(np.stack(labels, axis=0))
        events_lens = torch.as_tensor(events_lens)
        labels_lens = torch.as_tensor(labels_lens)

        if 'Gen1' in self.dataset:
            # permute to xytp
            events = events[:,:,[1,2,0,3]]
            # convert p to [-1,+1]
            events[:,:,-1] = events[:,:,-1] * 2 - 1

            # clamp gt coordinates outside of frame
            labels_x = labels[:,:,0]
            labels_y = labels[:,:,1]
            labels_w = labels[:,:,2]
            labels_h = labels[:,:,3]

            labels_w = torch.where(labels_x<0, labels_w + labels_x, labels_w)
            labels_x = torch.clamp(labels_x, 0, self.frame_size[1])
            labels_w = torch.where(labels_x + labels_w > self.frame_size[1], self.frame_size[1] - labels_x, labels_w)

            labels_h = torch.where(labels_y<0, labels_h + labels_y, labels_h)
            labels_y = torch.clamp(labels_y, 0, self.frame_size[0])
            labels_h = torch.where(labels_y + labels_h > self.frame_size[0], self.frame_size[0] - labels_y, labels_h)

            labels[:,:,0] = labels_x
            labels[:,:,1] = labels_y
            labels[:,:,2] = labels_w
            labels[:,:,3] = labels_h

        batch = {"events": events, "labels": labels, "lengths": events_lens, "labels_lengths":labels_lens}
        return batch