import os
import shutil
import torch
import numpy as np
import argparse
import yaml
from tqdm import tqdm
from model_detect_farsecnn import LitDetectFARSECNN
from utils.farsecnn_utils import load_farsecnn
from utils.data_utils import get_dataset
from utils.detection_utils import bbox_center_dim_to_topleft_botright, nms


def build_detection_annotations(test_cfg_path):
    with open(test_cfg_path, "r") as stream:
        try:
            test_cfg = yaml.safe_load(stream)
            print(f"Using test configuration %s" % test_cfg_path)
        except yaml.YAMLError as exc:
            print(exc)

    if torch.cuda.is_available() and test_cfg.get('use_gpu',True):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dataset_name = test_cfg.get('dataset_name')

    _,_,dataset = get_dataset(dataset_name)
    classes = dataset.classes

    cfg_path = test_cfg.get('model_cfg')
    checkpoint_path = test_cfg.get('checkpoint_path')
    bs = test_cfg.get('batch_size')

    net = LitDetectFARSECNN(cfg_path, bs=bs, dataset=dataset_name).to(device)
    net = load_farsecnn(net, checkpoint_path)
    net = net.eval()

    print('Generating detection annotations...')

    d_dest_path = os.path.join(test_cfg.get('output_path'), 'detections')
    if os.path.exists(d_dest_path):
        print("Destination path already exists, overwriting content.")
        shutil.rmtree(d_dest_path)
    os.makedirs(d_dest_path)

    for i in tqdm(range(0, len(dataset), bs)):
        samples = []
        for j in range(i, min(len(dataset), i + bs)):
            samples.append(dataset[j])
        batch = net.pad_batches(samples)
        batch = {k: t.to(device) for k, t in batch.items()}

        with torch.no_grad():
            det = net.detect(batch, confidence_threshold=0.0) # <batch_id, x_center, y_center, w, h, cf, class_id, timestamp>

        gt_bbox = batch['labels']
        for b in range(gt_bbox.shape[0]):
            d = det[det[:, 0] == b]

            # select only last timestamp
            t, t_idx = d[:, -1].unique(return_inverse=True)
            d = d[t_idx == t.shape[0] - 1]

            # discard batch_id and ts
            d = d[:, 1:-1]

            # non-max suppression
            d = nms(d)

            d_ltrb = bbox_center_dim_to_topleft_botright(d[:, :4]).long()
            d_ltrb = d_ltrb.clamp(min=0)
            d_ltrb[:, [0, 2]] = d_ltrb[:, [0, 2]].clamp(max=net.frame_size[1])
            d_ltrb[:, [1, 3]] = d_ltrb[:, [1, 3]].clamp(max=net.frame_size[0])
            d_c = d[:, -1].long()
            d_cf = d[:, -2]

            d_lines = [(f"%s %.5f %d %d %d %d\n" % (classes[c], cf, ltrb[0], ltrb[1], ltrb[2], ltrb[3])) for ltrb, c, cf in zip(d_ltrb, d_c, d_cf)]

            d_file = os.path.join(d_dest_path, str(i + b) + '.txt')
            with open(d_file, 'w') as f:
                f.writelines(d_lines)
    print('Done.')

    if not test_cfg.get('generate_groundtruth', False):
        exit(0)

    print('Generating groundtruth annotations...')

    gt_dest_path = os.path.join(test_cfg.get('output_path'), 'groundtruth')
    if os.path.exists(gt_dest_path):
        print("Destination path already exists, overwriting content.")
        shutil.rmtree(gt_dest_path)
    os.makedirs(gt_dest_path)

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        target = sample[1]

        gt_lines = []
        for gt in target:
            x1 = int(gt[0])
            y1 = int(gt[1])
            x2 = int(gt[0] + gt[2])
            y2 = int(gt[1] + gt[3])
            class_id = int(gt[4])

            gt_lines.append(f"%s %d %d %d %d\n" % (classes[class_id], x1, y1, x2, y2))

        gt_file = os.path.join(gt_dest_path, str(i) + '.txt')
        with open(gt_file, 'w') as f:
            f.writelines(gt_lines)
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_cfg", default='configs/test_cfg_detect.yaml',
                        help="Path to a config file for testing.")
    args = parser.parse_args()
    test_cfg_path = args.test_cfg

    build_detection_annotations(test_cfg_path)
