from model_farsecnn import LitFARSECNN
from model_detect_farsecnn import LitDetectFARSECNN
from lightning.pytorch.loggers.neptune import NeptuneLogger
from lightning.pytorch.trainer.trainer import Trainer
import torch
import tonic
import transforms
from utils.farsecnn_utils import load_farsecnn
import argparse
import yaml

def main(test_cfg_path):
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

    model_cfg_path = test_cfg.get('model_cfg')
    dataset = test_cfg.get('dataset_name')
    checkpoint_path = test_cfg.get('checkpoint_path')
    bs = test_cfg.get('batch_size', 1)
    log_mode = test_cfg.get('log_mode')

    transform_list = []
    for t in test_cfg.get('transforms', []):
        if t['name'] == 'RandomTranslate':
            target_size = tuple(t['target_size'])
            transform_list.append(transforms.RandomTranslate(target_size=target_size))
        elif t['name'] == 'RandomFlipLR':
            sensor_size = tuple(t['sensor_size'])
            p = t.get('probability', 0.5)
            transform_list.append(tonic.transforms.RandomFlipLR(sensor_size=sensor_size, p=p))
        elif t['name'] == 'UniformNoise':
            n = t.get('n', 0)
            transform_list.append(transforms.UniformNoise(n=n, use_range=True))
    transform = tonic.transforms.Compose(transform_list)

    if 'Gen1' in dataset:
        net = LitDetectFARSECNN(model_cfg_path, bs=bs, log_mode=log_mode, dataset=dataset, test_transforms=transform).to(device)
    else:
        net = LitFARSECNN(model_cfg_path, bs=bs, log_mode=log_mode, dataset=dataset, test_transforms=transform).to(device)

    net = load_farsecnn(net, checkpoint_path)
    print("Testing model: "+checkpoint_path)

    if log_mode == 'neptune':
        logger = NeptuneLogger(
            api_key=test_cfg['neptune_api_key'],
            project=test_cfg['neptune_project'],
            with_id=test_cfg['neptune_run_id'],
            prefix=''
        )
    else:
        logger = True

    trainer = Trainer(logger = logger)
    trainer.test(model=net, verbose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_cfg", default='configs/test_cfg.yaml',
                        help="Path to a config file for testing.")
    args = parser.parse_args()
    test_cfg_path = args.test_cfg

    main(test_cfg_path)
