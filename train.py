import os
import argparse
import yaml
from datetime import datetime
import torch
import lightning
import tonic.transforms
import transforms
from lightning.pytorch.loggers.neptune import NeptuneLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from model_detect_farsecnn import LitDetectFARSECNN
from model_farsecnn import LitFARSECNN
from model_init_farsecnn import LitInitFARSECNN
from utils.farsecnn_utils import load_farsecnn


def main(train_cfg_path):
    with open(train_cfg_path, "r") as stream:
        try:
            train_cfg = yaml.safe_load(stream)
            print(f"Running training configuration %s" % train_cfg_path)
        except yaml.YAMLError as exc:
            print(exc)

    lightning.seed_everything(seed=train_cfg.get('seed',0))

    dt_string = datetime.now().strftime("%Y%m%d_%H%M%S")

    use_gpu = train_cfg.get('use_gpu', True)
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_cfg_path = train_cfg.get('model_cfg')
    dataset = train_cfg.get('dataset_name')
    frame_size = train_cfg.get('frame_size')

    bs = train_cfg.get('batch_size', 1)
    grad_bs = train_cfg.get('gradient_batch_size', 1)
    if bs > grad_bs or (grad_bs % bs) != 0:
        new_grad_bs = bs * max(grad_bs // bs, 1)
        print(f"grad_bs = %d is not a multiple of bs = %d. Setting grad_bs = %d" % grad_bs, bs, new_grad_bs)
        grad_bs = new_grad_bs
    accumulate_grad = grad_bs // bs

    lr = train_cfg.get('learning_rate', 1e-3)

    transform_list = []
    for t in train_cfg.get('transforms',[]):
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


    log_mode = train_cfg.get('log_mode')
    callbacks = []
    if log_mode == 'neptune':
        logger = NeptuneLogger(
            api_key=train_cfg['neptune_api_key'],
            project=train_cfg['neptune_project'],
            name=dt_string,
            with_id=train_cfg.get('neptune_run_id'),
            tags=[dataset],
            log_model_checkpoints=train_cfg.get('neptune_log_model_checkpoints', False),
            prefix=''
        )

        logger.experiment["configs/model"].upload(model_cfg_path)
        logger.experiment['configs/train'].upload(train_cfg_path)
        callbacks.append(LearningRateMonitor())
    else:
        logger = True

    do_validation = train_cfg.get('do_validation', True)

    existing_ckpt = train_cfg.get('resume_ckpt')

    if train_cfg.get('save_checkpoint', True):
        if existing_ckpt:
            checkpoint_path = os.path.dirname(existing_ckpt)
        else:
            checkpoint_dir = train_cfg.get('checkpoint_dir', 'data/checkpoints')
            checkpoint_path = os.path.join(checkpoint_dir, dataset+'-'+dt_string)
        os.makedirs(checkpoint_path, exist_ok=True)
        print(f'Saving checkpoints to directory: %s' % checkpoint_path)
        callbacks.append(ModelCheckpoint(dirpath=checkpoint_path, monitor=('val/loss' if do_validation else 'train/loss'), save_weights_only=False))

    if train_cfg.get('early_stopping', False):
        monitor_metric = 'val/loss' if do_validation else 'train/loss'
        callbacks.append(EarlyStopping(monitor_metric, patience=train_cfg.get('early_stopping_patience', 10), min_delta=train_cfg.get('early_stopping_min_delta',0)))

    if train_cfg.get('detection', False):
        net = LitDetectFARSECNN(model_cfg_path, bs=bs, log_mode=log_mode, dataset=dataset, val_split=do_validation, start_lr=lr, frame_size=frame_size, train_transforms=transform).to(device)
    else:
        if train_cfg.get('init_state_mode', False):
            net = LitInitFARSECNN(model_cfg_path, bs=bs, log_mode=log_mode, dataset=dataset, start_lr=lr, frame_size=frame_size, train_transforms=transform).to(device)
        else:
            net = LitFARSECNN(model_cfg_path, bs=bs, log_mode=log_mode, dataset=dataset, start_lr=lr, frame_size=frame_size, train_transforms=transform).to(device)

    initialize_from_ckpt = train_cfg.get('initialize_from_ckpt')
    if initialize_from_ckpt:
        if existing_ckpt:
            print('Warning! initialize_from_ckpt was provided but will be overwritten by existing_ckpt')
        else:
            net = load_farsecnn(net, initialize_from_ckpt)
            print(f'Model initialized from %s' % initialize_from_ckpt)

    trainer = lightning.pytorch.trainer.trainer.Trainer(
        max_epochs=train_cfg.get('max_epoch', 200),
        accumulate_grad_batches=accumulate_grad,
        callbacks=callbacks,
        deterministic=False,
        logger=logger,
        limit_val_batches=(1.0 if  do_validation else 0.0),
        num_sanity_val_steps=(2 if do_validation and not train_cfg.get('init_state_mode', False) else 0)
    )

    trainer.fit(net, ckpt_path=existing_ckpt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_cfg", default='configs/train_cfg.yaml',
                        help="Path to a config file for the training run.")
    args = parser.parse_args()
    train_cfg_path = args.train_cfg

    main(train_cfg_path)

