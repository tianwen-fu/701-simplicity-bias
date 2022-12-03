import os
from logging import Logger, StreamHandler, FileHandler, Formatter
from datetime import datetime
import logging
import torch
import numpy as np
import random
import pprint
from copy import deepcopy
import tempfile

import utils
from synthetic_data.linear_slabs import LinearSlabDataset
from torch.utils.data import DataLoader
from models import Model
from utils import build_optimizer
from trainers import Trainer

__all__ = ['set_random_seed', 'run']


def _expand_multiplier(value, total_count):
    out = ()
    for item in value:
        if item['count'] == -1:
            count = total_count
            assert item is value[-1]
        else:
            count = item['count']
        out += (item['val'],) * count
        total_count -= count
    return out


def _expand_config(config):
    config = deepcopy(config)
    num_dim = config['data']['num_dim']
    config['data']['slabs'] = np.asarray(_expand_multiplier(config['data']['slabs'], num_dim), dtype=np.uint8)
    config['data']['noise_proportions'] = np.asarray(_expand_multiplier(config['data']['noise_proportions'], num_dim),
                                                     dtype=np.float32)
    config['data']['slab_probabilities'] = _expand_multiplier(config['data']['slab_probabilities'], num_dim)
    return config


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def _prepare_logger(name, log_dir):
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(__file__), 'log')
    work_dir = os.path.abspath(os.path.join(log_dir, name))
    os.makedirs(work_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    logger = Logger(name, logging.DEBUG)
    formatter = Formatter('[%(asctime)s] %(message)s')
    stdout_handler = StreamHandler()
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logging_file = os.path.join(work_dir, '{}.log'.format(timestamp))
    os.makedirs(os.path.dirname(logging_file), exist_ok=True)
    file_handler = FileHandler(logging_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger, work_dir


def _prepare_data(data_config, overfit_complex_features, device):
    # prepare dataset
    simple_axes = data_config.pop('simple_axes')
    batch_size = data_config.pop('batch_size')
    train_size = data_config.pop('train_samples')
    val_size = data_config.pop('val_samples')
    dataset = LinearSlabDataset.generate(num_samples=train_size + val_size, device=device, **data_config)
    train_data, val_data = dataset.split_train_val(train_size)
    s_randomized = train_data.randomize_axes(simple_axes)
    complex_axes = tuple(ax for ax in range(data_config['num_dim']) if ax not in simple_axes)
    sc_randomized = train_data.randomize_axes(complex_axes)
    if overfit_complex_features:
        train_data = s_randomized
    # convert to dataloaders
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(val_data, shuffle=False, batch_size=batch_size)
    s_randomized_dataloader = DataLoader(s_randomized, shuffle=False, batch_size=batch_size)
    sc_randomized_dataloader = DataLoader(sc_randomized, shuffle=False, batch_size=batch_size)
    return dict(
        Train=train_dataloader,
        Val=val_dataloader,
        S_Randomized=s_randomized_dataloader,
        Sc_Randomized=sc_randomized_dataloader
    )


def run(name, config, *, log_dir=None, seed=None, overfit_complex_features=False, device=None,
        wandb_project=None, wandb_entity=None, save_data=True):
    logger, work_dir = _prepare_logger(name, log_dir)

    logger.info(f'Config: {pprint.pformat(config)}')
    logger.info(f'Work directory: {work_dir}')
    wandb_run = None
    if wandb_project is not None:
        import wandb
        wandb_run = wandb.init(project=wandb_project, reinit=True, config={**config, 'seed': seed}, name=name,
                               entity=wandb_entity, sync_tensorboard=True)

    config = _expand_config(config)
    with open(os.path.join(work_dir, 'full_config.py'), 'w') as file:
        print(repr(config), file=file)

    if seed is not None:
        set_random_seed(seed)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataloaders = _prepare_data(config['data'], overfit_complex_features, device)
    for name, loader in dataloaders.items():
        dataset: LinearSlabDataset = loader.dataset
        dataset.visualize(title=name, save_as=os.path.join(work_dir, f'data_{name}.png'), show=False)
        if save_data:
            dataset.save_as(os.path.join(work_dir, f'data_{name}.npz'))

    # build model and optimizer
    model = Model(config['model']).to(device=device)
    logger.info(f'Model: {model}')
    if wandb_project is not None:
        wandb_run.watch(model)
    optimizer = utils.build_optimizer(model.parameters(), **config['optimizer'])
    scheduler = utils.build_scheduler(optimizer, **config['scheduler'])

    trainer = Trainer(dataloaders=dataloaders, model=model, device=device, work_dir=work_dir, logger=logger,
                      optimizer=optimizer, scheduler=scheduler, **config['trainer'])
    eval_results = trainer.run(wandb_logger=wandb_run)

    if wandb_project is not None:
        wandb_run.save(os.path.join(work_dir, '*'))
        wandb_run.finish()

    return eval_results


if __name__ == '__main__':
    from configs import lms_7_fcn300_2

    run('lms7', lms_7_fcn300_2)
