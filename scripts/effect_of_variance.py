import logging
import os
import sys
import numpy as np
import torch
from copy import deepcopy
from logging import Logger, StreamHandler, FileHandler
from datetime import datetime

codebase = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(codebase)

from synthetic_data.linear_slabs import LinearSlabDataset
from trainers import Trainer

timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
logger = Logger('variance_exps', logging.DEBUG)
logger.addHandler(StreamHandler())
logging_file = os.path.join(codebase, 'output', 'variance_exps_{}.log'.format(timestamp))
os.makedirs(os.path.dirname(logging_file), exist_ok=True)
logger.addHandler(FileHandler(logging_file))

TRAIN_SIZE = 100000

base_data_config = dict(
    num_samples=110000,
    num_dim=50,
    margins=0.1,
    width=1.0,
    random_orthonormal_transform=True,
    slabs=np.array([2] + [7] * 49),
    noise_proportions=np.array([0.1] + [0] * 49),
    slab_probabilities=[[1.0, 1.0]] + [[1 / 16.0, 0.25, 7 / 16.0, 0.5, 7 / 16.0, 0.25, 1 / 16.0]] * 49,
)

base_trainer_config = dict(
    model=dict(
        cls='fcn',
        num_layers=2,
        input_dim=50,
        output_dim=2,
        latent_dim=100,
        use_bn=False,
        dropout_probability=0.0,
        linear_init=None
    ),
    loss=dict(cls='CE'),
    device='cuda' if torch.cuda.is_available() else 'cpu',
    evaluate_interval=1000,
    save_interval=0,
    # work_dir='./training_logs/lms7_noisy_{}/'.format(datetime.datetime.now().strftime('%m%d%H%M')),
    loss_eps=1e-2,
    # logger=logger,
    max_steps=250000,
    optimizer=dict(
        cls='SGD',
        lr=0.1,
        weight_decay=5.0e-5,
        momentum=0.9
    ),
    train_data=dict(
        # dataset=equiv_train,
        batch_size=256,
        shuffle=True
    ), val_data=dict(
        # dataset=equiv_val,
        batch_size=256,
        shuffle=False
    ),
    additional_data=dict(
        s_randomized=dict(
            # dataset=equiv_train.randomize_axes((0,)),
            batch_size=256,
            shuffle=False
        ),
        sc_randomized=dict(
            # dataset=equiv_train.randomize_axes((1,)),
            batch_size=256,
            shuffle=False
        )
    )
)


def variance_configs(side_probabilities):
    configs = {}
    for side_prob in side_probabilities:
        data_conf = deepcopy(base_data_config)
        center_prob = (1.0 - 2 * side_prob) / 2.0
        slab_probs = [side_prob, 0.25, center_prob, 0.5, center_prob, 0.25, side_prob]
        data_conf['slab_probabilities'] = [[1.0, 1.0]] + [slab_probs] * 49
        dataset = LinearSlabDataset.generate(**data_conf)
        train_data, val_data = dataset.split_train_val(TRAIN_SIZE)

        trainer_conf = deepcopy(base_trainer_config)
        trainer_conf.update(
            logger=logger,
            work_dir='{}/output/training_logs_{}/variance/variance_{:.5f}/'.format(codebase, timestamp, side_prob)
        )
        trainer_conf['train_data']['dataset'] = train_data
        trainer_conf['val_data']['dataset'] = val_data
        trainer_conf['additional_data']['s_randomized']['dataset'] = train_data.randomize_axes((0,))
        trainer_conf['additional_data']['sc_randomized']['dataset'] = train_data.randomize_axes(tuple(range(1, 50)))
        configs[side_prob] = trainer_conf
    return configs


def training_sample_configs(num_samples):
    configs = {}
    for n in num_samples:
        data_conf = deepcopy(base_data_config)
        data_conf['num_samples'] = n
        dataset = LinearSlabDataset.generate(**data_conf)
        train_data, val_data = dataset.split_train_val(n - 10000)

        trainer_conf = deepcopy(base_trainer_config)
        trainer_conf.update(
            logger=logger,
            work_dir='{}/output/training_logs_{}/nsamples/nsamples_{}/'.format(codebase, timestamp, n)
        )
        trainer_conf['train_data']['dataset'] = train_data
        trainer_conf['val_data']['dataset'] = val_data
        trainer_conf['additional_data']['s_randomized']['dataset'] = train_data.randomize_axes((0,))
        trainer_conf['additional_data']['sc_randomized']['dataset'] = train_data.randomize_axes(tuple(range(1, 50)))
        configs[n] = trainer_conf
    return configs


def network_arch_configs(num_layers, latent_dims):
    configs = {}
    dataset = LinearSlabDataset.generate(**base_data_config)
    train_data, val_data = dataset.split_train_val(TRAIN_SIZE)
    for n_layers in num_layers:
        for latent_dim in latent_dims:
            trainer_conf = deepcopy(base_trainer_config)
            trainer_conf.update(
                logger=logger,
                work_dir='{}/output/training_logs_{}/nlayers/nlayers_{}_latentdim_{}/'.format(codebase, timestamp, n_layers, latent_dim)
            )
            trainer_conf['model']['num_layers'] = n_layers
            trainer_conf['model']['latent_dim'] = latent_dim
            trainer_conf['train_data']['dataset'] = train_data
            trainer_conf['val_data']['dataset'] = val_data
            trainer_conf['additional_data']['s_randomized']['dataset'] = train_data.randomize_axes((0,))
            trainer_conf['additional_data']['sc_randomized']['dataset'] = train_data.randomize_axes(tuple(range(1, 50)))
            configs[n_layers, latent_dim] = trainer_conf
    return configs


def input_dim_configs(input_dims):
    configs = {}
    for input_dim in input_dims:
        data_conf = deepcopy(base_data_config)
        data_conf['num_dim'] = input_dim
        data_conf['slabs'] = np.array([2] + [7] * (input_dim - 1))
        data_conf['noise_proportions'] = np.array([0.1] + [0] * (input_dim - 1))
        data_conf['slab_probabilities'] = [[1.0, 1.0]] + \
                                          [[1 / 16.0, 0.25, 7 / 16.0, 0.5, 7 / 16.0, 0.25, 1 / 16.0]] * (input_dim - 1)
        dataset = LinearSlabDataset.generate(**data_conf)
        train_data, val_data = dataset.split_train_val(TRAIN_SIZE)

        trainer_conf = deepcopy(base_trainer_config)
        trainer_conf.update(
            logger=logger,
            work_dir='{}/output/training_logs_{}/inputdim/inputdim_{}/'.format(codebase, timestamp, input_dim)
        )
        trainer_conf['model']['input_dim'] = input_dim
        trainer_conf['train_data']['dataset'] = train_data
        trainer_conf['val_data']['dataset'] = val_data
        trainer_conf['additional_data']['s_randomized']['dataset'] = train_data.randomize_axes((0,))
        trainer_conf['additional_data']['sc_randomized']['dataset'] = train_data.randomize_axes(
            tuple(range(1, input_dim)))
        configs[input_dim] = trainer_conf
    return configs


def num_slabs_configs():
    configs = {}
    data_conf = deepcopy(base_data_config)
    data_conf.update({'slabs': np.array([2] + [5] * 49),
                      'slab_probabilities': [[1.0, 1.0]] + [[0.125, 0.5, 0.75, 0.5, 0.125]] * 49})
    dataset = LinearSlabDataset.generate(**data_conf)
    train_data, val_data = dataset.split_train_val(TRAIN_SIZE)

    trainer_conf = deepcopy(base_trainer_config)
    trainer_conf.update(
        logger=logger,
        work_dir='{}/output/training_logs_{}/nslabs/nslabs_{}/'.format(codebase, timestamp, 5)
    )
    trainer_conf['train_data']['dataset'] = train_data
    trainer_conf['val_data']['dataset'] = val_data
    trainer_conf['additional_data']['s_randomized']['dataset'] = train_data.randomize_axes((0,))
    trainer_conf['additional_data']['sc_randomized']['dataset'] = train_data.randomize_axes(tuple(range(1, 50)))
    configs[5] = trainer_conf
    return configs


def main():
    for train_configs_generator in [
        variance_configs(np.linspace(1 / 16.0, 1 / 4.0, 10, endpoint=True)),
        training_sample_configs(np.arange(40000, 200000, 20000)),
        network_arch_configs([2, 3, 4, 5], [100, 200, 300]),
        input_dim_configs([2, 5, 10, 20, 30, 40, 50]),
        num_slabs_configs()
    ]:
        for trainer_conf in train_configs_generator.values():
            trainer = Trainer(**trainer_conf)
            trainer.run()


if __name__ == '__main__':
    main()
