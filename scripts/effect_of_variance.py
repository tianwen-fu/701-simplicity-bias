import copy
import logging
import os
import sys
import numpy as np
import torch
from copy import deepcopy
from logging import Logger, StreamHandler, FileHandler, Formatter
from datetime import datetime
import argparse

codebase = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(codebase)

from synthetic_data.linear_slabs import LinearSlabDataset
from trainers import Trainer

timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
logger = Logger('variance_exps', logging.DEBUG)
formatter = Formatter('[%(asctime)s] %(message)s')
stdout_handler = StreamHandler()
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)
logging_file = os.path.join(codebase, 'output', 'variance_exps_{}.log'.format(timestamp))
os.makedirs(os.path.dirname(logging_file), exist_ok=True)
file_handler = FileHandler(logging_file)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

TRAIN_SIZE = 40000

base_data_config = dict(
    num_samples=50000,
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
        latent_dim=300,
        use_bn=False,
        dropout_probability=0.0,
        linear_init=None
    ),
    loss=dict(cls='CE'),
    device='cuda' if torch.cuda.is_available() else 'cpu',
    premature_evaluate_interval=1000,
    evaluate_interval=5000,
    save_interval=0,
    # work_dir='./training_logs/lms7_noisy_{}/'.format(datetime.datetime.now().strftime('%m%d%H%M')),
    accuracy_threshold=0.995,
    # logger=logger,
    max_steps=250000,
    optimizer=dict(
        cls='SGD',
        lr=0.3,
        weight_decay=5.0e-4,
        momentum=0.0
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


def side_prob_configs(n_slabs, side_probabilities):
    configs = {}
    assert n_slabs in (5, 7)
    for side_prob in side_probabilities:
        try:
            data_conf = deepcopy(base_data_config)
            slabs = np.array([2] + [n_slabs] * 49)
            if n_slabs == 7:
                center_prob = (1.0 - 2 * side_prob) / 2.0
                slab_probs = [side_prob, 0.25, center_prob, 0.5, center_prob, 0.25, side_prob]
            else:
                center_prob = 1.0 - 2 * side_prob
                slab_probs = [side_prob, 0.5, center_prob, 0.5, side_prob]
            data_conf.update({'slabs': slabs,
                              'slab_probabilities': [[1.0, 1.0]] + [slab_probs] * 49})
            dataset = LinearSlabDataset.generate(**data_conf)
            train_data, val_data = dataset.split_train_val(TRAIN_SIZE)

            trainer_conf = deepcopy(base_trainer_config)
            trainer_conf.update(
                logger=logger,
                work_dir='{}/output/training_logs_{}/{}slabs_sideprob_{:.5f}/'.format(codebase, timestamp, n_slabs,
                                                                                      side_prob)
            )
            s_randomized = train_data.randomize_axes((0,))
            trainer_conf['train_data']['dataset'] = train_data
            trainer_conf['val_data']['dataset'] = val_data
            trainer_conf['additional_data']['s_randomized']['dataset'] = s_randomized
            trainer_conf['additional_data']['sc_randomized']['dataset'] = train_data.randomize_axes(tuple(range(1, 50)))

            ref_trainer_conf = deepcopy(base_trainer_config)
            ref_trainer_conf.update(
                logger=logger,
                work_dir='{}/output/training_logs_{}/{}slabs_ref_sideprob_{:.5f}/'.format(codebase, timestamp, n_slabs,
                                                                                          side_prob)
            )
            ref_trainer_conf['train_data']['dataset'] = s_randomized
            ref_trainer_conf['val_data']['dataset'] = val_data
            ref_trainer_conf['additional_data']['s_randomized']['dataset'] = s_randomized
            ref_trainer_conf['additional_data']['sc_randomized']['dataset'] = train_data.randomize_axes(
                tuple(range(1, 50)))
            configs[side_prob] = trainer_conf, ref_trainer_conf
        except Exception as e:
            logger.exception(f'Error for side_prob = {side_prob}', exc_info=sys.exc_info())
    return configs


def input_dim_configs(n_slabs, input_dims):
    configs = {}
    assert n_slabs in (5, 7)
    for input_dim in input_dims:
        try:
            data_conf = deepcopy(base_data_config)
            data_conf['num_dim'] = input_dim
            data_conf['slabs'] = np.array([2] + [n_slabs] * (input_dim - 1))
            data_conf['noise_proportions'] = np.array([0.1] + [0] * (input_dim - 1))
            if n_slabs == 7:
                slab_probs = [1 / 16.0, 0.25, 7 / 16.0, 0.5, 7 / 16.0, 0.25, 1 / 16.0]
            else:
                slab_probs = [0.125, 0.5, 0.75, 0.5, 0.125]
            data_conf['slab_probabilities'] = [[1.0, 1.0]] + [slab_probs] * (input_dim - 1)
            dataset = LinearSlabDataset.generate(**data_conf)
            train_data, val_data = dataset.split_train_val(TRAIN_SIZE)

            trainer_conf = deepcopy(base_trainer_config)
            trainer_conf.update(
                logger=logger,
                work_dir='{}/output/training_logs_{}/{}slabs_inputdim_{}/'.format(codebase, timestamp,
                                                                                  n_slabs, input_dim)
            )
            s_randomized = train_data.randomize_axes((0,))
            trainer_conf['model']['input_dim'] = input_dim
            trainer_conf['train_data']['dataset'] = train_data
            trainer_conf['val_data']['dataset'] = val_data
            trainer_conf['additional_data']['s_randomized']['dataset'] = s_randomized
            trainer_conf['additional_data']['sc_randomized']['dataset'] = train_data.randomize_axes(
                tuple(range(1, input_dim)))

            # reference, trained only on sc-randomized
            ref_trainer_conf = deepcopy(base_trainer_config)
            ref_trainer_conf.update(
                logger=logger,
                work_dir='{}/output/training_logs_{}/{}slabs_ref_inputdim_{}/'.format(codebase, timestamp,
                                                                                      n_slabs, input_dim)
            )
            ref_trainer_conf['model']['input_dim'] = input_dim
            ref_trainer_conf['train_data']['dataset'] = s_randomized
            ref_trainer_conf['val_data']['dataset'] = val_data
            ref_trainer_conf['additional_data']['s_randomized']['dataset'] = s_randomized
            ref_trainer_conf['additional_data']['sc_randomized']['dataset'] = train_data.randomize_axes(
                tuple(range(1, input_dim)))
            configs[input_dim] = trainer_conf, ref_trainer_conf
        except Exception as e:
            logger.exception(f'Error for input_dim={input_dim}', exc_info=sys.exc_info())
    return configs


def n_linear_configs(n_slabs, input_dims):
    configs = {}
    assert n_slabs in (5, 7)
    for linear_dim, slab_dim in input_dims:
        data_conf = deepcopy(base_data_config)
        data_conf['num_dim'] = linear_dim + slab_dim
        data_conf['slabs'] = np.array([2] * linear_dim + [n_slabs] * slab_dim)
        data_conf['noise_proportions'] = np.array([0.1] * linear_dim + [0] * slab_dim)
        if n_slabs == 7:
            slab_probs = [1 / 16.0, 0.25, 7 / 16.0, 0.5, 7 / 16.0, 0.25, 1 / 16.0]
        else:
            slab_probs = [0.125, 0.5, 0.75, 0.5, 0.125]
        data_conf['slab_probabilities'] = [[1.0, 1.0]] * linear_dim + [slab_probs] * slab_dim
        dataset = LinearSlabDataset.generate(**data_conf)
        train_data, val_data = dataset.split_train_val(TRAIN_SIZE)

        trainer_conf = deepcopy(base_trainer_config)
        trainer_conf.update(
            logger=logger,
            work_dir='{}/output/training_logs_{}/{}slabs_lineardim_{}_slabdim_{}/'.format(codebase, timestamp, n_slabs,
                                                                                          linear_dim, slab_dim)
        )
        trainer_conf['model']['input_dim'] = linear_dim + slab_dim
        trainer_conf['train_data']['dataset'] = train_data
        trainer_conf['val_data']['dataset'] = val_data
        s_randomized_data = train_data.randomize_axes(tuple(range(linear_dim)))
        trainer_conf['additional_data']['s_randomized']['dataset'] = s_randomized_data
        trainer_conf['additional_data']['sc_randomized']['dataset'] = train_data.randomize_axes(
            tuple(range(linear_dim, linear_dim + slab_dim)))

        ref_trainer_conf = deepcopy(base_trainer_config)
        ref_trainer_conf.update(
            logger=logger,
            work_dir='{}/output/training_logs_{}/{}slabs_ref_lineardim_{}_slabdim_{}/'.format(codebase, timestamp,
                                                                                              n_slabs,
                                                                                              linear_dim, slab_dim)
        )
        ref_trainer_conf['model']['input_dim'] = linear_dim + slab_dim
        ref_trainer_conf['train_data']['dataset'] = s_randomized_data
        ref_trainer_conf['val_data']['dataset'] = val_data
        ref_trainer_conf['additional_data']['s_randomized']['dataset'] = s_randomized_data
        ref_trainer_conf['additional_data']['sc_randomized']['dataset'] = train_data.randomize_axes(
            tuple(range(linear_dim, linear_dim + slab_dim)))
        configs[linear_dim, slab_dim] = trainer_conf, ref_trainer_conf
    return configs


# I actually would prefer calling the param "momentums"
# but the spell check has some Latin origin though ...
def hyperparam_tuning(n_slabs, latent_dims, num_layers, lrs, momenta):
    assert n_slabs == 7
    assert len(latent_dims) == len(num_layers) == len(lrs) == len(momenta)
    dataset = LinearSlabDataset.generate(**base_data_config)
    train_data, val_data = dataset.split_train_val(TRAIN_SIZE)
    s_randomized = train_data.randomize_axes((0,))
    sc_randomized = train_data.randomize_axes(tuple(range(1, 50)))
    configs = {}

    for latent_dim, nlayers, lr, momentum in zip(latent_dims, num_layers, lrs, momenta):
        try:
            trainer_conf = deepcopy(base_trainer_config)
            trainer_conf['model'].update(
                num_layers=nlayers,
                latent_dim=latent_dim
            )
            trainer_conf['optimizer'].update(
                lr=lr,
                momentum=momentum
            )
            ref_trainer_conf = copy.deepcopy(trainer_conf)
            trainer_conf.update(
                logger=logger,
                work_dir='{}/output/training_logs_{}/{}slabs_{}_{}_lr{}_m{}/'.format(
                    codebase, timestamp, n_slabs, latent_dim, nlayers, lr, momentum)
            )
            ref_trainer_conf.update(
                logger=logger,
                work_dir='{}/output/training_logs_{}/{}slabs_ref_{}_{}_lr{}_m{}/'.format(
                    codebase, timestamp, n_slabs, latent_dim, nlayers, lr, momentum)
            )
            trainer_conf['train_data']['dataset'] = train_data
            trainer_conf['val_data']['dataset'] = val_data
            trainer_conf['additional_data']['s_randomized']['dataset'] = s_randomized
            trainer_conf['additional_data']['sc_randomized']['dataset'] = sc_randomized
            ref_trainer_conf['train_data']['dataset'] = s_randomized
            ref_trainer_conf['val_data']['dataset'] = val_data
            ref_trainer_conf['additional_data']['s_randomized']['dataset'] = s_randomized
            ref_trainer_conf['additional_data']['sc_randomized']['dataset'] = sc_randomized

            configs[latent_dim, nlayers, lr, momentum] = trainer_conf, ref_trainer_conf
        except Exception as e:
            logger.exception(f'Error for params {latent_dim, nlayers, lr, momentum}', exc_info=sys.exc_info())

    return configs


def noise_proportion_configs(n_slabs, noise_props):
    configs = {}
    assert n_slabs in (5, 7)
    for noise_prop in noise_props:
        try:
            data_conf = deepcopy(base_data_config)
            slabs = np.array([2] + [n_slabs] * 49)
            if n_slabs == 7:
                slab_probs = [1 / 16.0, 0.25, 7 / 16.0, 0.5, 7 / 16.0, 0.25, 1 / 16.0]
            else:
                slab_probs = [0.125, 0.5, 0.75, 0.5, 0.125]
            data_conf.update({'slabs': slabs,
                              'noise_proportions': np.array([noise_prop] + [0] * 49),
                              'slab_probabilities': [[1.0, 1.0]] + [slab_probs] * 49})
            dataset = LinearSlabDataset.generate(**data_conf)
            train_data, val_data = dataset.split_train_val(TRAIN_SIZE)
            s_randomized = train_data.randomize_axes((0,))

            trainer_conf = deepcopy(base_trainer_config)
            trainer_conf.update(
                logger=logger,
                work_dir='{}/output/training_logs_{}/{}slabs_noiseprop_{:.3f}/'.format(codebase, timestamp, n_slabs,
                                                                                      noise_prop)
            )
            trainer_conf['train_data']['dataset'] = train_data
            trainer_conf['val_data']['dataset'] = val_data
            trainer_conf['additional_data']['s_randomized']['dataset'] = s_randomized
            trainer_conf['additional_data']['sc_randomized']['dataset'] = train_data.randomize_axes(tuple(range(1, 50)))

            ref_trainer_conf = deepcopy(base_trainer_config)
            ref_trainer_conf.update(
                logger=logger,
                work_dir='{}/output/training_logs_{}/{}slabs_ref_noiseprop_{:.3f}/'.format(codebase, timestamp, n_slabs,
                                                                                          noise_prop)
            )
            ref_trainer_conf['train_data']['dataset'] = s_randomized
            ref_trainer_conf['val_data']['dataset'] = val_data
            ref_trainer_conf['additional_data']['s_randomized']['dataset'] = s_randomized
            ref_trainer_conf['additional_data']['sc_randomized']['dataset'] = train_data.randomize_axes(
                tuple(range(1, 50)))
            configs[noise_prop] = trainer_conf, ref_trainer_conf
        except Exception as e:
            logger.exception(f'Error for noise_prop = {noise_prop}', exc_info=sys.exc_info())
    return configs

def nsamples_configs(n_slabs, nsamples):
    configs = {}
    assert n_slabs in (5, 7)
    for sample_size in nsamples:
        try:
            data_conf = deepcopy(base_data_config)
            slabs = np.array([2] + [n_slabs] * 49)
            if n_slabs == 7:
                slab_probs = [1 / 16.0, 0.25, 7 / 16.0, 0.5, 7 / 16.0, 0.25, 1 / 16.0]
            else:
                slab_probs = [0.125, 0.5, 0.75, 0.5, 0.125]
            data_conf.update({'slabs': slabs,
                              'slab_probabilities': [[1.0, 1.0]] + [slab_probs] * 49,
                              'num_samples': sample_size + 10000,})
            dataset = LinearSlabDataset.generate(**data_conf)
            train_data, val_data = dataset.split_train_val(sample_size)

            trainer_conf = deepcopy(base_trainer_config)
            trainer_conf.update(
                logger=logger,
                work_dir='{}/output/training_logs_{}/{}slabs_nsamples_{}/'.format(codebase, timestamp, n_slabs,
                                                                                                       sample_size)
            )
            s_randomized = train_data.randomize_axes((0,))
            trainer_conf['train_data']['dataset'] = train_data
            trainer_conf['val_data']['dataset'] = val_data
            trainer_conf['additional_data']['s_randomized']['dataset'] = s_randomized
            trainer_conf['additional_data']['sc_randomized']['dataset'] = train_data.randomize_axes(tuple(range(1, 50)))

            ref_trainer_conf = deepcopy(base_trainer_config)
            ref_trainer_conf.update(
                logger=logger,
                work_dir='{}/output/training_logs_{}/{}slabs_ref_nsamples_{}/'.format(codebase, timestamp, n_slabs,
                                                                                                           sample_size)
            )
            ref_trainer_conf['train_data']['dataset'] = s_randomized
            ref_trainer_conf['val_data']['dataset'] = val_data
            ref_trainer_conf['additional_data']['s_randomized']['dataset'] = s_randomized
            ref_trainer_conf['additional_data']['sc_randomized']['dataset'] = train_data.randomize_axes(
                tuple(range(1, 50)))
            configs[sample_size] = trainer_conf, ref_trainer_conf
        except Exception as e:
            logger.exception(f'Error for sample_size = {sample_size}', exc_info=sys.exc_info())
    return configs


def execute_config(func, *args, **kwargs):
    for trainer_conf, *ref_conf in func(*args, **kwargs).values():
        logger.info(f'Trainer configuration: \n {trainer_conf}')
        trainer = Trainer(**trainer_conf)
        trainer.run()
        if ref_conf is not None:
            ref_conf, = ref_conf
            logger.info(f'Reference configuration: \n {ref_conf}')
            trainer = Trainer(**ref_conf)
            trainer.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reverse', action='store_true')
    args = parser.parse_args()

    params = [
        # (input_dim_configs, 7, [2, 3, 5, 7] + list(range(10, 51, 5))),
        # (side_prob_configs, 7, np.linspace(1 / 32.0, 1 / 2.0, num=20, endpoint=False)),
        # (input_dim_configs, 5, [40, 50, 80, 100, 120, 150]),
        # (side_prob_configs, 5, np.linspace(1 / 64.0, 1 / 2.0, num=30, endpoint=False)),
        # (n_linear_configs, 5, [(l, 50 - l) for l in range(5, 50, 5)]),
        # (noise_proportion_configs, 7, np.linspace(0.1, 0.6, num=11, endpoint=True)),
        (nsamples_configs, 7, list(range(20000, 250000, 20000))),
    ]

    if args.reverse:
        params = params[::-1]
    for param in params:
        execute_config(*param)

    # execute_config(hyperparam_tuning, 7,
    #                [100, 200, 200, 200, 300, 300, 300, 300],
    #                [  2,   2,   2,   2,   2,   2,   2,   2],
    #                [0.3, 0.1, 0.1, 0.3, 0.1, 0.3, 0.1, 0.3],
    #                [0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.9, 0.9],
    #                )


if __name__ == '__main__':
    main()
