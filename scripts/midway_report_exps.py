import os
import sys
from argparse import ArgumentParser
from datetime import datetime
from copy import deepcopy
import traceback
import logging
from logging import StreamHandler

import runner
import numpy as np
from abc import ABCMeta, abstractmethod
from configs import lms_7_fcn300_2, lms_5_fcn300_2


def format_exp_name(config, seed):
    name = '{}slabs_n{}_p{:.4f}_d{}_noise{:.2f}_seed{}'.format(
        config['data']['slabs'][1]['val'],
        config['data']['train_samples'],
        config['data']['slab_probabilities'][1]['val'][0],
        config['data']['num_dim'],
        config['data']['noise_proportions'][0]['val'],
        seed)
    return name


class ExperimentSetup(metaclass=ABCMeta):
    def __init__(self, n_slabs, base_config=None):
        self.n_slabs = n_slabs
        self.base_config = base_config or (lms_5_fcn300_2 if n_slabs == 5 else lms_7_fcn300_2)

    @abstractmethod
    def generate_config(self, config, n_slabs, param):
        pass

    def __call__(self, params):
        configs = [self.generate_config(deepcopy(self.base_config), self.n_slabs, param) for param in params]
        return configs


class SideProb(ExperimentSetup):
    def generate_config(self, config, n_slabs, side_prob):
        if n_slabs == 7:
            center_prob = (1.0 - 2 * side_prob) / 2.0
            slab_probs = (side_prob, 0.25, center_prob, 0.5, center_prob, 0.25, side_prob)
        else:
            center_prob = 1.0 - 2 * side_prob
            slab_probs = (side_prob, 0.5, center_prob, 0.5, side_prob)
        config['data']['slab_probabilities'][1]['val'] = slab_probs
        return config


class InputDim(ExperimentSetup):
    def generate_config(self, config, n_slabs, num_dim):
        config['data']['num_dim'] = num_dim
        return config


class NoiseProportion(ExperimentSetup):
    def generate_config(self, config, n_slabs, noise_prop):
        config['data']['noise_proportions'][0]['val'] = noise_prop
        return config


class NumSamples(ExperimentSetup):
    def generate_config(self, config, n_slabs, num_samples):
        config['data']['train_samples'] = num_samples
        return config


class LinearDim(ExperimentSetup):
    def generate_config(self, config, n_slabs, linear_dim):
        slab_dim = config['data']['num_dim'] - linear_dim
        assert slab_dim > 0
        config['data']['slabs'][0]['count'] = linear_dim
        config['data']['noise_proportions'][0]['count'] = linear_dim
        config['data']['slab_probabilities'][0]['count'] = linear_dim
        config['data']['simple_axes'] = tuple(range(linear_dim))


setups = {
    '5slab_sideprob': SideProb(5)(np.linspace(1 / 64.0, 1 / 2.0, num=30, endpoint=False)),
    '7slab_sideprob': SideProb(7)(np.linspace(1 / 32.0, 1 / 2.0, num=20, endpoint=False)),
    '5slab_inputdim': InputDim(5)((40, 50, 80, 100, 120, 150)),
    '7slab_inputdim': InputDim(7)((40, 50, 80, 100, 120, 150)),
    '7slab_noiseprop': NoiseProportion(7)(np.linspace(0.1, 0.7, num=15, endpoint=True)),
    '7slab_nsamples': NumSamples(7)((20000, 40000, 100000, 200000, 500000))
}


def parse_args():
    parser = ArgumentParser('Midway Report Experiments')
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    root_log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'log'))
    parser.add_argument('--seeds', nargs='+', type=int, default=(0,))
    parser.add_argument('--log-dir', nargs='?', type=str, default=os.path.join(root_log_dir, timestamp))
    parser.add_argument('--experiments', nargs='+', type=str, choices=tuple(setups.keys()), required=True)
    parser.add_argument('--wandb-project', type=str)
    parser.add_argument('--wandb-entity', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(args.log_dir, 'meta.log'), level=logging.DEBUG)
    root_logger = logging.getLogger('MidwayReportExps')
    root_logger.addHandler(StreamHandler())
    # avoid running duplicate setups (maybe duplicate of baselines)
    completed_setups = set()
    for seed in args.seeds:
        for setup_name in args.experiments:
            for config in setups[setup_name]:
                try:
                    exp_name = format_exp_name(config, seed)
                    if exp_name in completed_setups:
                        root_logger.warning(f'Skipping duplicate experiment {exp_name} in {setup_name}')
                    else:
                        runner.run(format_exp_name(config, seed), config, log_dir=args.log_dir, seed=seed,
                                   wandb_project=args.wandb_project, wandb_entity=args.wandb_entity)
                except:
                    root_logger.error(f'Error running experiment {setup_name}', exc_info=sys.exc_info())
                    traceback.print_exc()


if __name__ == '__main__':
    main()
