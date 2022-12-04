import itertools
import logging
import os
import pickle
import sys
import traceback
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from logging import StreamHandler

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import runner
import numpy as np
from abc import ABCMeta, abstractmethod
from configs import lms_7_fcn300_2, lms_5_fcn300_2, ms_57_fcn300_2


def format_exp_name(config):
    name = '{}slabs_n{}_p{:.4f}_d{}_noise{:.2f}_fcn{}_{}_{}_lr{}_mom{}_seed{}'.format(
        config['data']['slabs'][1]['val'],
        config['data']['train_samples'],
        config['data']['slab_probabilities'][1]['val'][0],
        config['data']['num_dim'],
        config['data']['noise_proportions'][0]['val'],
        config['model']['latent_dim'],
        config['model']['num_layers'],
        config['model']['activation'],
        config['optimizer']['lr'],
        config['optimizer']['momentum'],
        config['seed'])
    if config['data']['slabs'][0]['val'] == 5:
        assert name.startswith('7slabs')
        name = name.replace('7slabs_', 'ms57_')
    return name


class ExperimentSetup(metaclass=ABCMeta):
    def __init__(self, n_slabs, first_n_slabs=2, base_config=None):
        self.n_slabs = n_slabs
        if first_n_slabs == 2:
            self._base_config = base_config or (lms_5_fcn300_2 if n_slabs == 5 else lms_7_fcn300_2)
        else:
            assert first_n_slabs == 5 and n_slabs == 7  # ms57
            self._base_config = ms_57_fcn300_2

    @property
    def base_config(self):
        return self._base_config

    @base_config.setter
    def base_config(self, new_base_config):
        self._base_config = new_base_config
        assert self._base_config['data']['slabs'][1]['val'] == self.n_slabs

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
        config['model']['input_dim'] = num_dim
        return config


class NoiseProportion(ExperimentSetup):
    def generate_config(self, config, n_slabs, noise_prop):
        config['data']['noise_proportions'][0]['val'] = noise_prop
        return config


class NumSamples(ExperimentSetup):
    def generate_config(self, config, n_slabs, num_samples):
        config['data']['train_samples'] = num_samples
        return config


class ModelSize(ExperimentSetup):
    def generate_config(self, config, n_slabs, latent_dim):
        config['trainer']['accuracy_threshold'] = 0.999
        config['model']['latent_dim'] = latent_dim
        return config


class InputDimConverge(ExperimentSetup):
    def generate_config(self, config, n_slabs, num_dim):
        config['data']['num_dim'] = num_dim
        config['model']['input_dim'] = num_dim
        config['trainer']['accuracy_threshold'] = 1.0
        config['trainer']['max_steps'] = 2000000
        config['scheduler']['cls'] = 'StepLR'
        config['scheduler']['step_size'] = 100000
        return config


class HyperparamTuning(ExperimentSetup):
    def __init__(self, n_slabs, n_samples, first_n_slabs=2, base_config=None):
        super().__init__(n_slabs, first_n_slabs, base_config)
        self.base_config = deepcopy(self.base_config)
        self.base_config['data']['train_samples'] = n_samples
        # use shorter schedule
        self.base_config['trainer']['max_steps'] = 80_000

    def generate_config(self, config, n_slabs, param):
        latent_dim, num_layers, lr, momentum = param
        config['model'].update(num_layers=num_layers, latent_dim=latent_dim)
        config['optimizer'].update(lr=lr, momentum=momentum)
        return config


class ActivationFunction(ExperimentSetup):
    def generate_config(self, config, n_slabs, activation):
        config['model']['activation'] = activation
        return config


def compose_configs(base_configs, other_setup: ExperimentSetup, other_params):
    configs = []
    for config in base_configs:
        other_setup.base_config = config
        configs.extend(list(other_setup(other_params)))
    return configs


setups = {
    '5slab_sideprob': SideProb(5)(np.linspace(1 / 64.0, 1 / 2.0, num=30, endpoint=False)),
    '7slab_sideprob': SideProb(7)(np.linspace(1 / 32.0, 1 / 2.0, num=20, endpoint=False)),
    '5slab_inputdim': InputDim(5)((40, 50, 80, 100, 120, 150)),
    '7slab_inputdim': InputDim(7)((40, 50, 80, 100, 120, 150)),
    '7slab_noiseprop': NoiseProportion(7)(np.linspace(0.1, 0.7, num=15, endpoint=True)),
    '7slab_nsamples': NumSamples(7)((20000, 40000, 100000, 125000, 150000, 175000, 200000, 500000)),
    '7slab_40dim': InputDimConverge(7)((40,)),
    'toy_conv': InputDimConverge(5)((40,)),
    '7slab_modelsize': ModelSize(7)((300, 400, 500, 700, 1000)),
    '7slab_largeSample_hyperparam': HyperparamTuning(7, 200000)(itertools.product(
        (300, 500, 1000),  # latent dim
        (2, 3, 5, 7, 10),  # num_layers
        (0.05, 0.1, 0.3, 0.5),  # lr
        (0.0, 0.9)
    )),
    '5slab_sideprob_act': compose_configs(
        ActivationFunction(5)(('sigmoid', 'tanh', 'ReLU')),
        SideProb(5), np.linspace(1 / 64.0, 1 / 2.0, num=10, endpoint=False)
    ),
    '7slab_sideprob_act': compose_configs(
        ActivationFunction(7)(('sigmoid', 'tanh', 'ReLU')),
        SideProb(7), np.linspace(1 / 32.0, 1 / 2.0, num=10, endpoint=False)
    ),
    'ms57_noiseprop': compose_configs(
        NumSamples(7, 5)((100000,)),
        NoiseProportion(7, 5), np.linspace(0.1, 0.7, num=15, endpoint=True)
    ),
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
    parser.add_argument('--exp-results-data', type=str, default=os.path.join(root_log_dir, timestamp, 'metrics.pkl'))
    parser.add_argument('--no-save-data', action='store_true')
    args = parser.parse_args()
    return args


def update_metric_records(record_file, config, result):
    with open(record_file, 'rb') as in_data:
        data = pickle.load(in_data)
    config = repr(config)
    if config in data:
        data[config].append(result)
    else:
        data[config] = [result]
    with open(record_file, 'wb') as out_data:
        pickle.dump(data, out_data)


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(args.log_dir, 'meta.log'), level=logging.DEBUG)
    root_logger = logging.getLogger('MidwayReportExps')
    root_logger.addHandler(StreamHandler())
    # database for final metrics
    if not os.path.isfile(args.exp_results_data):
        with open(args.exp_results_data, 'wb') as out_f:
            pickle.dump({}, out_f)
    # avoid running duplicate setups (maybe duplicate of baselines)
    completed_setups = set()
    for seed in args.seeds:
        for setup_name in args.experiments:
            for config in setups[setup_name]:
                exp_name = None
                try:
                    config['seed'] = seed
                    exp_name = format_exp_name(config)
                    root_logger.info(f'Running {exp_name}')
                    if exp_name in completed_setups:
                        root_logger.warning(f'Skipping duplicate experiment {exp_name} in {setup_name}')
                    else:
                        result = runner.run(exp_name, config, log_dir=args.log_dir, seed=seed,
                                            wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
                                            save_data=not args.no_save_data)
                        completed_setups.add(exp_name)

                        # save metrics
                        update_metric_records(args.exp_results_data, config, result)
                except KeyboardInterrupt:
                    sys.exit(1)
                except:
                    root_logger.error(f'Error running experiment {exp_name or setup_name}', exc_info=sys.exc_info())
                    traceback.print_exc()


if __name__ == '__main__':
    main()
