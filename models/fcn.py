__all__ = ['generate_fcn']

import torch
from torch import nn
from typing import Callable, Optional


def kaiming_init_relu(data: torch.Tensor):
    nn.init.kaiming_uniform_(data.data, nonlinearity='relu')


def generate_fcn(num_layers, input_dim, output_dim, latent_dim,
                 activation=nn.ReLU, use_bn=False, dropout_probability=0.0,
                 linear_init: Optional[Callable[[torch.Tensor], None]] = kaiming_init_relu) -> nn.Module:
    if isinstance(activation, str):
        if activation == 'ReLU':
            activation = nn.ReLU
        elif activation == 'sigmoid':
            activation = nn.Sigmoid
        elif activation == 'tanh':
            activation = nn.Tanh
        else:
            raise NotImplementedError(f'Unknown Activation {activation}')

    layers = [nn.Linear(input_dim, latent_dim), activation()]
    if dropout_probability > 0: layers.append(nn.Dropout(dropout_probability))
    if use_bn: layers.append(nn.BatchNorm1d(latent_dim))
    for _ in range(num_layers - 1):
        layers.extend([nn.Linear(latent_dim, latent_dim), activation()])
        if dropout_probability > 0: layers.append(nn.Dropout(dropout_probability))
        if use_bn: layers.append(nn.BatchNorm1d(latent_dim))
    layers.append(nn.Linear(latent_dim, output_dim))
    model = nn.Sequential(*layers)

    if linear_init is not None:
        for layer in model:
            if isinstance(layer, nn.Linear):
                linear_init(layer.weight)
    return model
