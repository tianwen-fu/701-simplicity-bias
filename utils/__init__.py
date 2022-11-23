import torch
import torch.optim
import torch.utils.data

__all__ = ['build_optimizer']


def build_optimizer(params, cls, **kwargs):
    if cls == 'SGD':
        return torch.optim.SGD(params=params, **kwargs)
    else:
        raise NotImplementedError('unknown optimizer {}'.format(cls))
