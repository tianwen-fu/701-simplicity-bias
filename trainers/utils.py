import torch
import torch.optim
import torch.utils.data
from synthetic_data.linear_slabs import LinearSlabDataset

__all__ = ['build_optimizer', 'build_dataset', 'build_dataloader']


def build_optimizer(params, cls, **kwargs):
    if cls == 'SGD':
        return torch.optim.SGD(params=params, **kwargs)
    else:
        raise NotImplementedError('unknown optimizer {}'.format(cls))


datasets_registry = {
    'linear_slabs': LinearSlabDataset.from_file
}


def build_dataset(cls, **kwargs):
    return datasets_registry[cls](**kwargs)


def build_dataloader(dataset, **kwargs):
    if isinstance(dataset, dict):
        dataset = build_dataset(**dataset)
    return torch.utils.data.DataLoader(dataset=dataset, **kwargs)
