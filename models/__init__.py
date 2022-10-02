__all__ = ['build_model', 'build_loss']

from torch import nn
from .fcn import generate_fcn

model_registry = {
    'fcn': generate_fcn
}

loss_registry = {
    'CE': nn.CrossEntropyLoss
}


def build_model(cls, **kwargs) -> nn.Module:
    func = model_registry[cls]
    return func(**kwargs)


def build_loss(cls, **kwargs) -> nn.Module:
    return loss_registry[cls](**kwargs)
