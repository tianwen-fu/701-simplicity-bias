__all__ = ['Model']

import torch
from torch import nn
from .fcn import generate_fcn
from copy import deepcopy

model_registry = {
    'fcn': generate_fcn
}

loss_registry = {
    'CrossEntropy': nn.CrossEntropyLoss
}


def build_model(cls, **kwargs) -> nn.Module:
    func = model_registry[cls]
    return func(**kwargs)


def build_loss(cls, **kwargs) -> nn.Module:
    return loss_registry[cls](**kwargs)


class Model(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        model_config = deepcopy(model_config)
        loss_config = model_config.pop('loss')
        self.classifier = build_model(**model_config)
        self.loss = build_loss(loss_config)

    def forward(self, x):
        return self.classifier(x)

    def get_loss(self, logits, labels) -> torch.Tensor:
        return self.loss(logits, labels)
