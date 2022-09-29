__all__ = ['Trainer']

import os
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data
from logging import Logger
from trainers.utils import build_optimizer


class Trainer:
    def __init__(self, train_data: dict, val_data: dict, model: nn.Module, loss: nn.Module, device,
                 evaluate_interval, save_interval, work_dir, loss_eps: float, logger: Logger,
                 max_steps, optimizer: dict):
        """
        train a model until loss convergence
        :param train_data: dict of params to be passed to the construction of the train dataloader
        :param val_data: dict of params to be passed to the construction of the val dataloader
        :param model:
        :param loss:
        :param device: 'cpu' or 'cuda'
        :param evaluate_interval: updates between two evaluations
        :param save_interval: updates between two checkpoints, 0 for no saving
        :param work_dir: the directory for storing checkpoints and tensorboard logs
        :param optimizer: ('cls' => 'SGD', **kwargs)
        """
        train_data = train_data.copy()
        val_data = val_data.copy()
        train_data['dataset'] = train_data['dataset']
        val_data['dataset'] = val_data['dataset']
        self.train_data = torch.utils.data.DataLoader(**train_data)
        self.val_data = torch.utils.data.DataLoader(**val_data)

        self.model = model.to(device)
        self.loss = loss.to(device)
        self.loss_eps = loss_eps
        self.evaluate_interval = evaluate_interval
        self.save_interval = save_interval
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
        self.writer = SummaryWriter(work_dir)
        self.logger = logger
        self.device = device
        self.optimizer = build_optimizer(self.model.parameters(), **optimizer)
        self.max_steps = max_steps

        logger.info('model: {}'.format(model))

    def preprocess(self, x_batch, y_batch):
        x_batch = x_batch.float().to(device=self.device)
        y_batch = y_batch.long().to(device=self.device)

        return x_batch, y_batch

    def train_step(self, x_batch, y_batch):
        self.model.train()
        self.optimizer.zero_grad()
        x_batch, y_batch = self.preprocess(x_batch, y_batch)
        logits = self.model(x_batch)
        loss = self.loss(logits, y_batch)
        loss.backward()
        self.optimizer.step()
        return loss

    def step_evaluate(self) -> Dict[str, float]:
        """
        quick evaluation in training
        :param ref_tensor: a tensor for determining dtype and device
        """
        results = {}
        with torch.no_grad():
            self.model.eval()
            for name, data in [('Train', self.train_data), ('Val', self.val_data)]:
                total_loss = np.array([0.0])
                correct = count = 0
                for x, y in data:
                    x, y = self.preprocess(x, y)
                    logits = self.model(x)
                    loss = self.loss(logits, y)
                    total_loss += loss.cpu().numpy()
                    pred = torch.argmax(logits, 1)
                    assert pred.shape == y.shape
                    correct += (pred == y).sum().cpu().item()
                    count += len(y)
                results['{}/SumLoss'.format(name)] = total_loss.item()
                results['{}/Accuracy'.format(name)] = float(correct) / count
        return results

    def stop_criteria(self, step_number, eval_results):
        loss = eval_results['Train/SumLoss']
        return step_number > self.max_steps or loss < self.loss_eps or not np.isfinite(loss)

    def final_evaluate(self):
        # TODO: compute AUCs, etc.
        pass

    def log_eval_results(self, step, eval_results: dict):
        for name, item in eval_results.items():
            self.writer.add_scalar(name, item, global_step=step)
        self.logger.info('Step: {}\n{}'.format(
            step,
            '\n'.join(['{}: {}'.format(k, v) for k, v in eval_results.items()])
        ))

    def run(self):
        step = 0
        stop = False
        while not stop:
            for x_batch, y_batch in self.train_data:
                loss = self.train_step(x_batch, y_batch)
                self.writer.add_scalar('Train/Loss', loss.cpu().item(),
                                       global_step=step)
                if self.save_interval > 0 and step % self.save_interval == 0:
                    torch.save(dict(model=self.model.state_dict(), optim=self.optimizer.state_dict()),
                               os.path.join(self.work_dir, 'step_{}.pth'.format(step)))
                if step % self.evaluate_interval == 0:
                    self.logger.info('Step {}: Loss {}'.format(step, loss.cpu().item()))
                    self.logger.info('Evaluating ...')
                    eval_results = self.step_evaluate()
                    self.log_eval_results(step, eval_results)
                    if self.stop_criteria(step, eval_results):
                        stop = True
                if stop:
                    break

                step += 1
        self.logger.info('Train finished with {} steps'.format(step + 1))
