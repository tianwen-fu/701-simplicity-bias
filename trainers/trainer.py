__all__ = ['Trainer', 'FixedScheduleTrainer']

import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from logging import Logger
from typing import Dict
from sklearn.metrics import roc_auc_score
from models import Model


class Trainer:
    def __init__(self, dataloaders: Dict[str, DataLoader], model: Model, device, premature_evaluate_interval,
                 evaluate_interval, save_interval, work_dir, accuracy_threshold: float,
                 logger: Logger, max_steps, optimizer: Optimizer):
        """
        train a model until loss convergence
        :param model:
        :param device: 'cpu' or 'cuda'
        :param evaluate_interval: updates between two evaluations
        :param save_interval: updates between two checkpoints, 0 for no saving
        :param work_dir: the directory for storing checkpoints and tensorboard logs
        :param optimizer: ('cls' => 'SGD', **kwargs)
        """
        self.train_data = dataloaders['Train']
        self.dataloaders = dataloaders

        self.model = model
        self.accuracy_threshold = accuracy_threshold
        self.premature_evaluate_interval = premature_evaluate_interval
        self.evaluate_interval = evaluate_interval
        self.save_interval = save_interval
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
        self.writer = SummaryWriter(work_dir)
        self.logger = logger
        self.device = device
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.patience = 1  # patience to wait for loss convergence

    def train_step(self, x_batch, y_batch):
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(x_batch)
        loss = self.model.get_loss(logits, y_batch)
        loss.backward()
        self.optimizer.step()

        # compute stats
        pred = torch.argmax(logits.detach(), 1)
        accuracy = ((pred == y_batch).sum().float() / len(y_batch)).cpu().item()
        return loss, dict(
            loss=loss.detach().cpu().item(),
            acc=accuracy)

    def short_evaluate(self) -> Dict[str, float]:
        """
        quick evaluation in training
        :param ref_tensor: a tensor for determining dtype and device
        """
        results = {}
        with torch.no_grad():
            self.model.eval()
            with torch.no_grad():
                for name, data in self.dataloaders.items():
                    total_loss = np.array([0.0])
                    logits = []
                    ys = []
                    for x, y in data:
                        logit = self.model(x)
                        loss = self.model.get_loss(logit, y)
                        total_loss += loss.cpu().numpy() * len(y)  # default: reduction='mean'
                        logits.append(logit.detach())
                        ys.append(y.detach())
                    logits = torch.cat(logits, dim=0)
                    y = torch.cat(ys, dim=0)
                    pred = torch.argmax(logits, 1)
                    assert pred.shape == y.shape
                    scores = logits[:, 1] - logits[:, 0]

                    results['{}/AUC'.format(name)] = roc_auc_score(y.cpu().numpy(), scores.cpu().numpy())
                    results['{}/AverageLoss'.format(name)] = total_loss.item() / y.shape[0]
                    results['{}/Accuracy'.format(name)] = (pred == y).sum().cpu().float() / y.shape[0]
        return results

    def stop_criteria(self, step_number, eval_results):
        loss = eval_results['Train/AverageLoss']
        acc = eval_results['Train/Accuracy']
        if acc > self.accuracy_threshold:
            if self.patience == 0:
                return True
            self.patience -= 1
        else:
            self.patience = 10000 // self.evaluate_interval
        return step_number > self.max_steps or not np.isfinite(loss)

    def log_eval_results(self, step, eval_results: dict):
        for name, item in eval_results.items():
            self.writer.add_scalar(name, item, global_step=step)
        self.logger.info('Step: {}\n{}'.format(
            step,
            '\n'.join(['{}: {}'.format(k, v) for k, v in eval_results.items()])
        ))

    def run(self, wandb_logger=None):
        self.logger.info(f'Started, logging to {self.work_dir}...')

        # flush tiny numbers to zero to prevent severe CPU performance degradation
        # see https://discuss.pytorch.org/t/training-time-gets-slower-and-slower-on-cpu/145483/5
        torch.set_flush_denormal(True)

        step = 0
        stop = False
        while not stop:
            for x_batch, y_batch in self.train_data:
                loss, stats = self.train_step(x_batch, y_batch)
                for k, v in stats.items():
                    self.writer.add_scalar('Train_Step/{}'.format(k), v,
                                           global_step=step)
                    if wandb_logger is not None:
                        wandb_logger.log({'Train_Step/{}'.format(k): v}, step=step, commit=False)
                if self.save_interval > 0 and step % self.save_interval == 0:
                    torch.save(dict(model=self.model.state_dict(), optim=self.optimizer.state_dict()),
                               os.path.join(self.work_dir, 'step_{}.pth'.format(step)))
                if (step < self.evaluate_interval and step % self.premature_evaluate_interval == 0) or (
                        step % self.evaluate_interval == 0):
                    self.logger.info('Step {}: Loss {}'.format(step, loss.cpu().item()))
                    self.logger.info('Evaluating ...')
                    eval_results = self.short_evaluate()
                    self.log_eval_results(step, eval_results)
                    if wandb_logger is not None:
                        wandb_logger.log(eval_results, step=step)
                    if self.stop_criteria(step, eval_results):
                        stop = True
                if stop:
                    break

                step += 1
        torch.save(dict(model=self.model.state_dict()), os.path.join(self.work_dir, 'final.pth'))
        self.logger.info('Train finished with {} steps'.format(step + 1))


class FixedScheduleTrainer(Trainer):
    def __init__(self, dataloaders: Dict[str, DataLoader], model: Model, device, premature_evaluate_interval,
                 evaluate_interval, save_interval, work_dir, accuracy_threshold: float, logger: Logger, max_steps,
                 optimizer: Optimizer):
        super().__init__(dataloaders, model, device, premature_evaluate_interval, evaluate_interval, save_interval,
                         work_dir, accuracy_threshold, logger, max_steps, optimizer)

    def stop_criteria(self, step_number, eval_results):
        return step_number > self.max_steps
