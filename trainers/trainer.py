__all__ = ['Trainer', 'FixedScheduleTrainer']

import os
from typing import Dict
import time
import psutil

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from logging import Logger
from trainers.utils import build_optimizer, build_dataloader
from models import build_model, build_loss
from typing import Dict
from sklearn.metrics import roc_auc_score


class Trainer:
    def __init__(self, train_data: dict, val_data: dict, model: dict, loss: dict, device,
                 evaluate_interval, save_interval, work_dir, accuracy_threshold: float, logger: Logger,
                 max_steps, optimizer: dict, additional_data: Dict[str, dict] = None):
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
        :param additional_data: additional data to evaluate on
        """
        train_data = train_data.copy()
        val_data = val_data.copy()
        self.train_data = build_dataloader(**train_data)
        self.val_data = build_dataloader(**val_data)

        self.model = build_model(**model).to(device=device)
        self.loss = build_loss(**loss).to(device)
        self.accuracy_threshold = accuracy_threshold
        self.evaluate_interval = evaluate_interval
        self.save_interval = save_interval
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
        self.writer = SummaryWriter(work_dir)
        self.logger = logger
        self.device = device
        self.optimizer = build_optimizer(self.model.parameters(), **optimizer)
        self.max_steps = max_steps
        self.patience = 1  # patience to wait for loss convergence

        # for computing throughput
        self.last_step = 0
        self.last_time = 0
        if additional_data is None: additional_data = {}
        self.additional_data = {k: build_dataloader(**dl) for k, dl in additional_data.items()}

        logger.info('model: {}'.format(self.model))

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
            dataloaders = {'Train': self.train_data, 'Val': self.val_data}
            dataloaders.update(self.additional_data)
            for name, data in dataloaders.items():
                total_loss = np.array([0.0])
                logits = []
                ys = []
                for x, y in data:
                    x, y = self.preprocess(x, y)
                    logit = self.model(x)
                    loss = self.loss(logit, y)
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

    def system_log(self, step):
        dic = {
            'System/CPULoadAvg5min': psutil.getloadavg()[1],
            'System/MemoryUsage': psutil.virtual_memory()[2]
        }
        if step > self.last_step:
            steps = step - self.last_step
            current_time = time.perf_counter()
            dic['System/StepsPerSec'] = steps / (current_time - self.last_time)
            self.last_time = current_time
        return dic

    def run(self):
        self.logger.info(f'Started, logging to {self.work_dir}...')
        step = 0
        stop = False
        self.last_time = time.perf_counter()
        self.last_step = 0
        while not stop:
            for x_batch, y_batch in self.train_data:
                loss, stats = self.train_step(x_batch, y_batch)
                for k, v in stats.items():
                    self.writer.add_scalar('Train_Step/{}'.format(k), v,
                                           global_step=step)
                if self.save_interval > 0 and step % self.save_interval == 0:
                    torch.save(dict(model=self.model.state_dict(), optim=self.optimizer.state_dict()),
                               os.path.join(self.work_dir, 'step_{}.pth'.format(step)))
                if step % self.evaluate_interval == 0:
                    self.logger.info('Step {}: Loss {}'.format(step, loss.cpu().item()))
                    self.logger.info('Evaluating ...')
                    eval_results = self.short_evaluate()
                    eval_results.update(self.system_log(step))
                    self.log_eval_results(step, eval_results)
                    if self.stop_criteria(step, eval_results):
                        stop = True
                if stop:
                    break

                step += 1
        self.logger.info('Train finished with {} steps'.format(step + 1))


class FixedScheduleTrainer(Trainer):
    def __init__(self, train_data: dict, val_data: dict, model: dict, loss: dict, device, evaluate_interval,
                 save_interval, work_dir, loss_eps: float, logger: Logger, max_steps, optimizer: dict,
                 additional_data: Dict[str, dict] = None):
        super().__init__(train_data, val_data, model, loss, device, evaluate_interval, save_interval, work_dir,
                         loss_eps, logger, max_steps, optimizer, additional_data)

    def stop_criteria(self, step_number, eval_results):
        return step_number > self.max_steps
