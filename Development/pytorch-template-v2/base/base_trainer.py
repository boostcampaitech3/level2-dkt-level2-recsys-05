import os
import torch
from abc import abstractmethod


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, optimizer, lr_scheduler, config):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_dir = cfg_trainer['save_dir']

        self.start_epoch = 1
        self.best_epoch = 0
        self.best_loss = 0
        self.best_acc = 0
        self.best_roc_auc = 0

    @abstractmethod
    def _train_epoch(self):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError
    
    @abstractmethod
    def _valid_epoch(self):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self, oof):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_loss, train_acc, train_roc_auc = self._train_epoch()
            valid_loss, valid_acc, valid_roc_auc = self._valid_epoch()
            if self.lr_scheduler:
                self.lr_scheduler.step()
            print(f'OOF-{oof}| Epoch: {epoch:3d}| train_loss: {train_loss:.5f}| train_acc: {train_acc:.5f}| train_roc_auc: {train_roc_auc:.5f}')
            print(f'                     valid_loss: {valid_loss:.5f}| valid_acc: {valid_acc:.5f}| valid_roc_auc: {valid_roc_auc:.5f}')

            if self.best_roc_auc < valid_roc_auc:
                self.best_epoch = epoch
                self.best_loss = valid_loss
                self.best_acc = valid_acc
                self.best_roc_auc = valid_roc_auc
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, f'oof_{oof}_' + self.config['name'] + '.pt'))

            # TODO
            # MLflow 연동
