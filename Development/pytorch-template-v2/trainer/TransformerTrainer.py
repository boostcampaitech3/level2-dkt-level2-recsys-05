import torch
import numpy as np
from base import BaseTrainer
from model.metric import accuracy, roc_auc


class TransformerTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        config,
        device,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
    ):
        super().__init__(model, criterion, optimizer, lr_scheduler, config)
        self.device = device
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader

    def _train_epoch(self):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        
        loss_val = 0
        targets = []
        outputs = []

        for data in self.data_loader:
            target = data['now_answerCode'].to(self.device)

            self.optimizer.zero_grad()

            output = self.model(data)

            loss = self.criterion(output[target != -1], target[target != -1])
            loss_val += loss.item()
            loss.backward()
            self.optimizer.step()

            targets.extend(target[:, -1].detach().cpu().numpy().tolist())
            outputs.extend(output[:, -1].detach().cpu().numpy().tolist())

        loss_val /= len(self.data_loader)
        return loss_val, accuracy(np.array(outputs), np.array(targets)), roc_auc(outputs, targets)

    def _valid_epoch(self):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()

        loss_val = 0
        targets = []
        outputs = []

        with torch.no_grad():
            # for batch_idx, (data, target) in enumerate(self.valid_data_loader):
            for data in self.data_loader:
                target = data['now_answerCode'].to(self.device)
                output = self.model(data)

                loss = self.criterion(output[:, -1], target[:, -1])
                loss_val += loss.item()
                targets.extend(target[:, -1].cpu().numpy().tolist())
                outputs.extend(output[:, -1].cpu().numpy().tolist())

        loss_val /= len(self.data_loader)
        return loss_val, accuracy(np.array(outputs), np.array(targets)), roc_auc(outputs, targets)