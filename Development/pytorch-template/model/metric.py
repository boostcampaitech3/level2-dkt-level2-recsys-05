import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def mnist_accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def accuracy(output, target):
    # output = output.detach().cpu().numpy()
    # target = target.detach().cpu().numpy()
    acc = accuracy_score(target, np.where(output >= 0.5, 1, 0))
    return acc


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def roc_auc(output, target):
    auc = roc_auc_score(target, output)
    return auc
