import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

def accuracy(output, target):
    return accuracy_score(target, np.where(output >= 0.5, 1, 0))


def roc_auc(output, target):
    return roc_auc_score(target, output)
