import torch.nn as nn

def bce_loss(output, target):
    loss = nn.BCELoss()
    return loss(output, target)
