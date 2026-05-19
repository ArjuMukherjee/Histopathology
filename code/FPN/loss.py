import torch
import torch.nn as nn

bce_loss = nn.BCEWithLogitsLoss()

def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1.0

    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def total_loss(pred, target):
    bce = bce_loss(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice