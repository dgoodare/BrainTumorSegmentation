import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    @staticmethod
    def forward(inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        # flatten input and target tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # calculate dice loss
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice
