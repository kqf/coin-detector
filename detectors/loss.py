import torch


class DetectionLoss(torch.nn.Module):
    def __init__(self, sublosses=None):
        super().__init__()
        self.sublosses = sublosses

    def forward(self, y_pred, y):
        return 0.1
