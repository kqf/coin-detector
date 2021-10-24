import torch

from typing import Callable
from dataclasses import dataclass


def default_losses():
    losses = {
        "boxes": torch.nn.MSELoss(),
        "classes": torch.nn.CrossEntropyLoss(),
    }
    return losses


def select(y_pred, y_true, anchor, positives, negatives, use_negatives=False):
    batch_, anchor_, obj_ = torch.where(positives)
    y_pred_pos = y_pred[batch_, anchor_]
    y_true_pos = y_true[batch_, obj_]
    anchor_pos = anchor[batch_, anchor_]

    if not use_negatives:
        return y_true_pos, y_pred_pos, anchor_pos

    y_pred_neg = y_pred[torch.where(negatives)]
    anchor_neg = anchor[torch.where(negatives)]

    # Zero is a background
    y_true_neg = torch.zeros_like(y_pred_neg)

    y_pred_tot = torch.cat([y_pred_pos, y_pred_neg], dim=0)
    anchor_tot = torch.cat([anchor_pos, anchor_neg], dim=0)
    y_true_tot = torch.cat([y_true_pos, y_true_neg], dim=0).long()
    return y_true_tot, y_pred_tot, anchor_tot


@dataclass
class WeightedLoss:
    loss: torch.nn.Module
    weight: float = 1.
    enc_pred: Callable = lambda x, _: x
    enc_true: Callable = lambda x, _: x

    def __call__(self, y_pred, y_true, anchors):
        y_pred_encoded = self.enc_pred(y_pred, anchors)
        y_true_encoded = self.enc_true(y_true, anchors)
        return self.weight * self.loss(y_pred_encoded, y_true_encoded)


class DetectionLoss(torch.nn.Module):
    def __init__(self, sublosses=None):
        super().__init__()
        self.sublosses = sublosses or default_losses()

    def forward(self, y_pred, y):
        preds, _ = y_pred
        losses = []
        for name, subloss in self.sublosses.items():
            # y_pred[batch, n_detections, dim1], y[batch, n_objects, dim2]
            losses.append(
                subloss(
                    preds[name][:, :, None],
                    y[name][:, None]
                )
            )
        return torch.stack(losses).sum()
