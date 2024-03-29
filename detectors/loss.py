from dataclasses import dataclass
from functools import partial
from typing import Callable

import torch
import torchvision
from sklearn.metrics import f1_score

from detectors.encode import encode, to_cchw, to_coords
from detectors.matching import match


def select(y_pred, y_true, anchor, positives, negatives, use_negatives=True):
    batch_, obj_, anchor_ = torch.where(positives)
    y_pred_pos = y_pred[batch_, anchor_]
    y_true_pos = y_true[batch_, obj_]
    anchor_pos = anchor[batch_, anchor_]

    if not use_negatives:
        return y_pred_pos, y_true_pos, anchor_pos

    y_pred_neg = y_pred[torch.where(negatives)]
    anchor_neg = anchor[torch.where(negatives)]

    # Zero is a background
    y_true_neg_shape = [y_pred_neg.shape[0]]
    if len(y_true_pos.shape) > 1:
        y_true_neg_shape.append(y_true_pos.shape[-1])

    # Assume that zero is the negative class, increase the labels by 1
    y_true_neg = torch.zeros(y_true_neg_shape, device=y_true_pos.device)

    y_pred_tot = torch.cat([y_pred_pos, y_pred_neg], dim=0)
    anchor_tot = torch.cat([anchor_pos, anchor_neg], dim=0)
    # Increase y_true_pos by 1 since negatives are zeros
    y_true_tot = torch.squeeze(torch.cat([y_true_pos + 1, y_true_neg], dim=0))
    return y_pred_tot, y_true_tot, anchor_tot


@dataclass
class WeightedLoss:
    loss: torch.nn.Module
    weight: float = 1.
    enc_pred: Callable = lambda x, _: x
    enc_true: Callable = lambda x, _: x
    needs_negatives: bool = False

    def __call__(self, y_pred, y_true, anchors):
        y_pred_encoded = self.enc_pred(y_pred, anchors)
        y_true_encoded = self.enc_true(y_true, anchors)
        return self.weight * self.loss(y_pred_encoded, y_true_encoded)


def default_losses():
    return {
        "boxes": WeightedLoss(
            torch.nn.MSELoss(),
            enc_true=lambda x, a: encode(to_cchw(x), a),
            # enc_true=encode,
            weight=0.01,
        ),
        "classes": WeightedLoss(
            partial(
                torchvision.ops.sigmoid_focal_loss,
                reduction="mean",
                alpha=0.8,
                gamma=0.5,
            ),
            enc_true=lambda y, _: torch.nn.functional.one_hot(
                y.reshape(-1).long(), num_classes=3).float(),
            needs_negatives=True,
        ),
    }


def default_metrics():
    def f1(y_pred, y_true, anchors_):
        return f1_score(
            y_pred.argmax(-1).detach().cpu().numpy(),
            y_true.cpu().numpy(),
            average="micro",
        )

    losses = {
        "classes": {"f1": f1},
        "boxes": {},
    }
    return losses


class DetectionLoss(torch.nn.Module):
    def __init__(self, sublosses=None, metrics=None):
        super().__init__()
        self.sublosses = sublosses or default_losses()
        # We need to register the losses to manage things properly
        self.registered = torch.nn.ModuleList([
            loss.loss for loss in self.sublosses.values()
            if isinstance(loss.loss, torch.nn.Module)
        ])
        self.metrics = metrics or default_metrics()

    def forward(self, y_pred, y):
        preds, anchors = y_pred
        # Bind targets with anchors

        positives, negatives, _ = match(
            y["boxes"],
            # to_coords(y["boxes"]),
            y["classes"] < 0,
            to_coords(anchors)
        )

        # fselect -- selects only matched positives / negatives
        fselect = partial(select, positives=positives, negatives=negatives)
        losses, metrics = {}, {}
        for name, subloss in self.sublosses.items():
            # fselect(
            #   y_pred[batch, n_detections, dim1],
            #   y_true[batch, n_objects, dim2],
            #   anchor[batch, n_detections, 4],
            # )
            # ~> y_pred_[n_samples, dim1]
            # ~> y_true_[n_samples, dim2]
            # ~> anchor_[n_samples, 4]

            y_pred_, y_true_, anchor_ = fselect(
                preds[name], y[name], anchors,
                use_negatives=subloss.needs_negatives
            )
            losses[name] = subloss(y_pred_, y_true_, anchor_)
            for mname, metric in self.metrics[name].items():
                metrics[f"{name}_{mname}"] = metric(y_pred_, y_true_, anchor_)

        losses["loss"] = torch.stack(tuple(losses.values())).sum()
        losses.update(metrics)
        return losses
