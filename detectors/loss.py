import torch


def default_losses():
    losses = {
        "boxes": torch.nn.MSELoss(),
        "classes": torch.nn.CrossEntropyLoss(),
    }
    return losses


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
