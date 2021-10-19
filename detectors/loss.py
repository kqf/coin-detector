import torch


def default_losses():
    losses = {
        "boxes": torch.nn.MSELoss,
        "classes": torch.nn.CrossEntropyLoss,
    }
    return losses


class DetectionLoss(torch.nn.Module):
    def __init__(self, sublosses=None):
        super().__init__()
        self.sublosses = sublosses or default_losses()

    def forward(self, y_pred, y):
        losses = []
        for name, subloss in self.sublosses.items():
            import ipdb; ipdb.set_trace(); import IPython; IPython.embed()  # noqa
            losses.append(subloss(y_pred[name], y[name]))
        return torch.cat(losses).sum()
