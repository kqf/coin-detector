import torch
from torchvision import models

from detectors.anchors import AnchorBoxes


class SqueezeCells(torch.nn.Module):
    def forward(self, x):
        batch, channel, *_ = x.shape
        return x.view(batch, -1, channel)


def default_heads(n_classes, kernel_size=1):
    return torch.nn.ModuleDict({
        "boxes":
            torch.nn.Sequential(
                torch.nn.Conv2d(3, 4, kernel_size),
                SqueezeCells(),
            ),
        "classes":
            torch.nn.Sequential(
                # Always +1 class, for background
                torch.nn.Conv2d(3, n_classes + 1, kernel_size),
                SqueezeCells(),
            ),
    })


class DummyDetector(torch.nn.Module):
    def __init__(self, heads=None, n_classes=2, kernel_size=5):
        super().__init__()
        self.backbone = models.resnet18()
        self.backbone.fc = torch.nn.Identity()
        self.heads = heads or default_heads(n_classes)
        self.anchors = AnchorBoxes()

    def forward(self, x):
        latent = self.backbone(x)
        outputs = {n: h(latent) for n, h in self.heads.items()}

        _, _, *image_shape = x.shape
        return outputs, self.anchors(image_shape, [latent])

    def to(self, device):
        self.anchors.to(device)
        return super().to(device)
