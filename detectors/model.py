import torch


def default_heads(n_classes):
    return torch.nn.ModuleDict({
        "boxes": torch.nn.Conv2d(100, 4),
        "classes": torch.nn.Conv2d(100, n_classes)
    })


class DummyDetector(torch.nn.Module):
    def __init__(self, heads=None, n_classes=2):
        super().__init__()
        self.backbone = torch.nn.AdaptiveAvgPool2d(6)
        self.heads = heads or default_heads(n_classes)

    def forward(self, x):
        latent = self.backbone(x)
        return {n: h(latent) for n, h in self.heads.items()}
