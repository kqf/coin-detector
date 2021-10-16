import torch


def default_heads(n_classes, kernel_size=6):
    return torch.nn.ModuleDict({
        "boxes": torch.nn.Conv2d(100, 4, kernel_size),
        "classes": torch.nn.Conv2d(100, n_classes, kernel_size)
    })


class DummyDetector(torch.nn.Module):
    def __init__(self, heads=None, n_classes=2, kernel_size=6):
        super().__init__()
        self.backbone = torch.nn.AdaptiveAvgPool2d(kernel_size)
        self.heads = heads or default_heads(n_classes, kernel_size)

    def forward(self, x):
        latent = self.backbone(x)
        return {n: h(latent) for n, h in self.heads.items()}
