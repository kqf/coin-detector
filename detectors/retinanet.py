import torch
from torchvision.models.detection.backbone_utils import mobilenet_backbone

from detectors.anchors import DEFAULT_ANCHORS, AnchorBoxes
from detectors.dummy import default_heads


class MobileRetinaNet(torch.nn.Module):
    def __init__(
            self,
            layer_idx=None,
            out_channels=256,
            n_classes=2,
            kernel_size=1,
            pretrained=True,
            anchors=None,
    ):
        super().__init__()
        self.fpn = mobilenet_backbone(
            "mobilenet_v2",
            pretrained=pretrained,
            fpn=True
        )
        self.heads = default_heads(
            n_classes=n_classes,
            channels=out_channels,
            kernel_size=kernel_size,
        )
        self.anchors = AnchorBoxes(anchors or DEFAULT_ANCHORS * 5)

    def forward(self, x):
        pyramids = self.fpn(x)

        outputs = {
            name: torch.cat([h(x) for x in pyramids.values()], dim=1)
            for name, h in self.heads.items()
        }

        _, _, *image_shape = x.shape
        anchors, _ = self.anchors(image_shape, pyramids.values())
        return outputs, anchors.to(x.device)
