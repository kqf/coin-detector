import torch
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter

from detectors.dummy import default_heads
from detectors.anchors import AnchorBoxes


class PyramidBlock(torch.nn.Module):
    def __init__(self, inp, out, scale_factor=2):
        super().__init__()
        self.p1 = torch.nn.Conv2d(inp, out, kernel_size=1, stride=1, padding=0)
        self.pu = torch.nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.p2 = torch.nn.Conv2d(out, out, kernel_size=3, stride=1, padding=1)

    def forward(self, x, skip=None):
        x1 = self.p1(x)
        if skip is not None:
            x1 = x1 + skip
        xu = self.pu(x1)
        x2 = self.p2(x1)
        return x2, xu


class Cumulative(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.steps = torch.nn.ModuleList(args)

    def forward(self, x):
        output = [x]
        for step in self.steps:
            output.append(step(output[-1]))
        return output[1:]


class Pyramial(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.steps = torch.nn.ModuleList(args)

    def forward(self, features, *args):
        output = []
        for x, step in zip(features, self.steps):
            x_out, *args = step(x, *args)
            output.append(x_out)
        return output


class FPN(torch.nn.Module):
    def __init__(self, c3, c4, c5, out_channels=256):
        super(FPN, self).__init__()
        self.p5_4_3 = Pyramial(
            # upsample c5 to get p5 from the fpn paper
            PyramidBlock(c5, out_channels),
            # add p5 elementwise to c4
            PyramidBlock(c4, out_channels),
            # add p4 elementwise to c3
            PyramidBlock(c3, out_channels, scale_factor=1),
        )

        self.p6_7 = Cumulative(
            # "p6 is obtained via a 3x3 stride-2 conv on c5"
            torch.nn.Conv2d(c5, out_channels,
                            kernel_size=3, stride=2, padding=1),
            # "p7 is computed by applying relu followed
            # by a 3x3 stride-2conv on p6
            torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Conv2d(out_channels, out_channels,
                                kernel_size=3, stride=2, padding=1),
            )
        )

    def forward(self, inputs):
        *_, c5 = inputs

        # x5, u5 = self.p5(c5)
        # x4, u4 = self.p4(c4, u5)
        # x3, __ = self.p3(c3, u4)

        x5, x4, x3 = self.p5_4_3(inputs[::-1])

        x6, x7 = self.p6_7(c5)
        return x3, x4, x5, x6, x7


class RetinaNet(torch.nn.Module):
    def __init__(
            self,
            layer_idx=None,
            out_channels=256,
            n_classes=2,
            kernel_size=1,
            pretrained=True
    ):
        super().__init__()
        backbone = resnet50(pretrained=pretrained)

        # layer_idx = layer_idx or [1, 2, 3]
        layer_idx = layer_idx or [2, 3, 4]
        return_layers = {f"layer{k}": str(v) for v, k in enumerate(layer_idx)}

        self.body = IntermediateLayerGetter(
            backbone, return_layers=return_layers)

        in_channels_stage2 = backbone.inplanes // 8
        in_channels_list = [in_channels_stage2 *
                            2 ** (i - 1) for i in layer_idx]
        self.fpn = FPN(*in_channels_list, out_channels=out_channels)
        self.heads = default_heads(
            n_classes=n_classes,
            channels=out_channels,
            kernel_size=kernel_size,
        )
        self.anchors = AnchorBoxes()

    def forward(self, x):
        body = self.body(x)
        pyramids = self.fpn(list(body.values()))

        outputs = {
            name: torch.cat([h(x) for x in pyramids], dim=1)
            for name, h in self.heads.items()
        }

        _, _, *image_shape = x.shape
        acnhors = self.anchors(image_shape, pyramids)
        return outputs, acnhors
