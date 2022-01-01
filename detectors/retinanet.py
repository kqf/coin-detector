import torch


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


class FPN(torch.nn.Module):
    def __init__(self, c3, c4, c5, feature_size=256):
        super(FPN, self).__init__()

        # upsample c5 to get p5 from the fpn paper
        self.p5 = PyramidBlock(c5, feature_size)

        # add p5 elementwise to c4
        self.p4 = PyramidBlock(c4, feature_size)

        # add p4 elementwise to c3
        self.p3 = PyramidBlock(c3, feature_size, scale_factor=1)

        self.p6_7 = Cumulative(
            # "p6 is obtained via a 3x3 stride-2 conv on c5"
            torch.nn.Conv2d(c5, feature_size,
                            kernel_size=3, stride=2, padding=1),

            # "p7 is computed by applying relu followed
            # by a 3x3 stride-2conv on p6
            torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Conv2d(feature_size, feature_size,
                                kernel_size=3, stride=2, padding=1),
            )
        )

    def forward(self, inputs):
        c3, c4, c5 = inputs

        x5, u5 = self.p5(c5)
        x4, u4 = self.p4(c4, u5)
        x3, __ = self.p3(c3, u4)

        x6, x7 = self.p6_7(c5)

        return x3, x4, x5, x6, x7
