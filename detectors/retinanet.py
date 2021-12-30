import torch


class FPN(torch.nn.Module):
    def __init__(self, c3, c4, c5, feature_size=256):
        super(FPN, self).__init__()

        # upsample c5 to get p5 from the fpn paper
        self.p5_1 = torch.nn.Conv2d(c5, feature_size,
                                    kernel_size=1, stride=1, padding=0)
        self.p5_upsampled = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_2 = torch.nn.Conv2d(feature_size, feature_size,
                                    kernel_size=3, stride=1, padding=1)

        # add p5 elementwise to c4
        self.p4_1 = torch.nn.Conv2d(c4, feature_size,
                                    kernel_size=1, stride=1, padding=0)
        self.p4_upsampled = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_2 = torch.nn.Conv2d(feature_size, feature_size,
                                    kernel_size=3, stride=1, padding=1)

        # add p4 elementwise to c3
        self.p3_1 = torch.nn.Conv2d(c3, feature_size,
                                    kernel_size=1, stride=1, padding=0)
        self.p3_2 = torch.nn.Conv2d(feature_size, feature_size,
                                    kernel_size=3, stride=1, padding=1)

        # "p6 is obtained via a 3x3 stride-2 conv on c5"
        self.p6 = torch.nn.Conv2d(c5, feature_size,
                                  kernel_size=3, stride=2, padding=1)

        # "p7 is computed by applying relu followed by a 3x3 stride-2conv on p6
        self.p7_1 = torch.nn.ReLU()
        self.p7_2 = torch.nn.Conv2d(feature_size, feature_size,
                                    kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        c3, c4, c5 = inputs

        p5_x = self.p5_1(c5)
        p5_upsampled_x = self.p5_upsampled(p5_x)
        p5_x = self.p5_2(p5_x)

        p4_x = self.p4_1(c4)
        p4_x = p5_upsampled_x + p4_x
        p4_upsampled_x = self.p4_upsampled(p4_x)
        p4_x = self.p4_2(p4_x)

        p3_x = self.p3_1(c3)
        p3_x = p3_x + p4_upsampled_x
        p3_x = self.p3_2(p3_x)

        p6_x = self.p6(c5)

        p7_x = self.p7_1(p6_x)
        p7_x = self.p7_2(p7_x)

        return p3_x, p4_x, p5_x, p6_x, p7_x
