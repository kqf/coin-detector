from collections import OrderedDict

import pytest
import torch
from torchvision.ops import FeaturePyramidNetwork

from detectors.retinanet import MobileRetinaNet


@pytest.fixture
def layer_outputs(batch_size=4):
    x1 = torch.ones(batch_size, 16, 64, 64)
    x2 = torch.ones(batch_size, 32, 32, 32)
    x3 = torch.ones(batch_size, 64, 16, 16)
    return x1, x2, x3


def initialize(model):
    for p in model.parameters():
        torch.nn.init.constant_(p, 1.)


def expected(shape, fill, edge, corner, upper_left=False):
    x = torch.full(shape, fill)
    x[:, :, :, 0] = edge
    x[:, :, 0] = edge
    x[:, :, 0, 0] = corner

    if upper_left:
        return x

    x[:, :, :, -1] = edge
    x[:, :, -1] = edge
    x[:, :, 0, -1] = corner
    x[:, :, -1, 0] = corner
    x[:, :, -1, -1] = corner
    return x


def test_default_fpn(layer_outputs, feature_size=256):
    model = FeaturePyramidNetwork([16, 32, 64], feature_size)
    initialize(model)

    inputs = OrderedDict([(str(i), l) for i, l in enumerate(layer_outputs)])
    output = OrderedDict(model(inputs))

    x3_exp = expected((4, 256, 64, 64), 264_961., 176_641, 117_761)
    torch.testing.assert_allclose(output["0"], x3_exp)

    x4_exp = expected((4, 256, 32, 32), 225_793., 150_529., 100_353.)
    torch.testing.assert_allclose(output["1"], x4_exp)

    x5_exp = expected((4, 256, 16, 16), 149_761., 99_841., 66_561.)
    torch.testing.assert_allclose(output["2"], x5_exp)


@pytest.fixture
def batch(image_size=480, batch_size=4):
    return torch.ones(batch_size, 3, image_size, image_size)


def test_mobileretinanet(batch, output_features=256, kernel_size=1):
    model = MobileRetinaNet(out_channels=output_features,
                            kernel_size=kernel_size)
    initialize(model)
    outputs, anchors = model(batch)

    n_anchors = 514
    assert outputs["boxes"].shape == (4, n_anchors, 4)
    assert outputs["classes"].shape == (4, n_anchors, 3)
    assert anchors.shape == (4, n_anchors, 4)
