import pytest
import torch

from detectors.retinanet import FPN


@pytest.fixture
def layer_outputs(batch_size=4):
    x1 = torch.ones(batch_size, 16, 64, 64)
    x2 = torch.ones(batch_size, 32, 32, 32)
    x3 = torch.ones(batch_size, 64, 16, 16)
    return x1, x2, x3


def initialize(model):
    for p in model.parameters():
        torch.nn.init.constant_(p, 1.)


def expected(shape, fill, edge, corner):
    x = torch.full(shape, fill)
    x[:, :, :, [0, -1]] = x[:, :, [0, -1]] = edge
    x[:, :, 0, 0] = corner
    x[:, :, 0, -1] = corner
    x[:, :, -1, 0] = corner
    x[:, :, -1, -1] = corner
    return x


def test_fpn(layer_outputs, feature_size=256):
    model = FPN(16, 32, 64, feature_size=256)
    initialize(model)
    o3, o4, o5 = layer_outputs
    x3, x4, x5, x6, x7 = model(layer_outputs)

    assert x3.shape[1] == 256
    assert x4.shape[1] == 256
    assert x5.shape[1] == 256

    assert x3.shape[2:] == o3.shape[2:]
    assert x4.shape[2:] == o4.shape[2:]
    assert x5.shape[2:] == o5.shape[2:]

    assert x6.shape == (4, feature_size, 8, 8)
    assert x7.shape == (4, feature_size, 4, 4)

    x3_exp = expected((4, 256, 64, 64), 264_961., 176_641, 117_761)
    torch.testing.assert_allclose(x3, x3_exp)

    x4_exp = expected((4, 256, 32, 32), 225_793., 150_529., 100_353.)
    torch.testing.assert_allclose(x4, x4_exp)

    x5_exp = expected((4, 256, 16, 16), 149_761., 99_841., 66_561.)
    torch.testing.assert_allclose(x5, x5_exp)
