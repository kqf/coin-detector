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
    x[:, :, 0, 0] = edge
    x[:, :, 0, -1] = edge
    x[:, :, -1, 0] = edge
    x[:, :, -1, -1] = edge
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

    assert torch.testing.assert_allclose(
        x3, expected((4, 256, 64, 64), 264961., 117761, 176641))
