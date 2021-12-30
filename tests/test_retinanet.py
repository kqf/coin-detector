import pytest
import torch

from detectors.retinanet import FPN


@pytest.fixture
def layer_outputs(batch_size=4):
    x1 = torch.ones(batch_size, 16, 64, 64)
    x2 = torch.ones(batch_size, 32, 32, 32)
    x3 = torch.ones(batch_size, 64, 16, 16)
    return x1, x2, x3


def test_fpn(layer_outputs, feature_size=256):
    model = FPN(16, 32, 64, feature_size=256)
    o3, o4, o5 = layer_outputs
    x3, x4, x5, x6, x7 = model(layer_outputs)

    assert x3.shape[1] == 256
    assert x4.shape[1] == 256
    assert x5.shape[1] == 256
    assert x6.shape[1] == 256
    assert x7.shape[1] == 256

    # assert x3.shape == o3.shape
    # assert x4.shape == o4.shape
    # assert x5.shape == o5.shape
