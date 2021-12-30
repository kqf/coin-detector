import pytest
import torch
from detectors.retinanet import FPN


@pytest.fixture
def layer_outputs(batch_size=4):
    x1 = torch.ones(batch_size, 16, 64, 64)
    x2 = torch.ones(batch_size, 32, 32, 32)
    x3 = torch.ones(batch_size, 64, 16, 16)
    return x1, x2, x3


def test_fpn(layer_outputs):
    model = FPN(16, 32, 64)
    model(layer_outputs)
