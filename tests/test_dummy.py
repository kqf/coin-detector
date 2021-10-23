import pytest
import torch
from detectors.dummy import DummyDetector
from detectors.anchors import DEFAULT_ANCHORS


@pytest.fixture
def batch(batch_size=16):
    return torch.zeros(batch_size, 3, 28, 28)


def test_dummy(batch, kernel_size=6):
    model = DummyDetector(kernel_size=kernel_size)
    outputs, anchors = model(batch)

    assert outputs["boxes"].shape == (16, 36, 4)
    assert outputs["classes"].shape == (16, 36, 2)
    n_anchors = len(DEFAULT_ANCHORS[0]) * kernel_size * kernel_size
    assert anchors.shape == (16, n_anchors, 6)
