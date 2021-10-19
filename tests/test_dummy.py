import pytest
import torch
from detectors.dummy import DummyDetector


@pytest.fixture
def batch(batch_size=16):
    return torch.zeros(batch_size, 3, 28, 28)


def test_dummy(batch):
    model = DummyDetector()
    outputs = model(batch)

    assert outputs["boxes"].shape == (16, 36, 4)
    assert outputs["classes"].shape == (16, 36, 2)
