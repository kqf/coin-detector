import pytest
import torch
from detectors.model import DummyDetector


@pytest.fixture
def batch(batch_size=16):
    return torch.zeros(batch_size, 3, 28, 28)


def test_dummy(batch):
    model = DummyDetector()
    model(batch)
