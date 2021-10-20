import pytest
import torch
from detectors.anchors import AnchorBoxes


@pytest.fixture
def image_shape():
    return 32, 32


@pytest.fixture
def features(image_shape, batch_size=16, channels=8):
    layer_h, layer_w = image_shape[0] // 2, image_shape[1] // 2
    return torch.zeros((batch_size, channels, layer_h, layer_w))


def test_anchors(image_shape, features):
    boxlayer = AnchorBoxes()
    boxlayer(image_shape, [features])
