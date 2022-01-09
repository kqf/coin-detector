import pytest
import torch
from detectors.anchors import DEFAULT_ANCHORS, AnchorBoxes


@pytest.fixture
def image_shape():
    return 32, 32


@pytest.fixture
def features(image_shape, batch_size=16, channels=3):
    layer_h, layer_w = image_shape[0] // 2, image_shape[1] // 2
    return torch.zeros((batch_size, channels, layer_h, layer_w))


def test_anchors(image_shape, features):
    boxlayer = AnchorBoxes()
    boxes = boxlayer(image_shape, [features])
    import ipdb; ipdb.set_trace(); import IPython; IPython.embed() # noqa
    
    batch_size, n_anchors, n_coords = boxes.shape
    assert n_coords == 5

    exp_batch_size, _, features_h, features_w = features.shape
    assert n_anchors == len(DEFAULT_ANCHORS[0]) * features_w * features_h
    assert batch_size == exp_batch_size
