import pytest
import torch
from detectors.anchors import AnchorBoxes
from detectors.matching import match


@pytest.fixture
def target_boxes(batch_size, n_targets):
    x = torch.ones((batch_size, n_targets, 4))
    x[:, 0, :] = torch.tensor([144.0, 144.0, 96., 96.0])
    x[:, 1, :] = torch.tensor([240.0, 240.0, 96., 96.0])
    return x


@pytest.fixture
def target_classes(batch_size, n_targets):
    x = torch.ones((batch_size, n_targets))
    x[:, 0] = 0
    x[:, 1] = 1
    return x


@pytest.fixture
def anchors(batch_size, image_shape=(480, 480), latent_size=5):
    layer = AnchorBoxes()
    latent = torch.zeros(batch_size, 1, latent_size, latent_size)
    anchors, _ = layer(image_shape, [latent])
    return anchors


@pytest.mark.parametrize("n_targets", [2])
@pytest.mark.parametrize("batch_size", [16])
def test_matches(
    target_boxes, target_classes, anchors, batch_size, n_targets,
):
    mask = target_classes > 1000

    _, n_anchors, _ = anchors.shape

    positives, negatives = match(target_boxes, mask, anchors)

    expected_positives = (anchors[:, :, None] == target_boxes[:, None]).all(dim=-1)
    assert (expected_positives == positives).all()
    assert positives.shape == (batch_size, n_targets, n_anchors)
    assert negatives.shape == (batch_size, n_anchors)
