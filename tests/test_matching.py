import torch
import pytest

from detectors.matching import match


@pytest.fixture
def y_pred(batch_size, n_anchors, n_outputs):
    return torch.ones((batch_size, n_anchors, n_outputs))


@pytest.fixture
def target_boxes(batch_size, n_targets):
    x = torch.ones((batch_size, n_targets, 4))
    x[:, 0, :] = [144.0, 144.0, 96., 96.0]
    x[:, 1, :] = [240.0, 240.0, 96., 96.0]
    return x


@pytest.fixture
def target_classes(batch_size, n_targets):
    x = torch.ones((batch_size, n_targets))
    x[:, 0, :] = 0
    x[:, 1, :] = 1
    return x


@pytest.fixture
def anchors(batch_size, n_anchors):
    return torch.ones((batch_size, n_anchors, 4))


@pytest.mark.parametrize("n_anchors", [144])
@pytest.mark.parametrize("n_outputs", [6])
@pytest.mark.parametrize("n_targets", [2])
@pytest.mark.parametrize("batch_size", [16])
def test_matches(
    y_pred, target_boxes, target_classes, anchors, batch_size,
    n_anchors, n_outputs, n_targets,
):
    mask = target_classes > 1000

    positives, negatives = match(target_boxes, mask, anchors)
    assert positives.shape == (batch_size, n_targets, n_anchors)
    assert negatives.shape == (batch_size, n_anchors)
