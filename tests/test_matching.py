import torch
import pytest

from detectors.matching import match


@pytest.fixture
def y_pred(batch_size, n_anchors, n_outputs):
    return torch.ones((batch_size, n_anchors, n_outputs))


@pytest.fixture
def target_boxes(batch_size, n_targets):
    return torch.ones((batch_size, n_targets, 4))


@pytest.fixture
def target_classes(batch_size, n_targets):
    return torch.ones((batch_size, n_targets))


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
