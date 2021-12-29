import torch
import pytest

from detectors.loss import select


@pytest.fixture
def y_pred(batch_size, n_anchors, n_outputs):
    return torch.zeros((batch_size, n_anchors, n_outputs)) + 1


@pytest.fixture
def y_true(batch_size, n_targets, n_outputs):
    return torch.zeros((batch_size, n_targets, n_outputs)) + 2


@pytest.fixture
def anchors(batch_size, n_anchors):
    return torch.zeros((batch_size, n_anchors)) + 3


@pytest.fixture
def positive(batch_size, n_anchors, n_targets):
    matched = torch.empty(batch_size, n_targets, n_anchors).uniform_(0, 1)
    return matched > 0.5


@pytest.fixture
def negative(batch_size, n_anchors):
    matched = torch.empty(batch_size, n_anchors).uniform_(0, 1)
    return matched > 0.5


def total_len(use_neg, positive, negative):
    n_pos = torch.where(positive)[0].shape[0]
    if not use_neg:
        return n_pos
    n_neg = torch.where(negative)[0].shape[0]
    return n_pos + n_neg


@pytest.mark.parametrize("batch_size", [16])
@pytest.mark.parametrize("n_outputs", [2])
@pytest.mark.parametrize("n_targets", [10])
@pytest.mark.parametrize("n_anchors", [144])
@pytest.mark.parametrize("use_neg", [
    # False,
    True
])
def test_selects_samples(y_pred, y_true, anchors, positive, negative, use_neg):
    y_pred_, y_true_, anchor_ = select(
        y_pred, y_true, anchors, positive, negative, use_negatives=use_neg
    )

    n_samples = total_len(use_neg, positive, negative)
    assert y_pred_.shape[0] == n_samples
    assert y_true_.shape[0] == n_samples
    assert anchor_.shape[0] == n_samples

    assert torch.all(y_pred_ == 1)
    assert torch.all(y_true_ <= 2)
    assert torch.all(anchor_ == 3)
