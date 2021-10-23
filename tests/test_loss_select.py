import torch
import pytest

from detectors.loss import select


@pytest.fixture
def y_pred(batch_size, n_anchors, n_outputs):
    return torch.ones((batch_size, n_anchors, n_outputs))


@pytest.fixture
def y_true(batch_size, n_targets, n_outputs):
    return torch.ones((batch_size, n_targets, n_outputs))


@pytest.fixture
def anchors(batch_size, n_anchors):
    return torch.ones((batch_size, n_anchors))


@pytest.fixture
def positive(batch_size, n_anchors, n_targets):
    matched = torch.empty(batch_size, n_anchors, n_targets).uniform_(0, 1)
    return matched > 0.5


@pytest.fixture
def negative(batch_size, n_anchors):
    matched = torch.empty(batch_size, n_anchors).uniform_(0, 1)
    return matched > 0.5


# @pytest.mark.skip()
@pytest.mark.parametrize("batch_size", [16])
@pytest.mark.parametrize("n_anchors", [144])
@pytest.mark.parametrize("n_outputs", [4])
@pytest.mark.parametrize("n_targets", [10])
def test_selects_samples(y_pred, y_true, anchors, positive, negative):
    y_pred_, y_true_, anchors_ = select(
        y_pred, y_true, anchors, positive, positive
    )
    n_samples = torch.where(positive)[0].shape[0]
    assert y_pred_.shape[0] == n_samples
    assert y_true_.shape[0] == n_samples
    assert anchors_.shape[0] == n_samples

    # y_pred_, y_true_, anchors_ = select(
    #     y_pred, y_true, anchors, positive, negative, use_negatives=True
    # )

    # n_samples = torch.where(positive)[0].shape[0] + negative.shape[0]
    # assert y_pred_.shape[0] == n_samples
    # assert y_true_.shape[0] == n_samples
    # assert anchors_.shape[0] == n_samples
