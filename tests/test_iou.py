import torch
import pytest

from detectors.iou import iou


@pytest.mark.parametrize("x1, x2, answer", [
    (
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        1.,
    ),
    (
        [0, 0, 1, 1],
        [0.5, 0.5, 1, 1],
        1. / 4.,
    ),
    (
        [0, 0, 1, 1],
        [0.25, 0.25, 0.75, 0.75],
        1. / 4.,
    ),
    (
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        0.,
    ),
    (
        [0, 0, 1, 1],
        [0.5, 0.5, 1.5, 1.5],
        1. / 7.,
    ),
])
@pytest.mark.parametrize("to_batch", [
    lambda x: torch.tensor([x]),
    lambda x: torch.tensor([x] * 16),
])
@pytest.mark.parametrize("shift", [0, 1, -1, 2, -2])
def test_iou(x1, x2, answer, to_batch, shift):
    b1, b2 = to_batch(x1) + shift, to_batch(x2) + shift
    # Check if calculates properly
    torch.testing.assert_allclose(iou(b1, b2), to_batch(answer))

    # Check if is symmetric
    torch.testing.assert_allclose(iou(b2, b1), to_batch(answer))
