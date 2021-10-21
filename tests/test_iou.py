import torch
import pytest

from detectors.iou import iou


@pytest.mark.parametrize("x1, x2, answer", [
    (
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        1.
    ),
    (
        [0, 0, 1, 1],
        [0.5, 0.5, 1, 1],
        1. / 4.
    ),
    (
        [0, 0, 1, 1],
        [0.25, 0.25, 0.75, 0.75],
        1. / 4.
    ),
    (
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        0.
    ),
])
@pytest.mark.parametrize("to_batch", [
    lambda x: torch.tensor([x]),
    lambda x: torch.tensor([x] * 16),
])
def test_iou(x1, x2, answer, to_batch):
    b1, b2 = to_batch(x1), to_batch(x2)
    torch.testing.assert_allclose(iou(b1, b2), to_batch(answer))
