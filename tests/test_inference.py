import torch
import pytest

from detectors.inference import infer, nms
from detectors.anchors import DEFAULT_ANCHORS


@pytest.fixture
def batch():
    anchors = torch.ones(8, 144, 6)
    x = {
        "boxes": torch.ones(8, 144, 4),
        "classes": torch.ones(8, 144, 2),
    }
    return x, anchors



@pytest.mark.skip
@pytest.mark.parametrize("bsize", [16])
def test_inferences(expected, batch, bsize):
    predictions = infer(batch, DEFAULT_ANCHORS, top_n=None,
                        min_iou=0.5, threshold=0.5)
    assert len(predictions) == bsize
    assert all([x.shape[-1] == 5 for x in predictions])

    for pred, nominal in zip(predictions, expected):
        assert pred.shape == nominal.shape

        # Check if nms works
    for sample in predictions:
        nms(sample)
