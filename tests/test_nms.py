import torch
import pytest
import numpy as np

from detectors.inference import nms


@pytest.fixture
def candidates(n_anchors=400, n_classes=4):
    x = np.zeros((n_anchors, 4))
    x[:, 0] = np.linspace(0.4, 0.6, n_anchors)
    x[:, 1] = np.linspace(0.4, 0.5, n_anchors)
    x[:, 2] = np.linspace(0.4, 0.5, n_anchors)
    x[:, 3] = 0.2

    predictions = {}
    predictions["boxes"] = torch.tensor(x)

    classes = np.zeros((n_anchors, n_classes))
    predictions["classes"] = torch.tensor(classes)
    return predictions


def test_nms(candidates):
    sup = nms(candidates)
    print(sup.shape)
    # top = candidates.shape[0] // 2

    # assert torch.equal(sup[0], candidates[top, 1:])
    # assert torch.equal(sup[-1], candidates[-1, 1:])
