import torch
import pytest
import numpy as np

from detection.inference import nms


@pytest.fixture
def candidates(n_candidates=13 * 3 + 26 * 3 + 52 * 3):
    x = np.zeros((n_candidates, 4))
    x[:, 0] = np.linspace(0.4, 0.6, n_candidates)
    x[n_candidates // 2, 0] = 1

    x[:, 1] = np.linspace(0.4, 0.5, n_candidates)
    x[:, 2] = np.linspace(0.4, 0.5, n_candidates)
    x[:, 3] = 0.2
    predictions["boxes"] = x

    classes = np.zeros((n_candidates, 4))
    predictions["classes"] = classes
    return predictions


def test_nms(candidates):
    sup = nms(candidates)
    top = candidates.shape[0] // 2

    assert torch.equal(sup[0], candidates[top, 1:])
    assert torch.equal(sup[-1], candidates[-1, 1:])
