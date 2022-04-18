import numpy as np
import pytest
import torch

from detectors.encode import decode
from detectors.inference import infer


@pytest.fixture
def candidates(batch_size=4, n_anchors=400, n_classes=4):
    x = np.zeros((batch_size, n_anchors, 4))
    x[..., 0] = np.linspace(0.4, 0.6, n_anchors)
    x[..., 1] = np.linspace(0.4, 0.5, n_anchors)
    x[..., 2] = np.linspace(0.4, 0.5, n_anchors)
    x[..., 3] = 0.2

    predictions = {}
    predictions["boxes"] = torch.tensor(x)

    classes = np.zeros((batch_size, n_anchors, n_classes))
    # Left it be always the first class
    classes[:, :, 1] = 0.9
    predictions["classes"] = torch.tensor(classes)

    anchors = torch.tensor(np.ones((batch_size, n_anchors, 4)))
    return predictions, anchors


# @pytest.mark.skip
def test_inference(candidates):
    sup = infer(candidates, decode=lambda x, _: x)
    assert len(sup) == candidates[-1].shape[0]
    import ipdb; ipdb.set_trace(); import IPython; IPython.embed()  # noqa
