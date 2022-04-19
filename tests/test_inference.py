import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from detectors.inference import infer
from detectors.shapes import arrows, box


def pplot(image, data, stem="image"):
    for i, (boxes, ilabels) in enumerate(data):
        for coords, confidence in zip(boxes, ilabels):
            box(image, *coords)
        print(">>>")
        plt.imshow(image)
        arrows()
        plt.show()
        plt.savefig(f"{stem}-{i}.png")


@pytest.fixture
def image():
    return np.zeros((640, 640, 3), dtype=np.uint8) + 255


@pytest.fixture
def candidates(image, batch_size=4, n_anchors=400, n_classes=4):
    x = np.zeros((batch_size, n_anchors, 4))
    x[..., 0] = np.linspace(280, 300, n_anchors)
    x[..., 1] = 280
    x[..., 2] = np.linspace(280, 300, n_anchors) + 20
    x[..., 3] = 280 + 20

    predictions = {}
    predictions["boxes"] = torch.tensor(x)

    classes = np.zeros((batch_size, n_anchors, n_classes))
    # Left it be always the first class
    classes[:, :, 1] = np.linspace(0.2, 0.8, n_anchors)
    predictions["classes"] = torch.tensor(classes)

    anchors = torch.tensor(np.ones((batch_size, n_anchors, 4)))
    pplot(image, data=zip(predictions["boxes"], predictions["classes"]))
    return predictions, anchors


# @pytest.mark.skip
def test_inference(image, candidates):
    sup = infer(candidates, decode=lambda x, _: x)
    assert len(sup) == candidates[-1].shape[0]
    pplot(image, data=sup, stem="filtered")
    for boxes, scores in sup:
        print(boxes.shape, scores.shape)
        print(boxes)
