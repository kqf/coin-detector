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
        plt.imshow(image)
        arrows()
        # plt.show()
        plt.savefig(f"{stem}-{i}.png")


@pytest.fixture
def image():
    return np.zeros((640, 640, 3), dtype=np.uint8) + 255


@pytest.fixture
def predictions(image, batch_size, n_anchors, n_classes):
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
    pplot(image, data=zip(predictions["boxes"], predictions["classes"]))
    return predictions


@pytest.fixture
def anchors(batch_size, n_anchors):
    return torch.tensor(np.ones((batch_size, n_anchors, 4)))


@pytest.fixture
def expected(predictions):
    x = predictions["boxes"]
    return x[:, [-1]]


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("n_anchors", [200])
@pytest.mark.parametrize("n_classes", [4])
def test_inference(image, predictions, anchors, expected):
    sup = infer((predictions, anchors), decode=lambda x, _: x)
    pplot(image, data=sup, stem="filtered")
    for (boxes, _), exptd in zip(sup, expected):
        np.testing.assert_allclose(boxes, exptd, rtol=1e-5)
