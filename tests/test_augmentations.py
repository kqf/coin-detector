import numpy as np
import pytest
from detectors.augmentations import transform


@pytest.fixture
def image():
    return np.zeros([3, 400, 400])


@pytest.fixture
def boxes():
    return np.array([[0.5, 0.5, 0.25, 0.25]])


@pytest.fixture
def labels():
    return np.array([1])[None, :]


def test_transforms(image, boxes, labels):
    convert = transform(train=True)
    timage, tboxes, tlabels = convert(image, boxes, labels)
