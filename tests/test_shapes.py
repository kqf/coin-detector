import matplotlib.pyplot as plt
import numpy as np
import pytest

from detectors.shapes import (box, box_mask, to_disc, to_ellipse, to_polygon,
                              to_recatangle)


@pytest.fixture
def image():
    shape = (400, 400)
    return np.zeros(shape)


@pytest.fixture
def bbox(cx, cy, w, h):
    return [cx, cy, w, h]


@pytest.mark.parametrize("method", [
    to_ellipse,
    to_recatangle,
    to_polygon,
    to_disc,
])
@pytest.mark.parametrize("cx", np.arange(0, 400, step=40))
@pytest.mark.parametrize("cy", np.arange(0, 400, step=40))
@pytest.mark.parametrize("w, h", [
    (80, 50),
    (50, 80),
    (50, 50),
])
def test_shapes(method, image, bbox):
    image = method(image, *bbox)
    box(image, *bbox, alpha=1)
    plt.imshow(image)
    plt.pause(0.000001)
    plt.cla()
    within_box = box_mask(image, *bbox)
    assert image[within_box].any(), "There is something within the box"
    assert not image[~within_box].any(), "There is nothing outside the box"
