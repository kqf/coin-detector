import pytest
import numpy as np
import matplotlib.pyplot as plt

from detectors.shapes import to_ellipse, to_recatangle, to_disc, to_polygon
from detectors.shapes import box


@pytest.fixture
def image():
    shape = (400, 400)
    img = np.zeros(shape)
    return img


def box_mask(image, cx, cy, w, h):
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = cy - h / 2, cy + h / 2

    xx, yy = np.meshgrid(
        np.arange(image.shape[0]),
        np.arange(image.shape[1]),
    )
    mask = (x0 <= xx) & (xx <= x1) & (y0 <= yy) & (yy <= y1)
    return mask


@pytest.mark.parametrize("method", [
    to_ellipse,
    to_recatangle,
    to_polygon,
    to_disc,
])
@pytest.mark.parametrize("cx", np.arange(0, 400, step=40))
@pytest.mark.parametrize("cy", np.arange(0, 400, step=40))
def test_shapes(method, image, cx, cy):
    bbox = [cx, cy, 80, 50]
    image = method(image, *bbox)
    box(image, *bbox)
    plt.show(block=False)
    plt.imshow(image)
    # plt.draw()
    # plt.pause(0.000001)
    within_box = box_mask(image, *bbox)
    assert image[within_box].any(), "There is something within the box"
    assert not image[~within_box].any(), "There is nothing outside the box"
