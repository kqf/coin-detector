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


def on_box(image, cx, cy, w, h):
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = cy - h / 2, cy + h / 2

    xx, yy = np.meshgrid(
        np.arange(image.shape[0]),
        np.arange(image.shape[1]),
    )
    mask = (x0 < xx) & (xx < x1) & (y0 < yy) & (yy < y1)
    return mask


@pytest.mark.parametrize("method", [
    to_ellipse,
    # to_recatangle,
    # to_disc,
    # to_polygon,
])
def test_shapes(method, image):
    bbox = [150, 50, 80, 50]
    image = method(image, *bbox)
    box(image, *bbox)
    plt.imshow(image)
    plt.show()
    mask = on_box(image, *bbox)
    assert image[mask].any()
    assert not image[~mask].any()
