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


@pytest.mark.parametrize("method", [
    to_ellipse,
    to_recatangle,
    to_disc,
    to_polygon,
])
def test_shapes(method, image):
    bbox = [150, 50, 80, 50]
    image = method(image, *bbox)
    box(image, *bbox)
    plt.imshow(image)
    plt.show()
