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


def test_shapes(image):
    bbox = [350, 50, 80, 50]
    image = to_ellipse(image, *bbox)
    box(*bbox)

    bbox = [350, 150, 50, 80]
    image = to_recatangle(image, *bbox)
    box(*bbox)

    bbox = [350, 250, 50, 80]
    image = to_disc(image, *bbox)
    box(*bbox)

    bbox = [350, 350, 50, 100]
    image = to_polygon(image, *bbox)
    box(*bbox)

    plt.imshow(image)
    box(*bbox)
    plt.show()
