import pytest
import matplotlib.pyplot as plt
from detectors.mc import make_image


@pytest.fixture
def blobs():
    blobs = [
        {
            "x_center": 50,
            "y_center": 50,
            "width": 90,
            "height": 90,
            "class_id": 0,
            "colors": [13, 46, 96],
        },
        {
            "x_center": 150,
            "y_center": 250,
            "width": 90,
            "height": 90,
            "class_id": 1,
            "colors": [94, 103, 36],
        },
    ]
    return blobs


def test_generates(blobs):
    image = make_image(shapes=blobs, image_shape=(400, 400))
    plt.imshow(image)
    plt.show()
