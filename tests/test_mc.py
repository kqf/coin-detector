import pytest
import matplotlib.pyplot as plt
from detectors.mc import make_image


@pytest.fixture
def blobs():
    blobs = [
        {
            "coco": [50, 50, 90, 90],
            "class_id": 0,
            "colors": [13, 46, 96],
        },
        {
            "coco": [150, 250, 90, 90],
            "class_id": 1,
            "colors": [94, 103, 36],
        },
    ]
    return blobs


def test_generates(blobs):
    image = make_image(shapes=blobs, image_shape=(400, 400))
    plt.imshow(image)
    plt.show()
