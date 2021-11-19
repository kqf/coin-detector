import torch
import numpy as np
import pytest
from detectors.augmentations import transform


@pytest.mark.parametrize("image, boxes, labels", [
    (
        np.zeros([3, 400, 400]),
        np.array([[0.5, 0.5, 0.25, 0.25]]),
        np.array([[1]])
    ),
    (
        np.zeros([3, 400, 400]),
        np.array([
            [0.5, 0.5, 0.25, 0.25],
            [0.7, 0.7, 0.25, 0.25],
        ]),
        np.array([[1], [0]])
    ),
])
@pytest.mark.parametrize("train", [True, False])
def test_transforms(train, image, boxes, labels):
    convert = transform(train=train)
    timage, tboxes, tlabels = convert(image, boxes, labels)

    # NB: height/width can change, while number of channels not.
    assert timage.shape[0] == image.shape[0]
    # Image is converted to torch already
    assert type(timage) == torch.Tensor

    assert tboxes.shape == tboxes.shape
    assert tboxes.dtype == boxes.dtype

    assert labels.shape == labels.shape
    assert tlabels.dtype == labels.dtype
