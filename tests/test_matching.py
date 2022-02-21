import matplotlib.pyplot as plt
import pytest
import torch

from detectors.anchors import AnchorBoxes
from detectors.encode import to_coords
from detectors.matching import match
from detectors.shapes import box


@pytest.fixture
def target_boxes(batch_size, n_targets):
    x = torch.ones((batch_size, n_targets, 4))
    x[:, 0, :] = torch.tensor([144.0, 144.0, 96., 96.0])
    x[:, 1, :] = torch.tensor([240.0, 240.0, 96., 96.0])
    return x


@pytest.fixture
def target_classes(batch_size, n_targets):
    x = torch.ones((batch_size, n_targets))
    x[:, 0] = 0
    x[:, 1] = 1
    return x


@pytest.fixture
def image():
    return torch.zeros((480, 480, 3)) + 255.


@pytest.fixture
def anchors(batch_size, image, latent_size=5):
    layer = AnchorBoxes()
    latent = torch.zeros(batch_size, 1, latent_size, latent_size)
    anchors, _ = layer(image.shape[:-1], [latent])
    return anchors


@pytest.mark.parametrize("n_targets", [2])
@pytest.mark.parametrize("batch_size", [16])
def test_matches(
    image, target_boxes, target_classes, anchors, batch_size, n_targets,
):
    mask = target_classes > 1000

    _, n_anchors, _ = anchors.shape

    positives, negatives, ious = match(
        to_coords(target_boxes),
        mask,
        to_coords(anchors)
    )

    for anchor in anchors[0]:
        print(anchor)
        box(image, *anchor)

    for target in target_boxes[0]:
        box(image, *target, color="b", lw=5, alpha=1)

    b_, obj_, anch_ = torch.where(positives)
    pos_boxes = anchors[b_, anch_]

    print("The corresponding ious", ious[b_, obj_, anch_][b_ == 0])
    print("The boxes", pos_boxes[b_ == 0])
    for bbox in pos_boxes[b_ == 0]:
        box(image, *bbox, color="r", alpha=1)

    plt.imshow(image)
    plt.show()
    return

    exp_positives = (anchors[:, None] == target_boxes[:, :, None]).all(dim=-1)
    assert (exp_positives == positives).all()
    assert positives.shape == (batch_size, n_targets, n_anchors)
    assert negatives.shape == (batch_size, n_anchors)
