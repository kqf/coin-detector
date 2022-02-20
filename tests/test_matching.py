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
    return torch.zeros((480, 480))


@pytest.fixture
def anchors(batch_size, image, latent_size=5):
    layer = AnchorBoxes()
    latent = torch.zeros(batch_size, 1, latent_size, latent_size)
    anchors, _ = layer(image.shape, [latent])
    return anchors


@pytest.mark.parametrize("n_targets", [2])
@pytest.mark.parametrize("batch_size", [16])
def test_matches(
    target_boxes, target_classes, anchors, batch_size, n_targets,
):
    mask = target_classes > 1000

    _, n_anchors, _ = anchors.shape

    positives, negatives = match(
        to_coords(target_boxes),
        mask,
        to_coords(anchors)
    )

    exp_positives = (anchors[:, None] == target_boxes[:, :, None]).all(dim=-1)
    import ipdb
    ipdb.set_trace()
    import IPython
    IPython.embed()  # noqa
    assert (exp_positives == positives).all()

    for image, labels in data:
        channels_last = image.cpu().numpy().transpose(1, 2, 0)
        masks = []
        for coords in labels["boxes"]:
            box(channels_last, *coords)

    assert positives.shape == (batch_size, n_targets, n_anchors)
    assert negatives.shape == (batch_size, n_anchors)
