import matplotlib.pyplot as plt
import torch
from detectors.augmentations import transform
from detectors.dataset import DetectionDataset, read_dataset
from detectors.encode import to_cchw
from detectors.shapes import arrows, box, box_mask


def test_dataset(fake_dataset):
    df = read_dataset(fake_dataset / "train.csv")
    data = DetectionDataset(df, transforms=transform())

    for image, labels in data:
        assert len(image.shape) == 3
        assert len(labels["boxes"].shape) == 2
        assert len(labels["classes"].shape) == 1

    for image, labels in data:
        channels_last = image.cpu().numpy().transpose(1, 2, 0)
        masks = []
        for coords in labels["boxes"]:
            coords = to_cchw(torch.Tensor(coords)).numpy()
            box(channels_last, *coords, alpha=0.8)
            masks.append(box_mask(channels_last, *coords))

        # Unused for image-based coords
        # has_object = np.stack(masks).any(axis=0)
        # nontrivial_pixel = channels_last.mean(axis=-1) < 1

        arrows()
        plt.imshow(channels_last)
        plt.show()

        # assert nontrivial_pixel[has_object].any()
        # assert not nontrivial_pixel[~has_object].any()
