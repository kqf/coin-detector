import numpy as np
import matplotlib.pyplot as plt
from detectors.dataset import DetectionDataset, read_dataset
from detectors.shapes import box, arrows
from detectors.augmentations import transform
from detectors.shapes import box_mask


def test_dataset(fake_dataset):
    df = read_dataset(fake_dataset / "train.csv")
    data = DetectionDataset(df, transforms=transform())

    for image, labels in data:
        assert len(image.shape) == 3
        assert len(labels["boxes"].shape) == 2
        assert len(labels["classes"].shape) == 1

    for image, labels in data:
        channels_last = image.cpu().numpy().transpose(1, 2, 0)
        plt.imshow(channels_last)
        masks = []
        for coords in labels["boxes"]:
            box(channels_last, *coords)
            masks.append(box_mask(channels_last, *coords))
        masks = np.stack(masks).any(axis=0)
        arrows()
        plt.show()
