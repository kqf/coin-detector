import matplotlib.pyplot as plt
from detectors.dataset import DetectionDataset, read_dataset
from detectors.shapes import box
from detectors.augmentations import transform


def test_dataset(fake_dataset):
    df = read_dataset(fake_dataset / "train.csv")
    data = DetectionDataset(df, transforms=transform())

    for image, labels in data:
        assert len(image.shape) == 3
        assert len(labels["boxes"].shape) == 2
        assert len(labels["classes"].shape) == 1

    for image, labels in data:
        plt.imshow(image.transpose(1, 2, 0))
        for coords in labels["boxes"]:
            box(*coords)
        plt.show()
