import pandas as pd
import matplotlib.pyplot as plt
from detectors.dataset import DetectionDataset


def test_dataset(fake_dataset):
    df = pd.read_csv(fake_dataset / "train.csv")
    data = DetectionDataset(df)

    for image, labels in data:
        assert len(image.shape) == 3
        assert len(labels["boxes"].shape) == 2
        assert len(labels["classes"].shape) == 1

    for image, labels in data:
        plt.imshow(image.transpose(1, 2, 0))
        plt.show()
