import pandas as pd
from detectors.dataset import DetectionDataset


def test_dataset(fake_dataset):
    df = pd.read_csv(fake_dataset / "train.csv")
    data = DetectionDataset(df)

    for image, labels in data:
        assert len(image.shape) == 3
        assert len(labels["boxes"].shape) == 2
        assert len(labels["labels"].shape) == 1
