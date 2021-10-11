import pandas as pd
from detectors.dataset import DetectionDataset


def test_dataset(fake_dataset):
    df = pd.read_csv(fake_dataset / "train.csv")
    data = DetectionDataset(df)

    for batch in data:
        print(batch)
