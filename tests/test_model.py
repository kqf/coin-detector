import pytest
import pandas as pd
from detectors.model import build_model
from detectors.dataset import DetectionDataset


# @pytest.mark.xfail
def test_model(fake_dataset):
    df = pd.read_csv(fake_dataset / "train.csv")
    train = DetectionDataset(df)

    model = build_model()
    model.fit(train)
