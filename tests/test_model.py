import pytest
from detectors.model import build_model
from detectors.dataset import DetectionDataset, read_dataset
from detectors.augmentations import transform


@pytest.fixture
def max_epochs(request):
    return request.config.getoption("--max-epochs")


def test_model(fake_dataset, max_epochs):
    df = read_dataset(fake_dataset / "train.csv")
    train = DetectionDataset(df, transforms=transform())

    model = build_model(max_epochs=max_epochs)
    model.fit(train)
    model.predict_proba(train)
