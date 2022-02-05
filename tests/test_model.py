import pytest
from detectors.model import build_model
from detectors.dataset import DetectionDataset, read_dataset
from detectors.augmentations import transform

from matplotlib import pyplot as plt
from detectors.shapes import box, arrows


@pytest.fixture
def max_epochs(request):
    return request.config.getoption("--max-epochs")


def test_model(fake_dataset, max_epochs):
    df = read_dataset(fake_dataset / "train.csv")
    train = DetectionDataset(df, transforms=transform())

    model = build_model(max_epochs=max_epochs)
    model.fit(train)
    predictions = model.predict_proba(train)

    # Now visually check the results
    for (image, labels), preds in zip(train, predictions):
        channels_last = image.cpu().numpy().transpose(1, 2, 0)
        for coords in preds:
            box(channels_last, *coords)
        plt.imshow(channels_last)
        arrows()
        plt.show()
