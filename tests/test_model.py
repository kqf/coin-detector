import pytest
from matplotlib import pyplot as plt

from detectors.augmentations import transform
from detectors.dataset import DetectionDataset, read_dataset
from detectors.model import build_model
from detectors.shapes import arrows, box


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
    counts = 0
    for (image, labels), (preds, classes) in zip(train, predictions):
        channels_last = image.cpu().numpy().transpose(1, 2, 0)
        for coords in preds:
            box(channels_last, *coords)
        plt.imshow(channels_last)
        arrows()
        plt.show()
        counts += 1
        plt.savefig(f"result-{counts}.png")
