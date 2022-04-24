import pytest
from matplotlib import pyplot as plt

from detectors.augmentations import transform
from detectors.dataset import DetectionDataset, read_dataset
from detectors.model import build_model
from detectors.shapes import arrows, box


@pytest.fixture
def max_epochs(request):
    return request.config.getoption("--max-epochs")


def pplot(data, preds):
    for i, ((image, labels), pp) in enumerate(zip(data, preds)):
        channels_last = image.cpu().numpy().transpose(1, 2, 0)
        for coords in labels["boxes"]:
            box(channels_last, *coords, color="b", alpha=0.4)

        for coords, classes in zip(*pp):
            print(f"Class: {classes}: {coords}")
            box(channels_last, *coords, alpha=0.5, color="r")

        plt.imshow(channels_last)
        arrows()
        plt.show()
        plt.savefig(f"result-{i}.png")


def fit(model, train):
    try:
        model.load_params(f_params="debug-weights.pt")
    except FileNotFoundError:
        model.fit(train)
        model.save_params(f_params="debug-weights.pt")
    return model


def test_model(fake_dataset, max_epochs):
    df = read_dataset(fake_dataset / "train.csv")
    train = DetectionDataset(df, transforms=transform())

    model = build_model(max_epochs=max_epochs)
    model.initialize()
    # fit(model, train)
    model.fit(train)

    predictions = model.predict_proba(train)
    for pp in predictions:
        print(f"Predicted number of boxes: {pp[1]}")
    pplot(data=train, preds=predictions)

    # Now visually check the results
