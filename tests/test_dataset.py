from detectors.dataset import DetectionDataset


def test_dataset(fake_dataset):
    data = DetectionDataset(fake_dataset)
    print(data)
