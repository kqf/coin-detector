import torch
import random

import pytest
import numpy as np
import pandas as pd

from detectors.mc import generate_to_directory


@pytest.fixture
def fixed_seed(seed=137):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


@pytest.fixture
def size():
    return 256


def to_image_path(path, x, extension=".png"):
    return (path / x).with_suffix(extension)


@pytest.fixture
def annotations(fixed_seed, tmp_path, width=2000, num_classes=3, n_samples=8):
    """
                               image_id          class_name  class_id rad_id   x_min   y_min   x_max   y_max
    0  50a418190bc3fb1ef1633bf9678929b3          No finding        14    R11     NaN     NaN     NaN     NaN
    1  21a10246a5ec7af151081d0cd6d65dc9          No finding        14     R7     NaN     NaN     NaN     NaN
    2  9a5094b2563a1ef3ff50dc5c7ff71345        Cardiomegaly         3    R10   691.0  1375.0  1653.0  1831.0
    3  051132a778e61a86eb147c7c6f564dfe  Aortic enlargement         0    R10  1264.0   743.0  1611.0  1019.0
    4  063319de25ce7edb9b1c6b8881290140          No finding        14    R10     NaN     NaN     NaN     NaN
    """  # noqa

    # NB: 2 here stands for the second blob on an image
    df = pd.DataFrame(index=range(n_samples * 2))
    df["class_id"] = df.index % num_classes

    n_images = len(df)
    shift = 1 + df.index / len(df)
    shift = 1
    df["image_id"] = df.index % n_samples
    df["image"] = df["image_id"].apply(
        lambda x: to_image_path(tmp_path, str(x))
    )

    df.loc[:n_images // 2 - 1, 'x_min'] = 200.0 * shift
    df.loc[:n_images // 2 - 1, 'x_max'] = 200.0 * shift + width * 0.28
    df.loc[:n_images // 2 - 1, 'y_min'] = 400.0 * shift
    df.loc[:n_images // 2 - 1, 'y_max'] = 400.0 * shift + width * 0.22

    df.loc[n_images // 2:, 'x_min'] = 450.0 * shift
    df.loc[n_images // 2:, 'x_max'] = 450.0 * shift + width * 0.28
    df.loc[n_images // 2:, 'y_min'] = 800.0 * shift
    df.loc[n_images // 2:, 'y_max'] = 800.0 * shift + width * 0.22

    df["h"] = width
    df["w"] = width

    x1, y1, x2, y2 = df[['x_min', 'y_min', 'x_max', 'y_max']].values.T
    df['x_center'] = (x1 + x2) / 2 / df["w"]
    df['y_center'] = (y1 + y2) / 2 / df["h"]
    df['width'] = (x2 - x1) / df["w"]
    df['height'] = (y2 - y1) / df["h"]
    return df


@pytest.fixture
def fake_dataset(tmp_path, annotations, size=256):
    generate_to_directory(annotations, tmp_path)
    yield tmp_path
