import numpy as np
import pandas as pd


def make_shape(
    x_min=50, y_min=50,
    x_max=90, y_max=90,
    h=2000, w=2000,
):

    Y, X = np.ogrid[:h, :w]

    w = (x_max - x_min)
    h = (y_max - y_min)

    cx = x_min + w / 2.
    cy = y_min + h / 2.

    xx = (X[..., None] - cx)
    yy = (Y[..., None] - cy)
    dists = np.sqrt((xx / w) ** 2 + (yy / h) ** 2)

    mask = dists <= 1. / 2.
    return mask.sum(axis=-1).astype(np.uint8)


def make_blob(
    x_min=50,
    y_min=50,
    x_max=90,
    y_max=90,
    h=2000,
    w=2000,
    channels=3,
    epsilon=0.1,
    class_id=0,
    **kwargs
):
    blob = make_shape(x_min, y_min, x_max, y_max, h, w)
    h, w = blob.shape

    extended = blob[..., None]
    # return extended + 255

    # Add a small term to add noise to the empty regions
    noise = np.random.poisson(extended * epsilon, size=(h, w, channels))

    # Convet to image scale
    return (extended + class_id * noise * 255.).astype(np.uint8)


def blob2image(blob, channels=3, epsilon=0.1, class_id=0):
    return blob


def annotations(n_points=32, h=2000, w=2000):
    x = np.random.uniform(0, w, (n_points, 2))
    y = np.random.uniform(0, h, (n_points, 2))
    df = pd.DataFrame({
        # NB: Replace np.arange by np.uniform(0, n_images, n_points)
        #     to get the dataset with multiple images
        "image_id": np.arange(n_points)
    })

    df["x_min"] = x.min(axis=1)
    df["y_min"] = y.min(axis=1)

    df["x_max"] = x.max(axis=1)
    df["y_max"] = x.max(axis=1)
    labels = (df["x_max"] - df["x_min"]) > (df["y_max"] - df["y_min"])
    df["class_id"] = labels.astype(int)
    df["class_name"] = labels.astype(str)
    return df
