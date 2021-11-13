import numpy as np
import pandas as pd


def make_shape(
    cx=50,
    cy=50,
    h=90,
    w=90,
    shape=(2000, 2000),
):
    print()
    Y, X = np.ogrid[:shape[0], :shape[1]]

    xx = (X[..., None] - cx)
    yy = (Y[..., None] - cy)
    dists = np.sqrt((xx / w) ** 2 + (yy / h) ** 2)

    mask = dists <= 1. / 2.
    return mask.sum(axis=-1).astype(np.uint8)


def make_blob(
    x_center=50,
    y_center=50,
    width=90,
    height=90,
    channels=3,
    epsilon=0.1,
    class_id=0,
    shape=(2000, 2000),
    **kwargs
):
    blob = make_shape(x_center, y_center, width, height, shape)
    h, w = blob.shape

    extended = blob[..., None]
    # return extended + 255

    # Add a small term to add noise to the empty regions
    noise = np.random.poisson(extended * epsilon, size=(h, w, channels))

    # Convet to image scale
    return (extended + class_id * noise * 255.).astype(np.uint8)


def blob2image(blob, channels=3, epsilon=0.1, class_id=0):
    return blob


def make_image(shapes, image_shape):
    blobs = []
    for row in shapes:
        blobs.append(make_blob(**row, shape=image_shape))
    blob = np.stack(blobs, axis=0).mean(axis=0)
    img = blob2image(blob)
    return img


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
