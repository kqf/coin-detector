import numpy as np
import pandas as pd


def make_shape(
    cx=50,
    cy=50,
    h=90,
    w=90,
    shape=(2000, 2000),
):
    Y, X = np.ogrid[:shape[0], :shape[1]]

    xx = (X[..., None] - cx)
    yy = (Y[..., None] - cy)
    dists = np.sqrt((xx / w) ** 2 + (yy / h) ** 2)

    mask = dists <= 1. / 2.
    return mask.sum(axis=-1).astype(np.bool8)


def make_colors(num_colors, channels=3, intensity_range=(0, 255)):
    intensity_range = (intensity_range, ) * channels
    colors = [np.random.randint(_min, _max, size=num_colors)
              for _min, _max in intensity_range]
    return np.transpose(colors)


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


def make_image(shapes, image_shape, channels=3, fmt="coco"):
    canvas_shape = (image_shape[0], image_shape[1], channels)
    image = np.full(canvas_shape, 255, dtype=np.uint8)
    for shape in shapes:
        idx = make_shape(*shape[fmt], shape=image_shape)
        image[idx] = shape["colors"]
    return image
