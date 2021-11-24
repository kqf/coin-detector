import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.draw import disk, ellipse, rectangle, polygon


def box(cx, cy, w, h):
    ax = plt.gca()
    patch = patches.Rectangle(
        (cx - w / 2, cy - h / 2), w, h,
        linewidth=2,
        edgecolor='r',
        facecolor='none'
    )
    ax.add_patch(patch)


def make_shape(
    img,
    cx=50,
    cy=50,
    h=90,
    w=90,
    image_id=0
):
    pass


def to_circle(img, cx, cy, w, h):
    shape = img.shape
    Y, X = np.ogrid[:shape[0], :shape[1]]

    xx = (X[..., None] - cx)
    yy = (Y[..., None] - cy)
    dists = np.sqrt((xx / w) ** 2 + (yy / h) ** 2)

    mask = dists <= 1. / 2.
    return mask.sum(axis=-1).astype(np.bool8)


def to_disc(img, cx, cy, w, h):
    rr, cc = disk((cy, cx), h / 2., shape=img.shape)
    img[rr, cc] = 1
    return img


def to_ellipse(img, cx, cy, w, h):
    rr, cc = ellipse(cy, cx, h / 2., w / 2., shape=img.shape)
    img[rr, cc] = 1
    return img


def to_recatangle(img, cx, cy, w, h):
    start = (cy - h // 2, cx - w // 2)
    end = (cy + h // 2, cx + w // 2)

    rr, cc = rectangle(start, end, shape=img.shape)
    img[rr, cc] = 1
    return img


def to_polygon(img, cx, cy, w, h, n=3, rot=0):
    i = np.arange(n)

    rows = cy + h / 2 * np.cos(2 * np.pi * i / n + rot)
    cols = cx + w / 2 * np.sin(2 * np.pi * i / n + rot)

    rr, cc = polygon(rows, cols, shape=img.shape)
    img[rr, cc] = 1
    return img
