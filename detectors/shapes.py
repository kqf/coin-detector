import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.draw import disk, ellipse, rectangle, polygon
from functools import partial


def _patch(xy, *args, **kwargs):
    return patches.Rectangle(xy, *args, **kwargs)


def box_mask(image, cx, cy, w, h):
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = cy - h / 2, cy + h / 2

    xx, yy = np.meshgrid(
        np.arange(image.shape[0]),
        np.arange(image.shape[1]),
    )
    mask = (x0 <= xx) & (xx <= x1) & (y0 <= yy) & (yy <= y1)
    return mask


def arrows():
    plt.grid(which="minor")
    ax = plt.gca()
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.tick_top()
    ax.minorticks_on()

    # make arrows
    ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
            transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot((1), (0), ls="", marker="v", ms=10, color="k",
            transform=ax.get_xaxis_transform(), clip_on=False)

    # plt.axes().yaxis.set_minor_locator(MultipleLocator(8))


def box(img, cx, cy, w, h, color="g"):
    imw, imh, *_ = img.shape
    ax = plt.gca()
    patch = _patch(
        (cx - w / 2, cy - h / 2), w, h,
        linewidth=2,
        edgecolor=color,
        facecolor='none',
        alpha=0.1
    )
    ax.add_patch(patch)

    # plt.arrow(
    #     cx + w, cy + h / 2 * 0.8, 0, -h * 0.8,
    #     length_includes_head=True,
    #     width=2, color="r", edgecolor="r"
    # )

    # plt.arrow(
    #     cx + w, cy - h, 0, -h,
    #     length_includes_head=True,
    #     width=2,
    #     color="b",
    #     edgecolor="b",
    # )


def to_circle(img, cx, cy, w, h):
    shape = img.shape
    Y, X = np.ogrid[:shape[0], :shape[1]]

    xx = (X[..., None] - cx)
    yy = (Y[..., None] - cy)
    dists = np.sqrt((xx / w) ** 2 + (yy / h) ** 2)

    mask = dists <= 1. / 2.
    return mask.sum(axis=-1).astype(np.bool8)


def to_disc(img, cx, cy, w, h):
    mask = np.zeros(img.shape[:2])
    diameter = min(w, h)
    rr, cc = disk((cx, cy), diameter / 2., shape=img.shape)
    mask[cc, rr] = 1
    return mask.astype(np.bool8)


def to_ellipse(img, cx, cy, w, h):
    mask = np.zeros(img.shape[:2])
    rr, cc = ellipse(cy, cx, h / 2., w / 2., shape=img.shape)
    mask[rr, cc] = 1
    return mask.astype(np.bool8)


def to_recatangle(img, cx, cy, w, h):
    mask = np.zeros(img.shape[:2])
    start = (int(cy - h / 2), int(cx - w / 2))
    end = (int(cy + h / 2), int(cx + w / 2))

    rr, cc = rectangle(start, end, shape=img.shape)
    mask[rr, cc] = 1
    return mask.astype(np.bool8)


def to_polygon(img, cx, cy, w, h, n=3, rot=0):
    mask = np.zeros(img.shape[:2])
    i = np.arange(n)

    cols = cy + h / 2 * np.cos(2 * np.pi * i / n + rot)
    rows = cx + w / 2 * np.sin(2 * np.pi * i / n + rot)

    rr, cc = polygon(cols, rows, shape=img.shape)
    mask[rr, cc] = 1
    return mask.astype(np.bool8)


_AVAILABLE_SHAPES = [
    to_circle,
    partial(to_polygon, n=4),
    to_polygon,
]


def make_shape(
    img,
    cx=50,
    cy=50,
    h=90,
    w=90,
    shape=0
):
    shape = _AVAILABLE_SHAPES[shape]
    return shape(img, cx, cy, h, w)
