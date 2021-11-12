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


def to_polygon(img, cx, cy, w, h):
    # starts = (cy - h // 2, cx - w // 2)
    # ends = (cy + h // 2, cx + w // 2)

    starts = np.array([
        cy - h // 2,
        cy - h // 2,
        cy + h // 2,
        cy + h // 2,
        # cy - h // 2,
    ])
    ends = np.array([
        cx - w // 2,
        cx + w // 2,
        cx + w // 2,
        cx - w // 2,
        # cx - w // 2,
    ])

    rr, cc = polygon(starts, ends)
    img[rr, cc] = 1
    return img


def main():
    shape = (400, 400)
    img = np.zeros(shape)

    bbox = [50, 50, 80, 50]
    img = to_ellipse(img, *bbox)
    box(*bbox)

    bbox = [350, 350, 50, 80]
    img = to_recatangle(img, *bbox)
    box(*bbox)

    bbox = [50, 350, 50, 80]
    img = to_disc(img, *bbox)
    box(*bbox)

    bbox = [350, 50, 50, 80]
    img = to_polygon(img, *bbox)
    box(*bbox)

    plt.imshow(img)
    box(*bbox)
    plt.show()


if __name__ == '__main__':
    main()
