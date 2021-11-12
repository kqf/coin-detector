import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.draw import disk, ellipse


def box(cx, cy, w, h):
    ax = plt.gca()
    patch = patches.Rectangle(
        (cx - w / 2, cy - h / 2), w, h,
        linewidth=1,
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


def main():
    shape = (400, 400)
    img = np.zeros(shape)

    bbox = [50, 50, 50, 80]
    img = to_ellipse(img, *bbox)
    plt.imshow(img)
    box(*bbox)
    plt.show()


if __name__ == '__main__':
    main()
