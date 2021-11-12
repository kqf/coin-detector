import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.draw import disk


def box(cx, cy, w, h):
    ax = plt.gca()
    patch = patches.Rectangle(
        (cx - w / 2, cy - h / 2), h, w,
        linewidth=1,
        edgecolor='r',
        facecolor='none'
    )
    ax.add_patch(patch)


def to_disc(img, cx, cy, h, w):
    rr, cc = disk((cx, cy), h / 2., shape=img.shape)
    img[rr, cc] = 1
    return img


def main():
    shape = (400, 400)
    img = np.zeros(shape)

    bbox = [200, 200, 50, 50]
    img = to_disc(img, *bbox)
    print(img)
    plt.imshow(img)
    box(*bbox)
    plt.show()


if __name__ == '__main__':
    main()
