import numpy as np
import matplotlib.pyplot as plt

from detectors.shapes import to_ellipse, to_recatangle, to_disc, to_polygon

from detectors.shapes import box


def main():
    shape = (400, 400)
    img = np.zeros(shape)

    bbox = [350, 50, 80, 50]
    img = to_ellipse(img, *bbox)
    box(*bbox)

    bbox = [350, 150, 50, 80]
    img = to_recatangle(img, *bbox)
    box(*bbox)

    bbox = [350, 250, 50, 80]
    img = to_disc(img, *bbox)
    box(*bbox)

    bbox = [350, 350, 50, 100]
    img = to_polygon(img, *bbox)
    box(*bbox)

    plt.imshow(img)
    box(*bbox)
    plt.show()


if __name__ == '__main__':
    main()
