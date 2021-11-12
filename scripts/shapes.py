import numpy as np
import matplotlib.pyplot as plt

from skimage.draw import disk


def main():
    shape = (400, 400)
    img = np.zeros(shape, dtype=np.uint8)
    rr, cc = disk((50, 50), 25, shape=shape)
    img[rr, cc] = 1
    print(img)
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    main()
