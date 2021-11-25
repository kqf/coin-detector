import numpy as np
from detectors.shapes import make_shape


def make_colors(num_colors, channels=3, intensity_range=(0, 255)):
    intensity_range = (intensity_range, ) * channels
    colors = [np.random.randint(_min, _max, size=num_colors)
              for _min, _max in intensity_range]
    return np.transpose(colors)


def make_image(
        shapes, image_shape, channels=3, fmt="coco", shape_col="class_id"):
    canvas_shape = (image_shape[0], image_shape[1], channels)
    image = np.full(canvas_shape, 255, dtype=np.uint8)
    for shape in shapes:
        print(shape)
        idx = make_shape(image, *shape[fmt], shape=shape[shape_col])
        image[idx] = shape["colors"]
    return image
