import numpy as np

from detectors.iou import iou


def one_hot(data, n_classes):
    encoded = np.zeros((data.shape[0], n_classes))
    encoded[np.arange(data.shape[0]), data] = 1
    return encoded


def nms(predictions, threshold=0.5, min_iou=0.5, top_n=None):
    # Filter out the boxes with low objectness score
    # classes[anchors, n_clases]
    classes = predictions["classes"]

    # classes[anchors, 4]
    boxes = predictions["boxes"]

    # n_anchors
    x = classes.argmax(-1)

    non_background = x != 0
    x = x[non_background]
    boxes = boxes[non_background]
    classes = classes[non_background]

    one_hot_classes = one_hot(x, classes.shape[-1])

    # Find non-maximum elements
    objectness_per_class = one_hot_classes * classes
    maximum = objectness_per_class.max(-1, keepdim=True).values
    not_maximum = objectness_per_class < maximum

    # IoUs
    ious = iou(boxes[:, None], boxes[None, :])

    # Putting it all together
    noise = not_maximum * (ious > min_iou).squeeze(-1)
    suppressed = (~noise).all(0)
    return x[suppressed, 1:]
