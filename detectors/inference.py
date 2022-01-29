from detectors.iou import iou


def nms(predictions, threshold=0.5, min_iou=0.5, top_n=None):
    # Filter out the boxes with low objectness score
    # classes[anchors, n_clases]
    classes = predictions["classes"]

    # classes[anchors, 4]
    boxes = predictions["boxes"]

    x = predictions[classes[..., 0] > threshold]

    # Ensure everything is calculated per class
    same_object = x[:, None, -1] == x[None, :, -1]

    # Find non-maximum elements
    objectness_per_class = same_object * x[None, :, 0]
    maximum = objectness_per_class.max(-1, keepdim=True).values
    not_maximum = objectness_per_class < maximum

    # IoUs
    ious = iou(boxes[:, None], boxes[None, :])

    # Putting it all together
    noise = same_object * not_maximum * (ious > min_iou).squeeze(-1)
    suppressed = (~noise).all(0)
    return x[suppressed, 1:]
