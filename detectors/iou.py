import torch


def box2area(bbox, eps=1e-16):
    # Asumming the bbox format is [x1, y1, x2, y2]
    width = bbox[..., 2] - bbox[..., 0] + eps
    hegiht = bbox[..., 3] - bbox[..., 1] + eps
    return width * hegiht


def iou(y_pred, y_true, eps=1e-16):
    # The bbox areas
    area_pred = box2area(y_pred)
    area_true = box2area(y_true)

    # Intersection co-ordinates
    x1_i = torch.maximum(y_pred[..., 0], y_true[..., 0])
    y1_i = torch.maximum(y_pred[..., 1], y_true[..., 1])
    x2_i = torch.minimum(y_pred[..., 2], y_true[..., 2])
    y2_i = torch.minimum(y_pred[..., 3], y_true[..., 3])

    intersection = torch.relu(x2_i - x1_i) * torch.relu(y2_i - y1_i)

    union = area_pred + area_true - intersection
    iou_ = intersection / (union + eps)

    return torch.relu(iou_)

