import numpy as np
from torchvision.ops import batched_nms


def infer(batch, thr=0.5, **kwargs):
    predictions = []
    preds, anchors = batch
    boxes, classes = preds["boxes"], preds["classes"]
    for boxes_, classes_ in zip(boxes, classes):
        scores_, class_ids_ = classes_.max(dim=-1)
        predictions.append(batched_nms(boxes_, scores_, class_ids_, thr))
    return predictions


def one_hot(data, n_classes):
    encoded = np.zeros((data.shape[0], n_classes))
    encoded[np.arange(data.shape[0]), data] = 1
    return encoded
