from torchvision.ops import batched_nms
from detectors.encode import decode


def infer(batch, thr=0.5, **kwargs):
    predictions = []
    preds, anchors = batch
    boxes, classes = preds["boxes"], preds["classes"]
    for boxes_, classes_, anchors_ in zip(boxes, classes, anchors):
        scores_, class_ids_ = classes_.max(dim=-1)
        decoded = decode(boxes_, anchors_)
        predictions.append(batched_nms(decoded, scores_, class_ids_, thr))
    return predictions
