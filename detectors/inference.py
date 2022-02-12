from torchvision.ops import batched_nms
from detectors.encode import to_cchw


def infer(batch, decode, threshold=0.9, **kwargs):
    predictions = []
    preds, anchors = batch
    boxes, classes = preds["boxes"], preds["classes"]
    for boxes_, classes_, anchors_ in zip(boxes, classes, anchors):
        scores_, class_ids_ = classes_.max(dim=-1)
        decoded = decode(boxes_, anchors_)
        selected = batched_nms(decoded, scores_, class_ids_, threshold)
        predictions.append((to_cchw(decoded[selected]), class_ids_[selected]))
    return predictions
