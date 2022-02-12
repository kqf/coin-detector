from torchvision.ops import batched_nms

from detectors.encode import to_coords


def infer(batch, decode, threshold=0.5, **kwargs):
    predictions = []
    preds, anchors = batch
    boxes, classes = preds["boxes"], preds["classes"]
    for boxes_, classes_, anchors_ in zip(boxes, classes, anchors):
        scores_, class_ids_ = classes_.max(dim=-1)
        decoded = decode(boxes_, anchors_)
        selected = batched_nms(decoded, scores_, class_ids_, threshold)
        output_boxes = to_coords(decoded[selected])
        predictions.append((output_boxes, class_ids_[selected]))

    return predictions
