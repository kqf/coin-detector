from torchvision.ops import batched_nms


def infer(batch, decode, background_class=0, threshold=0.2, **kwargs):
    predictions = []
    preds, anchors = batch
    boxes, classes = preds["boxes"], preds["classes"]
    for boxes_, classes_, anchors_ in zip(boxes, classes, anchors):
        scores_, class_ids_ = classes_.max(dim=-1)

        # Filter out the background detections
        pos = class_ids_ != background_class
        decoded = decode(boxes_, anchors_)[pos]
        scores_ = scores_[pos]
        class_ids_ = class_ids_[pos]

        selected = batched_nms(decoded, scores_, class_ids_, threshold)
        predictions.append((decoded[selected], class_ids_[selected]))

    return predictions
