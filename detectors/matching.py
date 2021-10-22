import torch
from detectors.iou import iou


def match_positives(score, pos_th):
    # socre[batch_size, n_obj, n_anchor]
    max_overlap = torch.abs(score.max(dim=1, keepdim=True)[0] - score) < 1.0e-6
    positive = max_overlap & (score > pos_th)
    return positive


def match(
    boxes,  # [batch_size, n_obj, 4]
    mask,  # [batch_size, n_obj]
    anchors,  # [batch_size, n_anchors, 4]
    on_image=None,  # [batch_size, n_anchors]
    criterion=iou,
    pos_th=0.5,
    neg_th=0.1,
    fill_value=-1,
):
    # criterion([batch_size, 1, n_anchors, 4], [batch_size, n_obj, 1, 4])
    # ~> overlap[batch_size, n_obj, n_anchor]
    overlap = criterion(anchors[:, None], boxes[:, :, None])

    # Remove all scores that are masked
    overlap[mask] = fill_value

    positive = match_positives(overlap, pos_th)

    # Check if within image
    if on_image is not None:
        positive = positive & on_image[..., None].bool()

    # Negatives are the anchors that have quite small
    # largest overlap with objects
    # overlap[batch_size, n_obj, n_anchor]
    overlap_, _ = overlap.max(dim=2)
    negative = overlap_ < neg_th

    return positive, negative
