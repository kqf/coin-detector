import torch
from detectors.iou import iou


def _padded(values, padding_mask, fill_value=-1):
    # Set fill_value for padded labes
    values[padding_mask] = fill_value
    # Restore the desired dimension order
    permuted = values.permute(0, 2, 1)

    return permuted


def match_positives(score, pos_th, on_image=None):
    max_overlap = torch.abs(score.max(dim=2, keepdim=True)[0] - score) < 1.0e-6
    positive = max_overlap & (score > pos_th)
    return positive


def match(
    boxes,
    classes,
    anchors,
    on_image=None,
    criterion=iou,
    pos_th=0.5,
    neg_th=0.1,
):
    # [b, n_obj, n_classes] -> b[b, n_obj]
    padding = torch.isnan(classes).all(-1)

    # criterion[b, n_obj, all_anchors] -> _padded[b, all_anchors, n_obj]
    overlap = _padded(criterion(anchors[:, None], boxes[:, :, None]), padding)

    positive = match_positives(overlap, pos_th)

    # Check if within image
    if on_image is not None:
        positive = positive & on_image[..., None].bool()

    # Define negatives as those with the largest overlap with any gt box
    overlap_, _ = overlap.max(dim=2)
    negative = overlap_ < neg_th

    return positive, negative
