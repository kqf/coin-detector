import torch


def encode(boxes, anchors):
    x0y0 = boxes[..., :2]
    x1y1 = boxes[..., 2:]

    # Convert to cchw
    wh = x1y1 - x0y0
    cc = x0y0 + wh / 2.

    # Apply nonlinearity
    encoded_cc = (cc - anchors[:, 0:2]) / anchors[:, 2:4]
    encoded_wh = torch.log(wh / anchors[:, 2:4])

    encoded = torch.cat([encoded_cc, encoded_wh], dim=-1)
    return encoded


def decode(boxes, anchors):
    encoded_cc = boxes[..., :2]
    encoded_wh = boxes[..., 2:]

    # Apply nonlinearity
    cc = (encoded_cc * anchors[..., 2:]) + anchors[..., :2]
    wh = torch.exp(encoded_wh) * anchors[..., 2:]

    # Convert to 0011
    x0y0 = cc - wh / 2.
    x1y1 = x0y0 + wh

    decoded = torch.cat([x0y0, x1y1], dim=-1)
    return decoded
