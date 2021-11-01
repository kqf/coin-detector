import torch


def to_coords(cchw):
    x = cchw.clone().detach()
    # center - width / 2 = x0
    x[..., 0] = x[..., 0] - x[..., 2] / 2.
    x[..., 1] = x[..., 1] - x[..., 3] / 2.

    # center + width / 2 = x1
    x[..., 2] = x[..., 0] + x[..., 2] / 2.
    x[..., 3] = x[..., 1] + x[..., 3] / 2.
    return x


def encode(boxes, anchors):
    # Convert to cchw
    wh = boxes[..., 2:]
    cc = boxes[..., :2]

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
