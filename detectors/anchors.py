from functools import reduce
from operator import mul

import torch

DEFAULT_ANCHORS = [
    [
        # (10, 10),
        # (5, 10),
        # (10, 5),
        (80, 80),
    ]
]


class AnchorBoxes(torch.nn.Module):
    def __init__(self, anchors=None):
        super().__init__()
        self.anchors = anchors or DEFAULT_ANCHORS
        self.device = torch.device("cpu")

    def forward(self, image_shape, layers):
        out_layers = []
        idx_tensor = []
        for idx, layer in enumerate(layers):
            b, _c, *layer_shape = layer.shape
            num_anchors = reduce(mul, layer_shape, 1)

            strides = torch.tensor(image_shape, device=self.device) / \
                torch.tensor(layer_shape, device=self.device)
            offsets = strides / 2

            # create meshgrid
            cell_y, cell_x = torch.meshgrid(
                torch.arange(layer_shape[0], device=self.device),
                torch.arange(layer_shape[1], device=self.device),
            )

            # create heights and widths
            anchors = []
            for (anchor_height, anchor_width) in self.anchors[idx]:
                height = torch.ones(layer_shape, device=self.device) \
                    * anchor_height

                width = torch.ones(layer_shape, device=self.device) \
                    * anchor_width

                x_coords = cell_x * strides[1] + offsets[1]
                y_coords = cell_y * strides[0] + offsets[0]

                # outs[layer_h, layer_w, 4]
                outs = torch.stack([x_coords, y_coords, width, height], dim=-1)

                # outs[layer_h, layer_w, 4] -> outs[layer_h * layer_w, 4]
                anchors.append(outs.reshape(num_anchors, -1))

            # list of num_layers tensors [n_anchors * layer_h * layer_w, 4]
            out_layers.append(torch.cat(anchors, dim=0))
            # create a lookup for which anchor comes from which layer
            idx_tensor.append(
                torch.full((num_anchors,), idx, device=self.device)
            )

        # single_example[n_layers * n_anchors * h * w, 4]
        single_example = torch.cat(out_layers, dim=0)
        single_indices = torch.cat(idx_tensor, dim=0)

        # batch[batch_size, n_layers * n_anchors * h * w, 4]
        # batch = single_example.repeat((b, 1, 1))
        # batch_indices = single_indices.repeat((b, 1))
        return single_example, single_indices

    def to(self, device):
        self.device = device
        return super().to(device)
