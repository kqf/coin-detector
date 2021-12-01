import cv2
import numpy as np
import albumentations as alb
import functools

from albumentations.core.transforms_interface import DualTransform
from albumentations.pytorch import ToTensorV2

_mean = [0, 0, 0]
_std = [1, 1, 1]


class DebugAugmentations(DualTransform):
    def __init__(self, always_apply=True, p=1):
        super().__init__(always_apply, p)

    def get_params(self):
        return {"scale": 1}

    def apply(self, img, *args, **params):
        return img

    def apply_to_bbox(self, bbox, **params):
        # ipdb goes here
        pass

    def apply_to_keypoint(self, keypoint, scale=0, **params):
        return keypoint

    def get_transform_init_args(self):
        return {}


def pipeline(train=True, mean=None, std=None, size=32 * 13):
    mean = mean or _mean
    std = std or _std

    transforms = [
        # alb.PadIfNeeded(
        #     min_height=int(size),
        #     min_width=int(size),
        #     # border_mode=cv2.BORDER_CONSTANT,
        # ),
        # # DebugAugmentations(),
        # alb.Resize(size, size),
        alb.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0)
    ]

    train_transforms = []
    if train:
        train_transforms = [
            # bad
            # alb.Flip(1),
            # bad
            # alb.RandomRotate90(),
            alb.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.2,
                rotate_limit=15,
                p=0.9,
                border_mode=cv2.BORDER_REFLECT
            ),
        ]

    return alb.Compose(
        train_transforms + transforms,
        bbox_params=alb.BboxParams(
            format="coco",
            label_fields=['labels']
        )
    )


def apply(image, boxes, labels, transformations):
    sample = {
        'image': image.transpose(1, 2, 0),
        'bboxes': boxes,
        'labels': labels
    }
    transformed = transformations(**sample)

    image = transformed['image']
    boxes = np.array(transformed['bboxes'], dtype=np.float32)
    labels = np.array(transformed['labels'], dtype=labels.dtype)
    return image, boxes, labels


def transform(train=True, mean=None, std=None, size=32 * 13):
    transformations = pipeline(train, mean, std, size)
    return functools.partial(apply, transformations=transformations)
