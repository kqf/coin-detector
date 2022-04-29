# import cv2
import functools

import albumentations as alb
import numpy as np
from albumentations.pytorch import ToTensorV2

_mean = [0, 0, 0]
_std = [1, 1, 1]


def pipeline(train=True, mean=None, std=None, size=32 * 13):
    mean = mean or _mean
    std = std or _std

    transforms = [
        # alb.PadIfNeeded(
        #     min_height=int(size),
        #     min_width=int(size),
        #     # border_mode=cv2.BORDER_CONSTANT,
        # ),
        alb.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0)
    ]

    train_transforms = []
    if train:
        train_transforms = [
            # This one is definitely broken
            # alb.HorizontalFlip(),
            # Vertical flips don't change the position of an image
            # they seem to change only annotations
            # This one is definitely broken
            # alb.VerticalFlip(),
            # This one is definitely broken
            # alb.RandomRotate90(),
            # bad
            # This one is definitely broken
            # alb.Flip(1),
            # This one is definitely broken
            # alb.ShiftScaleRotate(p=0.5),
            # Slightly worse
            # alb.ShiftScaleRotate(
            #     shift_limit=0.0625,
            #     scale_limit=0.2,
            #     rotate_limit=15,
            #     p=0.9,
            #     border_mode=cv2.BORDER_REFLECT
            # ),
            # This one is definitely broken
            # alb.RandomCrop(width=350, height=350),
            alb.OneOf([
                alb.HueSaturationValue(10, 15, 10),
                alb.RandomBrightnessContrast(),
            ], p=0.3)
        ]

    return alb.Compose(
        train_transforms + transforms,
        bbox_params=alb.BboxParams(
            format="pascal_voc",
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
