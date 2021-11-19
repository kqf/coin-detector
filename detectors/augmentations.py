import albumentations as alb
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


def transform(train=True, mean=None, std=None, size=32 * 13):
    mean = mean or _mean
    std = std or _std

    transforms = [
        alb.PadIfNeeded(
            min_height=int(size),
            min_width=int(size),
            # border_mode=cv2.BORDER_CONSTANT,
        ),
        # DebugAugmentations(),
        alb.Resize(size, size),
        alb.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0)
    ]

    train_transforms = []
    if train:
        train_transforms = [
            alb.Flip(0.5)
        ]

    return alb.Compose(
        train_transforms + transforms,
        bbox_params=alb.BboxParams(
            format="pascal_voc",
            label_fields=['labels']
        )
    )
