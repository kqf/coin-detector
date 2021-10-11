import cv2
import numpy as np
import torch


def read_image(file):
    bitmap = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    image = np.stack([bitmap, bitmap, bitmap]).astype('float32')
    return image.transpose(1, 2, 0)


class DetectionDataset(torch.datasets.Dataset):

    def __init__(self, df, transforms=None):
        super().__init__()
        self.image_ids = df["image_id"].unique()
        self.df = df

    def example(self, index):
        image_id = self.image_ids[index]
        records = self.df[(self.df['image_id'] == image_id)]
        records = records.reset_index(drop=True)

        file = f"{self.image_dir}/{image_id}.png"
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        image = np.stack([image, image, image])
        image = image.astype('float32')
        image = image.transpose(1, 2, 0)

        if records.loc[0, "class_id"] == 0:
            records = records.loc[[0], :]

        boxes = records[['x_center', 'y_center', 'width', 'height']].values

        labels = torch.tensor(records["class_id"].values, dtype=torch.int64)

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': boxes,
                'labels': labels
            }
            transformed = self.transforms(**sample)

            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']

        _, width, height = image.shape
        targets = build_targets(
            boxes, labels,
            self.anchors,
            self.scales,
            self.iou_threshold,
            num_anchors_per_scale=self.num_anchors_per_scale,
            im_size=width,
        )

        return image, boxes, targets

    def __len__(self):
        return self.image_ids.shape[0]

    def __getitem__(self, index):
        image, boxes, targets = self.example(index)
        return image, targets
