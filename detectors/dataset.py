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
        self.transforms = transforms
        self.df = df

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        records = self.df[(self.df['image_id'] == image_id)]
        records = records.reset_index(drop=True)

        file = f"{self.image_dir}/{image_id}.png"
        image = read_image(file)

        if records.loc[0, "class_id"] == 0:
            records = records.loc[[0], :]

        boxes = records[['x_center', 'y_center', 'width', 'height']].values

        labels = torch.tensor(
            records["class_id"].values,
            dtype=torch.int64
        )

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

        return image, boxes, labels

    def __len__(self):
        return self.image_ids.shape[0]