import cv2
import numpy as np
import torch


def read_image(file):
    bitmap = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    image = np.stack([bitmap, bitmap, bitmap]).astype('float32')
    return image.transpose(1, 2, 0)


class DetectionDataset(torch.utils.data.Dataset):

    def __init__(self, df, image_col="image_id", transforms=None):
        super().__init__()
        self.examples = df[image_col].unique()
        self.transforms = transforms
        self.df = df

    def __getitem__(self, index):
        image_id = self.examples[index]
        records = self.df[(self.df['image_id'] == image_id)]
        records = records.reset_index(drop=True)

        file = records.loc[0, "image"]
        image = read_image(file)

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
