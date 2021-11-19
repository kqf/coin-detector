import cv2
import numpy as np
import torch


def from_mask(file):
    bitmap = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    image = np.stack([bitmap, bitmap, bitmap]).astype('float32')
    return image


def read_image(file):
    bgr = cv2.imread(file)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.transpose(2, 1, 0).astype(np.float32)


def to_yolo(boxes):
    width, height = boxes[..., 2], boxes[..., 3]
    yolo = boxes.copy()
    yolo[..., 0] = yolo[..., 0] / width
    yolo[..., 1] = yolo[..., 1] / height
    yolo[..., 2] = yolo[..., 2] / width
    yolo[..., 3] = yolo[..., 3] / height
    return yolo, width, height

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
        yolo_boxes, width, height = to_yolo(boxes)

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

        return image, {"boxes": boxes.astype(np.float32), "classes": labels}

    def __len__(self):
        return self.examples.shape[0]
