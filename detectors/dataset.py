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


def to_yolo(boxes, width, height):
    yolo = boxes.copy()
    yolo[..., 0] = yolo[..., 0] / width
    yolo[..., 1] = yolo[..., 1] / height
    yolo[..., 2] = yolo[..., 2] / width
    yolo[..., 3] = yolo[..., 3] / height
    return yolo

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
        width, height = records[["w", "h"]].values
        yolo_boxes = to_yolo(boxes, width, height)

        labels = records["class_id"].values

        if self.transforms:
            image, boxes, labels = self.transforms(image, yolo_boxes, labels)

        return image, {"boxes": boxes, "classes": labels}

    def __len__(self):
        return self.examples.shape[0]
