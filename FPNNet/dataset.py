import os
import torch
from PIL import Image
from read_csv import csv_to_label_and_bbx


class NBIDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.boxes = csv_to_label_and_bbx(os.path.join(self.root, "annotations.csv"))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        annotations = self.boxes[self.imgs[idx]]
        boxes = annotations['bbx']
        labels = annotations['labels']

        # FloatTensor[N, 4]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Int64Tensor[N]
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((labels.size()[0],), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target, self.imgs[idx]

    def __len__(self):
        return len(self.imgs)

