"""PyTorch Dataset for object detection (Faster R-CNN)."""
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class DetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transforms=None):
        self.image_dir = Path(image_dir)
        self.transforms = transforms

        with open(annotation_file) as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}
        self.anns = {}
        for ann in coco["annotations"]:
            self.anns.setdefault(ann["image_id"], []).append(ann)
        self.img_ids = list(self.images.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        img = cv2.imread(str(self.image_dir / img_info["file_name"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        anns = self.anns.get(img_id, [])
        boxes = []
        for a in anns:
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])

        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.ones(len(boxes), dtype=np.int64)

        if self.transforms:
            transformed = self.transforms(image=img, bboxes=boxes.tolist(), labels=labels.tolist())
            img = transformed["image"]
            boxes = np.array(transformed["bboxes"], dtype=np.float32) if transformed["bboxes"] else np.zeros((0, 4), dtype=np.float32)
            labels = np.array(transformed["labels"], dtype=np.int64)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
        }
        return img if torch.is_tensor(img) else torch.from_numpy(img).permute(2, 0, 1).float() / 255.0, target
