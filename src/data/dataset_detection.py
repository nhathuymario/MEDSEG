"""Dataset PyTorch cho bài toán phát hiện đối tượng bằng Faster R-CNN.

Đọc ảnh và annotation theo định dạng COCO, sau đó trả về:
- image tensor dạng [C, H, W]
- target dict gồm boxes, labels, image_id đúng format torchvision detection.
"""
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class DetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_file, file_list=None, transforms=None):
        self.image_dir = Path(image_dir)
        self.transforms = transforms

        # COCO JSON thường có 2 danh sách chính: images và annotations.
        # Ta chuyển images thành dict theo id để truy xuất nhanh trong __getitem__.
        with open(annotation_file) as f:
            coco = json.load(f)

        allowed_files = set(file_list) if file_list is not None else None
        self.images = {
            img["id"]: img
            for img in coco["images"]
            if allowed_files is None or img["file_name"] in allowed_files
        }
        self.anns = {}
        # Gom tất cả annotation theo image_id để mỗi ảnh lấy được nhiều bbox.
        for ann in coco["annotations"]:
            if ann["image_id"] not in self.images:
                continue
            self.anns.setdefault(ann["image_id"], []).append(ann)
        self.img_ids = list(self.images.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        img = cv2.imread(str(self.image_dir / img_info["file_name"]))
        if img is None:
            raise FileNotFoundError(
                f"Detection image not found: {self.image_dir / img_info['file_name']}"
            )
        # OpenCV đọc ảnh theo BGR, còn Albumentations/model thường dùng RGB.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_height, image_width = img.shape[:2]

        anns = self.anns.get(img_id, [])
        boxes = []
        for a in anns:
            x, y, w, h = a["bbox"]
            # COCO lưu bbox dạng [x, y, width, height].
            # Faster R-CNN của torchvision cần [x1, y1, x2, y2].
            x1 = min(max(float(x), 0.0), float(image_width))
            y1 = min(max(float(y), 0.0), float(image_height))
            x2 = min(max(float(x + w), 0.0), float(image_width))
            y2 = min(max(float(y + h), 0.0), float(image_height))
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])

        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        # Dự án hiện dùng 1 lớp foreground, nên label 1 là object cần phát hiện.
        # Label 0 được Faster R-CNN dành cho background.
        labels = np.ones(len(boxes), dtype=np.int64)

        if self.transforms:
            # Albumentations cần nhận kèm bboxes và labels để resize/flip đồng bộ.
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
