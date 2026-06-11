"""PyTorch Dataset for segmentation (U-Net / Attention U-Net)."""
import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, file_list=None, transforms=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transforms = transforms

        if file_list is not None:
            self.images = sorted(file_list)
        else:
            self.images = sorted(
                [
                    f.name
                    for f in self.image_dir.iterdir()
                    if f.suffix.lower() in (".jpg", ".jpeg", ".png")
                ]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img = cv2.imread(str(self.image_dir / img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Try common mask naming conventions
        stem = Path(img_name).stem
        for suffix in [f"{stem}_segmentation.png", f"{stem}.png", f"{stem}_mask.png"]:
            mask_path = self.mask_dir / suffix
            if mask_path.exists():
                break

        if not mask_path.exists():
            raise FileNotFoundError(f"No mask found for image {img_name}")

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.transforms:
            transformed = self.transforms(image=img, mask=mask)
            img, mask = transformed["image"], transformed["mask"]
            mask = mask.unsqueeze(0) if mask.dim() == 2 else mask
        else:
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask
