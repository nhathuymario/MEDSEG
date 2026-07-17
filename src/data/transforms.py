"""Các phép biến đổi/augmentation ảnh y khoa bằng Albumentations.

Train transforms có augmentation để tăng đa dạng dữ liệu; validation transforms
chỉ resize + normalize để đánh giá ổn định.
"""
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


def _resize_with_padding(size):
    """Keep aspect ratio and pad instead of stretching every image square."""
    return [
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(
            min_height=size,
            min_width=size,
            position="center",
            border_mode=cv2.BORDER_CONSTANT,
            fill=(123, 116, 103),
            fill_mask=0,
        ),
    ]


def get_train_transforms(size=256):
    """Augmentation tổng quát cho segmentation dataset."""
    return A.Compose([
        *_resize_with_padding(size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.ElasticTransform(alpha=120, sigma=6, p=0.2),
        A.CLAHE(clip_limit=2.0, p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            p=0.2,
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(size=256):
    """Transform tối thiểu cho validation/test segmentation."""
    return A.Compose([
        *_resize_with_padding(size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_chest_xray_train_transforms(size=256):
    """Augmentation nhẹ cho X-quang ngực thẳng.

    Không dùng vertical flip/rotate mạnh vì có thể tạo ảnh giải phẫu không thực tế.
    """
    return A.Compose([
        *_resize_with_padding(size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.CLAHE(clip_limit=2.0, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_detection_transforms(size=512, train=True, augmentation="default"):
    """Transform cho Faster R-CNN, có khai báo bbox_params để sửa tọa độ bbox.

    Parameters
    ----------
    augmentation : str
        "default" — HorizontalFlip + RandomBrightnessContrast (nhẹ).
        "strong"  — Thêm VerticalFlip, Rotate, CLAHE, ColorJitter, GaussNoise
                     để chống overfitting khi train trên nhiều miền.
    """
    bbox = A.BboxParams(format="pascal_voc", label_fields=["labels"])
    if not train:
        return A.Compose([
            *_resize_with_padding(size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=bbox)

    if augmentation == "strong":
        return A.Compose([
            *_resize_with_padding(size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.15, hue=0.05, p=0.3),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=bbox)

    return A.Compose([
        *_resize_with_padding(size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=bbox)

