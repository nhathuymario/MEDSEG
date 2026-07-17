# Models

## 1. Faster R-CNN (Detection)

- **Backbone:** ResNet-50 + FPN (pretrained COCO)
- **Task:** Phát hiện bounding box tổn thương da
- **Classes:** 2 (background + lesion)
- **Input:** RGB 512×512
- **Output:** List `{boxes, scores, labels}`
- **Loss:** RPN loss + classification loss + box regression loss
- **Optimizer:** SGD (momentum=0.9, weight_decay=5e-4)
- **Gradient clipping:** max_norm = 5.0

### Checkpoint tốt nhất được chọn theo `val_mAP@0.5`

## 2. U-Net (Segmentation — baseline)

- **Architecture:** Encoder (4 levels) → Bottleneck → Decoder (4 levels)
- **Features:** [64, 128, 256, 512] → Bottleneck: 1024
- **Skip connections:** Concatenation
- **Input:** RGB 256×256
- **Output:** Binary mask (1 channel, sigmoid)

## 3. Attention U-Net (Segmentation — chính)

- **Giống U-Net** nhưng thêm Attention Gate trên skip connections
- **Attention Gate:** Học trọng số encoder features dùng decoder gating signal
- **Ưu điểm:** Lọc bỏ background không liên quan, tập trung vào biên tổn thương
- **Parameters:** F_g (gating channels), F_l (input channels), F_int (intermediate)

### Checkpoint tốt nhất được chọn theo `val_dice`

## Loss functions

| Loss | Công thức | Use case |
|------|-----------|----------|
| DiceLoss | 1 − 2·TP / (pred + target) | Overlap-based |
| DiceBCELoss | BCE + DiceLoss | **Mặc định cho segmentation** |
| FocalLoss | α(1−pt)^γ · BCE | Class imbalance |
| TverskyLoss | 1 − TP / (TP + α·FP + β·FN) | Tunable FP/FN balance |

## Augmentation (Detection)

| Mode | Transforms |
|------|------------|
| `default` | HorizontalFlip, RandomBrightnessContrast |
| `strong` | + VerticalFlip, Rotate90, CLAHE, ColorJitter, GaussNoise |

## Augmentation (Segmentation)

HorizontalFlip, VerticalFlip, Rotate90, RandomBrightnessContrast, ElasticTransform, CLAHE, CoarseDropout.

Chest X-ray chỉ dùng HorizontalFlip + BrightnessContrast + CLAHE (không rotate mạnh vì giải phẫu).
