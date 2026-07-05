# Kết quả Đánh giá (Evaluation Results)

*Lưu ý: File này sẽ được cập nhật liên tục sau các đợt training và evaluation.*

## Chest X-ray Lung Segmentation

- Dataset: Montgomery County CXR Set
- Model: Attention U-Net
- Input size: 256x256
- Split: 96 train / 20 validation / 22 test
- Training: 100 epochs, Adam, Dice + BCE, mixed precision
- Best validation Dice: 0.9771
- Checkpoint: `outputs/checkpoints/best_chest_xray_segmentation.pth`

### Test Results

| Metric | Score |
|---|---:|
| Dice | 0.9811 |
| IoU | 0.9628 |
| Sensitivity | 0.9804 |
| Specificity | 0.9932 |
| Pixel accuracy | 0.9897 |

These results measure lung-field segmentation, not lesion localization.

## 1. Detection (Faster R-CNN) - ISIC 2018
| Metric | Threshold | Value | Target | Status |
|--------|-----------|-------|--------|--------|
| mAP    | 0.5       | -     | >= 0.60| Pending |
| mAP    | 0.5:0.95  | -     | -      | Pending |
| FPS (Inference) | - | - | - | Pending |

## 2. Segmentation - ISIC 2018
| Model | Dice Score | IoU (Jaccard) | Sensitivity | Specificity | Target Dice | Status |
|-------|------------|---------------|-------------|-------------|-------------|--------|
| U-Net | - | - | - | - | >= 0.85 | Pending |
| Attention U-Net | - | - | - | - | >= 0.85 | Pending |

## 3. Lung Segmentation - Chest X-ray
| Model | Dice Score | IoU (Jaccard) | Sensitivity | Specificity | Target Dice | Status |
|-------|------------|---------------|-------------|-------------|-------------|--------|
| U-Net | - | - | - | - | >= 0.90 | Pending |

## 4. Full Pipeline Performance
- **Thời gian xử lý trung bình (End-to-End)**: Đang cập nhật.
- **Tốc độ trên CPU**: Đang cập nhật.
- **Tốc độ trên GPU**: Đang cập nhật.
