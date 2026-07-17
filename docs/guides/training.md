# Hướng dẫn Training

## Yêu cầu

- NVIDIA GPU có CUDA (khuyến nghị VRAM ≥ 8 GB cho U-Net, ≥ 12 GB cho Faster R-CNN)
- Đã cài `requirements.txt` và kích hoạt `.venv`
- Đã download + preprocess data (xem [commands.md](commands.md))

## Tổng quan config

| Config | Model | Dataset | Epochs | LR | Scheduler |
|--------|-------|---------|--------|----|-----------|
| `detection_config.yaml` | Faster R-CNN | ISIC 2018 | 80 | 1e-3 | StepLR(20) |
| `detection_multidomain_config.yaml` | Faster R-CNN | ISIC + iToBoS | 30 | 5e-5 | CosineWarm(T₀=10) |
| `segmentation_config.yaml` | Attention U-Net | ISIC 2018 | 100 | 1e-4 | Cosine |
| `chest_xray_segmentation_config.yaml` | Attention U-Net | Chest X-ray | 100 | 1e-4 | Cosine |
| `segmentation_unet_baseline_config.yaml` | U-Net | ISIC 2018 | 100 | 1e-4 | Cosine |

## Thứ tự train khuyến nghị

```
1. Detection ISIC          → best_detection.pth
2. Detection đa miền       → best_detection_multidomain.pth (dùng #1 làm initial checkpoint)
3. Segmentation ISIC       → best_segmentation.pth
4. Segmentation Chest X-ray → best_chest_xray_segmentation.pth
5. U-Net baseline (tùy chọn, để so sánh với Attention U-Net)
```

Detection đa miền **bắt buộc** phải train Detection ISIC trước vì dùng `initial_checkpoint`.

## Đầu ra sau train

| Loại | Đường dẫn | FE đọc |
|------|-----------|--------|
| Checkpoint | `outputs/<func>/checkpoints/*.pth` | Không |
| Training history | `outputs/<func>/logs/*_training_history.csv` | Tab **Training** |

### Cột CSV theo model

- **Detection:** `epoch, train_loss, val_map, lr`
- **Segmentation:** `epoch, train_loss, train_dice, val_loss, val_dice, lr`

## Cách đọc loss — chẩn đoán overfitting

| train_loss | val metric | Chẩn đoán |
|------------|-----------|-----------|
| ↓ giảm | ↑ tăng (val_dice/val_map) | ✅ Đang học tốt |
| ↓ giảm | ↓ giảm hoặc đứng yên | ⚠️ **Overfitting** |
| Cao | Cao | ❌ Underfit — chưa train đủ |

**Detection:** Faster R-CNN không trả val_loss. Dùng `val_map` (mAP@0.5) để đánh giá.

**Segmentation:** So sánh `val_dice` với `train_dice`. Nếu khoảng cách ngày càng lớn → overfitting.

## Chống overfitting

### Augmentation

Detection có 2 chế độ augmentation (config `augmentation: default | strong`):

| Chế độ | Các phép biến đổi |
|--------|-------------------|
| `default` | HorizontalFlip, RandomBrightnessContrast |
| `strong` | + VerticalFlip, Rotate90, CLAHE, ColorJitter, GaussNoise |

Segmentation augmentation đã mạnh sẵn: HFlip, VFlip, Rotate90, BrightnessContrast, ElasticTransform, CLAHE, CoarseDropout.

### LR Scheduler

| Tên | Cách hoạt động |
|-----|----------------|
| `step` | Giảm LR × 0.1 mỗi `step_size` epoch |
| `cosine` | Giảm theo đường cosine từ `lr` về 0 |
| `cosine_warm` | CosineAnnealingWarmRestarts: restart mỗi `T_0` epoch |

### Early stopping

Config `early_stopping_patience: N` — dừng train khi val metric không cải thiện sau N epoch liên tiếp. Chỉ hỗ trợ detection hiện tại.

### Warmup

Config `warmup_epochs: N` — tăng LR tuyến tính từ 0 đến `lr` trong N epoch đầu. Giúp ổn định khi fine-tune từ checkpoint khác.

## Ngưỡng đánh giá

| Model | Tham số | Giá trị |
|-------|---------|---------|
| Detection ISIC | `confidence_threshold` | 0.8 |
| Detection đa miền | `confidence_threshold` | 0.8 |
| Detection iToBoS | `confidence_threshold` | 0.8 |
| Segmentation ISIC | `threshold` | 0.8 |
| Segmentation CXR | `threshold` | 0.5 |
| IoU matching (detection) | `iou_threshold` | 0.5 |

Ngưỡng khai báo trong từng file `configs/*.yaml`.

## Troubleshooting train

| Lỗi | Nguyên nhân | Khắc phục |
|-----|-------------|-----------|
| CUDA Out of Memory | `batch_size` hoặc `image_size` quá lớn | Giảm trong config YAML |
| Loss = NaN | LR quá cao | Giảm `lr`, gradient clipping đã có sẵn (max_norm=5) |
| Lock file kẹt | Train cũ crash | `Remove-Item outputs/system/*.train.lock` |
| Val metric tăng rồi giảm | Overfitting | Bật `augmentation: strong`, giảm `lr`, thêm `early_stopping_patience` |
