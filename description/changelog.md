# Changelog

## 2026-06-11

### User's Requirement

- Hoàn thiện cả nhánh ISIC 2018 và Chest X-ray cho đề tài.
- Tự động tải, xử lý và train Chest X-ray.
- Cập nhật tài liệu theo thứ tự requirement -> features -> solutions -> logic -> implement -> test.

### Features

- Download Montgomery/Shenzhen tự động, có resume và retry.
- Preprocess Chest X-ray và ghép mask hai lá phổi.
- Dataset split được sử dụng thật trong training.
- Training/evaluation theo YAML config.
- Checkpoint riêng cho từng dataset.

### Tech Solutions

- NLM recursive directory downloader.
- OpenCV CLAHE + nearest-neighbor mask resize.
- Attention U-Net nhẹ cho GPU 4 GB.
- CUDA mixed precision.
- Evaluation Dice, IoU, sensitivity, specificity, pixel accuracy.

### Logic + AI

- Montgomery dùng cho lung-field segmentation.
- ISIC dùng cho lesion detection và lesion segmentation.
- Validation quyết định best checkpoint; test split chỉ dùng đánh giá cuối.

### Implement

- Cập nhật:
  - `src/data/download.py`
  - `src/data/preprocess.py`
  - `src/data/transforms.py`
  - `src/data/dataset_segmentation.py`
  - `src/models/unet.py`
  - `src/training/train_segmentation.py`
  - `scripts/train.py`
  - `scripts/evaluate.py`
  - `configs/chest_xray_segmentation_config.yaml`
- Tạo processed data và split Chest X-ray.
- Train checkpoint `best_chest_xray_segmentation.pth`.

### Test

- 11 tests pass.
- Chest X-ray test Dice 0.9811, IoU 0.9628.
- CUDA được nhận trên RTX 3050 Laptop GPU.

## 2026-06-07

### User's Requirement

- Khởi tạo MedSeg cho detection và segmentation ảnh y tế.

### Features

- Data pipeline, models, API, frontend và documentation.

### Tech Solutions

- PyTorch, FastAPI, React, pytest.

### Logic + AI

- Faster R-CNN detection -> ROI -> U-Net/Attention U-Net segmentation.

### Implement

- Tạo cấu trúc project và các module nền tảng.

### Test

- Tạo model-level, data-level, API-level và pipeline-level tests.
