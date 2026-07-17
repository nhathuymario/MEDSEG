# Kiến trúc hệ thống

## 3 lớp

```
┌─────────────────────────────────────────────┐
│           React Frontend (Vite :5173)        │
│  Dashboard │ Analyze │ Compare │ Metrics     │
├─────────────────────────────────────────────┤
│           FastAPI Backend (:8000)             │
│  /detect │ /segment │ /pipeline │ /metrics   │
├─────────────────────────────────────────────┤
│           PyTorch ML Models                  │
│  Faster R-CNN │ U-Net │ Attention U-Net      │
└─────────────────────────────────────────────┘
```

## Luồng inference

```
Ảnh y tế → Resize + Normalize
         → Faster R-CNN → Bounding boxes (ROI)
                          → Crop từng ROI
                          → Attention U-Net → Mask phân đoạn
                          → Map mask về tọa độ gốc
                          → Overlay lên ảnh gốc → Base64 → FE hiển thị
```

## Luồng data (train)

```
Download        → data/raw/
Preprocess      → data/processed/ (resize, binarize, COCO JSON)
Split           → data/splits/*.csv (train 70% / val 15% / test 15%)
Train           → outputs/**/checkpoints/*.pth
                → outputs/**/logs/*_training_history.csv
Evaluate        → outputs/**/metrics/*.summary.json
```

## Luồng FE hiển thị chỉ số

```
FE gọi GET /api/metrics
  → Backend đọc *_training_history.csv  → JSON training_history  → Tab "Training"
  → Backend đọc *.summary.json         → JSON summaries         → Tab Detection/Segmentation/Pipeline
```

## Module chính

| Thư mục | Vai trò |
|---------|---------|
| `src/data/` | Download, preprocess, dataset, transforms, split |
| `src/models/` | Faster R-CNN, U-Net, Attention U-Net, components |
| `src/training/` | Vòng lặp train detection + segmentation |
| `src/evaluation/` | Metrics: mAP, Dice, IoU, Sensitivity |
| `src/pipeline/` | Inference pipeline (detect → crop → segment → overlay) |
| `api/routers/` | FastAPI endpoints: health, detect, segment, pipeline, metrics |
| `api/services/` | Singleton model loader |
| `frontend/src/pages/` | Dashboard, Analyze, Compare, Metrics, History |
| `scripts/` | CLI: train.py, evaluate.py, evaluate_pipeline.py |
| `configs/` | YAML config cho từng model/dataset |
