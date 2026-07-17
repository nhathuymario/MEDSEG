# Project Walkthrough

## Cấu trúc dự án

```
MedSeg/
├── src/           # ML: data, models, training, evaluation, pipeline
├── api/           # FastAPI backend: routers, schemas, services
├── frontend/      # React frontend: Vite, pages, components
├── tests/         # Unit & integration tests
├── configs/       # YAML config cho từng model/dataset
├── scripts/       # CLI: train.py, evaluate.py, evaluate_pipeline.py
├── outputs/       # Checkpoints, logs, metrics (xem output_layout.md)
├── data/          # raw → processed → splits
├── docs/          # Tài liệu
├── notebooks/     # Jupyter notebooks phân tích
└── description/   # Mô tả dự án
```

## Module ML (`src/`)

| Package | Files | Vai trò |
|---------|-------|---------|
| `src/data/` | download, preprocess, dataset_detection, dataset_segmentation, transforms, split | Pipeline dữ liệu end-to-end |
| `src/models/` | faster_rcnn, unet, attention_unet, components/ | Kiến trúc model |
| `src/training/` | train_detection, train_segmentation | Vòng lặp huấn luyện |
| `src/evaluation/` | metrics_detection, metrics_segmentation | Tính mAP, Dice, IoU |
| `src/pipeline/` | inference | Pipeline inference: detect → crop → segment → overlay |

## API Backend (`api/`)

| File | Endpoint | Chức năng |
|------|----------|-----------|
| `routers/health.py` | `GET /health` | Health check + model status |
| `routers/detection.py` | `POST /detect` | Phát hiện tổn thương |
| `routers/segmentation.py` | `POST /segment` | Phân đoạn |
| `routers/pipeline.py` | `POST /pipeline` | Full pipeline detect → segment |
| `routers/metrics.py` | `GET /metrics` | Trả chỉ số + training history cho FE |
| `services/model_service.py` | — | Singleton loader cho detector + segmentor |

## Frontend (`frontend/src/`)

| File | Chức năng |
|------|-----------|
| `pages/Dashboard.jsx` | Tổng quan thống kê |
| `pages/Analyze.jsx` | Upload ảnh + chọn model + xem kết quả + slider threshold |
| `pages/Compare.jsx` | So sánh side-by-side giữa các model |
| `pages/Metrics.jsx` | Chỉ số đánh giá: training history, detection, segmentation, pipeline |
| `pages/History.jsx` | Lịch sử phân tích |
| `api/client.js` | API client (fetch wrapper) |
| `hooks/useAnalysis.js` | Custom hook xử lý upload + inference |

## Config files (`configs/`)

| File | Model | Dataset |
|------|-------|---------|
| `detection_config.yaml` | Faster R-CNN | ISIC 2018 |
| `detection_multidomain_config.yaml` | Faster R-CNN | ISIC + iToBoS |
| `detection_itobos_config.yaml` | Faster R-CNN | iToBoS (eval only) |
| `segmentation_config.yaml` | Attention U-Net | ISIC 2018 |
| `chest_xray_segmentation_config.yaml` | Attention U-Net | Chest X-ray |
| `segmentation_unet_baseline_config.yaml` | U-Net | ISIC 2018 |

## Tests

| Test | Mô tả |
|------|--------|
| `tests/test_models/` | Forward pass shape cho Faster R-CNN, U-Net, Attention U-Net |
| `tests/test_pipeline/` | Full pipeline detect → segment end-to-end |
| `tests/test_api/` | API endpoint tests |

## Luồng hoàn chỉnh

```
Download → Preprocess → Split → Train → Evaluate → Chạy BE+FE → Xem kết quả
```

Chi tiết lệnh: [commands.md](commands.md)
