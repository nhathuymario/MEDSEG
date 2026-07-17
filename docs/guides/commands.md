# Sổ tay lệnh MedSeg

Tất cả lệnh chạy từ `E:\xulianh\MedSeg` với venv đã kích hoạt.

## Môi trường

```powershell
cd E:\xulianh\MedSeg
.\.venv\Scripts\Activate.ps1
```

## Chạy web

```powershell
# Terminal 1 — Backend
uvicorn api.main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend && npm run dev
```

| URL | Mô tả |
|-----|--------|
| http://localhost:5173 | React Frontend |
| http://localhost:8000/docs | Swagger API |
| http://localhost:8000/api/health | Health check |
| http://localhost:8000/api/metrics | Metrics + training history JSON |

## Download dữ liệu

```powershell
python -m src.data.download --dataset isic2018
python -m src.data.download --dataset montgomery
python -m src.data.download --dataset chest_xray   # Montgomery + Shenzhen
```

## Preprocess

```powershell
# ISIC 2018 — resize, binarize mask, tạo COCO annotations
python -c "from pathlib import Path; from src.data.preprocess import resize_and_save, mask_to_binary, generate_coco_annotations; resize_and_save(Path('data/raw/isic2018/ISIC2018_Task1-2_Training_Input'), Path('data/processed/isic2018/images'), size=(256,256), ext='*.jpg'); mask_to_binary(Path('data/raw/isic2018/ISIC2018_Task1_Training_GroundTruth'), Path('data/processed/isic2018/masks'), size=(256,256)); generate_coco_annotations(Path('data/processed/isic2018/images'), Path('data/processed/isic2018/masks'), Path('data/processed/isic2018/annotations/train.json'))"

# Chest X-ray
python -m src.data.preprocess --dataset chest_xray
```

## Split train/val/test

```powershell
python -c "from pathlib import Path; from src.data.split import create_splits; create_splits(Path('data/processed/isic2018/images'), Path('data/splits/isic2018'))"
python -c "from pathlib import Path; from src.data.split import create_splits; create_splits(Path('data/processed/chest_xray/images'), Path('data/splits/chest_xray'))"
```

## Train

| Config | Lệnh |
|--------|------|
| Detection ISIC | `python scripts/train.py --config configs/detection_config.yaml` |
| Detection đa miền | `python scripts/train.py --config configs/detection_multidomain_config.yaml` |
| Segmentation ISIC | `python scripts/train.py --config configs/segmentation_config.yaml` |
| Segmentation CXR | `python scripts/train.py --config configs/chest_xray_segmentation_config.yaml` |
| U-Net baseline | `python scripts/train.py --config configs/segmentation_unet_baseline_config.yaml` |

Override epoch: `python scripts/train.py --config configs/detection_config.yaml --epochs 50`

## Evaluate

Evaluate tạo `*.summary.json` — file này FE đọc để hiển thị chỉ số.

```powershell
# Detection ISIC (390 ảnh test)
python scripts/evaluate.py --model detection --config configs/detection_config.yaml --checkpoint outputs/detection/checkpoints/best_detection.pth --split test --output-csv outputs/detection/metrics/isic2018_detection_test_per_image.csv

# Checkpoint đa miền trên holdout ISIC (390 ảnh)
python scripts/evaluate.py --model detection --config configs/detection_multidomain_isic_eval_config.yaml --checkpoint outputs/detection/checkpoints/best_detection_multidomain.pth --split test --output-csv outputs/detection/metrics/multidomain_on_isic_test_per_image.csv

# Detection clinical iToBoS (dùng checkpoint đa miền)
python scripts/evaluate.py --model detection --config configs/detection_itobos_config.yaml --checkpoint outputs/detection/checkpoints/best_detection_multidomain.pth --split test --output-csv outputs/detection/metrics/itobos_detection_test_per_image.csv

# Segmentation ISIC
python scripts/evaluate.py --model segmentation --config configs/segmentation_config.yaml --checkpoint outputs/segmentation/skin/checkpoints/best_segmentation.pth --split test --output-csv outputs/segmentation/skin/metrics/isic2018_test_per_image.csv

# Segmentation Chest X-ray
python scripts/evaluate.py --model segmentation --config configs/chest_xray_segmentation_config.yaml --checkpoint outputs/segmentation/chest_xray/checkpoints/best_chest_xray_segmentation.pth --split test --output-csv outputs/segmentation/chest_xray/metrics/chest_xray_test_per_image.csv

# Full pipeline Detection → Segmentation
python scripts/evaluate_pipeline.py --split test --output-csv outputs/pipeline/metrics/isic2018_pipeline_test_per_image.csv

# U-Net diagnostic toàn bộ ISIC (2594 ảnh)
python scripts/evaluate.py --model segmentation --config configs/segmentation_unet_baseline_config.yaml --checkpoint outputs/segmentation/skin/checkpoints/best_segmentation_unet_baseline.pth --split all --output-csv outputs/segmentation/skin/metrics/isic2018_unet_baseline_all_2594_per_image.csv
```

Giới hạn số ảnh: thêm `--max-images 50`

## Test

```powershell
python -m pytest tests/ -v                        # Toàn bộ
python -m pytest tests/test_api -v                 # API
python -m pytest tests/test_pipeline -v            # Pipeline
python -m pytest tests/test_models -v              # Models
```

## Xử lý lỗi lock

```powershell
Get-ChildItem outputs/system/*.train.lock                               # Xem lock
Remove-Item outputs/system/detection_multidomain_config.train.lock -Force  # Xóa
```

## File → FE mapping

Xem chi tiết tại [output_layout.md](../reference/output_layout.md).
