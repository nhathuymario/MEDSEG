# MedSeg - Sổ tay lệnh môi trường, train và evaluate

File này gom các lệnh hay dùng khi chạy project MedSeg trên PowerShell.

## 1. Vào project và kích hoạt môi trường

```powershell
cd E:\xulianh\MedSeg
.\.venv\Scripts\Activate.ps1
```

Kiểm tra Python đang dùng:

```powershell
python -c "import sys; print(sys.executable)"
```

Nếu muốn chạy trực tiếp bằng Python trong `.venv` mà không cần activate:

```powershell
.\.venv\Scripts\python.exe --version
```

## 2. Chạy backend

```powershell
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Hoặc:

```powershell
uvicorn api.main:app --reload --port 8000
```

Sau khi backend chạy, truy cập:

- Swagger API: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/api/health

Kiểm tra backend bằng PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

## 3. Chạy frontend

```powershell
cd E:\xulianh\MedSeg\frontend
npm install
npm run dev
```

Frontend thường chạy tại:

```text
http://localhost:5173
```

Build kiểm tra frontend:

```powershell
cd E:\xulianh\MedSeg\frontend
npm run build
```

## 4. Luồng xử lý dữ liệu

```text
Ảnh/mask gốc
-> data/raw/

Tiền xử lý
-> data/processed/

Chia tập train/val/test
-> data/splits/*.csv

Train model
-> outputs/checkpoints/*.pth

Evaluate model
-> outputs/metrics/*.csv

Backend load checkpoint
-> inference qua API
```

## 5. Download dữ liệu

Tải ISIC 2018:

```powershell
python -m src.data.download --dataset isic2018
```

Tải Montgomery Chest X-ray:

```powershell
python -m src.data.download --dataset montgomery
```

Tải bộ chest X-ray theo cấu hình chung:

```powershell
python -m src.data.download --dataset chest_xray
```

Theo dõi lỗi download Chest X-ray nếu có:

```powershell
Get-Content data/raw/chest_xray/download-error.log -Wait
```

## 6. Preprocess dữ liệu

Preprocess ISIC 2018:

```powershell
python -c "from pathlib import Path; from src.data.preprocess import resize_and_save, mask_to_binary, generate_coco_annotations; resize_and_save(Path('data/raw/isic2018/ISIC2018_Task1-2_Training_Input'), Path('data/processed/isic2018/images'), size=(256,256), ext='*.jpg'); mask_to_binary(Path('data/raw/isic2018/ISIC2018_Task1_Training_GroundTruth'), Path('data/processed/isic2018/masks'), size=(256,256)); generate_coco_annotations(Path('data/processed/isic2018/images'), Path('data/processed/isic2018/masks'), Path('data/processed/isic2018/annotations/train.json'))"
```

Preprocess Chest X-ray Montgomery:

```powershell
python -m src.data.preprocess --dataset chest_xray
```

## 7. Tạo train/val/test split

Tạo split cho ISIC 2018:

```powershell
python -c "from pathlib import Path; from src.data.split import create_splits; create_splits(Path('data/processed/isic2018/images'), Path('data/splits/isic2018'))"
```

Tạo split cho Chest X-ray:

```powershell
python -c "from pathlib import Path; from src.data.split import create_splits; create_splits(Path('data/processed/chest_xray/images'), Path('data/splits/chest_xray'))"
```

## 8. Train model

Train Chest X-ray segmentation:

```powershell
.\.venv\Scripts\python.exe scripts\train.py `
  --config configs\chest_xray_segmentation_config.yaml
```

Train ISIC segmentation:

```powershell
.\.venv\Scripts\python.exe scripts\train.py `
  --config configs\segmentation_config.yaml
```

Train ISIC detection:

```powershell
.\.venv\Scripts\python.exe scripts\train.py `
  --config configs\detection_config.yaml
```

Train nhanh 1 epoch để kiểm tra code:

```powershell
.\.venv\Scripts\python.exe scripts\train.py `
  --config configs\segmentation_config.yaml `
  --epochs 1
```

Sau khi train, checkpoint nằm tại:

```text
outputs/checkpoints/best_chest_xray_segmentation.pth
outputs/checkpoints/best_segmentation.pth
outputs/checkpoints/best_detection.pth
```

Training history nằm tại:

```text
outputs/logs/best_chest_xray_segmentation_training_history.csv
outputs/logs/best_segmentation_training_history.csv
outputs/logs/best_detection_training_history.csv
```

## 9. Evaluate model

Evaluate Chest X-ray segmentation:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py `
  --model segmentation `
  --config configs\chest_xray_segmentation_config.yaml `
  --checkpoint outputs\checkpoints\best_chest_xray_segmentation.pth `
  --split test `
  --output-csv outputs\metrics\chest_xray_test_per_image.csv
```

Evaluate ISIC segmentation:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py `
  --model segmentation `
  --config configs\segmentation_config.yaml `
  --checkpoint outputs\checkpoints\best_segmentation.pth `
  --split test `
  --output-csv outputs\metrics\isic2018_test_per_image.csv
```

Evaluate ISIC detection:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py `
  --model detection `
  --config configs\detection_config.yaml `
  --checkpoint outputs\checkpoints\best_detection.pth `
  --split test `
  --output-csv outputs\metrics\isic2018_detection_test_per_image.csv
```

File metric sau evaluate:

```text
outputs/metrics/chest_xray_test_per_image.csv
outputs/metrics/isic2018_test_per_image.csv
outputs/metrics/isic2018_detection_test_per_image.csv
```

## 10. Test code

Chạy toàn bộ test:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\ -v
```

Chạy test full pipeline:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_pipeline\test_full_pipeline.py -v
```

## 11. Cách đọc loss train/val

- `train_loss` giảm và `val_loss` cũng giảm: model đang học tốt.
- `train_loss` giảm nhưng `val_loss` tăng: model bị overfit.
- `train_loss` cao và `val_loss` cao: model underfit, train chưa đủ hoặc cấu hình chưa phù hợp.
- Với segmentation, nên đọc thêm `train_dice` và `val_dice`; Dice tăng dần là tín hiệu tốt.
- Với detection, `val_loss` không đơn giản vì Faster R-CNN của torchvision trả loss trong train mode. Vì vậy detection nên đọc `train_loss` khi train và dùng `mAP`, `Precision`, `Recall` trên val/test để đánh giá thật.

## 12. Các ngưỡng đánh giá đang dùng

Segmentation:

```text
threshold = 0.5
```

Detection:

```text
confidence_threshold = 0.5
iou_threshold = 0.5
```

Các ngưỡng này nằm trong:

```text
configs/segmentation_config.yaml
configs/chest_xray_segmentation_config.yaml
configs/detection_config.yaml
```
