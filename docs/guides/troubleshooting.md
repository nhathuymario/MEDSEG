# Khắc phục sự cố (Troubleshooting)

## 1. CORS — Frontend bị chặn khi gọi API

**Hiện tượng:** Trình duyệt báo `CORS policy: No 'Access-Control-Allow-Origin'`.

**Nguyên nhân:** Vite đổi port (5174, 5175...) khi 5173 bị chiếm. Backend chỉ cho phép port 5173.

**Khắc phục:** Cập nhật `allow_origins` trong `api/main.py`:
```python
allow_origins=[
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5173",
]
```

## 2. Confidence score cực thấp khi inference

**Hiện tượng:** Detection trả confidence 6–35% trên ảnh clinical rõ ràng. Threshold 0.8 → không có box nào.

**Nguyên nhân:** Mismatch preprocessing giữa train và inference.
- **Train:** Albumentations `A.Normalize(ImageNet)` + Faster R-CNN tự normalize → **double-normalize**
- **Inference cũ:** Chỉ `T.ToTensor()` → single-normalize → phân phối khác hẳn lúc train

**Khắc phục:** Trong `src/pipeline/inference.py`, dùng `TRANSFORM` có `Normalize` giống lúc train:
```python
TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
img_tensor = TRANSFORM(img).to(device)
```
Confidence quay về 95–98%.

## 3. Threshold không chỉnh được từ giao diện

**Hiện tượng:** Threshold cứng 0.8 ở backend, không điều chỉnh được.

**Khắc phục:**
1. Backend: thêm query param `threshold` vào `/detect` và `/pipeline`
2. Frontend `api/client.js`: truyền `?threshold=...` trong URL
3. Frontend `Analyze.jsx`: thêm slider cho người dùng kéo chọn

## 4. Lock file kẹt — không train được

**Hiện tượng:** `RuntimeError: Training is already running for ... (PID ...)`.

**Nguyên nhân:** Train cũ crash/kill nhưng lock file còn sót. Trên Windows, PID có thể bị tái sử dụng.

**Khắc phục:**
```powershell
Get-ChildItem outputs/system/*.train.lock
Remove-Item outputs/system/detection_multidomain_config.train.lock -Force
```

## 5. Training history trống trên FE

**Hiện tượng:** Tab Training hiển thị bảng nhưng không có dòng nào (0 epoch).

**Nguyên nhân:** File CSV ở đường dẫn chính chỉ có header (train bị crash sớm). File CSV ở đường dẫn legacy có data thật nhưng bị bỏ qua do `_first_existing` chọn file đầu tiên tồn tại.

**Khắc phục:** `_read_history` trong `api/routers/metrics.py` đã được sửa để ưu tiên file **có data rows** thay vì file tồn tại đầu tiên.

## 6. CUDA Out of Memory

**Khắc phục:** Giảm `batch_size` hoặc `image_size` trong config YAML. Detection multidomain mặc định `batch_size: 1` vì ảnh 512×512 tốn VRAM.

## 7. Val metric giảm sau vài epoch (Overfitting)

**Hiện tượng:** `train_loss` giảm đều nhưng `val_map` hoặc `val_dice` đạt đỉnh rồi giảm.

**Khắc phục:**
1. Bật `augmentation: strong` trong config detection
2. Giảm `lr` (vd: 1e-4 → 5e-5)
3. Đổi scheduler: `cosine_warm` thay vì `step`
4. Thêm `early_stopping_patience: 7`
5. Thêm `warmup_epochs: 2`

Xem thêm chi tiết tại [training.md](training.md).
