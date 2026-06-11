# API Reference

FastAPI backend cung cấp các REST API endpoints để giao tiếp với frontend.
Base URL (khi chạy local): `http://localhost:8000/api`

## 1. Health Check
`GET /health`
- **Mục đích**: Kiểm tra trạng thái của server và các model đã load.
- **Response**:
```json
{
  "status": "ok",
  "gpu_available": true,
  "models_loaded": ["detector", "segmentor"]
}
```

## 2. Detection Endpoint
`POST /detect`
- **Mục đích**: Nhận ảnh và trả về bounding boxes của vùng tổn thương.
- **Input**: FormData `file` (ảnh JPG/PNG/DICOM).
- **Response**:
```json
{
  "boxes": [[x1, y1, x2, y2]],
  "scores": [0.95],
  "labels": [1],
  "inference_time_ms": 120.5
}
```

## 3. Segmentation Endpoint
`POST /segment`
- **Mục đích**: Nhận ảnh và phân vùng tổn thương.
- **Input**: FormData `file`.
- **Response**:
```json
{
  "mask_base64": "iVBOR...",
  "overlay_base64": "iVBOR...",
  "dice_score": null,
  "inference_time_ms": 150.2
}
```

## 4. Full Pipeline Endpoint
`POST /pipeline`
- **Mục đích**: Chạy cả quy trình: Phát hiện -> Cắt ROI -> Phân vùng.
- **Input**: FormData `file`.
- **Response**:
```json
{
  "detection": {
    "boxes": [...],
    "scores": [...],
    "labels": [...],
    "inference_time_ms": 0
  },
  "segmentation_masks": ["iVBOR..."],
  "combined_overlay_base64": "iVBOR...",
  "total_time_ms": 300.5
}
```
