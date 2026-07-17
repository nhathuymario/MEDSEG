# API Reference

Base URL: `http://localhost:8000/api`

## Health Check

`GET /health`

```json
{ "status": "ok", "gpu_available": true, "models_loaded": ["detector", "segmentor"] }
```

## Detection

`POST /detect?threshold=0.5`

- **Input:** FormData `file` (JPG/PNG)
- **Query:** `threshold` (0.05–1.0, default 0.5) — ngưỡng confidence
- **Response:**
```json
{
  "boxes": [[x1, y1, x2, y2]],
  "scores": [0.95],
  "labels": [1],
  "inference_time_ms": 120.5
}
```

## Segmentation

`POST /segment`

- **Input:** FormData `file`
- **Response:**
```json
{
  "mask_base64": "iVBOR...",
  "overlay_base64": "iVBOR...",
  "dice_score": null,
  "inference_time_ms": 150.2
}
```

## Full Pipeline

`POST /pipeline?threshold=0.5`

- **Input:** FormData `file`
- **Query:** `threshold` (0.05–1.0) — ngưỡng detection
- **Response:**
```json
{
  "detection": { "boxes": [], "scores": [], "labels": [], "inference_time_ms": 0 },
  "segmentation_masks": ["iVBOR..."],
  "combined_overlay_base64": "iVBOR...",
  "total_time_ms": 300.5
}
```

## Metrics

`GET /metrics`

Trả về toàn bộ chỉ số đánh giá và training history. FE gọi endpoint này.

```json
{
  "summaries": {
    "isic2018_detection": { "map": 0.85, "precision": 0.90, ... },
    "multidomain_isic_detection": { ... },
    "itobos_detection": { ... },
    "isic2018_segmentation": { ... },
    "chest_xray_segmentation": { ... },
    "isic2018_pipeline": { ... },
    "isic2018_unet_baseline": { ... },
    "isic2018_unet_all_diagnostic": { ... }
  },
  "training_history": {
    "detection": { "epochs": 80, "rows": [...] },
    "detection_multidomain": { "epochs": 7, "rows": [...] },
    "isic2018_segmentation": { "epochs": 100, "rows": [...] },
    "chest_xray_segmentation": null
  },
  "training_jobs": {
    "detection_multidomain": { "status": "ready", "progress": "..." }
  }
}
```

## Evaluation Status

`GET /metrics/evaluation-status`

Trả trạng thái đánh giá đang chạy (nếu có).

## Run Evaluation

`POST /metrics/evaluate/{kind}?limit=50`

- **kind:** `detection`, `multidomain-isic-detection`, `itobos-detection`, `isic-segmentation`, `chest-xray-segmentation`, `isic-pipeline`, `isic-unet-all`
- **limit:** Số ảnh tối đa
- Chạy evaluate ở background, trả 202 ngay
