# Cấu trúc outputs

## Cây thư mục

```
outputs/
├── detection/
│   ├── checkpoints/
│   │   ├── best_detection.pth
│   │   └── best_detection_multidomain.pth
│   ├── logs/
│   │   ├── best_detection_training_history.csv
│   │   └── best_detection_multidomain_training_history.csv
│   └── metrics/
│       ├── isic2018_detection_test_per_image.csv
│       ├── isic2018_detection_test_per_image.summary.json
│       ├── multidomain_on_isic_test_per_image.csv
│       ├── multidomain_on_isic_test_per_image.summary.json
│       ├── itobos_detection_test_per_image.csv
│       └── itobos_detection_test_per_image.summary.json
├── segmentation/
│   ├── skin/
│   │   ├── checkpoints/
│   │   │   ├── best_segmentation.pth
│   │   │   └── best_segmentation_unet_baseline.pth
│   │   ├── logs/
│   │   │   ├── best_segmentation_training_history.csv
│   │   │   └── best_segmentation_unet_baseline_training_history.csv
│   │   └── metrics/
│   │       ├── isic2018_test_per_image.summary.json
│   │       ├── isic2018_unet_baseline_test_per_image.summary.json
│   │       └── isic2018_unet_baseline_all_2594_per_image.summary.json
│   └── chest_xray/
│       ├── checkpoints/best_chest_xray_segmentation.pth
│       ├── logs/best_chest_xray_segmentation_training_history.csv
│       └── metrics/chest_xray_test_per_image.summary.json
├── pipeline/
│   └── metrics/isic2018_pipeline_test_per_image.summary.json
├── checkpoints/                  ← legacy (fallback đa miền)
│   └── best_detection_multidomain.pth
├── logs/                         ← legacy (fallback đa miền)
│   └── best_detection_multidomain_training_history.csv
└── system/
    └── *.train.lock              ← lock file chống chạy trùng
```

## Mapping: File → FE tab

### Tab "Training" — đọc từ `*_training_history.csv`

| Bảng FE | File CSV |
|---------|----------|
| Detection đa miền Clinical + ISIC | `detection/logs/best_detection_multidomain_training_history.csv` |
| Detection ISIC | `detection/logs/best_detection_training_history.csv` |
| Phân đoạn ISIC | `segmentation/skin/logs/best_segmentation_training_history.csv` |
| Phân đoạn X-ray | `segmentation/chest_xray/logs/best_chest_xray_segmentation_training_history.csv` |

### Tab Detection / Segmentation / Pipeline — đọc từ `*.summary.json`

| Bảng FE | File JSON |
|---------|-----------|
| ISIC checkpoint → ISIC test | `detection/metrics/isic2018_detection_test_per_image.summary.json` |
| Multidomain checkpoint → ISIC test | `detection/metrics/multidomain_on_isic_test_per_image.summary.json` |
| Multidomain checkpoint → iToBoS test | `detection/metrics/itobos_detection_test_per_image.summary.json` |
| Phân đoạn ISIC | `segmentation/skin/metrics/isic2018_test_per_image.summary.json` |
| Phân đoạn X-ray | `segmentation/chest_xray/metrics/chest_xray_test_per_image.summary.json` |
| Full pipeline | `pipeline/metrics/isic2018_pipeline_test_per_image.summary.json` |
| U-Net baseline | `segmentation/skin/metrics/isic2018_unet_baseline_test_per_image.summary.json` |
| U-Net diagnostic | `segmentation/skin/metrics/isic2018_unet_baseline_all_2594_per_image.summary.json` |

## Quy ước

- Mỗi config khai báo `checkpoint_path` và `history_path` trong YAML
- `evaluate.py` tạo `*.csv` (per-image) và `*.summary.json` (tổng hợp) cùng thư mục
- API endpoint `GET /api/metrics` đọc tất cả file trên và trả về JSON cho FE
- Thư mục `outputs/checkpoints/` và `outputs/logs/` là **legacy fallback** cho job đa miền đã chạy trước khi chuyển cấu trúc
