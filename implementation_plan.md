# MedSeg - Phát hiện và Phân vùng Tổn thương Y tế

## Mục tiêu
Xây dựng hệ thống AI hỗ trợ chẩn đoán y khoa với 2 nhiệm vụ chính:
1. **Detection**: Phát hiện vùng nghi ngờ bệnh (Faster R-CNN)
2. **Segmentation**: Phân vùng chính xác tổn thương (U-Net / Attention U-Net)

## Datasets
| Dataset | Task | Size | Source |
|---------|------|------|--------|
| ISIC 2018 (Skin Lesion) | Segmentation + Detection | 2,594 training images + masks | Kaggle / ISIC Archive |
| Montgomery + Shenzhen Chest X-ray | Lung Segmentation | 138 + 662 CXR images | NLM / Kaggle |

## Cấu trúc Project

```
e:\xulianh\MedSeg\
├── README.md                          # Hướng dẫn sử dụng project
├── plan.md                            # Plan làm việc tổng quan
├── requirements.txt                   # Dependencies
├── setup.py                           # Package setup
│
├── configs/                           # Configuration files
│   ├── detection_config.yaml          # Faster R-CNN config
│   └── segmentation_config.yaml       # U-Net / Attention U-Net config
│
├── data/                              # Data directory (gitignored)
│   ├── raw/                           # Raw downloaded data
│   │   ├── isic2018/
│   │   └── chest_xray/
│   ├── processed/                     # Preprocessed data
│   │   ├── isic2018/
│   │   │   ├── images/
│   │   │   ├── masks/
│   │   │   └── annotations/           # COCO-format for detection
│   │   └── chest_xray/
│   │       ├── images/
│   │       └── masks/
│   └── splits/                        # Train/val/test split CSVs
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── data/                          # Data pipeline
│   │   ├── __init__.py
│   │   ├── download.py                # Dataset download scripts
│   │   ├── preprocess.py              # Preprocessing pipeline
│   │   ├── dataset_detection.py       # PyTorch Dataset for detection
│   │   ├── dataset_segmentation.py    # PyTorch Dataset for segmentation
│   │   ├── transforms.py             # Custom augmentations
│   │   └── split.py                   # Train/val/test splitting
│   │
│   ├── models/                        # Model architectures
│   │   ├── __init__.py
│   │   ├── faster_rcnn.py             # Faster R-CNN wrapper
│   │   ├── unet.py                    # Standard U-Net
│   │   ├── attention_unet.py          # Attention U-Net
│   │   └── components/                # Shared building blocks
│   │       ├── __init__.py
│   │       ├── conv_block.py
│   │       ├── attention_gate.py
│   │       └── losses.py              # DiceLoss, FocalLoss, etc.
│   │
│   ├── training/                      # Training logic
│   │   ├── __init__.py
│   │   ├── train_detection.py         # Faster R-CNN training
│   │   ├── train_segmentation.py      # U-Net training
│   │   ├── trainer.py                 # Base trainer class
│   │   └── callbacks.py              # Early stopping, checkpointing
│   │
│   ├── evaluation/                    # Evaluation & metrics
│   │   ├── __init__.py
│   │   ├── metrics_detection.py       # mAP, IoU for detection
│   │   ├── metrics_segmentation.py    # Dice, Jaccard, Sensitivity, Specificity
│   │   └── visualize.py              # Visualization utilities
│   │
│   └── pipeline/                      # Full pipeline
│       ├── __init__.py
│       ├── inference.py               # Single image inference
│       └── full_pipeline.py           # Detection → Segmentation pipeline
│
├── tests/                             # Testing
│   ├── test_models/                   # Model-level tests
│   │   ├── test_faster_rcnn.py
│   │   ├── test_unet.py
│   │   └── test_attention_unet.py
│   ├── test_data/                     # Data pipeline tests
│   │   ├── test_dataset.py
│   │   └── test_transforms.py
│   └── test_pipeline/                 # Full pipeline tests
│       ├── test_inference.py
│       └── test_full_pipeline.py
│
├── notebooks/                         # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_detection_analysis.ipynb
│   └── 03_segmentation_analysis.ipynb
│
├── docs/                              # Documentation folder
│   ├── architecture.md                # Kiến trúc hệ thống
│   ├── models.md                      # Mô tả chi tiết các model
│   ├── datasets.md                    # Mô tả datasets
│   ├── training_guide.md             # Hướng dẫn training
│   ├── evaluation_results.md         # Kết quả đánh giá
│   └── api_reference.md              # API reference
│
├── description/                       # Project description & chat logs
│   ├── project_description.md         # Mô tả tổng quan project
│   ├── meeting_notes.md               # Ghi chú các buổi thảo luận
│   └── changelog.md                   # Log thay đổi
│
├── outputs/                           # Training outputs (gitignored)
│   ├── checkpoints/                   # Model checkpoints
│   ├── logs/                          # Training logs / TensorBoard
│   └── predictions/                   # Prediction results
│
└── scripts/                           # Utility scripts
    ├── download_data.sh               # Download datasets
    ├── train.py                       # Main training entry point
    ├── evaluate.py                    # Main evaluation entry point
    └── predict.py                     # Prediction script
```

---

## Proposed Changes — 6 Phases

---

### Phase 1: Project Setup & Infrastructure
> **Mục tiêu**: Khởi tạo cấu trúc project, cài đặt dependencies, config management

#### [NEW] [requirements.txt](file:///e:/xulianh/MedSeg/requirements.txt)
- Core: `torch>=2.0`, `torchvision>=0.15`, `numpy`, `pandas`, `scikit-learn`
- Image: `opencv-python`, `Pillow`, `albumentations`
- Medical: `pydicom`, `SimpleITK` (optional cho DICOM)
- Visualization: `matplotlib`, `seaborn`, `tensorboard`
- Config: `PyYAML`, `omegaconf`
- Testing: `pytest`, `pytest-cov`
- Utilities: `tqdm`, `wandb` (optional)

#### [NEW] [setup.py](file:///e:/xulianh/MedSeg/setup.py)
- Package installation setup

#### [NEW] [configs/detection_config.yaml](file:///e:/xulianh/MedSeg/configs/detection_config.yaml)
```yaml
model:
  name: faster_rcnn
  backbone: resnet50
  pretrained: true
  num_classes: 2  # background + lesion
  rpn_anchor_sizes: [[32, 64, 128, 256, 512]]
  
data:
  dataset: isic2018
  image_size: [512, 512]
  batch_size: 4
  num_workers: 4
  
training:
  epochs: 50
  lr: 0.001
  weight_decay: 0.0005
  lr_scheduler: step
  step_size: 15
  
evaluation:
  iou_threshold: 0.5
  confidence_threshold: 0.5
```

#### [NEW] [configs/segmentation_config.yaml](file:///e:/xulianh/MedSeg/configs/segmentation_config.yaml)
```yaml
model:
  name: attention_unet  # or 'unet'
  in_channels: 3
  out_channels: 1
  features: [64, 128, 256, 512]
  
data:
  dataset: isic2018  # or 'chest_xray'
  image_size: [256, 256]
  batch_size: 8
  num_workers: 4
  
training:
  epochs: 100
  lr: 0.0001
  optimizer: adam
  loss: dice_bce  # combined Dice + BCE loss
  lr_scheduler: cosine
  
evaluation:
  threshold: 0.5
```

---

### Phase 2: Data Pipeline
> **Mục tiêu**: Download, preprocess, augmentation, và tạo PyTorch Datasets

#### [NEW] [src/data/download.py](file:///e:/xulianh/MedSeg/src/data/download.py)
- Hàm download ISIC 2018 từ Kaggle API hoặc direct URL
- Hàm download Montgomery + Shenzhen datasets
- Tự động extract và sắp xếp vào `data/raw/`
- Validation: kiểm tra file integrity sau download

#### [NEW] [src/data/preprocess.py](file:///e:/xulianh/MedSeg/src/data/preprocess.py)
- **ISIC 2018**: Resize → Normalize → Convert masks to binary
- **Chest X-ray**: Resize → CLAHE enhancement → Normalize
- Tạo COCO-format annotations từ binary masks (cho detection)
- Bounding box extraction từ segmentation masks

#### [NEW] [src/data/dataset_detection.py](file:///e:/xulianh/MedSeg/src/data/dataset_detection.py)
- Custom `torch.utils.data.Dataset` cho Faster R-CNN
- Output format: `{'boxes': tensor, 'labels': tensor, 'image_id': int}`
- Hỗ trợ COCO-format annotations

#### [NEW] [src/data/dataset_segmentation.py](file:///e:/xulianh/MedSeg/src/data/dataset_segmentation.py)
- Custom `torch.utils.data.Dataset` cho U-Net
- Output: `(image_tensor, mask_tensor)`
- Hỗ trợ on-the-fly augmentation

#### [NEW] [src/data/transforms.py](file:///e:/xulianh/MedSeg/src/data/transforms.py)
- Medical-specific augmentations sử dụng Albumentations:
  - `RandomRotate90`, `HorizontalFlip`, `VerticalFlip`
  - `ElasticTransform`, `GridDistortion`
  - `RandomBrightnessContrast`, `CLAHE`
  - `CoarseDropout` (cutout)

#### [NEW] [src/data/split.py](file:///e:/xulianh/MedSeg/src/data/split.py)
- Stratified split: 70% train / 15% val / 15% test
- Lưu split indices vào CSV trong `data/splits/`
- Đảm bảo reproducibility với random seed

---

### Phase 3: Detection — Faster R-CNN
> **Mục tiêu**: Implement và train Faster R-CNN cho lesion detection trên ISIC 2018

#### [NEW] [src/models/faster_rcnn.py](file:///e:/xulianh/MedSeg/src/models/faster_rcnn.py)
- Wrapper class `LesionDetector` sử dụng `torchvision.models.detection.fasterrcnn_resnet50_fpn`
- Pretrained backbone (COCO weights) → fine-tune on medical data
- Custom `AnchorGenerator` tối ưu cho kích thước tổn thương
- Custom `FastRCNNPredictor` cho 2 classes (background + lesion)
- Interface:
  ```python
  class LesionDetector(nn.Module):
      def __init__(self, num_classes=2, pretrained=True, anchor_sizes=None):
          ...
      def forward(self, images, targets=None):
          # Training: returns losses dict
          # Inference: returns predictions list
          ...
  ```

#### [NEW] [src/training/train_detection.py](file:///e:/xulianh/MedSeg/src/training/train_detection.py)
- Training loop cho Faster R-CNN
- Tích hợp: LR scheduler, gradient clipping, mixed precision (AMP)
- Logging: TensorBoard + console output
- Checkpointing: save best model theo mAP

#### [NEW] [src/evaluation/metrics_detection.py](file:///e:/xulianh/MedSeg/src/evaluation/metrics_detection.py)
- **mAP@0.5** (Mean Average Precision)
- **mAP@[0.5:0.95]** (COCO-style)
- **IoU** (Intersection over Union) per prediction
- **Precision / Recall** curves
- **Confusion matrix** (TP, FP, FN)

#### Testing — Model Level
```
tests/test_models/test_faster_rcnn.py
├── test_model_creation()           # Model instantiation
├── test_forward_pass()             # Forward pass with dummy data
├── test_output_format()            # Verify output dict keys
├── test_loss_computation()         # Training mode returns losses
├── test_inference_mode()           # Eval mode returns predictions
├── test_anchor_generator()         # Custom anchors work correctly
└── test_pretrained_loading()       # Pretrained weights load OK
```

---

### Phase 4: Segmentation — U-Net & Attention U-Net
> **Mục tiêu**: Implement U-Net + Attention U-Net cho precise lesion segmentation

#### [NEW] [src/models/components/conv_block.py](file:///e:/xulianh/MedSeg/src/models/components/conv_block.py)
- `DoubleConv`: Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU
- `DownBlock`: MaxPool → DoubleConv
- `UpBlock`: ConvTranspose2d → Concatenate → DoubleConv

#### [NEW] [src/models/components/attention_gate.py](file:///e:/xulianh/MedSeg/src/models/components/attention_gate.py)
- **Attention Gate** module:
  ```python
  class AttentionGate(nn.Module):
      def __init__(self, F_g, F_l, F_int):
          # F_g: gating signal channels (from decoder)
          # F_l: input feature channels (from encoder skip)
          # F_int: intermediate channels
          self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1)
          self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1)
          self.psi = nn.Conv2d(F_int, 1, kernel_size=1)
          self.sigmoid = nn.Sigmoid()
          self.relu = nn.ReLU(inplace=True)
      
      def forward(self, g, x):
          # g: gating signal, x: encoder feature
          g1 = self.W_g(g)
          x1 = self.W_x(x)
          psi = self.relu(g1 + x1)
          psi = self.sigmoid(self.psi(psi))
          return x * psi  # Weighted features
  ```

#### [NEW] [src/models/components/losses.py](file:///e:/xulianh/MedSeg/src/models/components/losses.py)
- `DiceLoss`: Dice coefficient loss
- `FocalLoss`: Focal loss cho class imbalance
- `DiceBCELoss`: Combined Dice + Binary Cross Entropy
- `TverskyLoss`: Tversky loss (α, β tunable for FP/FN balance)

#### [NEW] [src/models/unet.py](file:///e:/xulianh/MedSeg/src/models/unet.py)
- Standard U-Net architecture:
  - Encoder: 4 levels (64→128→256→512)
  - Bottleneck: 1024 channels
  - Decoder: 4 levels (512→256→128→64)
  - Skip connections: concatenation
  - Output: 1×1 Conv → Sigmoid

#### [NEW] [src/models/attention_unet.py](file:///e:/xulianh/MedSeg/src/models/attention_unet.py)
- Attention U-Net = U-Net + Attention Gates trên mỗi skip connection
- Kiến trúc tương tự U-Net nhưng thêm `AttentionGate` tại mỗi decoder level
- Cho phép model tập trung vào vùng tổn thương, suppress background noise

#### [NEW] [src/training/train_segmentation.py](file:///e:/xulianh/MedSeg/src/training/train_segmentation.py)
- Training loop cho U-Net / Attention U-Net
- Loss: DiceBCE (default) hoặc configurable
- LR scheduler: Cosine Annealing
- Mixed precision training
- Best model checkpointing theo Dice score

#### [NEW] [src/evaluation/metrics_segmentation.py](file:///e:/xulianh/MedSeg/src/evaluation/metrics_segmentation.py)
- **Dice Coefficient** (F1 score for segmentation)
- **Jaccard Index** (IoU)
- **Sensitivity** (True Positive Rate)
- **Specificity** (True Negative Rate)
- **Hausdorff Distance** (boundary accuracy)
- **Pixel Accuracy**

#### Testing — Model Level
```
tests/test_models/test_unet.py
├── test_model_creation()           # U-Net instantiation
├── test_forward_pass()             # Input → Output shape check
├── test_output_shape()             # Output matches input spatial dims
├── test_different_channels()       # 1-ch and 3-ch inputs
├── test_gradient_flow()            # Gradients propagate correctly
└── test_skip_connections()         # Feature concatenation works

tests/test_models/test_attention_unet.py
├── test_model_creation()           # Attention U-Net instantiation
├── test_attention_gate()           # AG produces valid attention maps
├── test_forward_pass()             # Full forward pass
├── test_output_shape()             # Output shape verification
├── test_attention_coefficients()   # Attention weights in [0,1]
└── test_comparison_with_unet()     # Same I/O interface as U-Net
```

---

### Phase 5: Full Pipeline & Integration
> **Mục tiêu**: Kết nối Detection → Segmentation thành pipeline hoàn chỉnh

#### [NEW] [src/pipeline/inference.py](file:///e:/xulianh/MedSeg/src/pipeline/inference.py)
- Single image inference cho từng model riêng lẻ
- Load checkpoint → preprocess → predict → post-process → visualize
- Support cả detection và segmentation

#### [NEW] [src/pipeline/full_pipeline.py](file:///e:/xulianh/MedSeg/src/pipeline/full_pipeline.py)
- **Full Pipeline Flow**:
  ```
  Input Image → Faster R-CNN (Detection)
       ↓
  Detected ROIs (bounding boxes)
       ↓
  Crop ROIs → Resize → Attention U-Net (Segmentation)
       ↓
  Segmentation masks per ROI
       ↓
  Map back to original image → Final result
  ```
- Class `MedSegPipeline`:
  ```python
  class MedSegPipeline:
      def __init__(self, detection_checkpoint, segmentation_checkpoint):
          ...
      def detect(self, image) -> List[BBox]
      def segment(self, image, bbox) -> np.ndarray  # mask
      def run(self, image) -> Dict  # full pipeline
      def visualize(self, image, results) -> np.ndarray
  ```

#### [NEW] [src/evaluation/visualize.py](file:///e:/xulianh/MedSeg/src/evaluation/visualize.py)
- Overlay bounding boxes lên ảnh gốc
- Overlay segmentation masks (color-coded)
- Side-by-side comparison: prediction vs ground truth
- Attention map visualization (cho Attention U-Net)
- Save results dạng hình ảnh

#### [NEW] [scripts/train.py](file:///e:/xulianh/MedSeg/scripts/train.py)
- CLI entry point: `python scripts/train.py --config configs/detection_config.yaml`
- Hỗ trợ cả detection và segmentation training

#### [NEW] [scripts/evaluate.py](file:///e:/xulianh/MedSeg/scripts/evaluate.py)
- CLI entry point: `python scripts/evaluate.py --model detection --checkpoint path/to/ckpt`
- In ra metrics table

#### [NEW] [scripts/predict.py](file:///e:/xulianh/MedSeg/scripts/predict.py)
- CLI: `python scripts/predict.py --image path/to/image --pipeline full`
- Output: annotated image + JSON results

#### Testing — Full Pipeline
```
tests/test_pipeline/test_inference.py
├── test_detection_inference()      # Single image detection
├── test_segmentation_inference()   # Single image segmentation
├── test_batch_inference()          # Batch processing
└── test_checkpoint_loading()       # Model loading from checkpoint

tests/test_pipeline/test_full_pipeline.py
├── test_pipeline_creation()        # Pipeline instantiation
├── test_detect_then_segment()      # Detection → Segmentation flow
├── test_empty_detection()          # Handle no detections gracefully
├── test_multiple_detections()      # Multiple ROIs segmented
├── test_visualization()            # Output visualization correct
├── test_end_to_end()              # Full E2E on sample image
└── test_performance_metrics()      # Combined metrics computation
```

---

### Phase 6: Documentation & Project Files
> **Mục tiêu**: Tạo đầy đủ documentation, README, plan, và description files

#### [NEW] [README.md](file:///e:/xulianh/MedSeg/README.md)
- Project overview & motivation
- Architecture diagram (Detection + Segmentation pipeline)
- Installation instructions
- Quick start guide
- Dataset preparation guide
- Training instructions
- Evaluation results (bảng metrics)
- Project structure tree
- License & references

#### [NEW] [plan.md](file:///e:/xulianh/MedSeg/plan.md)
- Kế hoạch làm việc chi tiết theo từng phase
- Timeline estimation
- Milestones & deliverables
- Risk assessment

#### [NEW] docs/ folder:
| File | Nội dung |
|------|----------|
| `docs/architecture.md` | Kiến trúc tổng quan, data flow diagram, model architecture diagrams |
| `docs/models.md` | Chi tiết Faster R-CNN, U-Net, Attention U-Net: hyperparameters, layer configs |
| `docs/datasets.md` | Mô tả ISIC 2018, Chest X-ray: statistics, preprocessing steps, augmentation |
| `docs/training_guide.md` | Hướng dẫn training step-by-step, troubleshooting |
| `docs/evaluation_results.md` | Bảng kết quả metrics, comparison charts, analysis |
| `docs/api_reference.md` | API reference cho các modules chính |

#### [NEW] description/ folder:
| File | Nội dung |
|------|----------|
| `description/project_description.md` | Mô tả tổng quan project, mục tiêu, scope |
| `description/meeting_notes.md` | Ghi chú các buổi thảo luận, quyết định thiết kế |
| `description/changelog.md` | Log các thay đổi quan trọng theo ngày |

---

## Verification Plan

### Automated Tests — Model Level
```bash
# Test từng model riêng lẻ
pytest tests/test_models/test_faster_rcnn.py -v
pytest tests/test_models/test_unet.py -v
pytest tests/test_models/test_attention_unet.py -v

# Test data pipeline
pytest tests/test_data/ -v
```

### Automated Tests — Full Pipeline
```bash
# Test pipeline integration
pytest tests/test_pipeline/ -v

# Test full E2E
pytest tests/test_pipeline/test_full_pipeline.py -v

# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=html
```

### Manual Verification
- Train Faster R-CNN trên ISIC 2018, đạt **mAP@0.5 ≥ 0.60**
- Train Attention U-Net trên ISIC 2018, đạt **Dice ≥ 0.85**
- Train U-Net trên Chest X-ray, đạt **Dice ≥ 0.90** (lung segmentation)
- Full pipeline chạy thành công: Detection → Segmentation → Visualization
- TensorBoard logs hiển thị training curves chính xác

---

## Open Questions

> [!IMPORTANT]
> **GPU Resources**: Bạn có GPU NVIDIA không? Nếu có thì là loại gì (VRAM bao nhiêu GB)? Điều này ảnh hưởng đến batch size và image resolution khi training.

> [!IMPORTANT]
> **Dataset Priority**: Bạn muốn bắt đầu với dataset nào trước — ISIC 2018 (skin lesion) hay Chest X-ray? Hay implement cả hai song song?

> [!NOTE]
> **Kaggle API**: Bạn đã có Kaggle API key chưa? Nếu có thì download dataset sẽ tự động hơn. Nếu không, tôi sẽ cung cấp script download manual.

> [!NOTE]
> **Language**: README và docs nên viết bằng tiếng Anh hay tiếng Việt?
