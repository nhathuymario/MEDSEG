# MedSeg — Work Plan

## Phase 1: Project Setup ✅
- [x] requirements.txt, .gitignore, configs
- [x] Project directory structure

## Phase 2: Data Pipeline ✅
- [x] Download scripts (ISIC 2018, Chest X-ray)
- [x] Preprocessing (resize, normalize, binary masks, COCO annotations)
- [x] PyTorch Datasets (Detection + Segmentation)
- [x] Albumentations transforms
- [x] Train/val/test split

## Phase 3: Detection — Faster R-CNN ✅
- [x] LesionDetector model (torchvision backbone)
- [x] Training loop with SGD, gradient clipping
- [x] mAP, IoU, Precision/Recall metrics
- [x] Model-level tests

## Phase 4: Segmentation — U-Net & Attention U-Net ✅
- [x] DoubleConv, DownBlock, UpBlock components
- [x] AttentionGate module
- [x] Loss functions (Dice, DiceBCE, Focal, Tversky)
- [x] U-Net + Attention U-Net architectures
- [x] Training loop with cosine annealing
- [x] Dice, IoU, Sensitivity, Specificity metrics
- [x] Model-level tests

## Phase 5: Full Pipeline ✅
- [x] Single-model inference (detection + segmentation)
- [x] MedSegPipeline: Detect → Crop → Segment → Visualize
- [x] Visualization utilities (bbox overlay, mask overlay)
- [x] Pipeline-level tests

## Phase 6: Documentation ✅
- [x] README.md
- [x] plan.md (this file)
- [x] docs/ folder
- [x] description/ folder

## Phase 7: FastAPI Backend ✅
- [x] REST API: /detect, /segment, /pipeline, /health
- [x] Pydantic schemas
- [x] Model service (singleton, GPU auto-detect)
- [x] CORS for React dev server

## Phase 8: React Frontend ✅
- [x] Vite + React 18 setup
- [x] Dark theme glassmorphism design system
- [x] Pages: Dashboard, Analyze, Compare, History, Results
- [x] Components: ImageViewer, ResultPanel
- [x] API client module
- [x] useAnalysis custom hook

## Next Steps
- [ ] Download and prepare ISIC 2018 dataset
- [ ] Train Faster R-CNN (target: mAP@0.5 ≥ 0.60)
- [ ] Train Attention U-Net (target: Dice ≥ 0.85)
- [ ] Deploy and demo full pipeline
