# System Architecture

## Overview
MedSeg is a full-stack medical image analysis system with three layers:

```
┌─────────────────────────────────────────┐
│           React Frontend (Vite)          │
│  Dashboard │ Analyze │ Compare │ History │
├─────────────────────────────────────────┤
│           FastAPI Backend               │
│  /detect │ /segment │ /pipeline │ /health│
├─────────────────────────────────────────┤
│           PyTorch ML Models              │
│  Faster R-CNN │ U-Net │ Attention U-Net  │
└─────────────────────────────────────────┘
```

## Data Flow
1. User uploads image via React frontend
2. Frontend sends image to FastAPI endpoint
3. Backend runs model inference (GPU/CPU)
4. Results (bbox + mask overlays) returned as base64
5. Frontend renders results on canvas

## Model Pipeline
```
Input Image → Faster R-CNN → Detected ROIs (bounding boxes)
                              ↓
                        Crop each ROI
                              ↓
                   Attention U-Net → Segmentation mask per ROI
                              ↓
                   Map masks back to original → Final overlay
```
