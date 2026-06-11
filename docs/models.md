# Models Documentation

## 1. Faster R-CNN (Detection)
- **Backbone**: ResNet-50 + FPN (pretrained on COCO)
- **Task**: Detect lesion bounding boxes
- **Classes**: 2 (background + lesion)
- **Input**: RGB image (512×512)
- **Output**: List of {boxes, scores, labels}
- **Loss**: RPN loss + classification loss + box regression loss

## 2. U-Net (Segmentation)
- **Architecture**: Encoder (4 levels) → Bottleneck → Decoder (4 levels)
- **Features**: [64, 128, 256, 512] → Bottleneck: 1024
- **Skip connections**: Concatenation
- **Input**: RGB image (256×256)
- **Output**: Binary mask (1 channel, sigmoid)

## 3. Attention U-Net (Segmentation)
- **Same as U-Net** but with Attention Gates on skip connections
- **Attention Gate**: Learns to weight encoder features using decoder gating signal
- **Benefit**: Suppresses irrelevant background, focuses on lesion boundaries
- **Parameters**: F_g (gating channels), F_l (input channels), F_int (intermediate)

## Loss Functions
| Loss | Formula | Use Case |
|------|---------|----------|
| DiceLoss | 1 - 2·TP / (pred + target) | Overlap-based |
| DiceBCELoss | BCE + DiceLoss | Combined (default) |
| FocalLoss | α(1-pt)^γ · BCE | Class imbalance |
| TverskyLoss | 1 - TP / (TP + α·FP + β·FN) | Tunable FP/FN balance |
