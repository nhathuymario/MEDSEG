# Datasets

## ISIC 2018 — Skin Lesion
- **Source**: ISIC Challenge 2018 Task 1
- **Size**: 2,594 dermoscopic images + binary segmentation masks
- **Format**: JPEG images + PNG masks
- **Resolution**: Variable (resized to 256×256 for segmentation, 512×512 for detection)
- **Split**: 70/15/15 (train/val/test)

## Montgomery County + Shenzhen — Chest X-ray
- **Montgomery**: 138 PA chest X-rays (80 normal, 58 TB)
- **Shenzhen**: 662 frontal X-rays (326 normal, 336 TB)
- **Task**: Lung field segmentation
- **Masks**: Manual lung segmentations available
- **Source**: NLM / Mendeley

## Preprocessing Pipeline
1. Resize to target dimensions
2. CLAHE enhancement (chest X-ray)
3. Normalize to ImageNet stats (mean/std)
4. Binary mask thresholding (>127)
5. COCO-format bbox generation from masks (for detection)
