# Datasets

## ISIC 2018 — Skin Lesion

- **Nguồn:** ISIC Challenge 2018 Task 1-2
- **Số ảnh:** 2,594 ảnh dermoscopy + binary mask
- **Format:** JPEG + PNG mask
- **Kích thước:** 256×256 (segmentation), 512×512 (detection)
- **Split:** 1815 train / 389 val / 390 test (70/15/15)
- **Dùng cho:** Detection ISIC, Segmentation ISIC, Full pipeline, U-Net baseline

## iToBoS 2024 — Clinical Skin Lesion

- **Nguồn:** iToBoS Grand Challenge 2024
- **Số ảnh:** 16,954 ảnh clinical (điện thoại, máy ảnh lâm sàng)
- **Annotation:** 59,997 bounding boxes (COCO JSON)
- **Kích thước:** 512×512
- **Split:** 8,473 train / 8,481 test
- **Dùng cho:** Detection đa miền (kết hợp với ISIC để fine-tune)
- **Đường dẫn:** `data/raw/clinical_skin/itobos/`

## Montgomery + Shenzhen — Chest X-ray

- **Montgomery:** 138 PA chest X-rays (80 bình thường, 58 TB)
- **Shenzhen:** 662 frontal X-rays (326 bình thường, 336 TB)
- **Task:** Phân đoạn vùng phổi
- **Mask:** Lung segmentation thủ công
- **Nguồn:** NLM / Mendeley

## Preprocessing pipeline

1. Resize về kích thước mục tiêu (256×256 hoặc 512×512)
2. CLAHE enhancement (chest X-ray)
3. Binarize mask (threshold > 127)
4. Normalize theo ImageNet stats (mean/std)
5. Tạo COCO-format bounding box từ mask (cho detection)
