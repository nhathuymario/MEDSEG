# Meeting Notes

## 2026-06-07 - Khởi tạo dự án

### User's Requirement

- Xây dựng hệ thống phát hiện và phân vùng tổn thương y tế.
- Sử dụng ISIC 2018 và Chest X-ray/Lung CT.
- Áp dụng Faster R-CNN và U-Net/Attention U-Net.

### Features

- Detection, segmentation, API, giao diện web và trang so sánh model.

### Tech Solutions

- PyTorch + FastAPI cho AI/backend.
- React + Vite cho frontend.
- ISIC 2018 cho da liễu; Montgomery/Shenzhen cho Chest X-ray.

### Logic + AI

- Detection tạo vùng quan tâm.
- Segmentation tạo mask chính xác trong vùng cần phân tích.

### Implement

- Thống nhất cấu trúc project và các phase triển khai.

### Test

- Yêu cầu test model, data pipeline và full pipeline.

## 2026-06-11 - Hoàn thiện Chest X-ray pipeline

### User's Requirement

- Đề tài phải có cả ISIC và Chest X-ray, không chỉ train ISIC.
- Tự động tải Chest X-ray và tiếp tục triển khai đến training.

### Features

- Downloader tự động cho Montgomery và Shenzhen.
- Resume/retry khi mạng gián đoạn.
- Preprocess Montgomery, ghép mask phổi trái/phải.
- Train và evaluate Attention U-Net theo train/val/test split.

### Tech Solutions

- Nguồn chính thức NLM.
- CLAHE, resize 256x256 và binary lung mask.
- Attention U-Net features `[32, 64, 128, 256]`.
- Adam, DiceBCE, cosine scheduler, mixed precision.
- CUDA 12.8 trên RTX 3050 Laptop 4 GB.

### Logic + AI

- Chỉ Montgomery được dùng cho lung segmentation vì có mask phổi trái/phải.
- Shenzhen không trộn vào bài toán này vì annotation không có cùng ý nghĩa nhãn.
- Split cố định 70/15/15 qua CSV để tái lập kết quả.

### Implement

- Sửa downloader và thêm CLI theo dataset.
- Thêm `prepare_chest_xray`.
- Thêm transform riêng, config riêng và checkpoint riêng.
- Sửa `train.py` dùng split/config thực tế.
- Hoàn thiện segmentation evaluation CLI.
- Sửa lỗi khởi tạo U-Net.

### Test

- 138 ảnh và 138 mask processed.
- Split: 96 train, 20 validation, 22 test.
- 100 epoch hoàn thành.
- Best validation Dice: 0.9771.
- Test Dice: 0.9811; IoU: 0.9628.
- 11 automated tests pass.

### Decision

- Chest X-ray phase được xem là hoàn thành cho bài toán phân vùng trường phổi.
- Bước tiếp theo là train/evaluate Faster R-CNN và Attention U-Net trên ISIC 2018.
