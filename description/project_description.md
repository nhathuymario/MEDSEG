# MedSeg — Project Description

## Mục tiêu
Xây dựng hệ thống AI hỗ trợ chẩn đoán y khoa với 2 nhiệm vụ:
1. **Detection**: Phát hiện vùng nghi ngờ bệnh bằng Faster R-CNN
2. **Segmentation**: Phân vùng chính xác tổn thương bằng U-Net / Attention U-Net

## Phạm vi
- Hỗ trợ phân tích skin lesion (ISIC 2018) và chest X-ray
- Web interface cho bác sĩ upload ảnh và xem kết quả
- So sánh hiệu quả giữa các model

## Công nghệ
- **ML**: PyTorch, torchvision, Albumentations
- **Backend**: FastAPI
- **Frontend**: React (Vite)
- **Testing**: pytest

## Ngày bắt đầu
2026-06-07
