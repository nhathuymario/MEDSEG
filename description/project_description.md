# MedSeg - Mô tả dự án

## 1. User's Requirement

Xây dựng hệ thống AI hỗ trợ chẩn đoán hình ảnh y tế với hai nhiệm vụ:

- Detection: phát hiện vùng nghi ngờ bệnh.
- Segmentation: phân vùng chính xác vùng tổn thương hoặc cấu trúc y khoa.
- Dữ liệu mục tiêu: ISIC 2018 và Chest X-ray.
- Mô hình: Faster R-CNN, U-Net và Attention U-Net.
- Có API và giao diện web để tải ảnh, phân tích và hiển thị kết quả.

## 2. Features

- Tải và chuẩn bị ISIC 2018.
- Tải tự động Montgomery và Shenzhen Chest X-ray từ NLM.
- Phát hiện tổn thương da bằng Faster R-CNN.
- Phân vùng tổn thương da bằng U-Net/Attention U-Net.
- Phân vùng trường phổi trên Chest X-ray.
- Đánh giá bằng mAP, Dice, IoU, sensitivity, specificity và pixel accuracy.
- FastAPI backend và React frontend.

## 3. Tech Solutions

- AI/ML: PyTorch, torchvision, Albumentations.
- Xử lý ảnh: OpenCV, NumPy.
- Backend: FastAPI, Pydantic.
- Frontend: React 18, Vite.
- Cấu hình: YAML.
- Kiểm thử: pytest.
- Phần cứng đã xác nhận: NVIDIA GeForce RTX 3050 Laptop GPU 4 GB.
- Môi trường train GPU: PyTorch 2.11.0 + CUDA 12.8.

## 4. Logic + AI

### ISIC 2018

Ảnh da liễu -> preprocess -> Faster R-CNN phát hiện ROI -> Attention U-Net phân vùng tổn thương -> overlay kết quả.

### Chest X-ray

Ảnh Montgomery -> CLAHE và resize -> ghép mask phổi trái/phải -> Attention U-Net phân vùng trường phổi -> đánh giá trên test split.

Lưu ý: mô hình Chest X-ray hiện phân vùng trường phổi, không phát hiện tổn thương phổi.

## 5. Implement

- Data pipeline, model, training loop, evaluation, API và frontend đã có.
- Chest X-ray pipeline đã hoàn thiện từ download đến checkpoint.
- ISIC 2018 đã được download, preprocess, tạo COCO annotation và split.
- Training/evaluation ISIC vẫn là công việc tiếp theo.

## 6. Test

- Chest X-ray: 138 cặp ảnh-mask, split 96/20/22.
- Attention U-Net đã train 100 epoch.
- Best validation Dice: 0.9771.
- Test Dice: 0.9811; IoU: 0.9628.
- 11 test data/U-Net/Attention U-Net đã pass.

Ngày bắt đầu: 2026-06-07. Cập nhật gần nhất: 2026-06-11.
