# Ghi chú các buổi họp (Meeting Notes)

## Ngày: 2026-06-07 (Khởi tạo dự án)
- **Nội dung thảo luận**:
  - Xác định yêu cầu của dự án MedSeg.
  - Thống nhất các bài toán: Phát hiện bệnh (Detection) và Phân vùng bệnh (Segmentation).
  - Chọn các datasets: ISIC 2018 cho da liễu và Chest X-ray cho phổi.
  - Công nghệ backend: PyTorch và FastAPI.
  - Công nghệ frontend: ReactJS (Vite).
- **Quyết định**:
  - Dùng Faster R-CNN cho Detection.
  - Dùng U-Net và Attention U-Net cho Segmentation.
  - Cần có so sánh giữa các model (Compare page).
- **Open Action Items**:
  - Xác định tài nguyên GPU hiện có để chỉnh params phù hợp (Batch size, Image Size).
  - Khởi động quá trình down data và test model.
