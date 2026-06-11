# Hướng dẫn Training (Training Guide)

Tài liệu này hướng dẫn cách training các model Faster R-CNN, U-Net và Attention U-Net.

## Yêu cầu Hệ thống
- **GPU**: NVIDIA GPU có hỗ trợ CUDA (khuyến nghị VRAM >= 8GB cho U-Net, >= 12GB cho Faster R-CNN).
- Cài đặt đầy đủ các thư viện trong `requirements.txt`.

## 1. Chuẩn bị Dữ liệu
Trước khi train, cần download và preprocess data:
```bash
# Chạy script download
bash scripts/download_data.sh

# Split data thành train/val/test
python src/data/split.py
```

## 2. Train Detection Model (Faster R-CNN)
Chạy script `train.py` với cấu hình của detection:
```bash
python scripts/train.py --config configs/detection_config.yaml
```
- Model checkpoint sẽ được lưu tại: `outputs/checkpoints/best_detection.pth`
- TensorBoard logs sẽ được lưu tại: `outputs/logs/detection/`

## 3. Train Segmentation Model (U-Net / Attention U-Net)
Chạy script `train.py` với cấu hình của segmentation:
```bash
python scripts/train.py --config configs/segmentation_config.yaml
```
*Lưu ý*: Bạn có thể đổi tên model (`unet` hoặc `attention_unet`) trong file `configs/segmentation_config.yaml`.
- Model checkpoint sẽ được lưu tại: `outputs/checkpoints/best_segmentation.pth`

## 4. Theo dõi Quá trình Train (TensorBoard)
Sử dụng TensorBoard để theo dõi loss và metrics:
```bash
tensorboard --logdir=outputs/logs/
```
Truy cập `http://localhost:6006` trên trình duyệt.

## 5. Troubleshooting (Sửa lỗi thường gặp)
- **CUDA Out of Memory**: Giảm `batch_size` hoặc `image_size` trong file cấu hình `.yaml`.
- **Loss không giảm (NaN)**: Giảm `lr` (learning rate) hoặc sử dụng gradient clipping.
