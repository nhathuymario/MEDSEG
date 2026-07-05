# BÁO CÁO CUỐI KÌ

## Hệ thống phát hiện và phân đoạn ảnh y tế MedSeg

**Môn học:** ..........................................................

**Giảng viên hướng dẫn:** .............................................

**Sinh viên thực hiện:** ..............................................

**Mã số sinh viên:** ..................................................

**Lớp:** ..............................................................

**Ngày hoàn thành:** 25/06/2026

---

## Tóm tắt

Đề tài xây dựng hệ thống MedSeg nhằm hỗ trợ phân tích ảnh y tế bằng học sâu. Hệ thống tập trung vào hai nhóm bài toán chính: phát hiện tổn thương da trên bộ dữ liệu ISIC 2018 bằng Faster R-CNN và phân đoạn vùng tổn thương hoặc vùng phổi bằng U-Net/Attention U-Net. Ngoài phần huấn luyện và đánh giá mô hình, đề tài còn triển khai backend FastAPI và frontend React để người dùng có thể tải ảnh, chọn workflow phân tích, xem kết quả overlay và theo dõi các chỉ số đánh giá.

Kết quả thực nghiệm cho thấy mô hình phân đoạn phổi X-quang đạt Dice 0.9811 và IoU 0.9628 trên tập test Montgomery CXR. Với ảnh da liễu ISIC 2018, mô hình phân đoạn tổn thương đạt Dice 0.8847 và IoU 0.7932 trên 390 ảnh test. Mô hình phát hiện tổn thương da Faster R-CNN đạt mAP@0.5 là 0.7142, Precision@0.5 là 0.8768 và Recall@0.5 là 0.7846.

---

## Mục lục

1. Giới thiệu đề tài
2. Mục tiêu và phạm vi
3. Cơ sở lý thuyết
4. Dữ liệu và tiền xử lý
5. Kiến trúc hệ thống
6. Huấn luyện mô hình
7. Đánh giá thực nghiệm
8. Giao diện và chức năng phần mềm
9. Hạn chế và hướng phát triển
10. Kết luận
11. Phụ lục

---

## 1. Giới thiệu đề tài

Ảnh y tế là nguồn dữ liệu quan trọng trong chẩn đoán và theo dõi bệnh. Tuy nhiên, việc đọc ảnh và khoanh vùng tổn thương thủ công thường tốn thời gian, phụ thuộc vào kinh nghiệm chuyên môn và khó mở rộng khi số lượng ảnh lớn. Các mô hình học sâu, đặc biệt là mạng tích chập và các kiến trúc segmentation, có khả năng học đặc trưng hình ảnh và hỗ trợ tự động hóa một phần quy trình phân tích.

Đề tài MedSeg được xây dựng với mục tiêu mô phỏng một hệ thống phân tích ảnh y tế có thể chạy được end-to-end, bao gồm dữ liệu, mô hình, pipeline inference, API backend và frontend. Hệ thống không nhằm thay thế bác sĩ, mà đóng vai trò công cụ hỗ trợ định vị và trực quan hóa vùng quan tâm trên ảnh.

---

## 2. Mục tiêu và phạm vi

### 2.1 Mục tiêu

- Xây dựng pipeline xử lý dữ liệu y tế cho hai nhóm ảnh: ảnh da liễu ISIC 2018 và ảnh X-quang phổi Montgomery/Shenzhen.
- Huấn luyện mô hình phát hiện tổn thương da bằng Faster R-CNN.
- Huấn luyện mô hình phân đoạn bằng Attention U-Net cho tổn thương da và vùng phổi.
- Triển khai backend FastAPI phục vụ inference qua API.
- Xây dựng frontend React cho phép upload ảnh, chọn workflow phân tích, xem kết quả và xuất báo cáo chỉ số.
- Đánh giá mô hình trên tập test độc lập và lưu số liệu per-image/trial-level.

### 2.2 Phạm vi

Hệ thống hiện tập trung vào:

- ISIC 2018: phát hiện bounding box tổn thương da và phân đoạn mask tổn thương.
- Montgomery Chest X-ray: phân đoạn vùng phổi.
- Frontend chạy local, backend chạy local, checkpoint lưu trên filesystem.

Hệ thống chưa sử dụng database. Các ảnh upload qua backend được xử lý trực tiếp để inference, không lưu lịch sử dài hạn trong cơ sở dữ liệu.

---

## 3. Cơ sở lý thuyết

### 3.1 Object Detection

Object detection là bài toán xác định vị trí đối tượng trong ảnh bằng bounding box. Trong đề tài này, mô hình Faster R-CNN được dùng để phát hiện vùng tổn thương da. Kết quả đầu ra gồm:

- `boxes`: tọa độ bounding box `[x1, y1, x2, y2]`.
- `scores`: độ tin cậy của dự đoán.
- `labels`: nhãn lớp.

### 3.2 Image Segmentation

Image segmentation là bài toán gán nhãn cho từng pixel trong ảnh. Trong đề tài, segmentation dùng để tạo mask vùng tổn thương da hoặc vùng phổi. Khác với detection chỉ khoanh hình chữ nhật, segmentation cho biết chính xác pixel nào thuộc vùng quan tâm.

### 3.3 U-Net và Attention U-Net

U-Net là kiến trúc encoder-decoder phổ biến trong segmentation ảnh y tế. Encoder trích xuất đặc trưng, decoder khôi phục kích thước không gian để tạo mask. Attention U-Net bổ sung attention gate nhằm giúp mô hình tập trung hơn vào vùng quan trọng và giảm nhiễu từ vùng nền.

### 3.4 Các chỉ số đánh giá

| Chỉ số | Công thức | Ý nghĩa |
|---|---|---|
| Dice | `2TP / (2TP + FP + FN)` | Độ trùng khớp giữa mask dự đoán và mask thật |
| IoU | `TP / (TP + FP + FN)` | Tỉ lệ giao nhau trên hợp nhất |
| Sensitivity/Recall | `TP / (TP + FN)` | Khả năng phát hiện đúng vùng thật |
| Specificity | `TN / (TN + FP)` | Khả năng nhận đúng vùng nền |
| Pixel Accuracy | `(TP + TN) / (TP + TN + FP + FN)` | Tỉ lệ pixel đúng tổng thể |
| mAP@0.5 | AP với IoU >= 0.5 | Chỉ số chính cho detection |

---

## 4. Dữ liệu và tiền xử lý

### 4.1 Cấu trúc thư mục dữ liệu

```text
data/
├── raw/
├── processed/
└── splits/
```

`data/raw/` chứa dữ liệu gốc sau khi tải về. `data/processed/` chứa ảnh và mask đã tiền xử lý để đưa vào model. `data/splits/` chứa các file CSV chia train/val/test.

### 4.2 Bộ dữ liệu ISIC 2018

ISIC 2018 được dùng cho bài toán tổn thương da. Dữ liệu gồm ảnh da liễu và ground truth mask. Sau xử lý, dữ liệu nằm tại:

- `data/processed/isic2018/images`
- `data/processed/isic2018/masks`
- `data/processed/isic2018/annotations/train.json`

Số lượng dữ liệu đã xử lý:

| Split | Số ảnh |
|---|---:|
| Train | 1815 |
| Validation | 389 |
| Test | 390 |

### 4.3 Bộ dữ liệu Chest X-ray

Chest X-ray dùng cho bài toán phân đoạn vùng phổi. Pipeline hiện sử dụng Montgomery CXR để tạo tập processed. Dữ liệu gồm ảnh X-quang và mask thủ công cho phổi trái/phải.

Sau xử lý:

- `data/processed/chest_xray/images`
- `data/processed/chest_xray/masks`

Số lượng dữ liệu:

| Split | Số ảnh |
|---|---:|
| Train | 96 |
| Validation | 20 |
| Test | 22 |

### 4.4 Tiền xử lý

Với Chest X-ray, script `src/data/preprocess.py` đọc ảnh từ `CXR_png`, đọc mask trái và phải từ `ManualMask/leftMask` và `ManualMask/rightMask`, sau đó gộp hai mask thành một mask phổi hoàn chỉnh. Ảnh và mask được resize về cùng kích thước, mask được nhị phân hóa thành nền và vùng phổi.

Với ISIC, ảnh và mask được chuẩn hóa để phục vụ segmentation. Annotation detection được tạo dưới dạng COCO JSON, trong đó bounding box được suy ra từ vùng mask tổn thương.

---

## 5. Kiến trúc hệ thống

### 5.1 Tổng quan

Hệ thống gồm ba lớp chính:

```text
React Frontend → FastAPI Backend → PyTorch Models
```

Frontend cung cấp giao diện upload ảnh và xem kết quả. Backend nhận file ảnh, load checkpoint, chạy inference và trả về kết quả dạng JSON/base64. Các model PyTorch thực hiện detection hoặc segmentation.

### 5.2 Backend

Backend sử dụng FastAPI, các endpoint chính gồm:

| Endpoint | Chức năng |
|---|---|
| `/api/health` | Kiểm tra trạng thái API, GPU và model đã load |
| `/api/detect` | Phát hiện tổn thương da bằng Faster R-CNN |
| `/api/segment` | Phân đoạn phổi X-quang |
| `/api/pipeline` | Chạy pipeline ISIC đầy đủ: detection + segmentation |

### 5.3 Frontend

Frontend sử dụng React và Vite. Các trang chính:

| Trang | Chức năng |
|---|---|
| Tổng quan | Hiển thị trạng thái API, GPU, model đã load, lịch sử gần đây |
| Phân tích | Upload ảnh và chọn workflow inference |
| So sánh | So sánh detection ISIC với pipeline ISIC đầy đủ trên cùng ảnh |
| Chỉ số | Hiển thị số liệu test, giải thích metric và xuất PDF |
| Lịch sử | Hiển thị các lượt phân tích đã chạy trong localStorage |

---

## 6. Huấn luyện mô hình

### 6.1 Checkpoint

Các checkpoint hiện có:

| Checkpoint | Ý nghĩa |
|---|---|
| `outputs/checkpoints/best_detection.pth` | Faster R-CNN phát hiện tổn thương da ISIC |
| `outputs/checkpoints/best_segmentation.pth` | Attention U-Net phân đoạn tổn thương da ISIC |
| `outputs/checkpoints/best_chest_xray_segmentation.pth` | Attention U-Net phân đoạn phổi X-quang |

### 6.2 Lệnh train

Train detection ISIC:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --config configs\detection_config.yaml
```

Train segmentation ISIC:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --config configs\segmentation_config.yaml
```

Train segmentation Chest X-ray:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --config configs\chest_xray_segmentation_config.yaml
```

---

## 7. Đánh giá thực nghiệm

### 7.1 Phân đoạn phổi X-quang

Mô hình: Attention U-Net  
Checkpoint: `outputs/checkpoints/best_chest_xray_segmentation.pth`  
Tập test: 22 ảnh Montgomery CXR  
CSV per-image: `outputs/metrics/chest_xray_test_per_image.csv`

| Chỉ số | Giá trị toàn test | Mean per-image | Std per-image |
|---|---:|---:|---:|
| Dice | 0.9811 | 0.9807 | 0.0115 |
| IoU | 0.9628 | 0.9623 | 0.0215 |
| Sensitivity | 0.9804 | 0.9807 | 0.0227 |
| Specificity | 0.9932 | 0.9931 | 0.0027 |
| Pixel Accuracy | 0.9897 | 0.9897 | 0.0064 |

### 7.2 Phân đoạn tổn thương da ISIC

Mô hình: Attention U-Net  
Checkpoint: `outputs/checkpoints/best_segmentation.pth`  
Tập test: 390 ảnh ISIC 2018  
CSV per-image: `outputs/metrics/isic2018_test_per_image.csv`

| Chỉ số | Giá trị toàn test | Mean per-image | Std per-image |
|---|---:|---:|---:|
| Dice | 0.8847 | 0.8707 | 0.1443 |
| IoU | 0.7932 | 0.7927 | 0.1728 |
| Sensitivity | 0.8611 | 0.9135 | 0.1495 |
| Specificity | 0.9774 | 0.9757 | 0.0377 |
| Pixel Accuracy | 0.9532 | 0.9532 | 0.0741 |

### 7.3 Phát hiện tổn thương da ISIC

Mô hình: Faster R-CNN  
Checkpoint: `outputs/checkpoints/best_detection.pth`  
Tập test: 390 ảnh ISIC 2018  
CSV per-image: `outputs/metrics/isic2018_detection_test_per_image.csv`

| Chỉ số | Giá trị |
|---|---:|
| mAP@0.5 | 0.7142 |
| Precision@0.5 | 0.8768 |
| Recall@0.5 | 0.7846 |
| TP | 306 |
| FP | 43 |
| FN | 84 |
| Best IoU mean | 0.6879 |
| Best IoU std | 0.3365 |

### 7.4 Lệnh đánh giá

Chest X-ray segmentation:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py `
  --model segmentation `
  --config configs\chest_xray_segmentation_config.yaml `
  --checkpoint outputs\checkpoints\best_chest_xray_segmentation.pth `
  --split test `
  --output-csv outputs\metrics\chest_xray_test_per_image.csv
```

ISIC segmentation:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py `
  --model segmentation `
  --config configs\segmentation_config.yaml `
  --checkpoint outputs\checkpoints\best_segmentation.pth `
  --split test `
  --output-csv outputs\metrics\isic2018_test_per_image.csv
```

ISIC detection:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py `
  --model detection `
  --config configs\detection_config.yaml `
  --checkpoint outputs\checkpoints\best_detection.pth `
  --split test `
  --output-csv outputs\metrics\isic2018_detection_test_per_image.csv
```

---

## 8. Giao diện và chức năng phần mềm

### 8.1 Dashboard

Trang tổng quan hiển thị số lượt phân tích, trạng thái GPU, số model đã load và trạng thái API. Trang này giúp kiểm tra nhanh backend có hoạt động và model có sẵn sàng hay không.

### 8.2 Analyze

Trang phân tích cho phép upload ảnh và chọn workflow:

- Phát hiện tổn thương da.
- Phân đoạn phổi X-quang.
- Pipeline ISIC đầy đủ.

Frontend đọc `/api/health` để biết workflow nào sẵn sàng, tránh hiển thị trạng thái sai so với backend.

### 8.3 Compare

Trang so sánh cho phép upload ảnh da liễu ISIC và chạy song song:

- Detection: chỉ vẽ bounding box.
- Full ISIC pipeline: vẽ bounding box và mask tổn thương.

Không so sánh trực tiếp ISIC với Chest X-ray vì hai bài toán khác loại ảnh, khác nhãn và khác mục tiêu y khoa.

### 8.4 Metrics

Trang chỉ số liệt kê các kết quả test thực tế, giải thích ý nghĩa metric và có nút xuất PDF bằng chức năng in của trình duyệt.

---

## 9. Hạn chế và hướng phát triển

### 9.1 Hạn chế

- Hệ thống chưa có database để lưu người dùng, ảnh upload và lịch sử phân tích lâu dài.
- Detection hiện tập trung vào ISIC, chưa train detection riêng cho Chest X-ray.
- Frontend lưu lịch sử phân tích bằng localStorage nên dữ liệu không đồng bộ giữa các thiết bị.
- Đánh giá detection suy ra bounding box ground truth từ mask tổn thương, phù hợp với dữ liệu hiện có nhưng chưa thay thế hoàn toàn annotation detection thủ công.
- Hệ thống chưa có xác thực người dùng và chưa triển khai production.

### 9.2 Hướng phát triển

- Thêm database để lưu lịch sử phân tích, kết quả inference và metadata ảnh.
- Bổ sung dashboard thống kê theo thời gian.
- Hỗ trợ định dạng DICOM cho ảnh y tế.
- Tách riêng model registry và quản lý version checkpoint.
- Huấn luyện thêm detection cho ảnh X-quang nếu bài toán yêu cầu phát hiện tổn thương bất thường.
- Triển khai Docker để dễ cài đặt và chạy trên máy khác.

---

## 10. Kết luận

Đề tài MedSeg đã xây dựng được một hệ thống phân tích ảnh y tế hoàn chỉnh ở mức prototype, bao gồm xử lý dữ liệu, huấn luyện mô hình, đánh giá định lượng, backend API và frontend trực quan. Kết quả thực nghiệm cho thấy mô hình phân đoạn phổi đạt độ chính xác cao trên tập test Montgomery CXR, trong khi mô hình ISIC đạt kết quả khả quan cho cả segmentation và detection.

Hệ thống có thể được sử dụng làm nền tảng tiếp tục phát triển các chức năng quản lý dữ liệu, lưu lịch sử phân tích, hỗ trợ nhiều định dạng ảnh y tế hơn và cải thiện khả năng triển khai thực tế.

---

## 11. Phụ lục

### 11.1 Chạy backend

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --reload --port 8000
```

API docs:

```text
http://localhost:8000/docs
```

### 11.2 Chạy frontend

```powershell
cd frontend
npm install
npm run dev
```

Frontend:

```text
http://localhost:5173
```

### 11.3 Chạy test source code

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

### 11.4 Các file quan trọng

| File/thư mục | Ý nghĩa |
|---|---|
| `src/data` | Download, preprocess, dataset, transform và split dữ liệu |
| `src/models` | Faster R-CNN, U-Net, Attention U-Net |
| `src/training` | Vòng lặp huấn luyện |
| `src/evaluation` | Metric đánh giá |
| `api` | Backend FastAPI |
| `frontend` | Frontend React |
| `outputs/checkpoints` | Checkpoint model đã train |
| `outputs/metrics` | CSV kết quả đánh giá per-image |

