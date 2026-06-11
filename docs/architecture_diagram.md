# Kiến Trúc và Tổ Chức Dữ Liệu - MedSeg

Dưới đây là các sơ đồ minh họa cách hệ thống MedSeg hoạt động và cách dữ liệu được tổ chức.

## 1. Kiến Trúc Hệ Thống Tổng Thể (System Architecture)

Hệ thống được chia làm 3 lớp chính: Frontend (Giao diện), Backend API (Xử lý request), và ML Models (Xử lý AI).

```mermaid
graph TD
    subgraph Client [Frontend - React/Vite]
        UI[Giao diện Người Dùng]
        API_Client[API Client Fetch]
    end

    subgraph Server [Backend - FastAPI]
        Router[API Routers]
        Model_Service[Model Service Singleton]
        Pipeline[Inference Pipeline]
    end

    subgraph AI [Machine Learning - PyTorch]
        FasterRCNN[Faster R-CNN - Detection]
        AttnUNet[Attention U-Net - Segmentation]
    end

    UI <-->|Tải ảnh & Xem kết quả| API_Client
    API_Client <-->|REST API JSON/Base64| Router
    Router --> Model_Service
    Model_Service --> Pipeline
    Pipeline --> FasterRCNN
    Pipeline --> AttnUNet
```

---

## 2. Luồng Phân Tích (Inference Pipeline Flow)

Đây là cách một bức ảnh y tế đi qua hệ thống AI từ khi được người dùng upload cho đến khi trả về kết quả cuối cùng trên màn hình.

```mermaid
flowchart TD
    Input([Ảnh Y Tế Đầu Vào]) --> Preprocess[Tiền xử lý Resize / Normalize]
    Preprocess --> Detection{Faster R-CNN}
    
    Detection -->|Output: Bounding Boxes| CropROI[Cắt vùng ROI theo BBox]
    
    CropROI --> Segmentation{Attention U-Net}
    
    Segmentation -->|Output: Binary Masks| MapBack[Ghép Mask vào tọa độ ảnh gốc]
    
    MapBack --> Combine[Overlay Mask & BBox lên ảnh gốc]
    Combine --> Base64[Encode sang Base64]
    
    Base64 --> Output([Trả về Frontend hiển thị])
```

---

## 3. Cấu Trúc và Tổ Chức Dữ Liệu (Data Organization)

Quá trình luân chuyển dữ liệu từ lúc tải về cho đến khi được đưa vào Dataset để training.

```mermaid
graph LR
    subgraph Internet [Nguồn Dữ Liệu]
        ISIC[ISIC 2018 Skin Lesion]
        CXR[Chest X-ray NLM]
    end

    subgraph Raw [data/raw/]
        R_ISIC[Zip/Images/Masks gốc]
    end

    subgraph Processed [data/processed/]
        P_Img[images: 256x256 / 512x512]
        P_Mask[masks: Binary PNG]
        P_Anno[annotations: COCO JSON]
    end

    subgraph Splits [data/splits/]
        CSV[train.csv / val.csv / test.csv]
    end

    subgraph PyTorch [PyTorch Datasets]
        Det_DS[DetectionDataset]
        Seg_DS[SegmentationDataset]
    end

    ISIC -->|download.py| Raw
    CXR -->|download.py| Raw
    
    Raw -->|preprocess.py \n- Resize \n- Binarize mask \n- Extract BBox| Processed
    
    Processed -->|split.py \n- Stratified 70/15/15| Splits
    
    Processed --> Det_DS
    Splits --> Det_DS
    
    Processed --> Seg_DS
    Splits --> Seg_DS
```

### Giải thích Tổ chức Data:
1. **`data/raw/`**: Lưu trữ nguyên bản các file zip hoặc ảnh vừa tải về từ nguồn (Kaggle, NLM). Không bao giờ chỉnh sửa trực tiếp trên file này.
2. **`data/processed/`**: Dữ liệu đã qua tiền xử lý bởi `preprocess.py` (đồng nhất kích thước, chuyển mask về ảnh nhị phân đen/trắng, tạo file JSON format COCO chứa bounding boxes để train Faster R-CNN).
3. **`data/splits/`**: Chứa các file CSV lưu danh sách tên file thuộc tập `train`, `val`, `test` để đảm bảo mỗi lần train data chia giống hệt nhau (Reproducibility).
4. **`PyTorch Datasets`**: Lúc train, `DetectionDataset` sẽ đọc ảnh và file COCO JSON, còn `SegmentationDataset` sẽ đọc ảnh và file Mask (PNG).
