# MedSeg — Medical Lesion Detection & Segmentation

AI-powered medical image analysis system supporting **lesion detection** (Faster R-CNN) and **precise segmentation** (U-Net / Attention U-Net) on skin lesion and chest X-ray images.

## Architecture

```
React Frontend (Vite:5173) ←→ FastAPI Backend (:8000) ←→ PyTorch Models
```

**Pipeline Flow:**
```
Input Image → Faster R-CNN (detect ROIs) → Crop → Attention U-Net (segment) → Overlay Result
```

## Quick Start

### 1. Backend Setup
```bash
pip install -r requirements.txt
# Start API server
uvicorn api.main:app --reload --port 8000
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev    # → http://localhost:5173
```

### 3. Training
```bash
# Detection
python scripts/train.py --config configs/detection_config.yaml

# ISIC segmentation
python scripts/train.py --config configs/segmentation_config.yaml

# Chest X-ray lung segmentation
python -m src.data.preprocess --dataset chest_xray
python -c "from pathlib import Path; from src.data.split import create_splits; create_splits(Path('data/processed/chest_xray/images'), Path('data/splits/chest_xray'))"
python scripts/train.py --config configs/chest_xray_segmentation_config.yaml
```

### 4. Testing
```bash
pytest tests/ -v --cov=src
```

## Datasets

| Dataset | Task | Images |
|---------|------|--------|
| ISIC 2018 | Skin lesion segmentation + detection | 2,594 |
| Montgomery + Shenzhen | Chest X-ray lung segmentation | 800 |

## Models

| Model | Task | Key Metric |
|-------|------|------------|
| Faster R-CNN (ResNet-50-FPN) | Lesion Detection | mAP@0.5 |
| U-Net | Segmentation | Dice Score |
| Attention U-Net | Segmentation | Dice Score |

## Project Structure

```
MedSeg/
├── src/           # ML source code (data, models, training, evaluation, pipeline)
├── api/           # FastAPI backend (routers, schemas, services)
├── frontend/      # React frontend (Vite, pages, components)
├── tests/         # Unit & integration tests
├── configs/       # YAML configuration files
├── docs/          # Documentation
├── description/   # Project description & logs
├── notebooks/     # Jupyter analysis notebooks
├── scripts/       # CLI training/evaluation scripts
└── outputs/       # Checkpoints, logs, predictions
```

## Tech Stack
- **ML**: PyTorch, torchvision, Albumentations
- **Backend**: FastAPI, Pydantic, uvicorn
- **Frontend**: React 18, Vite, React Router, Vanilla CSS
- **Testing**: pytest, httpx

## License
MIT
