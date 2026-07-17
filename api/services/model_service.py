"""Model loading & inference service (singleton)."""
import os
import torch
import base64
import io
import cv2
import numpy as np
from PIL import Image
from pathlib import Path


def _first_existing(*paths):
    """Return the first existing artifact path during output-folder migration."""
    return next((str(Path(path)) for path in paths if Path(path).exists()), str(Path(paths[0])))


class ModelService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self._initialized = True

    def load_model(self, name: str, model, checkpoint_path: str = None):
        if checkpoint_path:
            path = Path(checkpoint_path)
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval().to(self.device)
        self.models[name] = model

    def load_default_models(self):
        from src.models.attention_unet import AttentionUNet
        from src.models.faster_rcnn import LesionDetector

        chest_checkpoint = os.getenv(
            "CHEST_XRAY_CHECKPOINT",
            _first_existing(
                "outputs/segmentation/chest_xray/checkpoints/best_chest_xray_segmentation.pth",
                "outputs/checkpoints/best_chest_xray_segmentation.pth",
            ),
        )
        if (
            Path(chest_checkpoint).exists()
            and "chest_segmentor" not in self.models
        ):
            model = AttentionUNet(
                in_ch=3,
                out_ch=1,
                features=(32, 64, 128, 256),
            )
            self.load_model("chest_segmentor", model, chest_checkpoint)

        multidomain_checkpoint = _first_existing(
            "outputs/detection/checkpoints/best_detection_multidomain.pth",
            "outputs/checkpoints/best_detection_multidomain.pth",
        )
        detection_checkpoint = os.getenv(
            "ISIC_DETECTION_CHECKPOINT",
            multidomain_checkpoint
            if Path(multidomain_checkpoint).exists()
            else _first_existing(
                "outputs/detection/checkpoints/best_detection.pth",
                "outputs/checkpoints/best_detection.pth",
            ),
        )
        if Path(detection_checkpoint).exists() and "detector" not in self.models:
            detector = LesionDetector(num_classes=2, pretrained=False)
            self.load_model("detector", detector, detection_checkpoint)

        isic_segmentation_checkpoint = os.getenv(
            "ISIC_SEGMENTATION_CHECKPOINT",
            _first_existing(
                "outputs/segmentation/skin/checkpoints/best_segmentation.pth",
                "outputs/checkpoints/best_segmentation.pth",
            ),
        )
        if (
            Path(isic_segmentation_checkpoint).exists()
            and "isic_segmentor" not in self.models
        ):
            segmentor = AttentionUNet(
                in_ch=3,
                out_ch=1,
                features=(64, 128, 256, 512),
            )
            self.load_model(
                "isic_segmentor",
                segmentor,
                isic_segmentation_checkpoint,
            )

    def get_model(self, name: str):
        return self.models.get(name)

    @property
    def loaded_models(self):
        return list(self.models.keys())

    @staticmethod
    def image_to_base64(image: np.ndarray) -> str:
        _, buffer = cv2.imencode(".png", image)
        return base64.b64encode(buffer).decode("utf-8")

    @staticmethod
    def decode_upload(file_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    @staticmethod
    def preprocess_chest_xray(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


model_service = ModelService()
