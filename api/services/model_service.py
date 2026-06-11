"""Model loading & inference service (singleton)."""
import torch
import time
import base64
import io
import cv2
import numpy as np
from PIL import Image
from pathlib import Path


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
        if checkpoint_path and Path(checkpoint_path).exists():
            model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.eval().to(self.device)
        self.models[name] = model

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


model_service = ModelService()
