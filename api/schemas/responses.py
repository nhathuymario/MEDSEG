"""Pydantic response schemas."""
from pydantic import BaseModel
from typing import List, Optional


class DetectionResult(BaseModel):
    boxes: List[List[float]]
    scores: List[float]
    labels: List[int]
    overlay_base64: Optional[str] = None
    inference_time_ms: float


class SegmentationResult(BaseModel):
    mask_base64: str
    overlay_base64: str
    dice_score: Optional[float] = None
    inference_time_ms: float


class PipelineResult(BaseModel):
    detection: DetectionResult
    segmentation_masks: List[str]
    combined_overlay_base64: str
    total_time_ms: float


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    models_loaded: List[str]
