"""Segmentation endpoint."""
import time
from fastapi import APIRouter, UploadFile, File
from api.schemas.responses import SegmentationResult
from api.services.model_service import model_service
from src.pipeline.inference import run_segmentation
from src.evaluation.visualize import overlay_mask

router = APIRouter()


@router.post("/segment", response_model=SegmentationResult)
async def segment(file: UploadFile = File(...)):
    image = model_service.decode_upload(await file.read())
    segmentor = model_service.get_model("segmentor")

    t0 = time.time()
    if segmentor:
        mask = run_segmentation(segmentor, image, str(model_service.device))
        overlay = overlay_mask(image, mask)
    else:
        import numpy as np
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        overlay = image

    return SegmentationResult(
        mask_base64=model_service.image_to_base64(mask * 255),
        overlay_base64=model_service.image_to_base64(overlay),
        inference_time_ms=(time.time() - t0) * 1000,
    )
