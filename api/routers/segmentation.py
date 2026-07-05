"""Segmentation endpoint."""
import time
from fastapi import APIRouter, UploadFile, File, HTTPException
from api.schemas.responses import SegmentationResult
from api.services.model_service import model_service
from src.pipeline.inference import run_segmentation
from src.evaluation.visualize import overlay_mask

router = APIRouter()


@router.post("/segment", response_model=SegmentationResult)
async def segment(file: UploadFile = File(...)):
    image = model_service.decode_upload(await file.read())
    segmentor = model_service.get_model("chest_segmentor")
    if segmentor is None:
        raise HTTPException(
            status_code=503,
            detail="Chest X-ray segmentation checkpoint is not loaded.",
        )

    t0 = time.time()
    model_input = model_service.preprocess_chest_xray(image)
    mask = run_segmentation(segmentor, model_input, str(model_service.device))
    overlay = overlay_mask(image, mask)

    return SegmentationResult(
        mask_base64=model_service.image_to_base64(mask * 255),
        overlay_base64=model_service.image_to_base64(overlay),
        inference_time_ms=(time.time() - t0) * 1000,
    )
