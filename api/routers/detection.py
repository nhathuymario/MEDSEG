"""Detection endpoint."""
import time
from fastapi import APIRouter, UploadFile, File, HTTPException
from api.schemas.responses import DetectionResult
from api.services.model_service import model_service
from src.pipeline.inference import run_detection
from src.evaluation.visualize import draw_boxes

router = APIRouter()


@router.post("/detect", response_model=DetectionResult)
async def detect(file: UploadFile = File(...)):
    image = model_service.decode_upload(await file.read())
    detector = model_service.get_model("detector")
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "ISIC Faster R-CNN checkpoint is not loaded. "
                "Train the detection model before using this endpoint."
            ),
        )

    t0 = time.time()
    result = run_detection(detector, image, str(model_service.device))
    overlay = draw_boxes(image, result["boxes"], result["scores"])

    return DetectionResult(
        boxes=result["boxes"],
        scores=result["scores"],
        labels=result["labels"],
        overlay_base64=model_service.image_to_base64(overlay),
        inference_time_ms=(time.time() - t0) * 1000,
    )
