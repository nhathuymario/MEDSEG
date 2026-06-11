"""Detection endpoint."""
import time
from fastapi import APIRouter, UploadFile, File
from api.schemas.responses import DetectionResult
from api.services.model_service import model_service
from src.pipeline.inference import run_detection

router = APIRouter()


@router.post("/detect", response_model=DetectionResult)
async def detect(file: UploadFile = File(...)):
    image = model_service.decode_upload(await file.read())
    detector = model_service.get_model("detector")

    t0 = time.time()
    if detector:
        result = run_detection(detector, image, str(model_service.device))
    else:
        result = {"boxes": [], "scores": [], "labels": []}

    return DetectionResult(
        boxes=result["boxes"],
        scores=result["scores"],
        labels=result["labels"],
        inference_time_ms=(time.time() - t0) * 1000,
    )
