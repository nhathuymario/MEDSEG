"""Full pipeline endpoint: Detection → Segmentation."""
import time
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from api.schemas.responses import PipelineResult, DetectionResult
from api.services.model_service import model_service

router = APIRouter()


@router.post("/pipeline", response_model=PipelineResult)
async def run_pipeline(
    file: UploadFile = File(...),
    threshold: float = Query(default=0.5, ge=0.05, le=1.0, description="Detection confidence threshold")
):
    image = model_service.decode_upload(await file.read())

    t0 = time.time()
    detector = model_service.get_model("detector")
    segmentor = model_service.get_model("isic_segmentor")
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Full pipeline is unavailable because the ISIC Faster R-CNN "
                "checkpoint has not been trained/loaded."
            ),
        )
    if segmentor is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Full pipeline is unavailable because the ISIC segmentation "
                "checkpoint has not been trained/loaded."
            ),
        )

    from src.pipeline.full_pipeline import MedSegPipeline

    pipeline = MedSegPipeline(detector, segmentor, str(model_service.device))
    result = pipeline.run(image, det_thresh=threshold, seg_thresh=0.8, roi_margin=0.1)
    det = result["detection"]
    masks_b64 = [
        model_service.image_to_base64(item["mask"] * 255)
        for item in result["segmentations"]
    ]
    vis = pipeline.visualize(image, result)

    return PipelineResult(
        detection=DetectionResult(boxes=det["boxes"], scores=det["scores"], labels=det["labels"], inference_time_ms=0),
        segmentation_masks=masks_b64,
        combined_overlay_base64=model_service.image_to_base64(vis),
        total_time_ms=(time.time() - t0) * 1000,
    )
