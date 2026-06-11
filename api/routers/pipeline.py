"""Full pipeline endpoint: Detection → Segmentation."""
import time
from fastapi import APIRouter, UploadFile, File
from api.schemas.responses import PipelineResult, DetectionResult
from api.services.model_service import model_service

router = APIRouter()


@router.post("/pipeline", response_model=PipelineResult)
async def run_pipeline(file: UploadFile = File(...)):
    image = model_service.decode_upload(await file.read())

    t0 = time.time()
    detector = model_service.get_model("detector")
    segmentor = model_service.get_model("segmentor")

    # Detection
    from src.pipeline.inference import run_detection, run_segmentation
    from src.evaluation.visualize import draw_boxes, overlay_mask
    import numpy as np

    det = run_detection(detector, image, str(model_service.device)) if detector else {"boxes": [], "scores": [], "labels": []}

    # Segmentation per ROI
    masks_b64 = []
    full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for box in det["boxes"]:
        x1, y1, x2, y2 = map(int, box)
        roi = image[y1:y2, x1:x2]
        if roi.size == 0 or segmentor is None:
            continue
        mask = run_segmentation(segmentor, roi, str(model_service.device))
        full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], mask)
        masks_b64.append(model_service.image_to_base64(mask * 255))

    vis = draw_boxes(image, det["boxes"], det["scores"]) if det["boxes"] else image
    vis = overlay_mask(vis, full_mask)

    return PipelineResult(
        detection=DetectionResult(boxes=det["boxes"], scores=det["scores"], labels=det["labels"], inference_time_ms=0),
        segmentation_masks=masks_b64,
        combined_overlay_base64=model_service.image_to_base64(vis),
        total_time_ms=(time.time() - t0) * 1000,
    )
