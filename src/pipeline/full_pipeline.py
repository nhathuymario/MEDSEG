"""Full pipeline: Detection → Segmentation."""
import cv2
import numpy as np
import time
from .inference import run_detection, run_segmentation
from src.evaluation.visualize import draw_boxes, overlay_mask


class MedSegPipeline:
    def __init__(self, detector, segmentor, device="cpu"):
        self.detector = detector
        self.segmentor = segmentor
        self.device = device

    def run(self, image: np.ndarray, det_thresh=0.5, seg_thresh=0.5):
        """Run full pipeline: detect → crop ROIs → segment each ROI."""
        t0 = time.time()

        # Step 1: Detection
        det = run_detection(self.detector, image, self.device, det_thresh)

        # Step 2: Segment each detected ROI
        seg_results = []
        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for box in det["boxes"]:
            x1, y1, x2, y2 = map(int, box)
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            mask = run_segmentation(self.segmentor, roi, self.device, threshold=seg_thresh)
            full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], mask)
            seg_results.append({"bbox": [x1, y1, x2, y2], "mask": mask})

        total_ms = (time.time() - t0) * 1000
        return {"detection": det, "segmentations": seg_results, "full_mask": full_mask, "time_ms": total_ms}

    def visualize(self, image, results):
        vis = draw_boxes(image, results["detection"]["boxes"], results["detection"]["scores"])
        vis = overlay_mask(vis, results["full_mask"])
        return vis
