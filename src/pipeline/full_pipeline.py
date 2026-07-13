"""Pipeline đầy đủ: Detection -> crop ROI -> Segmentation."""
import cv2
import numpy as np
import time
from .inference import run_detection, run_segmentation
from src.evaluation.visualize import draw_boxes, overlay_mask


class MedSegPipeline:
    """Kết hợp detector và segmentor để phân đoạn trong vùng được phát hiện."""

    def __init__(self, detector, segmentor, device="cpu"):
        self.detector = detector
        self.segmentor = segmentor
        self.device = device

    @staticmethod
    def expand_and_clip_box(box, image_shape, margin=0.1):
        """Add ROI context and guarantee valid coordinates for any image size."""
        height, width = image_shape[:2]
        x1, y1, x2, y2 = map(float, box)
        box_width = max(1.0, x2 - x1)
        box_height = max(1.0, y2 - y1)
        x_margin = box_width * margin
        y_margin = box_height * margin
        return [
            max(0, int(np.floor(x1 - x_margin))),
            max(0, int(np.floor(y1 - y_margin))),
            min(width, int(np.ceil(x2 + x_margin))),
            min(height, int(np.ceil(y2 + y_margin))),
        ]

    def run(self, image: np.ndarray, det_thresh=0.8, seg_thresh=0.8, roi_margin=0.1):
        """Chạy pipeline: phát hiện ROI, crop từng ROI, rồi segment từng vùng."""
        t0 = time.time()

        # Bước 1: phát hiện các bbox nghi ngờ trên ảnh đầy đủ.
        det = run_detection(self.detector, image, self.device, det_thresh)

        # Bước 2: chạy segmentation trên từng ROI và ghép về full-size mask.
        seg_results = []
        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for box in det["boxes"]:
            x1, y1, x2, y2 = self.expand_and_clip_box(
                box, image.shape, roi_margin
            )
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            mask = run_segmentation(self.segmentor, roi, self.device, threshold=seg_thresh)
            full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], mask)
            seg_results.append({"bbox": [x1, y1, x2, y2], "mask": mask})

        total_ms = (time.time() - t0) * 1000
        return {
            "detection": det,
            "segmentations": seg_results,
            "full_mask": full_mask,
            "time_ms": total_ms,
        }

    def visualize(self, image, results):
        """Overlay bbox và mask cuối cùng lên ảnh để kiểm tra trực quan."""
        vis = draw_boxes(
            image,
            results["detection"]["boxes"],
            results["detection"]["scores"],
        )
        vis = overlay_mask(vis, results["full_mask"])
        return vis
