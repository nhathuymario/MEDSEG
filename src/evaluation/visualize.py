"""Visualization: overlay bboxes and masks on images."""
import cv2
import numpy as np


def draw_boxes(image, boxes, scores=None, color=(0, 255, 0), thickness=2):
    img = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if scores is not None:
            cv2.putText(img, f"{scores[i]:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def overlay_mask(image, mask, color=(0, 128, 255), alpha=0.4):
    img = image.copy()
    colored = np.zeros_like(img)
    colored[mask > 0.5] = color
    return cv2.addWeighted(img, 1, colored, alpha, 0)


def side_by_side(image, mask_pred, mask_gt=None, size=(512, 512)):
    img = cv2.resize(image, size)
    pred_overlay = overlay_mask(img, cv2.resize(mask_pred, size))
    panels = [img, pred_overlay]
    if mask_gt is not None:
        gt_overlay = overlay_mask(img, cv2.resize(mask_gt, size), color=(0, 255, 0))
        panels.append(gt_overlay)
    return np.hstack(panels)
