"""Metric đánh giá detection: IoU và mAP."""
import numpy as np


def compute_iou(box1, box2):
    """Tính Intersection over Union cho hai bbox [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)


def compute_ap(precisions, recalls):
    """Tính Average Precision bằng nội suy 11 điểm."""
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        prec_at_rec = [p for p, r in zip(precisions, recalls) if r >= t]
        ap += max(prec_at_rec) if prec_at_rec else 0
    return ap / 11


def compute_map(predictions, ground_truths, iou_threshold=0.5):
    """Tính mAP tại một ngưỡng IoU.

    predictions: list of {'boxes': [[x1,y1,x2,y2]], 'scores': [float]}
    ground_truths: list of {'boxes': [[x1,y1,x2,y2]]}
    """
    all_scores, all_tp = [], []
    total_gt = sum(len(gt["boxes"]) for gt in ground_truths)

    for pred, gt in zip(predictions, ground_truths):
        gt_matched = [False] * len(gt["boxes"])
        # Xét prediction theo score giảm dần như quy trình tính AP chuẩn.
        sorted_idx = np.argsort(-np.array(pred.get("scores", [])))

        for i in sorted_idx:
            best_iou, best_j = 0, -1
            for j, gt_box in enumerate(gt["boxes"]):
                iou = compute_iou(pred["boxes"][i], gt_box)
                if iou > best_iou:
                    best_iou, best_j = iou, j

            all_scores.append(pred["scores"][i])
            if best_iou >= iou_threshold and not gt_matched[best_j]:
                all_tp.append(1)
                gt_matched[best_j] = True
            else:
                all_tp.append(0)

    if not all_scores:
        return 0.0

    order = np.argsort(-np.array(all_scores))
    tp_cumsum = np.cumsum(np.array(all_tp)[order])
    fp_cumsum = np.cumsum(1 - np.array(all_tp)[order])

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / (total_gt + 1e-6)

    return compute_ap(precisions.tolist(), recalls.tolist())
