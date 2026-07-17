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
    """Compute all-point interpolated AP as area under the PR envelope.

    Unlike the legacy VOC 2007 11-point approximation, this does not award a
    fixed 1/11 AP to a detector whose maximum recall is only a tiny fraction.
    """
    if not precisions or not recalls:
        return 0.0
    mrec = np.concatenate(([0.0], np.asarray(recalls, dtype=float), [1.0]))
    mpre = np.concatenate(([0.0], np.asarray(precisions, dtype=float), [0.0]))
    for index in range(mpre.size - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    changing_recall = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changing_recall + 1] - mrec[changing_recall]) * mpre[changing_recall + 1]))


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
