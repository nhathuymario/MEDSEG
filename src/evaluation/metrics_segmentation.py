"""Metric đánh giá segmentation."""
import numpy as np


def compute_segmentation_metrics(pred: np.ndarray, target: np.ndarray, threshold=0.5):
    """Tính Dice, IoU, sensitivity, specificity và pixel accuracy."""
    pred_bin = (pred > threshold).astype(np.float32).flatten()
    target = target.astype(np.float32).flatten()

    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()
    tn = ((1 - pred_bin) * (1 - target)).sum()

    # Các hằng số nhỏ tránh chia cho 0 khi mask rỗng.
    dice = (2 * tp + 1) / (2 * tp + fp + fn + 1)
    iou = (tp + 1) / (tp + fp + fn + 1)
    sensitivity = (tp + 1e-6) / (tp + fn + 1e-6)
    specificity = (tn + 1e-6) / (tn + fp + 1e-6)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "dice": float(dice),
        "iou": float(iou),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "pixel_accuracy": float(accuracy),
    }
