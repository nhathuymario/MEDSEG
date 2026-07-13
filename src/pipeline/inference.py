"""Hàm inference cho từng model riêng lẻ."""
import torch
import cv2
import numpy as np
from torchvision import transforms as T


TRANSFORM = T.Compose([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
IMAGENET_MEAN_RGB = (123, 116, 103)


def letterbox_image(image: np.ndarray, size: int, fill=IMAGENET_MEAN_RGB):
    """Resize without distortion and pad to a square canvas.

    Returns the padded image and geometry needed to remove padding from a
    prediction. This makes inference consistent across portrait, landscape,
    small and large inputs while preserving the original aspect ratio.
    """
    height, width = image.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("Input image must have non-zero width and height")
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    pad_left = (size - resized_width) // 2
    pad_top = (size - resized_height) // 2
    canvas = np.empty((size, size, image.shape[2]), dtype=image.dtype)
    canvas[...] = np.asarray(fill, dtype=image.dtype)
    canvas[
        pad_top : pad_top + resized_height,
        pad_left : pad_left + resized_width,
    ] = resized
    return canvas, {
        "original_height": height,
        "original_width": width,
        "resized_height": resized_height,
        "resized_width": resized_width,
        "pad_top": pad_top,
        "pad_left": pad_left,
    }


def run_detection(model, image: np.ndarray, device="cpu", threshold=0.8):
    """Chạy Faster R-CNN trên một ảnh và trả về boxes/scores/labels."""
    model.eval().to(device)
    img_tensor = T.ToTensor()(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).to(device)

    with torch.no_grad():
        preds = model([img_tensor])[0]

    # Chỉ giữ prediction có confidence score vượt ngưỡng.
    mask = preds["scores"] >= threshold
    return {
        "boxes": preds["boxes"][mask].cpu().numpy().tolist(),
        "scores": preds["scores"][mask].cpu().numpy().tolist(),
        "labels": preds["labels"][mask].cpu().numpy().tolist(),
    }


def run_segmentation(model, image: np.ndarray, device="cpu", size=256, threshold=0.8):
    """Chạy U-Net/Attention U-Net trên một ảnh và trả về mask nhị phân."""
    model.eval().to(device)
    h, w = image.shape[:2]
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    padded, geometry = letterbox_image(img, size)
    tensor = TRANSFORM(padded).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(tensor)

    mask = torch.sigmoid(pred).squeeze().cpu().numpy()
    # Bỏ padding trước khi đưa mask về kích thước ảnh gốc. Nếu resize thẳng
    # canvas vuông về ảnh gốc, biên tổn thương sẽ bị kéo giãn.
    output_height, output_width = mask.shape[-2:]
    scale_y = output_height / size
    scale_x = output_width / size
    y1 = round(geometry["pad_top"] * scale_y)
    x1 = round(geometry["pad_left"] * scale_x)
    y2 = round((geometry["pad_top"] + geometry["resized_height"]) * scale_y)
    x2 = round((geometry["pad_left"] + geometry["resized_width"]) * scale_x)
    mask = mask[y1:y2, x1:x2]
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    return (mask > threshold).astype(np.uint8)
