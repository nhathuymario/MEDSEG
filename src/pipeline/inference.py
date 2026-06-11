"""Single-model inference utilities."""
import torch
import cv2
import numpy as np
from torchvision import transforms as T


TRANSFORM = T.Compose([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


def run_detection(model, image: np.ndarray, device="cpu", threshold=0.5):
    """Run Faster R-CNN on a single image. Returns boxes, scores, labels."""
    model.eval().to(device)
    img_tensor = T.ToTensor()(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).to(device)

    with torch.no_grad():
        preds = model([img_tensor])[0]

    mask = preds["scores"] >= threshold
    return {
        "boxes": preds["boxes"][mask].cpu().numpy().tolist(),
        "scores": preds["scores"][mask].cpu().numpy().tolist(),
        "labels": preds["labels"][mask].cpu().numpy().tolist(),
    }


def run_segmentation(model, image: np.ndarray, device="cpu", size=256, threshold=0.5):
    """Run U-Net/Attention U-Net on a single image. Returns binary mask."""
    model.eval().to(device)
    h, w = image.shape[:2]
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    tensor = TRANSFORM(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(tensor)

    mask = torch.sigmoid(pred).squeeze().cpu().numpy()
    mask = cv2.resize(mask, (w, h))
    return (mask > threshold).astype(np.uint8)
