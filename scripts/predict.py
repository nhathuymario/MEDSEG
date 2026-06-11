"""Prediction script for single image."""
import argparse
import cv2
import json
import torch
from src.models.faster_rcnn import LesionDetector
from src.models.attention_unet import AttentionUNet
from src.pipeline.full_pipeline import MedSegPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--pipeline", default="full")
    args = parser.parse_args()

    # Create dummy models for prediction
    detector = LesionDetector(num_classes=2, pretrained=False)
    segmentor = AttentionUNet()
    
    pipeline = MedSegPipeline(detector, segmentor)
    img = cv2.imread(args.image)
    if img is None:
        print(f"Failed to load image: {args.image}")
        return

    results = pipeline.run(img)
    vis = pipeline.visualize(img, results)
    
    cv2.imwrite("outputs/predictions/result.jpg", vis)
    print("Saved to outputs/predictions/result.jpg")


if __name__ == "__main__":
    import os
    os.makedirs("outputs/predictions", exist_ok=True)
    main()
