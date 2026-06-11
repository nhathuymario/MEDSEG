"""Main evaluation script."""
import argparse
import yaml
import torch
from src.models.faster_rcnn import LesionDetector
from src.models.attention_unet import AttentionUNet
from src.evaluation.metrics_detection import compute_map
from src.evaluation.metrics_segmentation import compute_segmentation_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["detection", "segmentation"])
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating {args.model} on {device}...")

    if args.model == "detection":
        model = LesionDetector(num_classes=2, pretrained=False)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval().to(device)
        print("Model loaded. (Evaluation logic placeholder)")
        # TODO: Implement full dataset loop for map
    else:
        model = AttentionUNet()
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval().to(device)
        print("Model loaded. (Evaluation logic placeholder)")
        # TODO: Implement full dataset loop for dice, iou


if __name__ == "__main__":
    main()
