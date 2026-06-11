"""Faster R-CNN wrapper for lesion detection."""
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class LesionDetector(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        self.model = fasterrcnn_resnet50_fpn(weights=weights)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, images, targets=None):
        return self.model(images, targets)

    @staticmethod
    def from_checkpoint(path, num_classes=2):
        import torch
        model = LesionDetector(num_classes, pretrained=False)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model
