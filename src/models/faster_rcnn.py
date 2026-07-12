"""Wrapper Faster R-CNN cho bài toán phát hiện vùng tổn thương/ROI."""
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class LesionDetector(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        # num_classes gồm background + số lớp foreground.
        # Với 1 lớp tổn thương: background=0, lesion=1 => num_classes=2.
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        self.model = fasterrcnn_resnet50_fpn(
            weights=weights,
            weights_backbone=None if not pretrained else None,
        )
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # Thay classification head mặc định của COCO bằng head cho số lớp của dự án.
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, images, targets=None):
        return self.model(images, targets)

    @staticmethod
    def from_checkpoint(path, num_classes=2):
        import torch
        model = LesionDetector(num_classes, pretrained=False)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model
