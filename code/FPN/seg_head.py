import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling.meta_arch.semantic_seg import SemSegFPNHead
from detectron2.layers import ShapeSpec

from backbone import ResNetBackbone
from ModifiedFPN import ModifiedFPN

class SegModel(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.backbone = ResNetBackbone()
        self.fpn = ModifiedFPN()

        # Detectron2 Semantic Head Configuration
        input_shape = {
            "p2": ShapeSpec(channels=256, stride=4),
            "p3": ShapeSpec(channels=256, stride=8),
            "p4": ShapeSpec(channels=256, stride=16),
            "p5": ShapeSpec(channels=256, stride=32),
        }

        self.sem_head = SemSegFPNHead(
            input_shape=input_shape,
            ignore_value=255,
            num_classes=num_classes,
            conv_dims=128,
            common_stride=4,
            norm="GN",
        )

    def forward(self, x):
        # Feature Extraction (ResNet Backbone)
        C2, C3, C4, C5 = self.backbone(x)
        
        # Multi-scale Fusion
        P2, P3, P4, P5 = self.fpn(C2, C3, C4, C5)

        features = {
            "p2": P2,
            "p3": P3,
            "p4": P4,
            "p5": P5,
        }

        seg = self.sem_head.layers(features)

        # Upsample to original input resolution (e.g., 512x512)
        seg = F.interpolate(seg, size=x.shape[2:], mode="bilinear", align_corners=False)
        
        return seg