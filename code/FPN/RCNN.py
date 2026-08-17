import torch.nn as nn
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from ModifiedFPN import ModifiedFPN
from backbone import build_backbone


class BackbonewithModifiedFPN(nn.Module):
    def __init__(self, backbone_name="resnet50", pretrained=True, fpn_out_channels=256):
        super().__init__()
        self.backbone = build_backbone(backbone_name, pretrained=pretrained)
        self.fpn = ModifiedFPN(in_channels=self.backbone.out_channels, out_channels=fpn_out_channels)
        self.out_channels = fpn_out_channels

    def forward(self, x):
        P2, P3, P4, P5, _ = self.fpn(*self.backbone(x))
        return {"0": P2, "1": P3, "2": P4, "3": P5}


def build_isg_maskrcnn(backbone_name="resnet50", num_classes=2, pretrained=True):
    """
    Constructs the auxiliary Instance Segmentation Guided (ISG) network using ModifiedFPN.
    """
    backbone_fpn = BackbonewithModifiedFPN(backbone_name=backbone_name, pretrained=pretrained)

    anchor_generator = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4
    )

    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=7,
        sampling_ratio=2
    )

    mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=14,
        sampling_ratio=2
    )

    model = MaskRCNN(
        backbone_fpn,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=box_roi_pool,
        mask_roi_pool=mask_roi_pool
    )
    return model