import torch.nn as nn
import torch.nn.functional as F
from backbone import build_backbone
from ModifiedFPN import ModifiedFPN


class SegModel(nn.Module):
    """
    Generator S architecture: Multi-scale 1/4 resolution summation with Instance Normalization.
    """
    def __init__(self, backbone_name="resnet50", pretrained=True, fpn_out_channels=256, num_classes=1):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = build_backbone(backbone_name, pretrained=pretrained)
        self.fpn = ModifiedFPN(in_channels=self.backbone.out_channels, out_channels=fpn_out_channels)

        # DeblurGAN-v2 Style Feature Head with Instance Normalization
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(fpn_out_channels, fpn_out_channels // 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(fpn_out_channels // 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fpn_out_channels // 2, fpn_out_channels // 4, kernel_size=3, padding=1),
            nn.InstanceNorm2d(fpn_out_channels // 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 4x Progressive Upsampling
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(fpn_out_channels // 4, fpn_out_channels // 4, kernel_size=3, padding=1),
            nn.InstanceNorm2d(fpn_out_channels // 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(fpn_out_channels // 4, num_classes, kernel_size=3, padding=1),
        )

    def extract_features(self, x):
        C2, C3, C4, C5 = self.backbone(x)
        return self.fpn(C2, C3, C4, C5)

    def forward(self, x):
        P2, P3, P4, P5, P6 = self.extract_features(x)

        # Resize all 5 pyramid levels to 1/4 resolution (P2 dimension)
        target_size = P2.shape[2:]
        u3 = F.interpolate(P3, size=target_size, mode="bilinear", align_corners=False)
        u4 = F.interpolate(P4, size=target_size, mode="bilinear", align_corners=False)
        u5 = F.interpolate(P5, size=target_size, mode="bilinear", align_corners=False)
        u6 = F.interpolate(P6, size=target_size, mode="bilinear", align_corners=False)

        # Summation across scales
        fused = P2 + u3 + u4 + u5 + u6

        feat = self.decoder_conv(fused)
        out_mask = self.final_upsample(feat)
        return out_mask