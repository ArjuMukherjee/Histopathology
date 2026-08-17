import torch
import torch.nn as nn
import torch.nn.functional as F
from DWTFusion import DWTEntropyFusion


class ModifiedFPN(nn.Module):
    """
    FPN with Group Normalization (32 groups), DWT Entropy Fusion, and P6 level.
    Matches the UNITPathSSL feature extraction backbone specification.
    """
    def __init__(self, in_channels=(256, 512, 1024, 2048), out_channels=256, num_groups=32):
        super().__init__()

        if len(in_channels) != 4:
            raise ValueError("ModifiedFPN requires exactly 4 backbone feature channels")

        # Lateral 1x1 Convolutions with Group Normalization
        self.lat2 = nn.Sequential(
            nn.Conv2d(in_channels[0], out_channels, kernel_size=1),
            nn.GroupNorm(num_groups, out_channels)
        )
        self.lat3 = nn.Sequential(
            nn.Conv2d(in_channels[1], out_channels, kernel_size=1),
            nn.GroupNorm(num_groups, out_channels)
        )
        self.lat4 = nn.Sequential(
            nn.Conv2d(in_channels[2], out_channels, kernel_size=1),
            nn.GroupNorm(num_groups, out_channels)
        )
        self.lat5 = nn.Sequential(
            nn.Conv2d(in_channels[3], out_channels, kernel_size=1),
            nn.GroupNorm(num_groups, out_channels)
        )

        # 3x3 Smooth Convolutions
        self.smooth2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.smooth3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.smooth4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.smooth5 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.p6 = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)

        # High-Frequency Wavelet Fusion Modules
        self.dwt2_fuse = DWTEntropyFusion(in_channels[0], out_channels)
        self.dwt3_fuse = DWTEntropyFusion(in_channels[1], out_channels)
        self.dwt4_fuse = DWTEntropyFusion(in_channels[2], out_channels)

        # Haar Filters
        self.register_buffer("ll", torch.tensor([[0.5, 0.5], [0.5, 0.5]]))
        self.register_buffer("lh", torch.tensor([[-0.5, -0.5], [0.5, 0.5]]))
        self.register_buffer("hl", torch.tensor([[-0.5, 0.5], [-0.5, 0.5]]))
        self.register_buffer("hh", torch.tensor([[0.5, -0.5], [-0.5, 0.5]]))

    def dwt(self, x):
        B, C, H, W = x.shape
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, W % 2, 0, H % 2))
        filters = torch.stack([self.ll, self.lh, self.hl, self.hh], dim=0).unsqueeze(1)
        filters = filters.repeat(C, 1, 1, 1)
        out = F.conv2d(x, filters, stride=2, groups=C)
        out = out.view(B, C, 4, out.shape[2], out.shape[3])
        return out[:, :, 1], out[:, :, 2], out[:, :, 3]

    def forward(self, C2, C3, C4, C5):
        P5_lat = self.lat5(C5)
        P4_lat = self.lat4(C4)
        P3_lat = self.lat3(C3)
        P2_lat = self.lat2(C2)

        P4 = P4_lat + F.interpolate(P5_lat, size=P4_lat.shape[2:], mode='nearest')
        P3 = P3_lat + F.interpolate(P4, size=P3_lat.shape[2:], mode='nearest')
        P2 = P2_lat + F.interpolate(P3, size=P2_lat.shape[2:], mode='nearest')

        LH2, HL2, HH2 = self.dwt(C2)
        LH3, HL3, HH3 = self.dwt(C3)
        LH4, HL4, HH4 = self.dwt(C4)

        F2 = F.interpolate(self.dwt2_fuse(LH2, HL2, HH2), size=P2.shape[2:], mode='bilinear', align_corners=False)
        F3 = F.interpolate(self.dwt3_fuse(LH3, HL3, HH3), size=P3.shape[2:], mode='bilinear', align_corners=False)
        F4 = F.interpolate(self.dwt4_fuse(LH4, HL4, HH4), size=P4.shape[2:], mode='bilinear', align_corners=False)

        P2 = self.smooth2(P2 + F2)
        P3 = self.smooth3(P3 + F3)
        P4 = self.smooth4(P4 + F4)
        P5 = self.smooth5(P5_lat)
        P6 = self.p6(P5)

        return P2, P3, P4, P5, P6