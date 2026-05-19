from DWTFusion import DWTEntropyFusion
import torch.nn as nn
import torch
import torch.nn.functional as F

class ModifiedFPN(nn.Module):
    def __init__(self):
        super().__init__()

        # Lateral convs: Reduce backbone channels to a consistent 256
        self.lat2 = nn.Conv2d(256, 256, 1)
        self.lat3 = nn.Conv2d(512, 256, 1)
        self.lat4 = nn.Conv2d(1024, 256, 1)
        self.lat5 = nn.Conv2d(2048, 256, 1)

        # DWT fusion modules: 
        # Note: DWT doesn't change channel count, it changes spatial resolution.
        # These must accept the channel count of the backbone layer.
        self.dwt2_fuse = DWTEntropyFusion(256, 256)
        self.dwt3_fuse = DWTEntropyFusion(512, 256)
        self.dwt4_fuse = DWTEntropyFusion(1024, 256)

        # Haar Wavelet Filters
        self.register_buffer("ll", torch.tensor([[0.5, 0.5],[0.5, 0.5]]))
        self.register_buffer("lh", torch.tensor([[-0.5,-0.5],[0.5,0.5]]))
        self.register_buffer("hl", torch.tensor([[-0.5,0.5],[-0.5,0.5]]))
        self.register_buffer("hh", torch.tensor([[0.5,-0.5],[-0.5,0.5]]))

    def dwt(self, x):
        B, C, H, W = x.shape
        # Handle odd dimensions if any (though target_size 512 avoids this)
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, W % 2, 0, H % 2))
        
        filters = torch.stack([self.ll, self.lh, self.hl, self.hh], dim=0).unsqueeze(1)
        filters = filters.repeat(C, 1, 1, 1)
        
        # Grouped convolution to apply DWT per channel
        out = F.conv2d(x, filters, stride=2, groups=C)
        out = out.view(B, C, 4, out.shape[2], out.shape[3])

        # out[:, :, 0] is LL (Low-low, approximation)
        # We return the high-frequency components: LH, HL, HH
        return out[:, :, 1], out[:, :, 2], out[:, :, 3]

    def forward(self, C2, C3, C4, C5):
        # 1. Lateral Connections
        P5 = self.lat5(C5)
        P4_lat = self.lat4(C4)
        P3_lat = self.lat3(C3)
        P2_lat = self.lat2(C2)

        # 2. Top-Down Path (Standard FPN)
        # Using the exact target shape ensures no 1-pixel mismatches
        P4 = P4_lat + F.interpolate(P5, size=P4_lat.shape[2:], mode='nearest')
        P3 = P3_lat + F.interpolate(P4, size=P3_lat.shape[2:], mode='nearest')
        P2 = P2_lat + F.interpolate(P3, size=P2_lat.shape[2:], mode='nearest')

        # 3. DWT Branch: Extract High-Frequency Edge Information
        LH2, HL2, HH2 = self.dwt(C2)
        LH3, HL3, HH3 = self.dwt(C3)
        LH4, HL4, HH4 = self.dwt(C4)

        # 4. Fusion: DWT components -> 256 channel feature maps
        F2 = self.dwt2_fuse(LH2, HL2, HH2)
        F3 = self.dwt3_fuse(LH3, HL3, HH3)
        F4 = self.dwt4_fuse(LH4, HL4, HH4)

        # 5. Upsample Fusion maps to match P-level resolutions
        # DWT reduces size by half, so we upsample back
        F2 = F.interpolate(F2, size=P2.shape[2:], mode='bilinear', align_corners=False)
        F3 = F.interpolate(F3, size=P3.shape[2:], mode='bilinear', align_corners=False)
        F4 = F.interpolate(F4, size=P4.shape[2:], mode='bilinear', align_corners=False)

        # 6. Final Integration
        P2 = P2 + F2
        P3 = P3 + F3
        P4 = P4 + F4

        return P2, P3, P4, P5