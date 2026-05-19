import torch
import torch.nn as nn
import torch.nn.functional as F

class DWTEntropyFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Maps the fused high-frequency components to the FPN's 256 channels
        self.conv_reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Learns a spatial attention map for the 3 DWT components
        self.attn_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1)
        )

    def local_entropy(self, x):
        """
        Calculates local entropy using a sliding window to maintain spatial info.
        This helps identify specific high-info areas like nuclei boundaries.
        """
        # Square the signal to represent energy/probability density
        p = torch.abs(x)
        # Smooth locally to create a probability distribution
        p = F.avg_pool2d(p, kernel_size=3, stride=1, padding=1)
        p = p / (p.sum(dim=(2, 3), keepdim=True) + 1e-8)
        
        # Entropy: -sum(p * log(p))
        ent = -p * torch.log(p + 1e-8)
        # Reduce channels to 1 to get a per-pixel 'importance' score
        return torch.mean(ent, dim=1, keepdim=True)

    def forward(self, LH, HL, HH):
        # 1. Instance Normalization (Helps stabilize training across different slides)
        LH = (LH - LH.mean(dim=(2,3), keepdim=True)) / (LH.std(dim=(2,3), keepdim=True) + 1e-6)
        HL = (HL - HL.mean(dim=(2,3), keepdim=True)) / (HL.std(dim=(2,3), keepdim=True) + 1e-6)
        HH = (HH - HH.mean(dim=(2,3), keepdim=True)) / (HH.std(dim=(2,3), keepdim=True) + 1e-6)

        # 2. Entropy Prior (Spatial Importance)
        # Each has shape [B, 1, H, W]
        e_lh = self.local_entropy(LH)
        e_hl = self.local_entropy(HL)
        e_hh = self.local_entropy(HH)
        entropy_weights = torch.cat([e_lh, e_hl, e_hh], dim=1) # [B, 3, H, W]

        # 3. Learnable Attention
        feat = torch.cat([LH, HL, HH], dim=1) # [B, C*3, H, W]
        attn = self.attn_conv(feat) # [B, 3, H, W]

        # 4. Softmax Weighting
        # Combining theoretical entropy with learned spatial attention
        weights = F.softmax(entropy_weights + attn, dim=1) # [B, 3, H, W]

        # 5. Fused Weighted Sum
        # weights[:, 0:1] is [B, 1, H, W], which broadcasts across LH's channels
        F_out = weights[:, 0:1] * LH + \
                weights[:, 1:2] * HL + \
                weights[:, 2:3] * HH

        # 6. Channel Reduction to FPN dimension (256)
        return self.conv_reduce(F_out)