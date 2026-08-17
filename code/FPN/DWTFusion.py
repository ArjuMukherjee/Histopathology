import torch
import torch.nn as nn
import torch.nn.functional as F


class DWTEntropyFusion(nn.Module):
    """Entropy-guided spatial attention fusion for Wavelet high-frequency components."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.attn_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1)
        )

    def local_entropy(self, x):
        p = torch.abs(x)
        p = F.avg_pool2d(p, kernel_size=3, stride=1, padding=1)
        p = p / (p.sum(dim=(2, 3), keepdim=True) + 1e-8)
        ent = -p * torch.log(p + 1e-8)
        return torch.mean(ent, dim=1, keepdim=True)

    def forward(self, LH, HL, HH):
        # 1. Instance Normalization across subbands
        LH = (LH - LH.mean(dim=(2, 3), keepdim=True)) / (LH.std(dim=(2, 3), keepdim=True) + 1e-6)
        HL = (HL - HL.mean(dim=(2, 3), keepdim=True)) / (HL.std(dim=(2, 3), keepdim=True) + 1e-6)
        HH = (HH - HH.mean(dim=(2, 3), keepdim=True)) / (HH.std(dim=(2, 3), keepdim=True) + 1e-6)

        # 2. Local Entropy Prior
        e_lh = self.local_entropy(LH)
        e_hl = self.local_entropy(HL)
        e_hh = self.local_entropy(HH)
        entropy_weights = torch.cat([e_lh, e_hl, e_hh], dim=1)

        # 3. Learnable Attention
        feat = torch.cat([LH, HL, HH], dim=1)
        attn = self.attn_conv(feat)

        # 4. Softmax Attention Fusion
        weights = F.softmax(entropy_weights + attn, dim=1)
        F_out = weights[:, 0:1] * LH + weights[:, 1:2] * HL + weights[:, 2:3] * HH

        return self.conv_reduce(F_out)