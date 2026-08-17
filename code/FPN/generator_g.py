import torch
import torch.nn as nn
import torch.nn.functional as F


class ModulatedConv2d(nn.Module):
    """
    Weight demodulation convolutional layer (StyleGAN2 / CoModGAN).
    Modulates convolution weights per sample with the joint style vector `s`.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, style_dim=512):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.affine = nn.Linear(style_dim, in_channels)

    def forward(self, x, style):
        B, C, H, W = x.shape
        # Compute per-sample style scaling factor
        s = self.affine(style).view(B, 1, self.in_channels, 1, 1) + 1.0

        # Weight modulation: [B, out_c, in_c, k, k]
        w = self.weight.unsqueeze(0) * s

        # Weight demodulation normalization
        d = torch.rsqrt(w.pow(2).sum(dim=[2, 3, 4], keepdim=True) + 1e-8)
        w_demod = w * d

        # Batched grouped convolution
        x = x.view(1, B * C, H, W)
        w_demod = w_demod.view(
            B * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
        )
        out = F.conv2d(x, w_demod, padding=self.padding, groups=B)
        return out.view(B, self.out_channels, H, W)


class CoModulatedGenerator(nn.Module):
    """
    Co-modulated Generator G: Pseudo Mask (x) + Style Noise (z) -> Synthetic Histopathology Image (y_hat).
    Architecture:
      - Conditioning Encoder E (Downsamples mask x to 4x4 spatial feature map and style vector E(x))
      - Mapping Network M (Transforms noise z ~ N(0, I) to stochastic style code w)
      - Affine Co-Modulation Fusion: s = A(E(x), M(z))
      - Synthesis Network D with weight-demodulated convolutions and multi-scale skip connections from E
    """
    def __init__(self, in_channels=1, out_channels=3, z_dim=512, style_dim=512):
        super().__init__()
        self.z_dim = z_dim
        self.style_dim = style_dim

        # 1. Stochastic Style Mapping Network M: z -> w (8-layer MLP)
        self.mapping_net = nn.Sequential(
            nn.Linear(z_dim, style_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 2. Conditioning Encoder E (Mask x -> multi-scale feature maps + E(x))
        self.enc1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.down1 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)   # e2: 256 -> 128 (128 ch)
        self.down2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # e3: 128 -> 64  (256 ch)
        self.down3 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)  # e4: 64  -> 32  (512 ch)
        self.down4 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)  # e5: 32  -> 16  (512 ch)
        self.down5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)  # e6: 16  -> 8   (512 ch)
        self.down6 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)  # e7: 8   -> 4   (512 ch)

        self.enc_fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512 * 4 * 4, style_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Initial 4x4 spatial feature map projection for D
        self.init_proj = nn.Linear(style_dim, 512 * 4 * 4)

        # Co-modulation Affine Transformation: [E(x), w] -> style vector s
        self.fusion = nn.Sequential(
            nn.Linear(style_dim * 2, style_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 3. Synthesis Network D with Modulated Convolutions & Skip Connections from E
        self.mod1 = ModulatedConv2d(512, 512, 3, style_dim)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.skip1 = nn.Conv2d(512, 512, kernel_size=1)

        self.mod2 = ModulatedConv2d(512, 512, 3, style_dim)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.skip2 = nn.Conv2d(512, 512, kernel_size=1)

        self.mod3 = ModulatedConv2d(512, 256, 3, style_dim)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.skip3 = nn.Conv2d(512, 256, kernel_size=1)

        self.mod4 = ModulatedConv2d(256, 128, 3, style_dim)
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.skip4 = nn.Conv2d(256, 128, kernel_size=1)

        self.mod5 = ModulatedConv2d(128, 64, 3, style_dim)
        self.up5 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.skip5 = nn.Conv2d(128, 64, kernel_size=1)

        self.mod6 = ModulatedConv2d(64, 32, 3, style_dim)
        self.up6 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # Final RGB color projection
        self.to_rgb = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, mask_x, z=None):
        B = mask_x.shape[0]
        if z is None:
            z = torch.randn(B, self.z_dim, device=mask_x.device)

        # 1. Conditioning Encoder E
        e1 = F.leaky_relu(self.enc1(mask_x), 0.2)
        e2 = F.leaky_relu(self.down1(e1), 0.2)  # [B, 128, 128, 128]
        e3 = F.leaky_relu(self.down2(e2), 0.2)  # [B, 256, 64, 64]
        e4 = F.leaky_relu(self.down3(e3), 0.2)  # [B, 512, 32, 32]
        e5 = F.leaky_relu(self.down4(e4), 0.2)  # [B, 512, 16, 16]
        e6 = F.leaky_relu(self.down5(e5), 0.2)  # [B, 512, 8, 8]
        e7 = F.leaky_relu(self.down6(e6), 0.2)  # [B, 512, 4, 4]

        E_x = self.enc_fc(e7)
        w = self.mapping_net(z)
        style_s = self.fusion(torch.cat([E_x, w], dim=1))

        # 2. Synthesis Network D
        h = self.init_proj(E_x).view(B, 512, 4, 4)
        h = F.leaky_relu(self.mod1(h, style_s), 0.2)

        h = self.up1(h) + self.skip1(e6)      # 4  -> 8
        h = F.leaky_relu(self.mod2(h, style_s), 0.2)

        h = self.up2(h) + self.skip2(e5)      # 8  -> 16
        h = F.leaky_relu(self.mod3(h, style_s), 0.2)

        h = self.up3(h) + self.skip3(e4)      # 16 -> 32
        h = F.leaky_relu(self.mod4(h, style_s), 0.2)

        h = self.up4(h) + self.skip4(e3)      # 32 -> 64
        h = F.leaky_relu(self.mod5(h, style_s), 0.2)

        h = self.up5(h) + self.skip5(e2)      # 64 -> 128
        h = F.leaky_relu(self.mod6(h, style_s), 0.2)

        h = self.up6(h)                       # 128 -> 256
        out_img = torch.tanh(self.to_rgb(h))  # Output range [-1, 1]
        return out_img