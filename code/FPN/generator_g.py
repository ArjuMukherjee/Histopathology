import torch
import torch.nn as nn
import torch.nn.functional as F


class EqConv2d(nn.Module):
    """
    Equalized-learning-rate conv (StyleGAN). Weights are stored ~N(0,1) and scaled by 1/sqrt(fan_in)
    at runtime so that Adam's fixed-size steps change activations by a bounded amount. With plain
    nn.Conv2d the same steps were ~30x larger relative to the weights -> activations exploded and
    the output tanh saturated to +/-1 within a few hundred iterations.
    """
    def __init__(self, in_ch, out_ch, k, stride=1, padding=0, lr_mul=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, k, k) / lr_mul)
        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.scale = lr_mul / (in_ch * k * k) ** 0.5
        self.lr_mul, self.stride, self.padding = lr_mul, stride, padding

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias * self.lr_mul, stride=self.stride, padding=self.padding)


class EqLinear(nn.Module):
    def __init__(self, in_f, out_f, lr_mul=1.0, bias_init=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) / lr_mul)
        self.bias = nn.Parameter(torch.full((out_f,), bias_init / lr_mul))
        self.scale, self.lr_mul = lr_mul / in_f ** 0.5, lr_mul

    def forward(self, x):
        return F.linear(x, self.weight * self.scale, self.bias * self.lr_mul)


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
        self.bias = nn.Parameter(torch.zeros(out_channels))
        # Equalized learning rate (StyleGAN2): raw randn weights saturated tanh at the output
        self.weight_scale = 1.0 / (in_channels * kernel_size ** 2) ** 0.5
        self.affine = EqLinear(style_dim, in_channels)

    def forward(self, x, style):
        B, C, H, W = x.shape
        # Compute per-sample style scaling factor
        s = self.affine(style).view(B, 1, self.in_channels, 1, 1) + 1.0

        # Weight modulation: [B, out_c, in_c, k, k]
        w = self.weight.unsqueeze(0) * self.weight_scale * s

        # Weight demodulation normalization
        d = torch.rsqrt(w.pow(2).sum(dim=[2, 3, 4], keepdim=True) + 1e-8)
        w_demod = w * d

        # Batched grouped convolution
        x = x.view(1, B * C, H, W)
        w_demod = w_demod.view(
            B * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
        )
        out = F.conv2d(x, w_demod, padding=self.padding, groups=B)
        return out.view(B, self.out_channels, H, W) + self.bias.view(1, -1, 1, 1)


class CoModulatedGenerator(nn.Module):
    """
    Co-modulated Generator G: Pseudo Mask (x) + Style Noise (z) -> Synthetic Histopathology Image (y_hat).
    Architecture:
      - Conditioning Encoder E (Downsamples mask x to 4x4 spatial feature map and style vector E(x))
      - Mapping Network M (Transforms noise z ~ N(0, I) to stochastic style code w)
      - Affine Co-Modulation Fusion: s = A(E(x), M(z))
      - Synthesis Network D with weight-demodulated convolutions and multi-scale skip connections from E
    """
    def __init__(self, in_channels=1, out_channels=3, z_dim=512, style_dim=512, enc_dim=1024, mapping_layers=8):
        super().__init__()
        self.z_dim = z_dim
        self.style_dim = style_dim
        cond_dim = enc_dim + style_dim          # [E(x), w] concatenated -> per-layer affine A (Eq. 4)

        # 1. Stochastic Style Mapping Network M: z -> w (8 FC layers, StyleGAN2, lr_mul=0.01)
        layers = []
        for _ in range(mapping_layers):
            layers += [EqLinear(z_dim if not layers else style_dim, style_dim, lr_mul=0.01), nn.LeakyReLU(0.2, inplace=True)]
        self.mapping_net = nn.Sequential(*layers)

        # 2. Conditioning Encoder E (Mask x -> multi-scale feature maps + E(x))
        self.enc1 = EqConv2d(in_channels, 64, 3, padding=1)
        self.down1 = EqConv2d(64, 128, 4, stride=2, padding=1)   # e2: 256 -> 128 (128 ch)
        self.down2 = EqConv2d(128, 256, 4, stride=2, padding=1)  # e3: 128 -> 64  (256 ch)
        self.down3 = EqConv2d(256, 512, 4, stride=2, padding=1)  # e4: 64  -> 32  (512 ch)
        self.down4 = EqConv2d(512, 512, 4, stride=2, padding=1)  # e5: 32  -> 16  (512 ch)
        self.down5 = EqConv2d(512, 512, 4, stride=2, padding=1)  # e6: 16  -> 8   (512 ch)
        self.down6 = EqConv2d(512, 512, 4, stride=2, padding=1)  # e7: 8   -> 4   (512 ch)
        self.enc_conv = EqConv2d(512, 512, 3, padding=1)          # Conv 3x3 at 4x4 (Fig. 3)

        # Fig. 3: 512x4x4 -> FC (1x1024) -> dropout 0.5  ==> E(x)
        self.enc_fc = nn.Sequential(
            nn.Flatten(),
            EqLinear(512 * 4 * 4, enc_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
        )

        # FC -> reshape to 512x4x4: replaces StyleGAN2's constant input
        self.init_proj = EqLinear(enc_dim, 512 * 4 * 4)

        # 3. Synthesis Network D with Modulated Convolutions & Skip Connections from E
        self.mod1 = ModulatedConv2d(512, 512, 3, cond_dim)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.skip1 = EqConv2d(512, 512, 1)

        self.mod2 = ModulatedConv2d(512, 512, 3, cond_dim)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.skip2 = EqConv2d(512, 512, 1)

        self.mod3 = ModulatedConv2d(512, 256, 3, cond_dim)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.skip3 = EqConv2d(512, 256, 1)

        self.mod4 = ModulatedConv2d(256, 128, 3, cond_dim)
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.skip4 = EqConv2d(256, 128, 1)

        self.mod5 = ModulatedConv2d(128, 64, 3, cond_dim)
        self.up5 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.skip5 = EqConv2d(128, 64, 1)

        self.mod6 = ModulatedConv2d(64, 32, 3, cond_dim)
        self.up6 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # Final RGB color projection - zero-init so G starts at tanh(0) = mid-gray and gradient flows
        self.to_rgb = EqConv2d(32, out_channels, 1)
        nn.init.zeros_(self.to_rgb.weight)
        self.skip_scale = 0.5 ** 0.5          # variance-preserving sum of upsampled path + encoder skip

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
        e7 = F.leaky_relu(self.enc_conv(e7), 0.2)

        E_x = self.enc_fc(e7)                                   # conditional style  E(x)  [B, 1024]
        w = self.mapping_net(z)                                 # stochastic style   M(z)  [B, 512]
        style_s = torch.cat([E_x, w], dim=1)                    # s = A(E(x), M(z)): A is each layer's affine (Eq. 4)

        # 2. Synthesis Network D
        h = self.init_proj(E_x).view(B, 512, 4, 4)              # replaces StyleGAN2 constant input
        h = F.leaky_relu(self.mod1(h, style_s), 0.2)

        h = (self.up1(h) + self.skip1(e6)) * self.skip_scale      # 4  -> 8
        h = F.leaky_relu(self.mod2(h, style_s), 0.2)

        h = (self.up2(h) + self.skip2(e5)) * self.skip_scale      # 8  -> 16
        h = F.leaky_relu(self.mod3(h, style_s), 0.2)

        h = (self.up3(h) + self.skip3(e4)) * self.skip_scale      # 16 -> 32
        h = F.leaky_relu(self.mod4(h, style_s), 0.2)

        h = (self.up4(h) + self.skip4(e3)) * self.skip_scale      # 32 -> 64
        h = F.leaky_relu(self.mod5(h, style_s), 0.2)

        h = (self.up5(h) + self.skip5(e2)) * self.skip_scale      # 64 -> 128
        h = F.leaky_relu(self.mod6(h, style_s), 0.2)

        h = self.up6(h)                       # 128 -> 256
        out_img = torch.tanh(self.to_rgb(h))  # Output range [-1, 1]
        return out_img