import os
import glob
import cv2
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from seg_head import SegModel
from generator_g import CoModulatedGenerator
from loss import UNITPathSSLLoss


class LargeUnpairedDataset(Dataset):
    def __init__(self, mask_dir, tissue_dir, target_size=(256, 256)):
        super().__init__()
        valid_exts = (".png", ".tif", ".tiff", ".jpg", ".jpeg")
        
        # Recursive search to capture images in nested subfolders
        self.mask_files = sorted([
            f for f in glob.glob(os.path.join(mask_dir, "**", "*.*"), recursive=True)
            if f.lower().endswith(valid_exts)
        ])
        self.tissue_files = sorted([
            f for f in glob.glob(os.path.join(tissue_dir, "**", "*.*"), recursive=True)
            if f.lower().endswith(valid_exts)
        ])
        self.target_size = tuple(target_size)

        if len(self.tissue_files) == 0 or len(self.mask_files) == 0:
            raise ValueError(f"Found {len(self.tissue_files)} tissues in '{tissue_dir}' and {len(self.mask_files)} masks in '{mask_dir}'.")
        
        print(f"Dataset ready: {len(self.tissue_files)} tissue images, {len(self.mask_files)} pseudo masks loaded.")

    def __len__(self):
        return max(len(self.mask_files), len(self.tissue_files))

    def __getitem__(self, idx):
        mask_path = self.mask_files[idx % len(self.mask_files)]
        tissue_path = self.tissue_files[idx % len(self.tissue_files)]

        # 1. Load and prepare pseudo mask (Domain X)
        m = cv2.imread(mask_path, 0)
        if m is None:
            raise FileNotFoundError(f"Could not load mask at: {mask_path}")
        m = cv2.resize(m, self.target_size, interpolation=cv2.INTER_NEAREST)
        m = (m > 0).astype("float32")
        mask_t = torch.from_numpy(m).unsqueeze(0)

        # 2. Load and prepare tissue patch (Domain Y)
        t = cv2.imread(tissue_path)
        if t is None:
            raise FileNotFoundError(f"Could not load tissue at: {tissue_path}")
        t = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)
        t = cv2.resize(t, self.target_size, interpolation=cv2.INTER_LINEAR)
        tissue_t = (torch.from_numpy(t).permute(2, 0, 1).float() / 127.5) - 1.0  # Normalize to [-1, 1]

        return mask_t, tissue_t


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class ADA(nn.Module):
    """
    Adaptive Discriminator Augmentation (paper Eq. 5, following StyleGAN2-ADA).
    Pixel blitting / geometric / color transforms are applied, each with probability p, to BOTH
    real y and fake y_hat before D_G. Augmentations are differentiable so gradients reach G.
    Overfitting heuristic r_t = E[sign(D_G(y_real))], updated once every 4 minibatches:
        p = max(p + sign(r_t - target) * (N * 4) / 500000, 0),   p init 0, N = batch size.
    """
    def __init__(self, target=0.6, interval=4, speed_denom=500000.0, p_max=1.0):
        super().__init__()
        self.target, self.interval, self.speed_denom, self.p_max = target, interval, speed_denom, p_max
        self.p = 0.0
        self._sign_acc, self._n_imgs, self._calls = 0.0, 0, 0

    @torch.no_grad()
    def update(self, d_real_logits):
        self._sign_acc += torch.sign(d_real_logits).mean().item()
        self._n_imgs += d_real_logits.shape[0]
        self._calls += 1
        if self._calls % self.interval == 0:
            r_t = self._sign_acc / self.interval
            step = (1.0 if r_t > self.target else -1.0) * self._n_imgs / self.speed_denom   # N * 4 / 500000
            self.p = float(min(max(self.p + step, 0.0), self.p_max))
            self._sign_acc, self._n_imgs = 0.0, 0

    def forward(self, img):
        if self.p <= 0:
            return img
        B = img.shape[0]
        dev = img.device

        def gate():  # per-sample bernoulli mask, shape [B,1,1,1]
            return (torch.rand(B, 1, 1, 1, device=dev) < self.p).float()

        # pixel blitting: x-flip, 90deg rotation, integer translation
        g = gate(); img = g * torch.flip(img, dims=[3]) + (1 - g) * img
        g = gate(); img = g * torch.rot90(img, 1, dims=[2, 3]) + (1 - g) * img
        g = gate()
        if g.sum() > 0:
            max_shift = int(0.125 * img.shape[-1])
            dx, dy = [int(v) for v in torch.randint(-max_shift, max_shift + 1, (2,))]
            img = g * torch.roll(img, shifts=(dy, dx), dims=(2, 3)) + (1 - g) * img

        # color: brightness, contrast, saturation
        g = gate(); img = img + g * (torch.randn(B, 1, 1, 1, device=dev) * 0.2)
        g = gate(); mean = img.mean(dim=(1, 2, 3), keepdim=True)
        c = torch.exp2(torch.randn(B, 1, 1, 1, device=dev) * 0.5)
        img = g * ((img - mean) * c + mean) + (1 - g) * img
        if img.shape[1] == 3:
            g = gate(); gray = img.mean(dim=1, keepdim=True)
            s = torch.exp2(torch.randn(B, 1, 1, 1, device=dev) * 1.0)
            img = g * ((img - gray) * s + gray) + (1 - g) * img

        # additive noise
        g = gate(); img = img + g * torch.randn_like(img) * (torch.rand(B, 1, 1, 1, device=dev) * 0.1)
        return img


def run_stage(stage_num, max_iters, lr, gen_g, gen_s, disc_g, disc_s, loader, device, accum_steps, save_dir, backbone,
              lambda1=2.0, lambda2=10.0, lambda3=0.0, gamma_g=10.0, gamma_s=10.0, r1_interval=16, ada=None):
    opt_g = optim.Adam(list(gen_g.parameters()) + list(gen_s.parameters()), lr=lr, betas=(0.0, 0.99), eps=1e-8)
    opt_d = optim.Adam(list(disc_g.parameters()) + list(disc_s.parameters()), lr=lr, betas=(0.0, 0.99), eps=1e-8)

    # Lazy R1: penalty is applied every `r1_interval` iters, so gamma is scaled by the interval to keep the same expected strength.
    criterion = UNITPathSSLLoss(lambda1=lambda1, lambda2=lambda2, lambda3=lambda3,
                                gamma_g=gamma_g * r1_interval, gamma_s=gamma_s * r1_interval)
    aug = ada if ada is not None else (lambda t: t)   # ADA sits in front of D_G only

    iter_cnt = 0
    data_iter = iter(loader)

    print(f"\n===== Starting Pretraining Stage {stage_num}: {max_iters} Iterations @ LR={lr} =====")

    while iter_cnt < max_iters:
        try:
            real_x, real_y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            real_x, real_y = next(data_iter)

        real_x, real_y = real_x.to(device), real_y.to(device)        # x: mask [0,1] 1ch | y: tissue [-1,1] 3ch
        do_r1 = (iter_cnt % r1_interval == 0)                        # lazy R1 (StyleGAN2 trick)

        # ---------------- 1. Discriminator step (Eq. 1 for D_G, Eq. 2 for D_S) ----------------
        with torch.no_grad():
            fake_y = gen_g(real_x)                                    # G(x)  in [-1,1]
            fake_x = torch.sigmoid(gen_s((real_y + 1.0) / 2.0))       # S(y)  in [0,1]  <-- D_S fake is S(y), not S(G(x))

        real_y_d = real_y.detach().requires_grad_(do_r1)
        real_x_d = real_x.detach().requires_grad_(do_r1)

        d_g_real = disc_g(aug(real_y_d))
        d_g_fake = disc_g(aug(fake_y))
        d_g_loss, d_g_log = criterion.d_g_loss(d_g_real, d_g_fake, real_y_d if do_r1 else None)
        if ada is not None:
            ada.update(d_g_real.detach())

        d_s_real = disc_s(real_x_d)
        d_s_fake = disc_s(fake_x)
        d_s_loss, d_s_log = criterion.d_s_loss(d_s_real, d_s_fake, real_x_d if do_r1 else None)

        loss_d = (d_g_loss + d_s_loss) / accum_steps
        if iter_cnt % accum_steps == 0:
            opt_d.zero_grad(set_to_none=True)
        loss_d.backward()

        # ---------------- 2. Generator step (G and S jointly) ----------------
        for p in list(disc_g.parameters()) + list(disc_s.parameters()):
            p.requires_grad_(False)

        fake_y = gen_g(real_x)                                        # G(x)
        fake_x_logit = gen_s((real_y + 1.0) / 2.0)                    # S(y) logits
        fake_x = torch.sigmoid(fake_x_logit)                          # S(y) in [0,1]

        x_rec = torch.sigmoid(gen_s((fake_y + 1.0) / 2.0))            # S(G(x))  ~ x   forward cycle
        y_rec = gen_g(fake_x)                                         # G(S(y))  ~ y   backward cycle

        d_g_fake = disc_g(aug(fake_y))                                # ADA applied to G's output too
        d_s_fake = disc_s(fake_x)

        loss_g_total, g_log = criterion(d_g_fake, d_s_fake, x_rec, real_x, y_rec, real_y)
        loss_g_total = loss_g_total / accum_steps
        if iter_cnt % accum_steps == 0:
            opt_g.zero_grad(set_to_none=True)
        loss_g_total.backward()

        for p in list(disc_g.parameters()) + list(disc_s.parameters()):
            p.requires_grad_(True)

        if (iter_cnt + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(list(gen_g.parameters()) + list(gen_s.parameters()), 1.0)
            torch.nn.utils.clip_grad_norm_(list(disc_g.parameters()) + list(disc_s.parameters()), 1.0)
            opt_d.step()
            opt_g.step()

        iter_cnt += 1
        if iter_cnt % 500 == 0 or iter_cnt == max_iters:
            print(f"Stage {stage_num} | Iter [{iter_cnt:05d}/{max_iters:05d}] | "
                  f"G_total {loss_g_total.item() * accum_steps:.3f} "
                  f"(adv_g {g_log['loss_adv_g']:.3f} adv_s {g_log['loss_adv_s']:.3f} "
                  f"cyc_x {g_log['loss_cyc_x']:.3f} cyc_y {g_log['loss_cyc_y']:.3f}) | "
                  f"D_G {d_g_log['d_g_adv']:.3f} (r1 {d_g_log['d_g_r1']:.3f}) "
                  f"D_S {d_s_log['d_s_adv']:.3f} (r1 {d_s_log['d_s_r1']:.3f}) | "
                  f"ADA p={ada.p if ada is not None else 0:.2f}")

    ckpt_path = os.path.join(save_dir, f"{backbone}_stage{stage_num}_pretrained.pth")
    torch.save(gen_s.state_dict(), ckpt_path)
    print(f"--> Saved Stage {stage_num} checkpoint to: {ckpt_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    dataset = LargeUnpairedDataset(args.mask_dir, args.tissue_dir, tuple(args.target_size))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # Initialize components
    gen_s = SegModel(backbone_name=args.backbone, pretrained=True).to(device)
    gen_g = CoModulatedGenerator(in_channels=1, out_channels=3).to(device)
    disc_g = PatchDiscriminator(in_channels=3).to(device)
    disc_s = PatchDiscriminator(in_channels=1).to(device)

    ada = ADA(target=args.ada_target).to(device) if args.use_ada else None
    common = dict(lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3,
                  gamma_g=args.gamma_g, gamma_s=args.gamma_s, r1_interval=args.r1_interval, ada=ada)

    # Stage 1 (40,000 iterations @ LR 4e-4)
    run_stage(1, args.stage1_iters, 4.0e-4, gen_g, gen_s, disc_g, disc_s, loader, device, args.accumulation_steps, args.save_dir, args.backbone, **common)

    # Stage 2 (Re-initialize S and DS, 25,000 iterations @ LR 1e-4)
    gen_s = SegModel(backbone_name=args.backbone, pretrained=True).to(device)
    disc_s = PatchDiscriminator(in_channels=1).to(device)
    run_stage(2, args.stage2_iters, 1.0e-4, gen_g, gen_s, disc_g, disc_s, loader, device, args.accumulation_steps, args.save_dir, args.backbone, **common)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tissue-dir", default="WSI_Patches")
    parser.add_argument("--mask-dir", default="data/pseudo_masks")
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--stage1-iters", type=int, default=40000)
    parser.add_argument("--stage2-iters", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=6)
    parser.add_argument("--target-size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-dir", default="checkpoints/pretrain")
    parser.add_argument("--lambda1", type=float, default=2.0, help="weight of L_GAN(S, D_S)")
    parser.add_argument("--lambda2", type=float, default=10.0, help="weight of L_cyc")
    parser.add_argument("--lambda3", type=float, default=0.0, help="weight of L_seg (ISG), 0 = disabled")
    parser.add_argument("--gamma-g", type=float, default=10.0, help="R1 gamma for D_G")
    parser.add_argument("--gamma-s", type=float, default=10.0, help="R1 gamma for D_S")
    parser.add_argument("--r1-interval", type=int, default=16, help="lazy R1 every N iters")
    parser.add_argument("--use-ada", action="store_true", default=True)
    parser.add_argument("--no-ada", dest="use_ada", action="store_false")
    parser.add_argument("--ada-target", type=float, default=0.6)
    main(parser.parse_args())