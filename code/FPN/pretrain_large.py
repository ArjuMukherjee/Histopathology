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


def run_stage(stage_num, max_iters, lr, gen_g, gen_s, disc_g, disc_s, loader, device, accum_steps, save_dir, backbone):
    opt_g = optim.Adam(list(gen_g.parameters()) + list(gen_s.parameters()), lr=lr, betas=(0.0, 0.99), eps=1e-8)
    opt_d = optim.Adam(list(disc_g.parameters()) + list(disc_s.parameters()), lr=lr, betas=(0.0, 0.99), eps=1e-8)

    criterion = UNITPathSSLLoss(lambda1=2.0, lambda2=10.0, lambda3=0.0)
    mse = nn.MSELoss()

    iter_cnt = 0
    data_iter = iter(loader)

    print(f"\n===== Starting Pretraining Stage {stage_num}: {max_iters} Iterations @ LR={lr} =====")

    while iter_cnt < max_iters:
        try:
            real_x, real_y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            real_x, real_y = next(data_iter)

        real_x, real_y = real_x.to(device), real_y.to(device)

        # 1. Discriminator Step
        opt_d.zero_grad()
        with torch.no_grad():
            fake_y = gen_g(real_x)
            fake_x = torch.sigmoid(gen_s((real_y + 1.0) / 2.0))

        d_g_loss = 0.5 * (mse(disc_g(real_y), torch.ones_like(disc_g(real_y))) +
                          mse(disc_g(fake_y.detach()), torch.zeros_like(disc_g(fake_y))))
        d_s_loss = 0.5 * (mse(disc_s(real_x), torch.ones_like(disc_s(real_x))) +
                          mse(disc_s(fake_x.detach()), torch.zeros_like(disc_s(fake_x))))

        loss_d = (d_g_loss + d_s_loss) / accum_steps
        loss_d.backward()

        # 2. Generator Step
        opt_g.zero_grad()
        fake_y = gen_g(real_x)
        d_g_fake = disc_g(fake_y)
        loss_gan_g = mse(d_g_fake, torch.ones_like(d_g_fake))

        # Forward Cycle: x -> G(x) -> S(G(x)) ~ x
        rec_x = gen_s((fake_y + 1.0) / 2.0)
        d_s_fake = disc_s(torch.sigmoid(rec_x))

        loss_unit, _ = criterion(d_s_fake, torch.sigmoid(rec_x), real_x)
        loss_g_total = (loss_gan_g + loss_unit) / accum_steps
        loss_g_total.backward()

        if (iter_cnt + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(gen_s.parameters(), 1.0)
            opt_d.step()
            opt_g.step()

        iter_cnt += 1
        if iter_cnt % 500 == 0 or iter_cnt == max_iters:
            print(f"Stage {stage_num} | Iter [{iter_cnt:05d}/{max_iters:05d}] | "
                  f"Loss Gen: {loss_g_total.item() * accum_steps:.4f} | Loss D: {loss_d.item() * accum_steps:.4f}")

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

    # Stage 1 (40,000 iterations @ LR 4e-4)
    run_stage(1, args.stage1_iters, 4.0e-4, gen_g, gen_s, disc_g, disc_s, loader, device, args.accumulation_steps, args.save_dir, args.backbone)

    # Stage 2 (Re-initialize S and DS, 25,000 iterations @ LR 1e-4)
    gen_s = SegModel(backbone_name=args.backbone, pretrained=True).to(device)
    disc_s = PatchDiscriminator(in_channels=1).to(device)
    run_stage(2, args.stage2_iters, 1.0e-4, gen_g, gen_s, disc_g, disc_s, loader, device, args.accumulation_steps, args.save_dir, args.backbone)


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
    main(parser.parse_args())