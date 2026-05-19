import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import openslide
import os
from DWTFusion import DWTEntropyFusion

# -------- Seed Setter --------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# -------- Random Seed --------
def set_random_seed():
    seed = random.randint(0, 10000)
    set_seed(seed)
    print(f"Using random seed: {seed}")
    return seed

# -------- Model Initialization --------
def init_model():
    conv_layer1 = nn.Conv2d(3, 9, 1)
    conv_layer2 = nn.Conv2d(9, 3, 1)
    fusion = DWTEntropyFusion(9, 3)
    return conv_layer1, conv_layer2, fusion

# -------- DWT FUNCTION --------
def dwt(x):
    ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device=x.device)
    lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], device=x.device)
    hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], device=x.device)
    hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], device=x.device)

    filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)

    B, C, H, W = x.shape
    filters = filters.repeat(C, 1, 1, 1)

    x = x.view(1, B*C, H, W)
    out = F.conv2d(x, filters, stride=2, groups=B*C)
    out = out.view(B, C, 4, H//2, W//2)

    return out[:, :, 1], out[:, :, 2], out[:, :, 3]

# -------- Load Slide --------
slide = openslide.OpenSlide("../slide.svs")

# -------- White Patch Filter --------
def is_not_white(patch, threshold=0.8):
    gray = np.mean(patch / 255.0)
    return gray < threshold

# -------- Patch Extraction --------
def get_random_patch():
    W, H = slide.dimensions

    while True:
        x = random.randint(0, W - 512)
        y = random.randint(0, H - 512)

        patch = slide.read_region((x, y), 0, (512, 512)).convert("RGB")
        patch_np = np.array(patch)

        if is_not_white(patch_np):
            return patch_np

# -------- Forward Pass --------
def run_model_on_patch(patch, conv_layer1, conv_layer2, fusion):
    img = torch.tensor(patch / 255.0, dtype=torch.float32)
    img = img.permute(2, 0, 1).unsqueeze(0)

    C = conv_layer1(img)
    P = conv_layer2(C)

    LH, HL, HH = dwt(C)
    fusion_out = F.interpolate(
        fusion(LH, HL, HH),
        size=(512, 512),
        mode='bilinear'
    )

    # P_final = torch.sigmoid(P + fusion_out)
    P_final = P + fusion_out

    out = P_final[0].detach().permute(1, 2, 0).numpy()
    out = np.clip(out, 0, 1)

    return out, P, P_final, img

# -------- Main Execution --------
if __name__ == "__main__":

    num_results = 10

    # Create output folder
    save_dir = "outputs"
    os.makedirs(save_dir, exist_ok=True)

    # Set random seed
    set_seed(999)

    # 🔥 Initialize model AFTER seed
    conv_layer1, conv_layer2, fusion = init_model()

    # -------- Run --------
    for run in range(num_results):

        plt.figure(figsize=(12, 12))

        for i in range(3):
            patch = get_random_patch()

            out, P, P_final, img = run_model_on_patch(
                patch, conv_layer1, conv_layer2, fusion
            )

            original = img[0].permute(1, 2, 0).numpy()
            p_out = np.clip(P[0].detach().permute(1, 2, 0).numpy(), 0, 1)
            p_final_out = np.clip(P_final[0].detach().permute(1, 2, 0).numpy(), 0, 1)
            # p_out = P[0].detach().permute(1, 2, 0).numpy()
            # p_final_out = P_final[0].detach().permute(1, 2, 0).numpy()
            # p_final_out = (p_final_out - p_final_out.min()) / (p_final_out.max() - p_final_out.min() + 1e-8)

            # -------- Plot --------
            plt.subplot(3, 3, 3*i + 1)
            plt.imshow(original)
            plt.title(f"Patch {i+1} - Original")
            plt.axis('off')

            plt.subplot(3, 3, 3*i + 2)
            plt.imshow(p_out)
            plt.title(f"Patch {i+1} - P")
            plt.axis('off')

            plt.subplot(3, 3, 3*i + 3)
            plt.imshow(p_final_out)
            plt.title(f"Patch {i+1} - P_final")
            plt.axis('off')

        plt.tight_layout()

        save_path = os.path.join(save_dir, f"result_{run+1}.png")
        plt.savefig(save_path)
        plt.close()

    print(f"Saved {num_results} figures in 'outputs' folder")