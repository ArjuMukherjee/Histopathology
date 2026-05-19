import openslide
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt

# Load WSI
slide = openslide.OpenSlide("slide.svs")

patch_size = 512
num_patches = 10

width, height = slide.dimensions

# Output folder
os.makedirs("WSI_Patches", exist_ok=True)

saved = 0

while saved < num_patches:

    # Random patch location
    x = random.randint(0, width - patch_size)
    y = random.randint(0, height - patch_size)

    # Extract patch
    patch = slide.read_region((x, y), 0, (patch_size, patch_size))
    patch = np.array(patch)[:, :, :3]

    # Convert to grayscale
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

    # -------- White patch filtering --------
    white_pixels = np.sum(gray > 220)
    white_ratio = white_pixels / (patch_size * patch_size)

    if white_ratio > 0.8:
        continue

    # -------- Dark / Light separation --------
    dark_threshold = 100
    light_threshold = 180

    dark_region = gray.copy()
    dark_region[gray > dark_threshold] = 0

    light_region = gray.copy()
    light_region[gray < light_threshold] = 0

    saved += 1

    # -------- Create figure --------
    fig, axes = plt.subplots(1, 3, figsize=(12,4))

    axes[0].imshow(patch)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(dark_region, cmap="gray")
    axes[1].set_title("Dark Regions")
    axes[1].axis("off")

    axes[2].imshow(light_region, cmap="gray")
    axes[2].set_title("Light Regions")
    axes[2].axis("off")

    plt.tight_layout()

    # Save figure
    plt.savefig(f"WSI_Patches/Patch_{saved}.png", dpi=300)
    plt.close()

print(f"Finished saving {saved} figures.")