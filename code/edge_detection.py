import openslide
import numpy as np
import cv2
import pywt
import matplotlib.pyplot as plt
import random

# Load WSI
slide = openslide.OpenSlide("slide.svs")

patch_size = 512
num_patches = 10

width, height = slide.dimensions

for i in range(num_patches):

    # Random location
    x = random.randint(0, width - patch_size)
    y = random.randint(0, height - patch_size)

    # Extract patch
    patch = slide.read_region((x, y), 0, (patch_size, patch_size))
    patch = np.array(patch)[:, :, :3]

    # Convert to grayscale
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

    # ---------------- Edge Algorithms ---------------- #

    # 1. Canny
    canny = cv2.Canny(gray, 100, 200)

    # 2. Sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    sobel = np.sqrt(sobelx**2 + sobely**2)

    # 3. Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # 4. DWT edges
    LL, (LH, HL, HH) = pywt.dwt2(gray, 'haar')
    dwt_edges = np.abs(LH) + np.abs(HL) + np.abs(HH)

    # ---------------- Visualization ---------------- #

    fig, axes = plt.subplots(1, 5, figsize=(18,4))

    images = [gray, canny, sobel, laplacian, dwt_edges]
    titles = ["Original", "Canny", "Sobel", "Laplacian", "DWT Edges"]

    for j in range(5):
        axes[j].imshow(images[j], cmap='gray')
        axes[j].set_title(titles[j])
        axes[j].axis("off")

    plt.tight_layout()

    # Save figure
    plt.savefig(f"Edges_detected/Patch_Edges_{i+1}.png", dpi=300)
    plt.close()

print("Edge detection completed for 3 patches.")