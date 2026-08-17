import os
import cv2
import numpy as np
import argparse


def generate_polygon_mask(size=(512, 512), num_nuclei=120, min_radius=6, max_radius=16, num_vertices=8):
    mask = np.zeros(size, dtype=np.uint8)
    h, w = size

    # Random glandular clusters
    num_glands = np.random.randint(2, 5)
    gland_centers = np.random.randint(50, size[0] - 50, size=(num_glands, 2))

    for _ in range(num_nuclei):
        if np.random.rand() < 0.6 and len(gland_centers) > 0:
            g_center = gland_centers[np.random.randint(0, num_glands)]
            angle = np.random.uniform(0, 2 * np.pi)
            r_dist = np.random.uniform(20, 60)
            cx = int(np.clip(g_center[0] + r_dist * np.cos(angle), 20, w - 20))
            cy = int(np.clip(g_center[1] + r_dist * np.sin(angle), 20, h - 20))
        else:
            cx = np.random.randint(20, w - 20)
            cy = np.random.randint(20, h - 20)

        # Polygon vertices with Bézier-like jitter
        angles = np.sort(np.random.uniform(0, 2 * np.pi, num_vertices))
        radii = np.random.uniform(min_radius, max_radius, num_vertices)
        pts = []
        for a, r in zip(angles, radii):
            px = int(cx + r * np.cos(a))
            py = int(cy + r * np.sin(a))
            pts.append([px, py])

        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)

    return mask


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic pseudo masks for UNIT pretraining.")
    parser.add_argument("--save-dir", default="data/pseudo_masks")
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--size", type=int, nargs=2, default=[512, 512])
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Generating {args.num_samples} synthetic masks in '{args.save_dir}'...")

    for idx in range(args.num_samples):
        m = generate_polygon_mask(size=tuple(args.size))
        cv2.imwrite(os.path.join(args.save_dir, f"pseudo_mask_{idx:05d}.png"), m)

    print("Pseudo mask generation complete.")


if __name__ == "__main__":
    main()