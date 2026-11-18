import os
import cv2
import numpy as np
import argparse


def generate_polygon_mask(size=(512, 512), num_nuclei=120, min_radius=4, max_radius=10,
                          num_vertices=16, distributiveness=0.6):
    """distributiveness in [0, 1]:
    0.0 -> tight clumps, nuclei may touch/overlap, strong clustering
    1.0 -> evenly spread, well-separated nuclei
    """
    mask = np.zeros(size, dtype=np.uint8)
    h, w = size
    d = float(np.clip(distributiveness, 0.0, 1.0))

    # Random glandular clusters (centers may sit near the edges too)
    num_glands = np.random.randint(2, 5)
    gland_centers = np.column_stack([
        np.random.randint(0, w, num_glands),
        np.random.randint(0, h, num_glands),
    ])

    placed_centers = []
    cluster_prob = 0.7 - 0.4 * d            # low d -> mostly clustered
    spread_lo, spread_hi = 10 + 30 * d, 40 + 90 * d  # low d -> tight clusters

    actual_nuclei = num_nuclei + np.random.randint(-15, 16)  # vary count +-15

    for _ in range(actual_nuclei):
        # random per-nucleus separation: gaps and closeness both occur
        base_sep = max_radius * (0.4 + 1.6 * d)          # 0 -> overlap allowed, 1 -> wide gaps
        min_sep = base_sep * np.random.uniform(0.5, 1.5)  # random gaps / random closeness

        cx, cy = None, None
        for _attempt in range(10):
            if np.random.rand() < cluster_prob and len(gland_centers) > 0:
                g_center = gland_centers[np.random.randint(0, num_glands)]
                angle = np.random.uniform(0, 2 * np.pi)
                r_dist = np.random.uniform(spread_lo, spread_hi)
                tx = int(np.clip(g_center[0] + r_dist * np.cos(angle), 0, w - 1))
                ty = int(np.clip(g_center[1] + r_dist * np.sin(angle), 0, h - 1))
            else:
                tx = np.random.randint(0, w)
                ty = np.random.randint(0, h)

            if all((tx - px) ** 2 + (ty - py) ** 2 >= min_sep ** 2 for px, py in placed_centers):
                cx, cy = tx, ty
                break
        if cx is None:
            continue  # couldn't find a spot respecting this nucleus' separation; skip
        placed_centers.append((cx, cy))

        # Jittered vertices -> convex hull (nuclei are convex; clipped at borders -> partial nuclei at edges)
        angles = np.sort(np.random.uniform(0, 2 * np.pi, num_vertices))
        radii = np.random.uniform(min_radius, max_radius, num_vertices)
        pts = []
        for a, r in zip(angles, radii):
            px = int(cx + r * np.cos(a))
            py = int(cy + r * np.sin(a))
            pts.append([px, py])

        hull = cv2.convexHull(np.array(pts, dtype=np.int32))
        cv2.fillPoly(mask, [hull], 255)

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