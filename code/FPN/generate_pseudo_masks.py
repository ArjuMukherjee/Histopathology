import os
import cv2
import numpy as np
import argparse


def generate_polygon_mask(size=(512, 512), num_nuclei=None, min_radius=4, max_radius=9,
                          num_vertices=48, distributiveness=0.33, nuclei_range=(15, 115),
                          oblong_prob=0.7, elongation_range=(1.4, 2.6), boundary_px=1):
    """distributiveness in [0, 1]:
    0.0 -> tight clumps, nuclei may touch, strong clustering
    1.0 -> evenly spread, well-separated nuclei
    num_nuclei=None -> count drawn uniformly from nuclei_range (default 15..115).
    oblong_prob      -> fraction of nuclei drawn as randomly oriented ellipses (axis ratio from elongation_range).
    boundary_px      -> every nucleus is separated from its neighbours by at least this many background pixels,
                        like the clear inter-nuclear boundary in real MoNuSeg masks (no merged blobs).
    """
    h, w = size
    d = float(np.clip(distributiveness, 0.0, 1.0))
    if num_nuclei is None:
        actual_nuclei = int(np.random.randint(nuclei_range[0], nuclei_range[1] + 1))
    else:
        actual_nuclei = int(num_nuclei)

    # Random glandular clusters (centers may sit near the edges too)
    num_glands = np.random.randint(2, 5)
    gland_centers = np.column_stack([
        np.random.randint(0, w, num_glands),
        np.random.randint(0, h, num_glands),
    ])

    placed_centers = []
    cluster_prob = 0.7 - 0.4 * d            # low d -> mostly clustered
    spread_lo, spread_hi = 10 + 30 * d, 40 + 90 * d  # low d -> tight clusters

    mask = np.zeros(size, dtype=np.uint8)
    occupied = np.zeros(size, dtype=np.uint8)    # mask dilated by boundary_px: the 'no-go' zone for new nuclei
    ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * boundary_px + 1, 2 * boundary_px + 1))

    for _ in range(actual_nuclei):
        base_sep = max_radius * (0.4 + 1.6 * d)
        min_sep = base_sep * np.random.uniform(0.5, 1.5)

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
            continue
        placed_centers.append((cx, cy))

        # Shape: ellipse (oblong with prob oblong_prob, otherwise near-circular) whose radius is modulated by
        # 2-3 low-frequency harmonics -> smooth, gently curved, slightly asymmetric outline like a real nucleus.
        if np.random.rand() < oblong_prob:
            elong = np.random.uniform(*elongation_range)
        else:
            elong = np.random.uniform(1.0, 1.25)
        a_ax = np.random.uniform(min_radius, max_radius) * np.sqrt(elong)   # semi-major
        b_ax = a_ax / elong                                                 # semi-minor
        theta = np.random.uniform(0, np.pi)
        t = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
        r_mod = np.ones_like(t)
        for k in (2, 3):                                        # only low harmonics: no wiggles / jaggies
            r_mod += np.random.uniform(0.0, 0.06) * np.cos(k * t + np.random.uniform(0, 2 * np.pi))
        ex, ey = a_ax * np.cos(t) * r_mod, b_ax * np.sin(t) * r_mod
        xs = cx + ex * np.cos(theta) - ey * np.sin(theta)
        ys = cy + ex * np.sin(theta) + ey * np.cos(theta)
        pts = np.stack([xs, ys], 1).round().astype(np.int32)

        # Draw the nucleus, then erase any pixel that falls inside an existing nucleus' boundary ring,
        # so neighbouring nuclei never merge: there is always >= boundary_px of background between them.
        nuc = np.zeros(size, dtype=np.uint8)
        cv2.fillPoly(nuc, [pts], 255)
        nuc[occupied > 0] = 0
        if cv2.countNonZero(nuc) < 6:      # almost fully carved away -> drop it
            placed_centers.pop()
            continue
        mask |= nuc
        occupied |= cv2.dilate(nuc, ring_k)

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