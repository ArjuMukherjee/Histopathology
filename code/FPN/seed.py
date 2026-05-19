import numpy as np
from test import (
    get_random_patch,
    run_model_on_patch,
    set_seed,
    init_model
)

# -------- Settings --------
NUM_SEEDS = 1000

red_seeds = []
pass_count = 0

print("Selecting a fixed patch for evaluation...")
patch = get_random_patch()

print("Starting seed search...\n")

# -------- Loop --------
for seed in range(NUM_SEEDS):
    set_seed(seed)

    # 🔥 Initialize model AFTER setting seed
    conv_layer1, conv_layer2, fusion = init_model()

    # Run model
    out, _, _, _ = run_model_on_patch(
        patch, conv_layer1, conv_layer2, fusion
    )

    # Mean RGB
    R, G, B = np.mean(out, axis=(0, 1))

    # Red dominance condition
    if R > G * 1.1 and R > B * 1.1:
        red_seeds.append(seed)
        pass_count += 1

    # Progress
    if (seed + 1) % 100 == 0:
        print(f"Checked {seed + 1}/{NUM_SEEDS} | Passes: {pass_count}")

# -------- Results --------
print("\n=========== FINAL RESULTS ===========\n")

print("Red-dominant seeds:")
print(red_seeds)

print(f"\nTotal passes: {pass_count} out of {NUM_SEEDS}")
print(f"Pass %: {(pass_count / NUM_SEEDS) * 100:.2f}%")

if red_seeds:
    print(f"Seed range: {min(red_seeds)} to {max(red_seeds)}")
else:
    print("No strong red seeds found.")