import matplotlib.pyplot as plt
import pandas as pd

df_scratch = pd.read_csv(
    "checkpoints/ablation_scratch/resnet50_training_metrics.csv"
)
df_imagenet = pd.read_csv(
    "checkpoints/ablation_imagenet/resnet50_training_metrics.csv"
)
df_ours = pd.read_csv("checkpoints/final/resnet50_training_metrics.csv")

plt.figure(figsize=(10, 6))
plt.plot(
    df_scratch["epoch"],
    df_scratch["val_dice"],
    label="Scratch (Random)",
    color="gray",
    linestyle=":",
)
plt.plot(
    df_imagenet["epoch"],
    df_imagenet["val_dice"],
    label="ImageNet Pretrained",
    color="blue",
    linestyle="--",
)
plt.plot(
    df_ours["epoch"],
    df_ours["val_dice"],
    label="UNITPathSSL (Ours)",
    color="green",
    linewidth=2.5,
)

plt.title(
    "Validation Dice Score Convergence: UNITPathSSL vs Baselines", fontsize=14
)
plt.xlabel("Epoch")
plt.ylabel("Validation Dice")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("checkpoints/pretrain_advancement_comparison.png", dpi=300)
print("Saved comparison curve!")