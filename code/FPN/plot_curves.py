import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_training_curves(csv_path, output_image_path=None):
  if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Metrics CSV file not found at: {csv_path}")

  df = pd.read_csv(csv_path)

  if output_image_path is None:
    base_dir = os.path.dirname(csv_path)
    output_image_path = os.path.join(base_dir, "training_curves.png")

  epochs = df["epoch"]
  fig, axes = plt.subplots(2, 2, figsize=(14, 10))

  # 1. Training vs Validation Loss
  axes[0, 0].plot(
      epochs,
      df["train_loss"],
      label="Train Loss",
      color="#d62728",
      linewidth=1.8,
  )
  axes[0, 0].plot(
      epochs,
      df["val_loss"],
      label="Val Loss",
      color="#8c1111",
      linestyle="--",
      linewidth=1.8,
  )
  axes[0, 0].set_title("Training vs. Validation Loss", fontsize=12, pad=8)
  axes[0, 0].set_xlabel("Epoch")
  axes[0, 0].set_ylabel("Loss")
  axes[0, 0].legend(loc="upper right")
  axes[0, 0].grid(True, alpha=0.3)

  # 2. Validation Dice & IoU Score
  axes[0, 1].plot(
      epochs,
      df["val_dice"],
      label="Val Dice Score",
      color="#1f77b4",
      linewidth=1.8,
  )
  axes[0, 1].plot(
      epochs,
      df["val_iou"],
      label="Val IoU (Jaccard)",
      color="#0e3d64",
      linestyle="--",
      linewidth=1.8,
  )
  axes[0, 1].set_title(
      "Validation Dice & IoU Over Epochs", fontsize=12, pad=8
  )
  axes[0, 1].set_xlabel("Epoch")
  axes[0, 1].set_ylabel("Score")
  axes[0, 1].legend(loc="lower right")
  axes[0, 1].grid(True, alpha=0.3)

  # 3. Accuracy, Precision & Recall
  axes[1, 0].plot(
      epochs,
      df["val_accuracy"],
      label="Accuracy",
      color="#2ca02c",
      linewidth=1.8,
  )
  axes[1, 0].plot(
      epochs,
      df["val_precision"],
      label="Precision",
      color="#17becf",
      linestyle="--",
      linewidth=1.8,
  )
  axes[1, 0].plot(
      epochs,
      df["val_recall"],
      label="Recall",
      color="#bcbd22",
      linestyle=":",
      linewidth=1.8,
  )
  axes[1, 0].set_title(
      "Classification & Segmentation Accuracy", fontsize=12, pad=8
  )
  axes[1, 0].set_xlabel("Epoch")
  axes[1, 0].set_ylabel("Score")
  axes[1, 0].legend(loc="lower right")
  axes[1, 0].grid(True, alpha=0.3)

  # 4. Learning Rate Schedule
  axes[1, 1].plot(
      epochs,
      df["lr"],
      label="Learning Rate",
      color="#9467bd",
      linewidth=1.8,
  )
  axes[1, 1].set_title(
      "Cosine Annealing Learning Rate Schedule", fontsize=12, pad=8
  )
  axes[1, 1].set_xlabel("Epoch")
  axes[1, 1].set_ylabel("LR")
  axes[1, 1].set_yscale("log")
  axes[1, 1].legend(loc="upper right")
  axes[1, 1].grid(True, alpha=0.3)

  plt.tight_layout()
  plt.savefig(output_image_path, dpi=300)
  plt.close()
  print(f"--> Saved training curve plot to: {output_image_path}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Plot training metric curves from CSV"
  )
  parser.add_argument(
      "--csv-path",
      default="checkpoints/final/resnet50_training_metrics.csv",
      help="Path to the training metrics CSV file",
  )
  parser.add_argument(
      "--save-path",
      default=None,
      help="Optional output PNG path. If omitted, saves beside the CSV.",
  )
  args = parser.parse_args()

  plot_training_curves(args.csv_path, args.save_path)