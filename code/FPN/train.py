import argparse
import csv
import os
import time
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from dataset import MoNuSegDataset, list_image_files, split_image_list
from loss import total_loss
from seg_head import SegModel


def dice_score(preds, targets, smooth=1e-6):
  preds = preds.astype(np.float32)
  targets = targets.astype(np.float32)
  intersection = np.sum(preds * targets)
  return (2.0 * intersection + smooth) / (
      np.sum(preds) + np.sum(targets) + smooth
  )

def set_bn_eval(m):
  classname = m.__class__.__name__
  if "BatchNorm" in classname:
    m.eval()

def evaluate(model, loader, device):
  model.eval()
  running_loss = 0.0
  all_preds_bin = []
  all_masks = []
  dice_scores = []

  with torch.no_grad():
    for imgs, masks in loader:
      imgs, masks = imgs.to(device), masks.to(device)
      preds = model(imgs)
      loss = total_loss(preds, masks)
      running_loss += loss.item()

      probs = torch.sigmoid(preds)
      print(f"DEBUG: prob min={probs.min().item():.3f}, max={probs.max().item():.3f}, mean={probs.mean().item():.3f}")
      pred_bin = (probs > 0.5).float().cpu().numpy()
      mask_np = masks.cpu().numpy()

      dice_scores.append(dice_score(pred_bin, mask_np))
      all_preds_bin.append(pred_bin.astype(np.uint8).flatten())
      all_masks.append(mask_np.astype(np.uint8).flatten())

  all_masks = np.concatenate(all_masks)
  all_preds_bin = np.concatenate(all_preds_bin)

  val_loss = running_loss / len(loader)
  mean_dice = np.mean(dice_scores)
  iou = jaccard_score(all_masks, all_preds_bin, zero_division=0)
  acc = accuracy_score(all_masks, all_preds_bin)
  prec = precision_score(all_masks, all_preds_bin, zero_division=0)
  rec = recall_score(all_masks, all_preds_bin, zero_division=0)
  f1 = f1_score(all_masks, all_preds_bin, zero_division=0)

  return val_loss, mean_dice, iou, acc, prec, rec, f1


def train(args):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  os.makedirs(args.save_dir, exist_ok=True)

  all_files = list_image_files(args.train_img_dir)
  train_files, val_files, _ = split_image_list(
      all_files, test_size=0.1, val_size=0.15, random_state=42
  )

  train_ds = MoNuSegDataset(
      args.train_img_dir,
      args.train_mask_dir,
      target_size=args.target_size,
      augment=args.augment,
      augment_factor=args.augment_factor,
      augment_prob=args.augment_prob,
      rotation_range=args.rotation_range,
      scale_range=args.scale_range,
      image_list=train_files,
  )
  val_ds = MoNuSegDataset(
      args.train_img_dir,
      args.train_mask_dir,
      target_size=args.target_size,
      augment=False,
      image_list=val_files,
  )

  train_loader = DataLoader(
      train_ds,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers,
      pin_memory=True,
  )
  val_loader = DataLoader(
      val_ds,
      batch_size=args.batch_size,
      shuffle=False,
      num_workers=args.num_workers,
      pin_memory=True,
  )

  model = SegModel(
      backbone_name=args.backbone,
      pretrained=args.pretrained,
      num_classes=args.num_classes,
  ).to(device)

  # Load UNITPathSSL Pretrained Weights if provided
  if args.pretrained_weights and os.path.exists(args.pretrained_weights):
    print(
        f"Loading UNITPathSSL pretrained checkpoint from:"
        f" {args.pretrained_weights}"
    )
    checkpoint = torch.load(args.pretrained_weights, map_location=device)
    state_dict = (
        checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    )
    model.load_state_dict(state_dict, strict=False)

  optimizer = optim.Adam(
      model.parameters(), lr=args.lr, weight_decay=args.weight_decay
  )
  scheduler = optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=args.epochs, eta_min=1e-6
  )

  best_val_dice = 0.0
  best_checkpoint = os.path.join(args.save_dir, f"{args.backbone}_best.pth")
  last_checkpoint = os.path.join(args.save_dir, f"{args.backbone}_last.pth")
  log_csv_path = os.path.join(
      args.save_dir, f"{args.backbone}_training_metrics.csv"
  )

  csv_headers = [
      "epoch",
      "train_loss",
      "val_loss",
      "val_dice",
      "val_iou",
      "val_accuracy",
      "val_precision",
      "val_recall",
      "val_f1",
      "lr",
      "epoch_time_sec",
  ]
  with open(log_csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(csv_headers)

  print(
      f"Starting training: Backbone={args.backbone}, Epochs={args.epochs},"
      f" BatchSize={args.batch_size}, LR={args.lr}"
  )

  epoch_durations = []
  total_start_time = time.time()

  for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    model.train()
    model.apply(
        set_bn_eval
    )
    running_loss = 0.0
    optimizer.zero_grad()

    for i, (imgs, masks) in enumerate(train_loader):
      imgs, masks = imgs.to(device), masks.to(device)
      preds = model(imgs)
      loss = total_loss(preds, masks) / args.accumulation_steps
      loss.backward()

      if (i + 1) % args.accumulation_steps == 0 or (i + 1) == len(train_loader):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

      running_loss += loss.item() * args.accumulation_steps

    current_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()

    train_loss = running_loss / len(train_loader)
    val_loss, val_dice, val_iou, val_acc, val_prec, val_rec, val_f1 = evaluate(
        model, val_loader, device
    )

    epoch_elapsed = time.time() - epoch_start_time
    epoch_durations.append(epoch_elapsed)

    # Save to CSV log file immediately
    with open(log_csv_path, mode="a", newline="") as f:
      writer = csv.writer(f)
      writer.writerow([
          epoch,
          f"{train_loss:.6f}",
          f"{val_loss:.6f}",
          f"{val_dice:.6f}",
          f"{val_iou:.6f}",
          f"{val_acc:.6f}",
          f"{val_prec:.6f}",
          f"{val_rec:.6f}",
          f"{val_f1:.6f}",
          f"{current_lr:.8f}",
          f"{epoch_elapsed:.2f}",
      ])

    print(
        f"Epoch [{epoch:03d}/{args.epochs:03d}] ({epoch_elapsed:.1f}s) | "
        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
        f"Val Dice: {val_dice:.4f} | IoU: {val_iou:.4f} | Acc: {val_acc:.4f}"
    )

    # Always save latest model checkpoint
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_dice": val_dice,
        },
        last_checkpoint,
    )

    # Save best checkpoint whenever validation dice improves
    if val_dice > best_val_dice:
      best_val_dice = val_dice
      torch.save(model.state_dict(), best_checkpoint)
      print(f"--> Saved Best Model Checkpoint (Val Dice: {best_val_dice:.4f})")

  total_elapsed = time.time() - total_start_time
  avg_epoch_time = sum(epoch_durations) / len(epoch_durations)

  print(f"\n==================== Training Summary ====================")
  print(f"Total Training Time:    {total_elapsed/60:.2f} mins ({total_elapsed:.1f}s)")
  print(f"Average Time / Epoch:   {avg_epoch_time:.2f}s")
  print(f"Best Validation Dice:   {best_val_dice:.4f}")
  print(f"Metrics Logged to:      {log_csv_path}")
  print(f"Best Checkpoint:        {best_checkpoint}")
  print(f"==========================================================\n")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--train-img-dir", required=True)
  parser.add_argument("--train-mask-dir", required=True)
  parser.add_argument(
      "--backbone",
      default="resnet50",
      choices=[
          "resnet18",
          "resnet34",
          "resnet50",
          "mobilenet_v2",
          "efficientnet_b0",
          "densenet121",
      ],
  )
  parser.add_argument("--pretrained-weights", type=str, default=None)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--batch-size", type=int, default=2)
  parser.add_argument("--accumulation-steps", type=int, default=2)
  parser.add_argument("--weight-decay", type=float, default=0.0)
  parser.add_argument("--epochs", type=int, default=60)
  parser.add_argument("--target-size", type=int, nargs=2, default=[512, 512])
  parser.add_argument("--augment", action="store_true")
  parser.add_argument("--augment-prob", type=float, default=0.5)
  parser.add_argument("--augment-factor", type=int, default=2)
  parser.add_argument(
      "--rotation-range", type=float, nargs=2, default=[-15, 15]
  )
  parser.add_argument("--scale-range", type=float, nargs=2, default=[0.9, 1.1])
  parser.add_argument("--num-classes", type=int, default=1)
  parser.add_argument("--pretrained", action="store_true")
  parser.add_argument("--num-workers", type=int, default=4)
  parser.add_argument("--save-dir", default="checkpoints/final")

  train(parser.parse_args())