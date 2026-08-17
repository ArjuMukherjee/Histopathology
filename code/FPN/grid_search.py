import argparse
import csv
import os
import sys
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
    roc_auc_score,
)
from sklearn.model_selection import KFold, ParameterGrid
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from seg_head import SegModel
from dataset import MoNuSegDataset, list_image_files, split_image_list
from loss import total_loss


def dice_score(preds, targets, smooth=1e-6):
    preds = preds.astype(np.float32)
    targets = targets.astype(np.float32)
    intersection = np.sum(preds * targets)
    return (2.0 * intersection + smooth) / (np.sum(preds) + np.sum(targets) + smooth)


def evaluate(model, loader, device):
    model.eval()
    running_loss = 0.0
    all_preds_bin, all_masks, all_probs, dice_scores = [], [], [], []

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = total_loss(preds, masks)
            running_loss += loss.item()

            probs = torch.sigmoid(preds)
            pred_bin = (probs > 0.5).float()
            pred_np = pred_bin.cpu().numpy()
            mask_np = masks.cpu().numpy()

            dice_scores.append(dice_score(pred_np, mask_np))
            all_probs.append(probs.cpu().numpy().flatten())
            all_preds_bin.append(pred_np.astype(np.uint8).flatten())
            all_masks.append(mask_np.astype(np.uint8).flatten())

    all_masks = np.concatenate(all_masks)
    all_preds_bin = np.concatenate(all_preds_bin)
    all_probs = np.concatenate(all_probs)

    return running_loss / len(loader), all_masks, all_preds_bin, all_probs, np.mean(dice_scores)


def compute_metrics(y_true, y_pred, y_prob):
    try:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        iou = jaccard_score(y_true, y_pred, zero_division=0)
    except ValueError:
        precision = recall = f1 = accuracy = iou = float("nan")

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    return precision, recall, f1, accuracy, iou, auc


def train_one_epoch(model, loader, optimizer, accumulation_steps, device):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    for i, (imgs, masks) in enumerate(loader):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = total_loss(preds, masks) / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps

    return running_loss / len(loader)


def measure_inference_time(model, loader, device, warmup_batches=2):
    model.eval()
    total_time = 0.0
    total_images = 0

    with torch.no_grad():
        for idx, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device)
            if idx < warmup_batches:
                _ = model(imgs)
                continue
            start = time.time()
            _ = model(imgs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.time() - start
            total_images += imgs.shape[0]

    return total_time / total_images if total_images > 0 else float("nan")


def run_grid_search(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_files = list_image_files(args.train_img_dir)
    train_list, val_list, _ = split_image_list(all_files, test_size=args.test_size, val_size=args.val_size, random_state=args.random_state)

    grid = list(ParameterGrid({
        "backbone": args.backbones,
        "lr": args.learning_rates,
        "batch_size": args.batch_sizes,
        "weight_decay": args.weight_decays,
    }))

    summary_path = os.path.join(args.save_dir, "Results")
    os.makedirs(summary_path, exist_ok=True)
    summary_file = os.path.join(summary_path, "grid_search_summary.csv")

    fieldnames = [
        "backbone", "lr", "batch_size", "weight_decay", "fold",
        "train_loss", "val_loss", "val_dice", "precision", "recall",
        "f1", "accuracy", "iou", "auc", "train_time_s", "pred_time_s_per_image", "total_images"
    ]

    with open(summary_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        best_combo, best_mean_dice = None, -1.0

        for combo in grid:
            print(f"\nEvaluating: {combo}")
            fold_dices, fold_train_times, fold_pred_times = [], [], []

            if args.use_kfold:
                kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.random_state)
                image_arr = np.array(all_files)

                for fold_idx, (t_idx, v_idx) in enumerate(kf.split(image_arr), start=1):
                    train_ds = MoNuSegDataset(args.train_img_dir, args.train_mask_dir, target_size=args.target_size, augment=args.augment, image_list=image_arr[t_idx].tolist())
                    val_ds = MoNuSegDataset(args.train_img_dir, args.train_mask_dir, target_size=args.target_size, augment=False, image_list=image_arr[v_idx].tolist())

                    train_loader = DataLoader(train_ds, batch_size=combo["batch_size"], shuffle=True, num_workers=args.num_workers, pin_memory=True)
                    val_loader = DataLoader(val_ds, batch_size=combo["batch_size"], shuffle=False, num_workers=args.num_workers, pin_memory=True)

                    model = SegModel(backbone_name=combo["backbone"], pretrained=args.pretrained, num_classes=args.num_classes).to(device)
                    optimizer = optim.Adam(model.parameters(), lr=combo["lr"], weight_decay=combo["weight_decay"])

                    fold_start = time.time()
                    for _ in range(args.epochs):
                        train_one_epoch(model, train_loader, optimizer, args.accumulation_steps, device)
                    fold_train_time = time.time() - fold_start

                    val_loss, y_true, y_pred, y_prob, val_dice = evaluate(model, val_loader, device)
                    pred_time = measure_inference_time(model, val_loader, device)
                    precision, recall, f1, accuracy, iou, auc = compute_metrics(y_true, y_pred, y_prob)

                    fold_train_times.append(fold_train_time)
                    fold_pred_times.append(pred_time)
                    fold_dices.append(val_dice)

                    writer.writerow({
                        "backbone": combo["backbone"], "lr": combo["lr"], "batch_size": combo["batch_size"],
                        "weight_decay": combo["weight_decay"], "fold": fold_idx, "train_loss": "",
                        "val_loss": f"{val_loss:.6f}", "val_dice": f"{val_dice:.6f}", "precision": f"{precision:.6f}",
                        "recall": f"{recall:.6f}", "f1": f"{f1:.6f}", "accuracy": f"{accuracy:.6f}",
                        "iou": f"{iou:.6f}", "auc": f"{auc:.6f}", "train_time_s": f"{fold_train_time:.4f}",
                        "pred_time_s_per_image": f"{pred_time:.6f}", "total_images": len(v_idx),
                    })

            mean_dice = np.mean(fold_dices)
            print(f"Result: mean_dice={mean_dice:.4f} | train_time={np.mean(fold_train_times):.1f}s | pred_time={np.mean(fold_pred_times)*1000:.2f}ms")

            if mean_dice > best_mean_dice:
                best_mean_dice = mean_dice
                best_combo = {**combo, "mean_dice": mean_dice}

    print(f"\nBest Config Found: {best_combo}")
    with open(os.path.join(summary_path, "best_combo.txt"), "w") as f:
        f.write(str(best_combo) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-img-dir", required=True)
    parser.add_argument("--train-mask-dir", required=True)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--use-kfold", action="store_true")
    parser.add_argument("--backbones", nargs="+", default=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[1e-4, 3e-4])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--weight-decays", nargs="+", type=float, default=[0.0, 1e-4])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--target-size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-dir", default=".")

    run_grid_search(parser.parse_args())