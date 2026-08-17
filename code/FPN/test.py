import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, jaccard_score, roc_auc_score

from seg_head import SegModel
from dataset import MoNuSegDataset, list_image_files
from loss import total_loss


def dice_score(preds, targets, smooth=1e-6):
    preds = preds.astype(np.float32)
    targets = targets.astype(np.float32)
    intersection = np.sum(preds * targets)
    return (2.0 * intersection + smooth) / (np.sum(preds) + np.sum(targets) + smooth)


def evaluate_test_set(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_files = list_image_files(args.test_img_dir)
    test_ds = MoNuSegDataset(args.test_img_dir, args.test_mask_dir, target_size=args.target_size, augment=False, image_list=test_files)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SegModel(backbone_name=args.backbone, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    all_preds, all_masks, all_probs = [], [], []
    dice_list = []
    running_loss = 0.0

    print(f"Evaluating {len(test_files)} test images using checkpoint: {args.checkpoint}")

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = total_loss(preds, masks)
            running_loss += loss.item()

            probs = torch.sigmoid(preds)
            pred_bin = (probs > 0.5).float().cpu().numpy()
            mask_np = masks.cpu().numpy()

            dice_list.append(dice_score(pred_bin, mask_np))
            all_probs.append(probs.cpu().numpy().flatten())
            all_preds.append(pred_bin.astype(np.uint8).flatten())
            all_masks.append(mask_np.astype(np.uint8).flatten())

    all_masks = np.concatenate(all_masks)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    precision = precision_score(all_masks, all_preds, zero_division=0)
    recall = recall_score(all_masks, all_preds, zero_division=0)
    f1 = f1_score(all_masks, all_preds, zero_division=0)
    acc = accuracy_score(all_masks, all_preds)
    iou = jaccard_score(all_masks, all_preds, zero_division=0)
    auc = roc_auc_score(all_masks, all_probs)
    mean_dice = np.mean(dice_list)

    print("\n================== TEST SET EVALUATION RESULTS ==================")
    print(f"Mean Dice Score : {mean_dice:.4f}")
    print(f"Mean IoU (Jaccard): {iou:.4f}")
    print(f"F1-Score         : {f1:.4f}")
    print(f"Precision        : {precision:.4f}")
    print(f"Recall           : {recall:.4f}")
    print(f"Accuracy         : {acc:.4f}")
    print(f"AUC-ROC          : {auc:.4f}")
    print(f"Mean Loss        : {running_loss / len(test_loader):.4f}")
    print("=================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-img-dir", default="code/MonuSeg/MonuSeg/Test/TissueImages")
    parser.add_argument("--test-mask-dir", default="code/MonuSeg/MonuSeg/Test/GroundTruth")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--target-size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    evaluate_test_set(parser.parse_args())