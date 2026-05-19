import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    jaccard_score,
    roc_auc_score
)

import numpy as np
import os

from seg_head import SegModel
from dataset import MoNuSegDataset
from loss import total_loss


# =========================
# Hyperparameters
# =========================
epochs = 100
batch = 8


# =========================
# Dice Score Function
# =========================
def dice_score(preds, targets, smooth=1e-6):
    """
    Dice Score between predicted mask and ground truth mask
    """

    preds = preds.astype(np.float32)
    targets = targets.astype(np.float32)

    intersection = np.sum(preds * targets)

    dice = (2.0 * intersection + smooth) / (
        np.sum(preds) + np.sum(targets) + smooth
    )

    return dice


# =========================
# Evaluation Function
# =========================
def evaluate(model, loader, device):

    model.eval()

    running_loss = 0.0

    all_preds_bin = []
    all_masks = []
    all_probs = []

    dice_scores = []

    if len(loader) == 0:
        print("Warning: Loader is empty!")

        return (
            0.0,
            np.array([]),
            np.array([]),
            np.array([]),
            0.0
        )

    with torch.no_grad():

        for imgs, masks in loader:

            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs)

            loss = total_loss(preds, masks)

            running_loss += loss.item()

            # -------------------------
            # Probability Mask
            # -------------------------
            probs = torch.sigmoid(preds)

            # -------------------------
            # Binary Prediction Mask
            # -------------------------
            pred_bin = (probs > 0.5).float()

            # -------------------------
            # Convert to numpy
            # -------------------------
            pred_np = pred_bin.cpu().numpy()
            mask_np = masks.cpu().numpy()

            # -------------------------
            # Dice Score
            # -------------------------
            batch_dice = dice_score(pred_np, mask_np)

            dice_scores.append(batch_dice)

            # -------------------------
            # Flatten for Metrics
            # -------------------------
            prob_flat = probs.cpu().numpy().flatten()

            pred_flat = pred_np.astype(np.uint8).flatten()

            mask_flat = mask_np.astype(np.uint8).flatten()

            # -------------------------
            # Subsample pixels
            # -------------------------
            step = 10

            all_probs.append(prob_flat[::step])
            all_preds_bin.append(pred_flat[::step])
            all_masks.append(mask_flat[::step])

    mean_dice = np.mean(dice_scores)

    return (
        running_loss / len(loader),
        np.concatenate(all_masks),
        np.concatenate(all_preds_bin),
        np.concatenate(all_probs),
        mean_dice
    )


# =========================
# Dataset Paths
# =========================
train_img = "../MonuSeg/MonuSeg/Training/TissueImages"
train_mask = "../MonuSeg/MonuSeg/Training/GroundTruth"

test_img = "../MonuSeg/MonuSeg/Test/TissueImages"
test_mask = "../MonuSeg/MonuSeg/Test/GroundTruth"


# =========================
# Dataset & DataLoader
# =========================
train_ds = MoNuSegDataset(train_img, train_mask)
test_ds = MoNuSegDataset(test_img, test_mask)

train_loader = DataLoader(
    train_ds,
    batch_size=batch,
    shuffle=True
)

test_loader = DataLoader(
    test_ds,
    batch_size=batch,
    shuffle=False
)


# =========================
# Device
# =========================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using Device: {device}")


# =========================
# Model
# =========================
model = SegModel().to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4
)


# =========================
# Create Folders
# =========================
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("Results", exist_ok=True)


# =========================
# History
# =========================
train_history = []
test_history = []
dice_history = []


# =========================
# Training
# =========================
print("\nStarting Training...\n")

for epoch in range(epochs):

    model.train()

    running_train_loss = 0.0

    for imgs, masks in train_loader:

        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        preds = model(imgs)

        loss = total_loss(preds, masks)

        loss.backward()

        optimizer.step()

        running_train_loss += loss.item()

    # -------------------------
    # Average Train Loss
    # -------------------------
    avg_train_loss = running_train_loss / len(train_loader)

    # -------------------------
    # Evaluation
    # -------------------------
    (
        avg_test_loss,
        y_true,
        y_pred,
        y_prob,
        avg_dice
    ) = evaluate(model, test_loader, device)

    # -------------------------
    # Store History
    # -------------------------
    train_history.append(avg_train_loss)

    test_history.append(avg_test_loss)

    dice_history.append(avg_dice)

    # -------------------------
    # Save Checkpoint
    # -------------------------
    # torch.save(
    #     model.state_dict(),
    #     f"checkpoints/fpn_epoch_{epoch+1}.pth"
    # )

    # -------------------------
    # Print Progress
    # -------------------------
    print(
        f"Epoch [{epoch+1}/{epochs}] | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Test Loss: {avg_test_loss:.4f} | "
        f"Dice Score: {avg_dice:.4f}"
    )


# =========================
# Save Final Model
# =========================
torch.save(
    model.state_dict(),
    "checkpoints/fpn_monuseg_final.pth"
)


# =========================
# Plot Loss Curve
# =========================
plt.figure(figsize=(10, 5))

plt.plot(train_history, label="Train Loss")

plt.plot(test_history, label="Test Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.title("Training vs Test Loss")

plt.legend()

plt.savefig("Results/loss_curve.png")

plt.close()


# =========================
# Plot Dice Curve
# =========================
plt.figure(figsize=(10, 5))

plt.plot(dice_history, label="Dice Score")

plt.xlabel("Epochs")
plt.ylabel("Dice Score")

plt.title("Dice Score vs Epochs")

plt.legend()

plt.savefig("Results/dice_curve.png")

plt.close()


# =========================
# Confusion Matrix
# =========================
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Background", "Nuclei"]
)

disp.plot(cmap=plt.cm.Blues)

plt.title("Pixel-level Confusion Matrix")

plt.savefig("Results/confusion_matrix.png")

plt.close()


# =========================
# Final Metrics
# =========================
precision = precision_score(y_true, y_pred)

recall = recall_score(y_true, y_pred)

f1 = f1_score(y_true, y_pred)

accuracy = accuracy_score(y_true, y_pred)

iou = jaccard_score(y_true, y_pred)

auc = roc_auc_score(y_true, y_prob)


# =========================
# Print Final Metrics
# =========================
print("\n" + "=" * 40)

print(" FINAL EVALUATION METRICS ")

print("=" * 40)

print(f"Accuracy     : {accuracy:.4f}")

print(f"Precision    : {precision:.4f}")

print(f"Recall       : {recall:.4f}")

print(f"F1-Score     : {f1:.4f}")

print(f"Dice Score   : {avg_dice:.4f}")

print(f"IoU          : {iou:.4f}")

print(f"AUC          : {auc:.4f}")

print("=" * 40)

print("\nTraining Complete!")

print("Saved Files:")
print("1. Results/loss_curve.png")
print("2. Results/dice_curve.png")
print("3. Results/confusion_matrix.png")
print("4. checkpoints/")