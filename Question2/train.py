"""
train.py - Train UNet for CityScape Image Segmentation
- 80/20 train-test split with seed 42
- 15+ epochs training
- Saves training loss, mIOU, mDice plots
"""

import os
import sys
import glob
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unet_model import UNet

# ============================================================
# Custom Dataset
# ============================================================
class CityscapesDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Read Image
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 96), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32) / 255.0

        # Read Mask
        mask = cv2.imread(self.mask_paths[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (128, 96), interpolation=cv2.INTER_NEAREST)
        mask = np.max(mask, axis=-1)

        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).long()
        return img, mask


# ============================================================
# Metrics
# ============================================================
def compute_miou(preds, targets, num_classes=23):
    """Compute mean Intersection over Union"""
    ious = []
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        if union == 0:
            continue  # Skip classes not present
        ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0


def compute_mdice(preds, targets, num_classes=23):
    """Compute mean Dice score"""
    dices = []
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)
        intersection = np.logical_and(pred_cls, target_cls).sum()
        total = pred_cls.sum() + target_cls.sum()
        if total == 0:
            continue  # Skip classes not present
        dices.append(2 * intersection / total)
    return np.mean(dices) if dices else 0.0


# ============================================================
# Training
# ============================================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # b23cs1075-dlops-major
    data_dir = os.path.join(base_dir, "q2")
    plots_dir = os.path.join(script_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Load image and mask paths
    image_paths = sorted(glob.glob(os.path.join(data_dir, "CameraRGB", "*.png")))
    mask_paths = sorted(glob.glob(os.path.join(data_dir, "CameraMask", "*.png")))
    print(f"Found {len(image_paths)} images and {len(mask_paths)} masks")

    # 80-20 split with seed 42
    train_imgs, test_imgs, train_masks, test_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    print(f"Train: {len(train_imgs)}, Test: {len(test_imgs)}")

    # Save test split info for later use
    split_info = {
        "test_images": test_imgs,
        "test_masks": test_masks
    }
    with open(os.path.join(script_dir, "test_split.json"), "w") as f:
        json.dump(split_info, f)

    # Create datasets and dataloaders
    train_dataset = CityscapesDataset(train_imgs, train_masks)
    test_dataset = CityscapesDataset(test_imgs, test_masks)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Model
    model = UNet(n_channels=3, n_classes=23).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    num_epochs = 25
    train_losses = []
    train_mious = []
    train_mdices = []

    print(f"\nStarting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds)
            all_targets.append(masks)

        scheduler.step()

        # Epoch metrics
        avg_loss = running_loss / len(train_loader)
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        epoch_miou = compute_miou(all_preds, all_targets)
        epoch_mdice = compute_mdice(all_preds, all_targets)

        train_losses.append(avg_loss)
        train_mious.append(epoch_miou)
        train_mdices.append(epoch_mdice)

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} | mIOU: {epoch_miou:.4f} | mDice: {epoch_mdice:.4f}")

    # Save model
    model_path = os.path.join(script_dir, "unet_cityscapes.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    # ============================================================
    # Generate Plots
    # ============================================================
    epochs_range = range(1, num_epochs + 1)

    # Training Loss Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_losses, 'b-o', linewidth=2, markersize=4)
    plt.title('Training Loss Curve', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "training_loss.png"), dpi=150)
    plt.close()

    # mIOU Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_mious, 'g-o', linewidth=2, markersize=4)
    plt.title('Training mIOU Curve', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('mIOU', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "training_miou.png"), dpi=150)
    plt.close()

    # mDice Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_mdices, 'r-o', linewidth=2, markersize=4)
    plt.title('Training mDice Curve', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('mDice', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "training_mdice.png"), dpi=150)
    plt.close()

    print(f"Plots saved to {plots_dir}")

    # ============================================================
    # Evaluate on Test Set
    # ============================================================
    print("\nEvaluating on test set...")
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds)
            all_targets.append(masks)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    test_miou = compute_miou(all_preds, all_targets)
    test_mdice = compute_mdice(all_preds, all_targets)

    print(f"\n{'='*60}")
    print(f"TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"mIOU:  {test_miou:.4f}")
    print(f"mDice: {test_mdice:.4f}")
    print(f"{'='*60}")

    # Save test metrics
    metrics = {
        "test_miou": test_miou,
        "test_mdice": test_mdice,
        "train_losses": train_losses,
        "train_mious": train_mious,
        "train_mdices": train_mdices,
    }
    with open(os.path.join(script_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to metrics.json")

    if test_miou < 0.48 or test_mdice < 0.48:
        print(f"\n⚠️  WARNING: mIOU or mDice is below 0.48! Consider training for more epochs.")


if __name__ == "__main__":
    train()
