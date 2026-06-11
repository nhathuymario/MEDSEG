"""Training loop for U-Net / Attention U-Net segmentation."""
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models.components.losses import DiceBCELoss


def dice_score(pred, target, thresh=0.5):
    pred = (torch.sigmoid(pred) > thresh).float()
    intersection = (pred * target).sum()
    return (2 * intersection + 1) / (pred.sum() + target.sum() + 1)


def train_segmentation(model, train_dataset, val_dataset=None, config=None):
    cfg = config or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epochs = cfg.get("epochs", 100)
    lr = cfg.get("lr", 1e-4)
    batch_size = cfg.get("batch_size", 8)
    num_workers = cfg.get("num_workers", 0)
    checkpoint_path = Path(
        cfg.get(
            "checkpoint_path",
            "outputs/checkpoints/best_segmentation.pth",
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        if val_dataset
        else None
    )

    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_dice = 0
    for epoch in range(epochs):
        model.train()
        total_loss, total_dice = 0, 0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)
            pred = model(imgs)
            loss = criterion(pred, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_dice += dice_score(pred, masks).item()

        scheduler.step()
        n = len(train_loader)
        print(f"Epoch {epoch+1}: loss={total_loss/n:.4f}, dice={total_dice/n:.4f}")

        # Validation
        if val_loader:
            model.eval()
            val_dice = 0
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    pred = model(imgs)
                    val_dice += dice_score(pred, masks).item()
            val_dice /= len(val_loader)
            print(f"  Val Dice: {val_dice:.4f}")
            if val_dice > best_dice:
                best_dice = val_dice
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  Saved best model (dice={best_dice:.4f})")

    return model
