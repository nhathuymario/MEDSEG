"""Vòng lặp huấn luyện cho U-Net / Attention U-Net segmentation."""
import csv
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models.components.losses import DiceBCELoss


def dice_score(pred, target, thresh=0.5):
    """Tính Dice score trên batch sau khi ngưỡng hóa xác suất dự đoán."""
    pred = (torch.sigmoid(pred) > thresh).float()
    intersection = (pred * target).sum()
    return (2 * intersection + 1) / (pred.sum() + target.sum() + 1)


def train_segmentation(model, train_dataset, val_dataset=None, config=None):
    """Huấn luyện segmentation model và lưu checkpoint tốt nhất theo val Dice."""
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
    history_path = Path(
        cfg.get(
            "history_path",
            f"outputs/logs/{checkpoint_path.stem}_training_history.csv",
        )
    )
    use_amp = cfg.get("mixed_precision", True) and device.type == "cuda"

    # DataLoader cho segmentation trả về batch tensor ảnh và mask.
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
    # CosineAnnealingLR giảm learning rate mượt theo số epoch.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_dice = 0
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_dice",
                "val_loss",
                "val_dice",
                "lr",
            ],
        )
        writer.writeheader()

    for epoch in range(epochs):
        model.train()
        total_loss, total_dice = 0, 0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            # AMP chỉ bật khi dùng CUDA để tiết kiệm VRAM và tăng tốc train.
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(imgs)
                loss = criterion(pred, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_dice += dice_score(pred, masks).item()

        scheduler.step()
        n = len(train_loader)
        train_loss = total_loss / n
        train_dice = total_dice / n
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}: loss={train_loss:.4f}, dice={train_dice:.4f}")

        # Validation: không cập nhật gradient, chỉ tính loss và Dice.
        val_loss = None
        val_dice = None
        if val_loader:
            model.eval()
            total_val_loss, total_val_dice = 0, 0
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        pred = model(imgs)
                        loss = criterion(pred, masks)
                    total_val_loss += loss.item()
                    total_val_dice += dice_score(pred, masks).item()
            val_loss = total_val_loss / len(val_loader)
            val_dice = total_val_dice / len(val_loader)
            print(f"  Val Dice: {val_dice:.4f}")
            if val_dice > best_dice:
                best_dice = val_dice
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  Saved best model (dice={best_dice:.4f})")

        with open(history_path, "a", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "train_dice",
                    "val_loss",
                    "val_dice",
                    "lr",
                ],
            )
            writer.writerow(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_dice": train_dice,
                    "val_loss": val_loss if val_loss is not None else "",
                    "val_dice": val_dice if val_dice is not None else "",
                    "lr": current_lr,
                }
            )

    print(f"Saved training history to {history_path}")
    return model
