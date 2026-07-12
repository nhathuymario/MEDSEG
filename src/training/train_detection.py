"""Vòng lặp huấn luyện Faster R-CNN detection."""
import csv
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.evaluation.metrics_detection import compute_map


def detection_collate(batch):
    """Torchvision detection nhận list ảnh và list target, không phải tensor batch."""
    return list(zip(*batch))


def train_detection(model, train_dataset, val_dataset=None, config=None):
    """Huấn luyện detector và lưu checkpoint tốt nhất theo val mAP nếu có."""
    cfg = config or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epochs = cfg.get("epochs", 50)
    lr = cfg.get("lr", 0.001)
    batch_size = cfg.get("batch_size", 4)
    num_workers = cfg.get("num_workers", 0)
    iou_threshold = cfg.get("iou_threshold", 0.5)
    checkpoint_path = Path(
        cfg.get("checkpoint_path", "outputs/checkpoints/best_detection.pth")
    )
    history_path = Path(
        cfg.get(
            "history_path",
            f"outputs/logs/{checkpoint_path.stem}_training_history.csv",
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=detection_collate,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=detection_collate,
        )
        if val_dataset
        else None
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=cfg.get("weight_decay", 5e-4))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.get("step_size", 15), gamma=0.1)

    best_loss = float("inf")
    best_map = -1.0
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "val_map", "lr"],
        )
        writer.writeheader()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Khi truyền targets, Faster R-CNN trả về dict các loss thành phần.
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            # Clip gradient để tránh bước update quá lớn khi loss dao động.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        val_map = None
        if val_loader:
            model.eval()
            predictions = []
            ground_truths = []
            with torch.no_grad():
                for images, targets in val_loader:
                    images = [img.to(device) for img in images]
                    outputs = model(images)
                    for output, target in zip(outputs, targets):
                        predictions.append(
                            {
                                "boxes": output["boxes"].cpu().numpy().tolist(),
                                "scores": output["scores"].cpu().numpy().tolist(),
                            }
                        )
                        ground_truths.append(
                            {"boxes": target["boxes"].cpu().numpy().tolist()}
                        )
            val_map = compute_map(
                predictions,
                ground_truths,
                iou_threshold=iou_threshold,
            )

        message = f"Epoch {epoch+1}: loss={avg_loss:.4f}, lr={current_lr:.6f}"
        if val_map is not None:
            message += f", val_mAP@{iou_threshold}={val_map:.4f}"
        print(message)

        with open(history_path, "a", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["epoch", "train_loss", "val_map", "lr"],
            )
            writer.writerow(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "val_map": val_map if val_map is not None else "",
                    "lr": current_lr,
                }
            )

        improved = False
        if val_map is not None and val_map > best_map:
            best_map = val_map
            improved = True
        elif val_map is None and avg_loss < best_loss:
            best_loss = avg_loss
            improved = True

        if improved:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            if val_map is not None:
                print(f"  Saved best model (val_mAP={best_map:.4f})")
            else:
                print(f"  Saved best model (loss={best_loss:.4f})")

    print(f"Saved training history to {history_path}")
    return model
