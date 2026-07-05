"""Training loop for Faster R-CNN detection."""
import csv
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_detection(model, train_dataset, val_dataset=None, config=None):
    cfg = config or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epochs = cfg.get("epochs", 50)
    lr = cfg.get("lr", 0.001)
    batch_size = cfg.get("batch_size", 4)
    checkpoint_path = Path(
        cfg.get("checkpoint_path", "outputs/checkpoints/best_detection.pth")
    )
    history_path = Path(
        cfg.get(
            "history_path",
            f"outputs/logs/{checkpoint_path.stem}_training_history.csv",
        )
    )

    def collate(batch):
        return list(zip(*batch))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=cfg.get("weight_decay", 5e-4))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.get("step_size", 15), gamma=0.1)

    best_loss = float("inf")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "lr"])
        writer.writeheader()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, lr={current_lr:.6f}")

        with open(history_path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "lr"])
            writer.writerow(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "lr": current_lr,
                }
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved best model (loss={best_loss:.4f})")

    print(f"Saved training history to {history_path}")
    return model
