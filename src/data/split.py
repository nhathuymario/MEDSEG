"""Tiện ích chia dữ liệu thành train/val/test."""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def create_splits(image_dir: Path, output_dir: Path, train=0.7, val=0.15, seed=42):
    """Chia file ảnh thành CSV train/val/test.

    Mặc định tỉ lệ là 70% train, 15% validation, 15% test. random_state giúp
    lần chạy sau tạo ra cùng kết quả nếu danh sách file không đổi.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Lấy toàn bộ tên file ảnh và sort để thứ tự đầu vào luôn ổn định.
    files = sorted([f.name for f in image_dir.iterdir() if f.is_file()])

    train_files, temp = train_test_split(files, train_size=train, random_state=seed)
    # temp còn lại gồm val + test; val_ratio là tỉ lệ val bên trong phần temp.
    val_ratio = val / (1 - train)
    val_files, test_files = train_test_split(temp, train_size=val_ratio, random_state=seed)

    for name, split in [("train", train_files), ("val", val_files), ("test", test_files)]:
        df = pd.DataFrame({"filename": split})
        df.to_csv(output_dir / f"{name}.csv", index=False)
        print(f"{name}: {len(split)} files")

    return train_files, val_files, test_files
