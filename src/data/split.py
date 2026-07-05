"""Train/val/test split utilities."""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Chia dữ liệu thành các tập train, val, test theo tỷ lệ 70%, 15%, 15%
# Seed được đặt để đảm bảo kết quả có thể tái tạo
def create_splits(image_dir: Path, output_dir: Path, train=0.7, val=0.15, seed=42):
    """Create stratified splits and save as CSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Lấy toàn bộ tên file ảnh trong data\processed\images
    # Sắp xếp tên file bằng sorted() để đảm bảo thứ tự nhất quán
    files = sorted([f.name for f in image_dir.iterdir() if f.is_file()])

    train_files, temp = train_test_split(files, train_size=train, random_state=seed)
    val_ratio = val / (1 - train)
    val_files, test_files = train_test_split(temp, train_size=val_ratio, random_state=seed)

    for name, split in [("train", train_files), ("val", val_files), ("test", test_files)]:
        df = pd.DataFrame({"filename": split})
        df.to_csv(output_dir / f"{name}.csv", index=False)
        print(f"{name}: {len(split)} files")

    return train_files, val_files, test_files
