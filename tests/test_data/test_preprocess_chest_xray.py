import cv2
import numpy as np

from src.data.preprocess import prepare_chest_xray


def test_prepare_chest_xray_combines_lung_masks(tmp_path):
    raw_dir = tmp_path / "raw"
    image_dir = raw_dir / "CXR_png"
    left_dir = raw_dir / "ManualMask" / "leftMask"
    right_dir = raw_dir / "ManualMask" / "rightMask"
    for directory in (image_dir, left_dir, right_dir):
        directory.mkdir(parents=True)

    image = np.full((20, 20), 100, dtype=np.uint8)
    left_mask = np.zeros((20, 20), dtype=np.uint8)
    right_mask = np.zeros((20, 20), dtype=np.uint8)
    left_mask[2:10, 2:8] = 255
    right_mask[2:10, 12:18] = 255

    cv2.imwrite(str(image_dir / "sample.png"), image)
    cv2.imwrite(str(left_dir / "sample.png"), left_mask)
    cv2.imwrite(str(right_dir / "sample.png"), right_mask)

    output_dir = tmp_path / "processed"
    assert prepare_chest_xray(raw_dir, output_dir, size=(16, 16)) == 1

    output_image = cv2.imread(str(output_dir / "images" / "sample.png"))
    output_mask = cv2.imread(
        str(output_dir / "masks" / "sample.png"),
        cv2.IMREAD_GRAYSCALE,
    )
    assert output_image.shape == (16, 16, 3)
    assert output_mask.shape == (16, 16)
    assert set(np.unique(output_mask)) == {0, 255}
    assert output_mask[:, :8].max() == 255
    assert output_mask[:, 8:].max() == 255
