import numpy as np
from src.data.transforms import get_train_transforms

def test_train_transforms():
    tfm = get_train_transforms()
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8)

    res = tfm(image=img, mask=mask)
    assert res["image"].shape == (3, 256, 256)
    assert res["mask"].shape == (256, 256)  # before unsqueeze in dataset
