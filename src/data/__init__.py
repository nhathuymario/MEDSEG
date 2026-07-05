from .dataset_detection import DetectionDataset
from .dataset_segmentation import SegmentationDataset
from .transforms import (
    get_chest_xray_train_transforms,
    get_train_transforms,
    get_val_transforms,
)
from .split import create_splits


def download_isic2018(*args, **kwargs):
    from .download import download_isic2018 as download

    return download(*args, **kwargs)


def download_chest_xray(*args, **kwargs):
    from .download import download_chest_xray as download

    return download(*args, **kwargs)


def prepare_chest_xray(*args, **kwargs):
    from .preprocess import prepare_chest_xray as prepare

    return prepare(*args, **kwargs)
